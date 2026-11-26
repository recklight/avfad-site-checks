"""Training with both places' room noise mixed in: does the shortcut shrink? Exploratory.

Definitions and readings are in experiments/deviations.md, written before any noisy copy was
made. Each speaker's /a/ gets one copy with an HSA room and one with an ESSUA room, at a level
drawn within 6 dB of the intervention's 0 dB. Copies only ever train; tests are clean files.

    copies    handcrafted features of the two noisy copies of every /a/ file
    run       C1, everyone and the intervention with the copies in every training fold
    collect   comparisons with the plain detector; writes noise_augmentation/summary.json
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from voice_severity_bench.evaluation import class_scores  # noqa: E402
from voice_severity_bench.models import fit_model  # noqa: E402
from voice_severity_bench.splitting import repeated_split_indices  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "features"))

import exploratory_avfad as ex  # noqa: E402
import run_first_batch as batch  # noqa: E402
from arms import write_json  # noqa: E402
from common import load_signal  # noqa: E402
from extract_handcrafted import FEATURE_COLUMNS, features_of  # noqa: E402
from freeze import verify_freeze  # noqa: E402
from freeze_svd import verify_svd_freeze  # noqa: E402
from harness import ROOT, Arm, load_plan, make_config  # noqa: E402

COPY_PLACES = ("HSA", "ESSUA")
LEVEL_SPREAD_DB = 6.0
AUG_SEED = 20261112
EXPLORATORY = ROOT / "features" / "out_exploratory"
COPIES = EXPLORATORY / "avfad_a_augmented.csv"
RESULTS = HERE / "results" / "noise_augmentation"
S1_OUT = ROOT / "s1_segments" / "out"
MAIN_CASE = "ESSUA_with_HSA_+0"

# Fixed before any result; see deviations.md.
FLIP_FRACTION = 0.5
GAP_SHRINK = 0.05
II_TOLERANCE = 0.02
II_BETTER = 0.05


# ------------------------------------------------------------------ copies


def _copy_row(file_id, wav, room, target_db, offset_db, place, seed_key) -> dict:
    x, rate = load_signal(wav)
    pieces, level = ex.room_quiet(*room)
    rng = np.random.default_rng([AUG_SEED, *seed_key])
    total = int(sum(p.size for p in pieces))
    noise = ex.loop_to_length(pieces, x.size, int(rng.integers(total)), rate)
    noise = noise * 10.0 ** ((target_db + offset_db - level) / 20.0)
    return {
        "file_id": file_id,
        "copy": place,
        "room_file": room[0],
        "offset_db": offset_db,
        **dict(zip(FEATURE_COLUMNS, features_of([x + noise], rate), strict=True)),
    }


def copy_plan(
    file_ids: list[str], room_counts: dict[str, int]
) -> list[tuple[str, str, int, float]]:
    """(file, place, room index, level offset) of every copy; fixed by the seed alone."""
    rng = np.random.default_rng(AUG_SEED)
    plan = []
    for file_id in file_ids:
        for place in COPY_PLACES:
            room = int(rng.integers(room_counts[place]))
            offset = float(rng.uniform(-LEVEL_SPREAD_DB, LEVEL_SPREAD_DB))
            plan.append((file_id, place, room, offset))
    return plan


def run_copies(jobs: int) -> None:
    speakers = batch.load_speakers().sort_values("file_id")
    files = pd.read_csv(S1_OUT / "files.csv").set_index("file_id")
    spans = pd.read_csv(S1_OUT / "spans.csv")
    quiet = spans[spans["kind"] == "nonphonation_g100"]
    target = ex.hsa_target_db(speakers)
    rooms = {}
    for place in COPY_PLACES:
        rooms[place] = [
            (
                row.file_id,
                row.wav,
                list(
                    zip(
                        quiet.loc[quiet["file_id"] == row.file_id, "start_s"],
                        quiet.loc[quiet["file_id"] == row.file_id, "end_s"],
                        strict=True,
                    )
                ),
            )
            for row in ex.noise_files(place).itertuples()
        ]
    plan = copy_plan(speakers["file_id"].tolist(), {p: len(r) for p, r in rooms.items()})
    started = time.perf_counter()
    rows = Parallel(n_jobs=jobs)(
        delayed(_copy_row)(
            file_id,
            files.loc[file_id, "wav"],
            rooms[place][room],
            target,
            offset,
            place,
            (i, COPY_PLACES.index(place)),
        )
        for i, (file_id, place, room, offset) in enumerate(plan)
    )
    EXPLORATORY.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(COPIES, index=False, lineterminator="\n")
    write_json(
        RESULTS / "copies.json",
        {
            "target_db": target,
            "copies": len(rows),
            "rooms": {p: len(r) for p, r in rooms.items()},
            "rooms_used": {p: len({r for _, q, r, _ in plan if q == p}) for p in COPY_PLACES},
            "seconds": round(time.perf_counter() - started),
        },
    )
    print(len(rows), "copies in", round(time.perf_counter() - started), "s")


# ------------------------------------------------------------------ training with copies


def stacked(table: pd.DataFrame, clean: pd.DataFrame, copies: pd.DataFrame):
    """Rows: the clean files of ``table``, then each place's copies in the same order."""
    order = table["file_id"].tolist()
    blocks = [clean.set_index("file_id").loc[order, FEATURE_COLUMNS].to_numpy()]
    for place in COPY_PLACES:
        part = copies[copies["copy"] == place].set_index("file_id")
        blocks.append(part.loc[order, FEATURE_COLUMNS].to_numpy())
    X = np.vstack(blocks)
    y = np.tile(table["label"].to_numpy().astype(str), len(blocks))
    groups = np.tile(table["speaker"].to_numpy().astype(str), len(blocks))
    return X, y, groups


def training_rows(train: np.ndarray, n: int, blocks: int) -> np.ndarray:
    """A fold's training speakers in every block: clean and both copies."""
    return np.concatenate([train + k * n for k in range(blocks)])


def _fold(X, y, groups, n, train, test, config, classes, cases) -> dict:
    rows = training_rows(train, n, X.shape[0] // n)
    model = fit_model(X[rows], y[rows], config, groups=groups[rows])
    out = {
        "test": test,
        "predicted": np.asarray(model.predict(X[test])).astype(str),
        "scores": class_scores(model, X[test], classes)[:, classes.index("pathological")],
        "cases": [],
    }
    tested = set(groups[test].tolist())
    for name, ids, Z in cases:
        keep = np.array([g in tested for g in ids])
        if keep.any():
            flags = np.asarray(model.predict(Z[keep])).astype(str) == "pathological"
            out["cases"].append({"case": name, "speaker": ids[keep], "flag": flags})
    return out


def arm_with_copies(name, table, clean, copies, config, cases=(), jobs=1):
    table = table.sort_values("file_id").reset_index(drop=True)
    X, y, groups = stacked(table, clean, copies)
    n = len(table)
    y_clean, g_clean = y[:n], groups[:n]
    splits = repeated_split_indices(
        y_clean,
        g_clean,
        task="classification",
        n_splits=config.evaluation.n_splits,
        n_repeats=config.evaluation.n_repeats,
        seed=config.seed,
    )
    classes = tuple(sorted(np.unique(y_clean).tolist()))
    outputs = Parallel(n_jobs=jobs)(
        delayed(_fold)(X, y, groups, n, train, test, config, classes, cases)
        for folds in splits
        for train, test in folds
    )
    repeats = len(splits)
    predicted = np.empty((n, repeats), dtype=object)
    scores = np.full((n, repeats), np.nan)
    per_fold = len(splits[0])
    for index, output in enumerate(outputs):
        r = index // per_fold
        predicted[output["test"], r] = output["predicted"]
        scores[output["test"], r] = output["scores"]
    arm = scored_arm(name, y_clean, g_clean, predicted.astype(str), scores)
    return arm, [case for output in outputs for case in output["cases"]]


def scored_arm(name, y, groups, predicted, scores) -> Arm:
    from voice_severity_bench.evaluation import classification_metrics

    per_repeat = [
        classification_metrics(
            y,
            predicted[:, r],
            np.column_stack([1 - scores[:, r], scores[:, r]]),
            classes=("normal", "pathological"),
            positive_class="pathological",
        )
        for r in range(predicted.shape[1])
    ]
    mean = {k: float(np.mean([m[k] for m in per_repeat])) for k in per_repeat[0]}
    return Arm(
        name=name,
        y_true=y,
        groups=groups,
        predictions=predicted,
        scores=scores,
        uar=mean["uar"],
        roc_auc=mean["roc_auc"],
        sensitivity=mean["sensitivity"],
        specificity=mean["specificity"],
        worst_share_deviation=float("nan"),
        chosen_c={},
    )


def intervention_cases(table: pd.DataFrame) -> list:
    """The clean normal speakers and the intervention's noisy tables, keyed by speaker."""
    lookup = table.set_index("file_id")["speaker"]
    cases = []
    for name, place in (("clean_ESSUA", "ESSUA"), ("clean_HSA", "HSA")):
        part = table[(table["place"] == place) & (table["label"] == "normal")]
        cases.append(
            (name, part["speaker"].to_numpy().astype(str), part[FEATURE_COLUMNS].to_numpy())
        )
    for people, room in (("ESSUA", "HSA"), ("HSA", "HSA"), ("ESSUA", "ESSUA")):
        for dose in ex.DOSES_DB:
            noisy = pd.read_csv(ex.NOISE_TABLES / f"{people}_with_{room}_noise_{dose:+d}.csv")
            cases.append(
                (
                    f"{people}_with_{room}_{dose:+d}",
                    noisy["file_id"].map(lookup).to_numpy().astype(str),
                    noisy[FEATURE_COLUMNS].to_numpy(),
                )
            )
    return cases


def run_arms(jobs: int) -> None:
    plan = load_plan()
    config = make_config(plan, batch.LABEL_POSITIVE)
    speakers = batch.load_speakers()
    clean = batch.load_features("full_original")
    copies = pd.read_csv(COPIES)
    everyone = speakers.merge(clean[["file_id", *FEATURE_COLUMNS]], on="file_id", how="left")
    hsa = speakers["place"] == "HSA"
    patients = speakers[hsa & (speakers["label"] == "pathological")]
    hsa_normals = speakers[hsa & (speakers["label"] == "normal")]
    draws = pd.read_csv(HERE / "results" / "first_batch" / "c1_draws.csv", dtype={"speaker": str})
    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    every_arm, case_rows = arm_with_copies(
        "all", everyone, clean, copies, config, intervention_cases(everyone), jobs
    )
    arms = {
        "all": every_arm,
        "ii": arm_with_copies(
            "ii", pd.concat([patients, hsa_normals]), clean, copies, config, (), jobs
        )[0],
    }
    for index, drawn in draws.groupby("draw"):
        chosen = speakers[speakers["speaker"].isin(drawn["speaker"])]
        arms[f"i_{index:02d}"] = arm_with_copies(
            f"i_{index:02d}", pd.concat([patients, chosen]), clean, copies, config, (), jobs
        )[0]
    print("arms in", round(time.perf_counter() - started), "s", flush=True)
    rows = []
    for arm in arms.values():
        n, repeats = arm.predictions.shape
        rows.append(
            pd.DataFrame(
                {
                    "arm": arm.name,
                    "speaker": np.repeat(arm.groups, repeats),
                    "label": np.repeat(arm.y_true, repeats),
                    "repeat": np.tile(np.arange(repeats), n),
                    "predicted": arm.predictions.ravel(),
                    "score": arm.scores.ravel(),
                }
            )
        )
    pd.concat(rows).to_csv(RESULTS / "predictions.csv.gz", index=False)
    pd.concat([pd.DataFrame(row) for row in case_rows], ignore_index=True).to_csv(
        RESULTS / "intervention_predictions.csv.gz", index=False
    )


# ------------------------------------------------------------------ reading


def arms_from(path: Path) -> dict[str, Arm]:
    table = pd.read_csv(path, dtype={"speaker": str})
    arms = {}
    for name, part in table.groupby("arm"):
        wide_p = part.pivot(index="speaker", columns="repeat", values="predicted")
        wide_s = part.pivot(index="speaker", columns="repeat", values="score")
        labels = part.drop_duplicates("speaker").set_index("speaker").loc[wide_p.index, "label"]
        arms[name] = scored_arm(
            name,
            labels.to_numpy().astype(str),
            wide_p.index.to_numpy().astype(str),
            wide_p.to_numpy().astype(str),
            wide_s.to_numpy(),
        )
    return arms


def flips(path: Path, people: str, case: str) -> pd.Series:
    long = pd.read_csv(path, dtype={"speaker": str})
    means = long.groupby(["case", "speaker"])["flag"].mean().unstack("case")
    return (means[case] - means[f"clean_{people}"]).dropna()


def paired_mean_ci(values: np.ndarray, seed: int, n: int = 10_000) -> list[float]:
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(values.size, size=values.size)].mean() for _ in range(n)]
    return [float(v) for v in np.quantile(means, [0.025, 0.975])]


def reading(flip_aug, flip_base, flip_ci, gap_change, gap_change_ci, ii_change) -> dict:
    halved = bool(flip_aug < FLIP_FRACTION * flip_base and flip_ci[1] < 0)
    shrank = bool(gap_change <= -GAP_SHRINK and gap_change_ci[1] < 0)
    kept = bool(ii_change >= -II_TOLERANCE)
    if halved and shrank and kept:
        verdict = "works"
    elif halved or shrank:
        verdict = "partly"
    else:
        verdict = "does not work"
    return {
        "flips_halved": halved,
        "gap_shrank": shrank,
        "ii_not_worse": kept,
        "ii_better": bool(ii_change >= II_BETTER),
        "verdict": verdict,
    }


def run_collect() -> None:
    plan = load_plan()
    seed = plan["statistics"]["seeds"]["bootstrap"]
    n = plan["confirmatory_family"]["tests"]["C1"]["test"]["resamples"]
    aug = arms_from(RESULTS / "predictions.csv.gz")
    plain_all = json.loads(
        (HERE / "results" / "exploratory" / "avfad_all.json").read_text(encoding="utf-8")
    )["all_handcrafted"]
    base = arms_from(HERE / "results" / "first_batch" / "c1_predictions.csv.gz")
    aug_i = [a for k, a in aug.items() if k.startswith("i_")]
    base_i = [a for k, a in base.items() if k.startswith("i_")]
    gap_aug = batch.c1_bootstrap(aug_i, aug["ii"], n, seed)
    gap_base = batch.c1_bootstrap(base_i, base["ii"], n, seed)
    samples_change = gap_aug.pop("samples") - gap_base.pop("samples")
    finite = samples_change[np.isfinite(samples_change)]
    ii_change = batch.c1_bootstrap([aug["ii"]], base["ii"], n, seed)
    ii_change.pop("samples")
    flip_aug = flips(RESULTS / "intervention_predictions.csv.gz", "ESSUA", MAIN_CASE)
    flip_base = flips(
        HERE / "results" / "exploratory" / "intervention_predictions.csv.gz", "ESSUA", MAIN_CASE
    )
    common = flip_aug.index.intersection(flip_base.index)
    change = (flip_aug.loc[common] - flip_base.loc[common]).to_numpy(dtype=float)
    flip_ci = paired_mean_ci(change, seed)
    report = {
        "all": {"augmented": {"uar": aug["all"].uar, "roc_auc": aug["all"].roc_auc}},
        "ii": {"augmented": aug["ii"].uar, "plain": base["ii"].uar, "change": ii_change},
        "i": {
            "augmented_mean": float(np.mean([a.uar for a in aug_i])),
            "plain_mean": float(np.mean([a.uar for a in base_i])),
        },
        "gap": {
            "augmented": gap_aug,
            "plain": gap_base,
            "change": float(gap_aug["difference"] - gap_base["difference"]),
            "change_ci": [float(v) for v in np.quantile(finite, [0.025, 0.975])],
        },
        "intervention": {
            "case": MAIN_CASE,
            "augmented": float(flip_aug.mean()),
            "plain": float(flip_base.mean()),
            "speakers": int(common.size),
            "change": float(change.mean()),
            "change_ci": flip_ci,
        },
    }
    # No per-speaker predictions were kept for the plain detector on everyone; the summary only.
    report["all"]["plain"] = {"uar": plain_all["uar"], "roc_auc": plain_all["roc_auc"]}
    report["reading"] = reading(
        report["intervention"]["augmented"],
        report["intervention"]["plain"],
        flip_ci,
        report["gap"]["change"],
        report["gap"]["change_ci"],
        report["ii"]["augmented"] - report["ii"]["plain"],
    )
    write_json(RESULTS / "summary.json", report)
    print(json.dumps(report, indent=1)[:3000])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("copies", "run", "collect"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    verify_freeze()
    verify_svd_freeze()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.step == "copies":
        run_copies(args.jobs)
    elif args.step == "run":
        run_arms(args.jobs)
    else:
        run_collect()


if __name__ == "__main__":
    main()
