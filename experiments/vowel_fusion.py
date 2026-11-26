"""C1 on AVFAD /i/ and /u/ with frozen HuBERT, and the three vowels averaged per speaker.

Exploratory. Definitions and readings are in experiments/deviations.md (2026-11-10), written
before any /i/ or /u/ score inside AVFAD was seen. /a/ is not refitted: its arms come from
results/sensitivity/hubert/c1_predictions.csv.gz. Only the shuffled-label arms of /a/ run
again, because that run kept their UAR and not their scores.

    run       the (ii) arm, the 50 draws of (i) and 5 label shuffles of (ii) per vowel
    collect   C1 per vowel and for the average; writes results/vowel_fusion/summary.json

    .venv/Scripts/python.exe -I -X utf8 -B experiments/vowel_fusion.py run --jobs 10
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
from voice_severity_bench.bootstrap import group_weights, weighted_metric  # noqa: E402
from voice_severity_bench.splitting import repeated_split_indices  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_first_batch as batch  # noqa: E402
from arms import evaluate_layers, load_hubert, merge_features, write_json  # noqa: E402
from freeze import verify_freeze  # noqa: E402
from harness import Arm, load_plan, make_config, match_cells, predictions_table  # noqa: E402
from sensitivity_avfad import CHOSEN_LAYERS, hsa_groups  # noqa: E402

RESULTS = HERE / "results" / "vowel_fusion"
STORED_A = HERE / "results" / "sensitivity" / "hubert" / "c1_predictions.csv.gz"
NEW_VOWELS = ("i", "u")
N_SHUFFLES = 5
PAIRED_SEED = 20261117
N_RESAMPLES = 10_000
POSITIVE, NEGATIVE = "pathological", "normal"
# Readings, written before the run (deviations.md, 2026-11-10).
NOT_ONLY_A_II = 0.70
GAP_POINTS = 0.10
HELPS_POINTS = 0.05
SHUFFLE_BAND = (0.47, 0.53)


def file_id(speaker: str, vowel: str) -> str:
    return f"avfad__{speaker}" if vowel == "a" else f"avfad__{speaker}-{vowel}"


def with_vowel(members: pd.DataFrame, vowel: str) -> pd.DataFrame:
    return members.assign(file_id=[file_id(s, vowel) for s in members["speaker"].astype(str)])


def shuffled_labels(members: pd.DataFrame, seed: int) -> pd.DataFrame:
    """One permutation over speakers, so a speaker gets the same label in every vowel."""
    ordered = members.sort_values("speaker").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    return ordered.assign(label=rng.permutation(ordered["label"].to_numpy()))


def arm_sets(plan: dict) -> list[tuple[str, pd.DataFrame]]:
    """(ii), the 50 draws of (i) in the order of the /a/ run, and the shuffles of (ii)."""
    spec = plan["confirmatory_family"]["tests"]["C1"]
    seeds = plan["statistics"]["seeds"]
    patients, hsa_normals, all_normals = hsa_groups(batch.load_speakers())
    rng = np.random.default_rng(seeds["c1_draws"])
    draws = [match_cells(hsa_normals, all_normals, rng)[0] for _ in range(spec["draws"])]
    stored = pd.read_csv(HERE / "results" / "first_batch" / "c1_draws.csv", dtype={"speaker": str})
    for index, drawn in enumerate(draws):
        kept = stored.loc[stored["draw"] == index, "speaker"]
        if sorted(drawn["speaker"].astype(str)) != sorted(kept):
            raise RuntimeError(f"draw {index} differs from first_batch/c1_draws.csv")
    ii = pd.concat([patients, hsa_normals])
    sets = [("ii", ii)]
    sets += [(f"i_{i:02d}", pd.concat([patients, drawn])) for i, drawn in enumerate(draws)]
    sets += [
        (f"shuffled_{k}", shuffled_labels(ii, seeds["negative_control"] + k))
        for k in range(N_SHUFFLES)
    ]
    return sets


def _fit(vowel: str, name: str, members: pd.DataFrame, config) -> tuple[str, str, Arm, float]:
    start = time.perf_counter()
    features = load_hubert("full", CHOSEN_LAYERS, material=f"avfad_{vowel}")
    table = merge_features(with_vowel(members, vowel), features, "label")
    arm = evaluate_layers(name, table, CHOSEN_LAYERS, config)
    return vowel, name, arm, time.perf_counter() - start


def run(jobs: int) -> None:
    plan = load_plan()
    config = make_config(plan, batch.LABEL_POSITIVE)
    sets = arm_sets(plan)
    todo = [(v, name, members) for v in NEW_VOWELS for name, members in sets]
    todo += [("a", name, members) for name, members in sets if name.startswith("shuffled")]
    RESULTS.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    done = Parallel(n_jobs=jobs, return_as="generator_unordered")(
        delayed(_fit)(v, name, members, config) for v, name, members in todo
    )
    arms: dict[str, list[Arm]] = {}
    for count, (vowel, name, arm, seconds) in enumerate(done, start=1):
        arms.setdefault(vowel, []).append(arm)
        print(
            f"{vowel} {name} {seconds:.0f} s; {count} of {len(todo)} after "
            f"{time.perf_counter() - start:.0f} s",
            flush=True,
        )
    for vowel, part in arms.items():
        part.sort(key=lambda a: a.name)
        predictions_table(part).to_csv(RESULTS / f"predictions_{vowel}.csv.gz", index=False)
        pd.DataFrame([a.row() for a in part]).to_csv(RESULTS / f"arms_{vowel}.csv", index=False)


# ------------------------------------------------------------------ collect


def arm_from_table(table: pd.DataFrame, name: str) -> Arm:
    part = table[table["arm"] == name]
    wide = part.pivot(index="speaker", columns="repeat", values="score").sort_index()
    predicted = part.pivot(index="speaker", columns="repeat", values="predicted").sort_index()
    labels = part.groupby("speaker")["label"].first().loc[wide.index].to_numpy().astype(str)
    return make_arm(
        name,
        wide.index.to_numpy().astype(str),
        labels,
        wide.to_numpy(),
        predicted.to_numpy().astype(str),
    )


def make_arm(name, groups, y_true, scores, predictions=None) -> Arm:
    if predictions is None:
        predictions = np.where(scores >= 0.5, POSITIVE, NEGATIVE)
    return Arm(
        name=name,
        y_true=y_true,
        groups=groups,
        predictions=predictions,
        scores=scores,
        uar=float(weighted_metric(y_true, predictions, metric="uar")),
        roc_auc=float(weighted_metric(y_true, scores, metric="roc_auc", positive_class=POSITIVE)),
        sensitivity=float(
            weighted_metric(y_true, predictions, metric="sensitivity", positive_class=POSITIVE)
        ),
        specificity=float(
            weighted_metric(y_true, predictions, metric="specificity", positive_class=POSITIVE)
        ),
        worst_share_deviation=float("nan"),
        chosen_c={},
    )


def average_arm(name: str, parts: list[Arm]) -> Arm:
    """Mean positive-class score of the same speakers over the vowels, cut at 0.5."""
    first = parts[0]
    for other in parts[1:]:
        if not (
            np.array_equal(other.groups, first.groups)
            and np.array_equal(other.y_true, first.y_true)
        ):
            raise ValueError(f"{name}: speakers or labels differ between vowels")
    return make_arm(name, first.groups, first.y_true, np.mean([p.scores for p in parts], axis=0))


def same_folds(parts: list[tuple[str, Arm]], config) -> bool:
    """Every vowel's arm gets the same speakers in every test fold of every repeat.

    The rows are put back in the order the fit saw them (``merge_features`` sorts by file id).
    """

    def folds(vowel: str, arm: Arm):
        order = np.argsort([file_id(g, vowel) for g in arm.groups], kind="stable")
        y, groups = arm.y_true[order], arm.groups[order]
        splits = repeated_split_indices(
            y,
            groups,
            task="classification",
            n_splits=config.evaluation.n_splits,
            n_repeats=config.evaluation.n_repeats,
            seed=config.seed,
        )
        return [[frozenset(groups[test]) for _, test in repeat] for repeat in splits]

    reference = folds(*parts[0])
    return all(folds(*p) == reference for p in parts[1:])


def paired_difference(new: Arm, old: Arm, metric: str, seed: int) -> dict:
    if not np.array_equal(new.groups, old.groups):
        raise ValueError("the two arms hold different speakers")
    weights = group_weights(new.groups, new.groups.size, N_RESAMPLES, seed=seed)
    use = "scores" if metric == "roc_auc" else "predictions"
    extra = {"positive_class": POSITIVE} if metric == "roc_auc" else {}
    samples = weighted_metric(new.y_true, getattr(new, use), weights, metric=metric, **extra)
    samples = samples - weighted_metric(
        old.y_true, getattr(old, use), weights, metric=metric, **extra
    )
    finite = samples[np.isfinite(samples)]
    low, high = np.quantile(finite, [0.025, 0.975])
    estimate = float(
        weighted_metric(new.y_true, getattr(new, use), metric=metric, **extra)
        - weighted_metric(old.y_true, getattr(old, use), metric=metric, **extra)
    )
    return {
        "difference": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_resamples": N_RESAMPLES,
        "seed": seed,
    }


def material_reading(uar_ii: float, gap: dict) -> str:
    if uar_ii < NOT_ONLY_A_II and gap["difference"] >= GAP_POINTS and gap["ci_low"] > 0:
        return "not only /a/"
    return "does not meet the C1 materials thresholds"


def helps_reading(paired: dict) -> str:
    if paired["ci_low"] <= 0 <= paired["ci_high"]:
        return "more material does not help"
    if paired["difference"] >= HELPS_POINTS and paired["ci_low"] > 0:
        return "more material helps"
    return "small"


def gap_reading(gap: dict) -> str:
    if gap["ci_low"] <= 0:
        return "the gap is gone"
    if gap["difference"] >= GAP_POINTS:
        return "the gap stays"
    return "the gap shrinks"


def shuffle_passes(uars: list[float]) -> bool:
    return SHUFFLE_BAND[0] <= float(np.mean(uars)) <= SHUFFLE_BAND[1]


def summarise(name: str, arms: list[Arm], plan: dict) -> dict:
    spec = plan["confirmatory_family"]["tests"]["C1"]
    by_name = {a.name: a for a in arms}
    arm_ii = by_name["ii"]
    arms_i = [by_name[f"i_{i:02d}"] for i in range(spec["draws"])]
    shuffled = [by_name[f"shuffled_{k}"] for k in range(N_SHUFFLES)]
    gap = batch.c1_bootstrap(arms_i, arm_ii, N_RESAMPLES, plan["statistics"]["seeds"]["bootstrap"])
    gap.pop("samples")
    uars = [a.uar for a in arms_i]
    shuffled_uars = [a.uar for a in shuffled]
    return {
        "material": name,
        "ii": {
            "uar": arm_ii.uar,
            "roc_auc": arm_ii.roc_auc,
            "sensitivity": arm_ii.sensitivity,
            "specificity": arm_ii.specificity,
        },
        "i": {
            "uar_mean": float(np.mean(uars)),
            "uar_min": float(np.min(uars)),
            "uar_max": float(np.max(uars)),
            "roc_auc_mean": float(np.mean([a.roc_auc for a in arms_i])),
        },
        "gap_i_minus_ii": gap,
        "shuffled_ii": {
            "uar": shuffled_uars,
            "uar_mean": float(np.mean(shuffled_uars)),
            "passes": shuffle_passes(shuffled_uars),
        },
    }


def collect() -> dict:
    plan = load_plan()
    config = make_config(plan, batch.LABEL_POSITIVE)
    tables = {
        v: pd.read_csv(RESULTS / f"predictions_{v}.csv.gz", dtype={"speaker": str})
        for v in ("a", *NEW_VOWELS)
    }
    stored_a = pd.read_csv(STORED_A, dtype={"speaker": str})
    tables["a"] = pd.concat([stored_a, tables["a"]], ignore_index=True)
    names = sorted(set(tables["i"]["arm"]))
    arms = {v: [arm_from_table(t, n) for n in names] for v, t in tables.items()}
    stored = json.loads((STORED_A.parent / "c1_summary.json").read_text(encoding="utf-8"))
    rebuilt = next(a for a in arms["a"] if a.name == "ii").uar
    if abs(rebuilt - stored["uar_ii"]) > 1e-12:
        raise RuntimeError(f"/a/ (ii) rebuilt as {rebuilt}, stored {stored['uar_ii']}")
    checks = {}
    for name in names:
        parts = [(v, next(a for a in arms[v] if a.name == name)) for v in ("a", *NEW_VOWELS)]
        checks[name] = same_folds(parts, config)
    if not all(checks.values()):
        raise RuntimeError(f"folds differ: {[n for n, ok in checks.items() if not ok]}")
    fused = [
        average_arm(name, [next(a for a in arms[v] if a.name == name) for v in ("a", *NEW_VOWELS)])
        for name in names
    ]
    predictions_table(fused).to_csv(RESULTS / "predictions_average.csv.gz", index=False)
    report = {v: summarise(f"/{v}/", arms[v], plan) for v in ("a", *NEW_VOWELS)}
    report["average"] = summarise("/a/ /i/ /u/ averaged", fused, plan)
    for v in NEW_VOWELS:
        report[v]["reading"] = material_reading(report[v]["ii"]["uar"], report[v]["gap_i_minus_ii"])
    fused_ii = next(a for a in fused if a.name == "ii")
    a_ii = next(a for a in arms["a"] if a.name == "ii")
    report["average_minus_a_ii"] = {
        "uar": paired_difference(fused_ii, a_ii, "uar", PAIRED_SEED),
        "roc_auc": paired_difference(fused_ii, a_ii, "roc_auc", PAIRED_SEED),
    }
    average = report["average"]
    report["readings"] = {
        "readable": average["shuffled_ii"]["passes"],
        "more_material": helps_reading(report["average_minus_a_ii"]["uar"]),
        "reaches_0_70": average["ii"]["uar"] >= NOT_ONLY_A_II,
        "gap": gap_reading(average["gap_i_minus_ii"]),
    }
    report["same_folds_in_every_vowel"] = True
    write_json(RESULTS / "summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("run", "collect"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    verify_freeze()
    if args.step == "run":
        run(args.jobs)
    else:
        print(json.dumps(collect(), indent=1))


if __name__ == "__main__":
    main()
