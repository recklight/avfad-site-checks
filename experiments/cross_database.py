"""Trained on AVFAD /a/, tested on SVD a_n: does the (i) advantage of C1 travel? Exploratory.

Definitions and readings are in experiments/deviations.md, written before any cross-database
score existed. The detectors are the frozen ones (inner 3 folds choose C, and the layer for
HuBERT); the only outer split is AVFAD for training and SVD for testing.

    features    AVFAD /a/ full_norm handcrafted features into features/out_exploratory/
    run         every training set, scored on SVD; --kind handcrafted|hubert
    collect     the readings; writes results/cross_database/summary.json
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "features"))

import extract_handcrafted as hand  # noqa: E402
import published_splits as ps  # noqa: E402
import run_first_batch as batch  # noqa: E402
import run_svd as svd  # noqa: E402
from arms import layer_columns, load_hubert, write_json  # noqa: E402
from extract_handcrafted import FEATURE_COLUMNS  # noqa: E402
from freeze import verify_freeze  # noqa: E402
from freeze_svd import verify_svd_freeze  # noqa: E402
from harness import ROOT, load_plan, make_config  # noqa: E402
from sensitivity_avfad import CHOSEN_LAYERS  # noqa: E402
from within_strata_auc import mann_whitney_auc, pooled_auc  # noqa: E402

EXPLORATORY = ROOT / "features" / "out_exploratory"
RESULTS = HERE / "results" / "cross_database"
N_RESAMPLES = 1000
SEED = 20261111

# Fixed before any result; see deviations.md.
TRAVELS = 0.10
STAYS = 0.05


def run_features(jobs: int) -> None:
    work = hand.jobs_for("avfad", "a", ["full_norm"])
    with ProcessPoolExecutor(jobs) as pool:
        outputs = list(pool.map(hand.extract_file, work, chunksize=4))
    rows, skipped = [], 0
    for job, output in zip(work, outputs, strict=True):
        values, info = output["full_norm"]
        if values is None:
            skipped += 1
            continue
        rows.append({"file_id": job[0], **info, **dict(zip(FEATURE_COLUMNS, values, strict=True))})
    EXPLORATORY.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        EXPLORATORY / "avfad_a_full_norm.csv", index=False, lineterminator="\n"
    )
    print(len(rows), "files,", skipped, "with nothing to keep")


def training_sets() -> dict[str, pd.DataFrame]:
    speakers = batch.load_speakers()
    hsa = speakers["place"] == "HSA"
    patients = speakers[hsa & (speakers["label"] == "pathological")]
    hsa_normals = speakers[hsa & (speakers["label"] == "normal")]
    essua_normals = speakers[(speakers["place"] == "ESSUA") & (speakers["label"] == "normal")]
    draws = pd.read_csv(HERE / "results" / "first_batch" / "c1_draws.csv", dtype={"speaker": str})
    sets = {
        "ii": pd.concat([patients, hsa_normals]),
        "all": speakers,
        "essua_normals_hsa_patients": pd.concat([patients, essua_normals]),
    }
    for index, drawn in draws.groupby("draw"):
        chosen = speakers[speakers["speaker"].isin(drawn["speaker"])]
        sets[f"i_{index:02d}"] = pd.concat([patients, chosen])
    return {name: t.sort_values("file_id").reset_index(drop=True) for name, t in sets.items()}


def feature_tables(kind: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict | None]:
    if kind == "handcrafted":
        avfad = pd.read_csv(EXPLORATORY / "avfad_a_full_norm.csv")[["file_id", *FEATURE_COLUMNS]]
        test = pd.read_csv(ROOT / "features" / "out" / "svd_a_n_full_norm.csv")
        return avfad, test[["file_id", *FEATURE_COLUMNS]], list(FEATURE_COLUMNS), None
    avfad = load_hubert("full", CHOSEN_LAYERS)
    test = load_hubert("full", CHOSEN_LAYERS, material="svd_a_n")
    blocks = {f"l{layer:02d}": layer_columns(layer) for layer in CHOSEN_LAYERS}
    columns = [c for block in blocks.values() for c in block]
    return avfad, test, columns, blocks


def svd_test_set() -> pd.DataFrame:
    meta = svd.load_svd_speakers("a_n")
    return meta.assign(speaker="svd" + meta["speaker"].astype(str)).reset_index(drop=True)


def score_on_svd(train: pd.DataFrame, test: pd.DataFrame, columns, blocks, config) -> np.ndarray:
    """Fit on every training row (fold 0) and score every test row (fold 1)."""
    table = pd.concat([train.assign(fold=0), test.assign(fold=1)], ignore_index=True)
    table = table.assign(path=table["file_id"], group=table["speaker"])
    tuned = config.override({"model.block_selection": True}) if blocks is not None else config
    output = ps.fit_listed_fold(table, columns, blocks, tuned, 1)
    scores = np.full(len(test), np.nan)
    scores[output["items"] - len(train)] = output["scores"]
    return scores


def _one(name, train, test, columns, blocks, config) -> tuple[str, np.ndarray]:
    return name, score_on_svd(train, test, columns, blocks, config)


def run_scores(kind: str, jobs: int) -> None:
    plan = load_plan()
    config = make_config(plan, batch.LABEL_POSITIVE)
    avfad, svd_features, columns, blocks = feature_tables(kind)
    test = svd_test_set().merge(svd_features, on="file_id", how="left", validate="one_to_one")
    if test[columns[0]].isna().any():
        raise ValueError("an SVD speaker has no features")
    sets = {}
    for name, members in training_sets().items():
        train = members.merge(avfad, on="file_id", how="left", validate="one_to_one")
        if train[columns[0]].isna().any():
            raise ValueError(f"{name}: an AVFAD speaker has no features")
        sets[name] = train
    started = time.perf_counter()
    done = Parallel(n_jobs=jobs)(
        delayed(_one)(name, train, test, columns, blocks, config) for name, train in sets.items()
    )
    print(kind, len(done), "training sets in", round(time.perf_counter() - started), "s")
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = test[["file_id", "speaker", "label", "age", "sex", "year"]].copy()
    for name, scores in done:
        frame[name] = scores
    frame.to_csv(RESULTS / f"scores_{kind}.csv.gz", index=False)


def resample_indices(sick: np.ndarray, cells: np.ndarray, n: int, seed: int) -> list[np.ndarray]:
    """Speakers drawn with replacement inside each stratum and label; shared by every arm."""
    rng = np.random.default_rng(seed)
    groups = [
        np.flatnonzero((cells == cell) & (sick == label))
        for cell in np.unique(cells)
        for label in (True, False)
    ]
    return [np.concatenate([rng.choice(g, g.size) for g in groups if g.size]) for _ in range(n)]


def auc(scores: np.ndarray, sick: np.ndarray) -> float:
    return mann_whitney_auc(scores[sick], scores[~sick])


def uar_at_half(scores: np.ndarray, sick: np.ndarray) -> float:
    return float((np.mean(scores[sick] >= 0.5) + np.mean(scores[~sick] < 0.5)) / 2)


def evaluate(frame: pd.DataFrame, n: int = N_RESAMPLES, seed: int = SEED) -> dict:
    sick = (frame["label"] == "pathological").to_numpy()
    cells = svd.stratum(frame)
    names = [c for c in frame.columns if c in ("ii", "all", "essua_normals_hsa_patients")]
    draws = sorted(c for c in frame.columns if c.startswith("i_"))
    matrix = {name: frame[name].to_numpy() for name in [*names, *draws]}

    def both(scores, chosen=None):
        s, y, c = (
            (scores, sick, cells)
            if chosen is None
            else (scores[chosen], sick[chosen], cells[chosen])
        )
        return auc(s, y), pooled_auc(s, y, c)

    point = {name: both(matrix[name]) for name in matrix}
    resamples = resample_indices(sick, cells, n, seed)
    samples = {name: np.array([both(matrix[name], idx) for idx in resamples]) for name in matrix}
    i_point = np.mean([point[d] for d in draws], axis=0)
    i_samples = np.mean([samples[d] for d in draws], axis=0)
    report = {}
    for name in names:
        report[name] = {
            "auc": point[name][0],
            "auc_ci": ps.interval(samples[name][:, 0]),
            "within_strata_auc": point[name][1],
            "within_strata_auc_ci": ps.interval(samples[name][:, 1]),
            "uar_at_half": uar_at_half(matrix[name], sick),
        }
    report["i"] = {
        "draws": len(draws),
        "auc": float(i_point[0]),
        "auc_ci": ps.interval(i_samples[:, 0]),
        "auc_range": [
            float(min(point[d][0] for d in draws)),
            float(max(point[d][0] for d in draws)),
        ],
        "within_strata_auc": float(i_point[1]),
        "within_strata_auc_ci": ps.interval(i_samples[:, 1]),
        "uar_at_half_mean": float(np.mean([uar_at_half(matrix[d], sick) for d in draws])),
    }
    difference = i_samples - samples["ii"]
    report["d_ext"] = {
        "auc": float(i_point[0] - point["ii"][0]),
        "auc_ci": ps.interval(difference[:, 0]),
        "within_strata_auc": float(i_point[1] - point["ii"][1]),
        "within_strata_auc_ci": ps.interval(difference[:, 1]),
    }
    report["svd"] = {
        "speakers": int(sick.size),
        "pathological": int(sick.sum()),
        "strata": int(np.unique(cells).size),
    }
    return report


def internal_gap(kind: str) -> dict:
    if kind == "handcrafted":
        record = json.loads(
            (HERE / "results" / "first_batch" / "summary.json").read_text(encoding="utf-8")
        )["c1"]
    else:
        record = json.loads(
            (HERE / "results" / "sensitivity" / "hubert" / "c1_summary.json").read_text(
                encoding="utf-8"
            )
        )
    return {
        "auc_i": record["roc_auc_i_mean"],
        "auc_ii": record["roc_auc_ii"],
        "d_int": record["roc_auc_i_mean"] - record["roc_auc_ii"],
    }


def reading(d_ext: float, ci_low: float) -> str:
    if d_ext < STAYS or ci_low <= 0:
        return "does not travel: consistent with a place shortcut"
    if d_ext >= TRAVELS:
        return "part of it travels"
    return "middle"


def run_collect() -> None:
    report = {}
    for kind in ("handcrafted", "hubert"):
        path = RESULTS / f"scores_{kind}.csv.gz"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        record = evaluate(frame)
        record["internal"] = internal_gap(kind)
        record["d_ext_over_d_int"] = record["d_ext"]["auc"] / record["internal"]["d_int"]
        record["reading"] = reading(record["d_ext"]["auc"], record["d_ext"]["auc_ci"][0])
        record["reading_within_strata"] = reading(
            record["d_ext"]["within_strata_auc"], record["d_ext"]["within_strata_auc_ci"][0]
        )
        conditioned = json.loads(
            (HERE / "results" / "exploratory" / "conditioned.json").read_text(encoding="utf-8")
        )
        record["svd_internal_reference"] = conditioned.get("svd_a_n")
        report[kind] = record
        print(kind, json.dumps({k: record[k] for k in ("d_ext", "reading", "internal")}, indent=1))
    write_json(RESULTS / "summary.json", report)
    reverse = {}
    for kind in ("handcrafted", "hubert"):
        path = RESULTS / f"reverse_scores_{kind}.csv.gz"
        if path.exists():
            record = evaluate_reverse(pd.read_csv(path, dtype={"speaker": str}))
            record["reading"] = reverse_reading(record)
            reverse[kind] = record
    if reverse:
        write_json(RESULTS / "reverse.json", reverse)
        print(json.dumps(reverse, indent=1)[:2500])


REVERSE_SEED = 20261113
SEPARATES_INSIDE = 0.75
HARD_INSIDE = 0.65
LOOKS_SICKER = 0.60


def run_reverse(kind: str) -> None:
    """Train on SVD a_n once, score every AVFAD /a/ speaker."""
    plan = load_plan()
    config = make_config(plan, batch.LABEL_POSITIVE)
    avfad, svd_features, columns, blocks = feature_tables(kind)
    train = svd_test_set().merge(svd_features, on="file_id", how="left", validate="one_to_one")
    test = batch.load_speakers().merge(avfad, on="file_id", how="left", validate="one_to_one")
    if train[columns[0]].isna().any() or test[columns[0]].isna().any():
        raise ValueError("a speaker has no features")
    started = time.perf_counter()
    scores = score_on_svd(train, test, columns, blocks, config)
    print(kind, "reverse in", round(time.perf_counter() - started), "s", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = test[["file_id", "speaker", "place", "label", "age", "sex"]].assign(score=scores)
    frame.to_csv(RESULTS / f"reverse_scores_{kind}.csv.gz", index=False)


def evaluate_reverse(frame: pd.DataFrame, n: int = N_RESAMPLES, seed: int = REVERSE_SEED) -> dict:
    from harness import age_sex_cell

    hsa = (frame["place"] == "HSA").to_numpy()
    essua = (frame["place"] == "ESSUA").to_numpy()
    sick = (frame["label"] == "pathological").to_numpy()
    score = frame["score"].to_numpy()
    cells = age_sex_cell(frame).to_numpy().astype(str)
    groups = {
        "hsa_patients": hsa & sick,
        "hsa_normals": hsa & ~sick,
        "essua_normals": essua & ~sick,
        "others": ~(hsa | (essua & ~sick)),
    }

    def measures(index: np.ndarray) -> np.ndarray:
        s, k, h, e, c = score[index], sick[index], hsa[index], essua[index], cells[index]
        return np.array(
            [
                mann_whitney_auc(s[h & k], s[h & ~k]),
                pooled_auc(s[h], k[h], c[h]),
                mann_whitney_auc(s[h & k], s[e & ~k]),
                mann_whitney_auc(s[h & ~k], s[e & ~k]),
                mann_whitney_auc(s[k], s[~k]),
            ]
        )

    point = measures(np.arange(score.size))
    rng = np.random.default_rng(seed)
    members = [np.flatnonzero(m) for m in groups.values()]
    samples = np.array(
        [measures(np.concatenate([rng.choice(g, g.size) for g in members])) for _ in range(n)]
    )
    names = (
        "hsa_patients_vs_hsa_normals",
        "hsa_inside_strata",
        "hsa_patients_vs_essua_normals",
        "hsa_normals_vs_essua_normals",
        "everyone",
    )
    report = {
        name: {"auc": float(point[i]), "ci": ps.interval(samples[:, i])}
        for i, name in enumerate(names)
    }
    report["counts"] = {k: int(v.sum()) for k, v in groups.items()}
    return report


def reverse_reading(record: dict) -> dict:
    inside = record["hsa_patients_vs_hsa_normals"]["auc"]
    sicker = record["hsa_normals_vs_essua_normals"]
    if inside >= SEPARATES_INSIDE:
        place = "an outside model separates inside HSA"
    elif inside < HARD_INSIDE:
        place = "hard inside HSA for an outside model too"
    else:
        place = "middle"
    if sicker["auc"] >= LOOKS_SICKER and sicker["ci"][0] > 0.5:
        normals = "HSA normals look sicker to an outside model"
    elif sicker["ci"][0] <= 0.5 <= sicker["ci"][1]:
        normals = "no sign that HSA normals look sicker"
    else:
        normals = "middle"
    return {"inside_hsa": place, "hsa_normals": normals}


VOICED_SEED = 20261114
VOICED_HUBERT = ROOT / "features" / "out_hubert" / "voiced_a_full_16k.npz"


def _voiced_one(job: tuple[str, str]) -> tuple[str, np.ndarray]:
    from common import load_signal
    from extract_hubert import embed
    from scipy.signal import resample_poly

    file_id, wav = job
    x, rate = load_signal(wav)
    if rate != 8000:
        raise ValueError(f"{file_id}: expected 8 kHz, got {rate}")
    pooled, _ = embed([resample_poly(x, 2, 1)])
    return file_id, pooled


def voiced_hubert(jobs: int) -> None:
    files = pd.read_csv(ROOT / "s1_segments" / "out" / "files.csv")
    chosen = files[(files["database"] == "voiced") & (files["status"] == "ok")].sort_values(
        "file_id"
    )
    with ProcessPoolExecutor(jobs) as pool:
        done = list(pool.map(_voiced_one, zip(chosen["file_id"], chosen["wav"], strict=True)))
    np.savez(
        VOICED_HUBERT,
        file_id=np.array([f for f, _ in done]),
        pooled=np.stack([p for _, p in done]),
    )
    print(len(done), "VOICED files")


def run_voiced(jobs: int) -> None:
    import section8_rest as s8
    from arms import layer_columns as columns_of

    if not VOICED_HUBERT.exists():
        voiced_hubert(jobs)
    plan = load_plan()
    config = make_config(plan, batch.LABEL_POSITIVE)
    avfad, _, columns, blocks = feature_tables("hubert")
    data = np.load(VOICED_HUBERT)
    values = np.concatenate([data["pooled"][:, layer, :] for layer in CHOSEN_LAYERS], axis=1)
    voiced = pd.DataFrame(values, columns=[c for layer in CHOSEN_LAYERS for c in columns_of(layer)])
    voiced.insert(0, "file_id", data["file_id"].astype(str))
    labels = s8.voiced_labels()
    labels = labels.assign(label=labels["label"].replace({"healthy": "normal"}))
    test = labels.assign(speaker="voiced" + labels["speaker"]).merge(
        voiced, on="file_id", how="inner", validate="one_to_one"
    )
    sets = {
        name: members.merge(avfad, on="file_id", how="left", validate="one_to_one")
        for name, members in training_sets().items()
    }
    done = Parallel(n_jobs=jobs)(
        delayed(_one)(name, train, test, columns, blocks, config) for name, train in sets.items()
    )
    frame = test[["file_id", "speaker", "label"]].copy()
    for name, scores in done:
        frame[name] = scores
    frame.to_csv(RESULTS / "voiced_scores_hubert.csv.gz", index=False)
    sick = (frame["label"] == "pathological").to_numpy()
    flat = np.zeros(sick.size, dtype=int).astype(str)
    names = [c for c in frame.columns if c in ("ii", "all", "essua_normals_hsa_patients")]
    draws = sorted(c for c in frame.columns if c.startswith("i_"))
    point = {c: auc(frame[c].to_numpy(), sick) for c in [*names, *draws]}
    resamples = resample_indices(sick, flat, N_RESAMPLES, VOICED_SEED)
    samples = {
        c: np.array([auc(frame[c].to_numpy()[i], sick[i]) for i in resamples])
        for c in [*names, *draws]
    }
    i_point = float(np.mean([point[d] for d in draws]))
    i_samples = np.mean([samples[d] for d in draws], axis=0)
    report = {
        "speakers": int(sick.size),
        "pathological": int(sick.sum()),
        **{c: {"auc": point[c], "ci": ps.interval(samples[c])} for c in names},
        "i": {
            "auc": i_point,
            "ci": ps.interval(i_samples),
            "range": [float(min(point[d] for d in draws)), float(max(point[d] for d in draws))],
        },
        "d_ext": {"auc": i_point - point["ii"], "ci": ps.interval(i_samples - samples["ii"])},
        "all_minus_ii": {
            "auc": point["all"] - point["ii"],
            "ci": ps.interval(samples["all"] - samples["ii"]),
        },
    }
    report["reading"] = reading(report["d_ext"]["auc"], report["d_ext"]["ci"][0])
    write_json(RESULTS / "voiced.json", report)
    print(json.dumps(report, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("features", "run", "reverse", "voiced", "collect"))
    parser.add_argument("--kind", choices=("handcrafted", "hubert"), default="handcrafted")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    verify_freeze()
    verify_svd_freeze()
    if args.step == "features":
        run_features(args.jobs)
    elif args.step == "run":
        run_scores(args.kind, args.jobs)
    elif args.step == "reverse":
        run_reverse(args.kind)
    elif args.step == "voiced":
        run_voiced(args.jobs)
    else:
        run_collect()


if __name__ == "__main__":
    main()
