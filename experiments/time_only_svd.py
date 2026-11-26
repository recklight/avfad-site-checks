"""SVD's own shortcut in time: a detector that only learns the recording period of the normals.

Exploratory. Definitions and readings are in experiments/deviations.md (2026-11-10), written
before this ran. SVD is one place, but most of its normals were recorded in 1997-1998 and most of
its patients later. The detector never sees a patient: it is fitted on the normals of the other
parts to tell 1997-1998 from 1999-2004, and its "late" score is read as a pathology score.

    .venv/Scripts/python.exe -I -X utf8 -B experiments/time_only_svd.py
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "features"))

import diagnosis_breakdown as db  # noqa: E402
import run_svd as svd  # noqa: E402
import within_strata_auc as wsa  # noqa: E402
from arms import layer_columns, load_hubert, write_json  # noqa: E402
from extract_handcrafted import FEATURE_COLUMNS  # noqa: E402
from freeze import verify_freeze  # noqa: E402
from sensitivity_avfad import CHOSEN_LAYERS  # noqa: E402

RESULTS = HERE / "results" / "time_only_svd"
SEED = 20261152
N_PARTS = 5
N_REPEATS = 10
N_RESAMPLES = 10_000
LAST_EARLY_YEAR = 1998
# Reading, written before the run (deviations.md, 2026-11-10).
SHORTCUT = 0.65
SHORTCUT_LOW = 0.60


def table_of(detector: str) -> tuple[pd.DataFrame, list[str]]:
    meta = svd.load_svd_speakers("a_n")[["file_id", "label", "year"]]
    if meta["year"].isna().any():
        raise RuntimeError("an SVD session has no year")
    if detector == "hubert":
        features = load_hubert("full", CHOSEN_LAYERS, material="svd_a_n")
        columns = [c for layer in CHOSEN_LAYERS for c in layer_columns(layer)]
    else:
        features = pd.read_csv(HERE.parent / "features" / "out" / "svd_a_n_full_norm.csv")
        columns = list(FEATURE_COLUMNS)
    table = meta.merge(
        features[["file_id", *columns]], on="file_id", how="left", validate="one_to_one"
    )
    if table[columns[0]].isna().any():
        raise RuntimeError("an SVD session has no features")
    table = table.assign(late=table["year"] > LAST_EARLY_YEAR)
    return table.sort_values("file_id").reset_index(drop=True), columns


def period_scores(
    values: np.ndarray, late: np.ndarray, normal: np.ndarray, test: np.ndarray
) -> np.ndarray:
    """Late-vs-early decision values of the ``test`` rows, fitted on the other rows' normals."""
    train = ~test & normal
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    clf.fit(values[train], late[train])
    return clf.decision_function(values[test])


def _repeat(detector: str, repeat: int) -> tuple[str, np.ndarray]:
    table, columns = table_of(detector)
    values = table[columns].to_numpy(dtype=float)
    late = table["late"].to_numpy()
    normal = (table["label"] == "normal").to_numpy()
    strata = table["late"].astype(str) + "_" + table["label"].astype(str)
    folds = StratifiedKFold(N_PARTS, shuffle=True, random_state=SEED + repeat)
    score = np.full(len(table), np.nan)
    for _, test_index in folds.split(values, strata):
        test = np.zeros(len(table), dtype=bool)
        test[test_index] = True
        score[test] = period_scores(values, late, normal, test)
    return detector, score


def auc_with_ci(score: np.ndarray, sick: np.ndarray, rng: np.random.Generator) -> dict:
    kernel = db.comparisons(score[sick], score[~sick])
    w_sick = db.resample_counts(int(sick.sum()), N_RESAMPLES, rng)
    w_well = db.resample_counts(int((~sick).sum()), N_RESAMPLES, rng)
    low, high = np.percentile(db.weighted_auc(kernel, w_sick, w_well), [2.5, 97.5])
    return {"auc": float(kernel.mean()), "ci_low": float(low), "ci_high": float(high)}


def reading(record: dict) -> str:
    if record["auc"] >= SHORTCUT and record["ci_low"] > SHORTCUT_LOW:
        return "SVD has a shortcut in time too"
    if record["ci_low"] <= 0.5 <= record["ci_high"]:
        return "SVD has no shortcut in time"
    return "as measured"


def main(jobs: int = 12) -> None:
    verify_freeze()
    start = time.perf_counter()
    todo = [(d, r) for d in ("handcrafted", "hubert") for r in range(N_REPEATS)]
    done = Parallel(n_jobs=jobs)(delayed(_repeat)(d, r) for d, r in todo)
    rng = np.random.default_rng(SEED)
    summary: dict = {}
    table, _ = table_of("handcrafted")
    sick = (table["label"] == "pathological").to_numpy()
    late = table["late"].to_numpy()
    summary["period_only"] = auc_with_ci(late.astype(float), sick, rng)
    summary["counts"] = {
        f"{'late' if p else 'early'}_{lab}": int(((late == p) & (table["label"] == lab)).sum())
        for p in (False, True)
        for lab in ("normal", "pathological")
    }
    for detector in ("handcrafted", "hubert"):
        runs = [s for d, s in done if d == detector]
        score = np.mean(runs, axis=0)
        record = auc_with_ci(score, sick, rng)
        record["reading"] = reading(record)
        record["within_period"] = wsa.within_strata(
            np.column_stack(runs), sick, late.astype(str), SEED
        )
        normal = ~sick
        record["period_auc_among_normals"] = float(
            db.comparisons(score[normal & late], score[normal & ~late]).mean()
        )
        summary[detector] = record
        w = record["within_period"]
        print(
            detector,
            f"all {record['auc']:.3f} [{record['ci_low']:.3f}, {record['ci_high']:.3f}]",
            record["reading"],
            f"| within period {w['auc']:.3f} [{w['ci'][0]:.3f}, {w['ci'][1]:.3f}]",
            f"| late vs early normals {record['period_auc_among_normals']:.3f}",
        )
    p = summary["period_only"]
    print(
        "period only", f"{p['auc']:.3f} [{p['ci_low']:.3f}, {p['ci_high']:.3f}]", summary["counts"]
    )
    summary["seconds"] = round(time.perf_counter() - start, 1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / "summary.json", summary)


if __name__ == "__main__":
    main()
