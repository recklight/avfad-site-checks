"""AUC inside the age x sex x year strata of SVD a_n, as a reference for the HSA-only numbers.

Exploratory. Definitions and readings are in experiments/deviations.md (2026-11-10), written
before this ran. The detectors are those of C1; SVD is one place, so its within-strata AUC is what
a single-site sustained /a/ gives once the demographics are held equal.

    .venv/Scripts/python.exe -I -X utf8 -B experiments/svd_inside.py [small]

``small`` was added after the first run: the same, with HSA's training numbers per fold.
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from voice_severity_bench import models as vsb_models  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "features"))

import diagnosis_breakdown as db  # noqa: E402
import phonation_c1 as pc  # noqa: E402
import pool_training as pt  # noqa: E402
import published_splits as ps  # noqa: E402
import run_first_batch as batch  # noqa: E402
import run_svd as svd  # noqa: E402
import within_strata_auc as wsa  # noqa: E402
from arms import layer_columns, load_hubert, write_json  # noqa: E402
from extract_handcrafted import FEATURE_COLUMNS  # noqa: E402
from freeze import verify_freeze  # noqa: E402
from harness import load_plan, make_config  # noqa: E402
from sensitivity_avfad import CHOSEN_LAYERS  # noqa: E402

RESULTS = HERE / "results" / "svd_inside"
HSA = HERE / "results" / "exploratory" / "within_strata_auc.json"
HSA_KEYS = {"handcrafted": "avfad_hsa_handcrafted", "hubert": "avfad_hsa_hubert"}
# ``small`` (added after the first run): HSA's training numbers per fold, 82 x 0.8 and 329 x 0.8.
SMALL = {"normal": 66, "pathological": 263}
SMALL_SEED = 20261154


def table_of(detector: str) -> tuple[pd.DataFrame, list[str], dict | None]:
    meta = svd.load_svd_speakers("a_n")
    if detector == "hubert":
        features = load_hubert("full", CHOSEN_LAYERS, material="svd_a_n")
        blocks = {f"l{layer:02d}": layer_columns(layer) for layer in CHOSEN_LAYERS}
        columns = [c for block in blocks.values() for c in block]
    else:
        features = svd.load_table("a_n", "full_norm")
        blocks, columns = None, list(FEATURE_COLUMNS)
    table = meta.merge(
        features[["file_id", *columns]], on="file_id", how="left", validate="one_to_one"
    )
    if table[columns[0]].isna().any():
        raise RuntimeError("an SVD session has no features")
    table = table.assign(
        path=table["file_id"], group=table["speaker"].astype(str), label=table["label"].astype(str)
    )
    return table.sort_values("file_id").reset_index(drop=True), columns, blocks


def small_training(table: pd.DataFrame, fold: int, rng: np.random.Generator) -> pd.DataFrame:
    """The test fold plus HSA-sized training: SMALL normals and patients drawn from the rest."""
    test = table["fold"].to_numpy() == fold
    keep = test.copy()
    for label, n in SMALL.items():
        pool = np.flatnonzero(~test & (table["label"] == label).to_numpy())
        keep[rng.choice(pool, size=n, replace=False)] = True
    return table[keep]


def _fold(detector: str, repeat: int, fold: int, config, small: bool = False):
    start = time.perf_counter()
    if detector == "handcrafted":
        vsb_models.SEARCH_SPACES[pc.LOGISTIC] = {k: list(v) for k, v in pc.HANDCRAFTED_GRID.items()}
    table, columns, blocks = table_of(detector)
    table = table.assign(fold=pt.fold_ids(table, config)[:, repeat])
    tuned = config.override({"model.block_selection": True}) if blocks is not None else config
    if small:
        rng = np.random.default_rng([SMALL_SEED, repeat, fold])
        part = small_training(table.assign(row=np.arange(len(table))), fold, rng)
        output = ps.fit_listed_fold(part.reset_index(drop=True), columns, blocks, tuned, fold)
        items = part["row"].to_numpy()[output["items"]]
    else:
        output = ps.fit_listed_fold(table, columns, blocks, tuned, fold)
        items = output["items"]
    return detector, repeat, items, output["scores"], time.perf_counter() - start


def reading(hsa: float, svd_ci: list[float]) -> str:
    if svd_ci[0] <= hsa <= svd_ci[1]:
        return "inside HSA is about as hard as inside SVD"
    if hsa < svd_ci[0]:
        return "inside HSA is harder than inside SVD"
    return "as measured"


def small_reading(hsa: float, svd_ci: list[float]) -> str:
    if svd_ci[0] <= hsa <= svd_ci[1]:
        return "with the same numbers, HSA is as hard as SVD"
    if hsa < svd_ci[0]:
        return "with the same numbers, HSA is still harder"
    return "as measured"


def main(jobs: int = 12, small: bool = False) -> None:
    verify_freeze()
    config = make_config(load_plan(), batch.LABEL_POSITIVE)
    hsa = json.loads(HSA.read_text(encoding="utf-8"))
    summary: dict = {}
    for detector in ("handcrafted", "hubert"):
        start = time.perf_counter()
        todo = [
            (r, k)
            for r in range(config.evaluation.n_repeats)
            for k in range(config.evaluation.n_splits)
        ]
        done = Parallel(n_jobs=jobs)(delayed(_fold)(detector, r, k, config, small) for r, k in todo)
        table, _, _ = table_of(detector)
        matrix = np.full((len(table), config.evaluation.n_repeats), np.nan)
        for _, repeat, items, scores, _ in done:
            matrix[items, repeat] = scores
        if np.isnan(matrix).any():
            raise RuntimeError("a session was never tested")
        sick = (table["label"] == batch.LABEL_POSITIVE).to_numpy()
        cells = svd.stratum(table)
        inside = wsa.within_strata(matrix, sick, cells, wsa.SEED)
        plain = float(
            np.mean(
                [
                    db.comparisons(matrix[sick, r], matrix[~sick, r]).mean()
                    for r in range(matrix.shape[1])
                ]
            )
        )
        reference = hsa[HSA_KEYS[detector]]
        summary[detector] = {
            "auc": plain,
            "within_strata": inside,
            "hsa_within_strata": reference,
            "reading": (small_reading if small else reading)(reference["auc"], inside["ci"]),
            "seconds": round(time.perf_counter() - start, 1),
        }
        print(
            detector,
            f"auc {plain:.3f}",
            f"inside strata {inside['auc']:.3f} [{inside['ci'][0]:.3f}, {inside['ci'][1]:.3f}]",
            f"| HSA {reference['auc']:.3f}",
            summary[detector]["reading"],
            f"{summary[detector]['seconds']} s",
            flush=True,
        )
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / ("summary_small.json" if small else "summary.json"), summary)


if __name__ == "__main__":
    main(small=sys.argv[1:] == ["small"])
