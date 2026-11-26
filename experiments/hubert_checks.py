"""Two checks of the HuBERT score inside HSA; reasons are in experiments/deviations.md.

lower-c      the lower bound of the C grid was chosen in 41 of 50 folds, so go lower
no-surgery   the same arm without laryngeal surgery, next to the handcrafted 0.512
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
from voice_severity_bench import models as vsb_models  # noqa: E402
from joblib import Parallel, delayed, parallel_config  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import arms  # noqa: E402
import run_first_batch as batch  # noqa: E402
from freeze import verify_freeze  # noqa: E402
from harness import load_plan, make_config  # noqa: E402
from sensitivity_avfad import CHOSEN_LAYERS, RESULTS, hsa_groups  # noqa: E402

LOWER_GRID = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]


def _arm(name, table, config, grid):
    if grid is None:
        return arms.evaluate_layers(name, table, CHOSEN_LAYERS, config)

    original = arms._use_hubert_grid

    def use_lower_grid() -> None:
        vsb_models.SEARCH_SPACES[("classification", "logistic")] = {"model__C": list(grid)}

    arms._use_hubert_grid = use_lower_grid
    try:
        return arms.evaluate_layers(name, table, CHOSEN_LAYERS, config)
    finally:
        arms._use_hubert_grid = original


def main() -> None:
    verify_freeze()
    plan = load_plan()
    config = make_config(plan, batch.LABEL_POSITIVE)
    speakers = batch.load_speakers()
    features = arms.load_hubert("full", CHOSEN_LAYERS)
    patients, hsa, _ = hsa_groups(speakers)
    everyone = arms.merge_features(pd.concat([patients, hsa]), features, "label")
    kept = speakers[speakers["laryngeal_surgery"] == 0]
    patients_kept, hsa_kept, _ = hsa_groups(kept)
    clean = arms.merge_features(pd.concat([patients_kept, hsa_kept]), features, "label")
    with parallel_config(backend="loky", inner_max_num_threads=1):
        lower, no_surgery = Parallel(n_jobs=2)(
            [
                delayed(_arm)("ii_lower_c", everyone, config, LOWER_GRID),
                delayed(_arm)("ii_no_surgery", clean, config, None),
            ]
        )
    report = {
        "ii_lower_c": {**lower.row(), "grid": LOWER_GRID},
        "ii_no_surgery": no_surgery.row(),
    }
    (RESULTS / "hubert" / "checks.json").write_text(
        json.dumps(report, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
