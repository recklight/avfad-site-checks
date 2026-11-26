"""The reading of the SVD-inside comparison; no real data is read."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "features"))

import svd_inside as si  # noqa: E402


def test_reading():
    assert si.reading(0.60, [0.58, 0.66]) == "inside HSA is about as hard as inside SVD"
    assert si.reading(0.55, [0.58, 0.66]) == "inside HSA is harder than inside SVD"
    assert si.reading(0.70, [0.58, 0.66]) == "as measured"


def test_small_training_keeps_the_test_fold_and_draws_hsa_numbers():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 1500
    table = pd.DataFrame(
        {
            "fold": np.arange(n) % 5,
            "label": np.where(np.arange(n) % 2 == 0, "normal", "pathological"),
        }
    )
    part = si.small_training(table, 2, rng)
    assert (part["fold"] == 2).sum() == (table["fold"] == 2).sum()
    rest = part[part["fold"] != 2]
    assert (rest["label"] == "normal").sum() == si.SMALL["normal"]
    assert (rest["label"] == "pathological").sum() == si.SMALL["pathological"]


def test_small_reading():
    assert si.small_reading(0.60, [0.55, 0.65]).startswith("with the same numbers, HSA is as")
    assert si.small_reading(0.50, [0.55, 0.65]).endswith("still harder")
