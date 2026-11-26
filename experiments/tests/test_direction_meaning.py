"""Synthetic checks of the place-direction description; no feature file or recording is read."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "features"))

import direction_meaning as dm  # noqa: E402


def test_correlations_are_spearman():
    x = pd.Series(np.arange(10.0))
    table = pd.DataFrame({"a": np.arange(10.0) ** 3, "b": -np.arange(10.0)})
    out = dm.correlations(x, table, ["a", "b"])
    assert out == pytest.approx({"a": 1.0, "b": -1.0})


@pytest.mark.parametrize(
    ("floor", "other", "expected"),
    [
        (-0.35, 0.1, "the place direction follows the noise floor within one place"),
        (0.1, 0.4, "the place direction follows other measures within one place"),
        (0.1, -0.2, "little relation to these measures within one place"),
    ],
)
def test_reading(floor, other, expected):
    within = {
        "essua_normals": {"0": {"floor_db": 0.05, "duration_s": 0.0}},
        "hsa_everyone": {"0": {"floor_db": floor, "duration_s": other}},
    }
    got = dm.reading(within)
    assert got["reading"] == expected
    assert got["other_columns_at_or_above"] == (["duration_s"] if abs(other) >= 0.3 else [])


def test_threshold_is_the_written_one():
    assert dm.FOLLOWS == 0.30 and dm.FLOOR == "floor_db"


@pytest.mark.parametrize(
    ("aucs", "expected"),
    [
        ([0.52, 0.61], "the place direction separates pathology on SVD too"),
        ([0.39, 0.5], "the place direction separates pathology on SVD too"),
        ([0.45, 0.58], "the place direction has little to do with pathology on SVD"),
    ],
)
def test_svd_reading(aucs, expected):
    assert dm.svd_reading(aucs) == expected
