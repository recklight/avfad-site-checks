"""The SVD time detector never learns from a patient; no real data is read."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "features"))

import time_only_svd as ts  # noqa: E402


def test_patients_outside_the_test_part_change_nothing():
    rng = np.random.default_rng(0)
    late = np.array([False] * 50 + [True] * 50)
    normal = np.array([True, False] * 50)
    values = rng.normal(size=(100, 4))
    values[late, 1] += 1.5
    test = np.zeros(100, dtype=bool)
    test[::5] = True
    before = ts.period_scores(values, late, normal, test)
    moved = values.copy()
    moved[~test & ~normal] = 77.0
    assert np.allclose(before, ts.period_scores(moved, late, normal, test))
    assert before[late[test]].mean() > before[~late[test]].mean()


def test_reading():
    assert ts.reading({"auc": 0.70, "ci_low": 0.65, "ci_high": 0.75}) == (
        "SVD has a shortcut in time too"
    )
    assert ts.reading({"auc": 0.52, "ci_low": 0.48, "ci_high": 0.56}) == (
        "SVD has no shortcut in time"
    )
    assert ts.reading({"auc": 0.62, "ci_low": 0.58, "ci_high": 0.66}) == "as measured"
