"""Synthetic checks of the called-normal AUC; no prediction file is read."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "features"))

import published_axis as pa  # noqa: E402


def test_auc_called_normal():
    projection = np.array([3.0, 2.0, 1.0, 0.0, -1.0])
    called = np.array([True, True, False, False, False])
    point, boot = pa.auc_called_normal(projection, called, np.ones((4, 5)))
    assert point == 1.0
    assert np.allclose(boot, 1.0)


def test_reading():
    assert pa.reading({"auc": 0.7, "ci_low": 0.55, "ci_high": 0.85}) == (
        "the published system follows the place direction inside one place too"
    )
    assert pa.reading({"auc": 0.55, "ci_low": 0.4, "ci_high": 0.7}) == "not visible"
    assert pa.reading({"auc": 0.58, "ci_low": 0.52, "ci_high": 0.66}) == "as measured"
