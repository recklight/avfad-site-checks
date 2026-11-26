"""Synthetic checks of the ranking after alignment; no score file is read."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "features"))

import aligned_ranking as ar  # noqa: E402


def test_draws_rho_averages_the_draws():
    rng = np.random.default_rng(0)
    patients = pd.Index([f"P{i}" for i in range(200)])
    reference = rng.normal(size=200)
    arms = {
        "ii": pd.Series(reference + rng.normal(size=200), index=patients),
        "i_00": pd.Series(reference, index=patients),
        "i_01": pd.Series(-reference, index=patients),
    }
    weights = np.ones((5, 200))
    point, boot = ar.draws_rho(arms, patients, reference, weights)
    assert abs(point) < 1e-12
    assert np.allclose(boot, 0.0)


def test_readings():
    assert ar.change_reading({"difference": 0.15, "ci_low": 0.05, "ci_high": 0.25}) == (
        "after the alignment (i) ranks patients more like the outside detector"
    )
    assert ar.change_reading({"difference": 0.02, "ci_low": -0.03, "ci_high": 0.07}) == (
        "the ranking does not change"
    )
    assert ar.change_reading({"difference": 0.05, "ci_low": 0.01, "ci_high": 0.09}) == "as measured"
    assert ar.versus_ii_reading({"ci_low": -0.04, "ci_high": 0.02}) == "catches up with (ii)"
    assert ar.versus_ii_reading({"ci_low": -0.2, "ci_high": -0.1}) == "still agrees less than (ii)"
    assert ar.versus_ii_reading({"ci_low": 0.01, "ci_high": 0.1}) == "agrees more than (ii)"
