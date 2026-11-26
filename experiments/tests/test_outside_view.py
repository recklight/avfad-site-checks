"""Synthetic checks of the outside view; no score file or recording is read."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "features"))

import outside_view as ov  # noqa: E402


def test_unit_weights_give_the_spearman_correlation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=80)
    y = 0.6 * x + rng.normal(size=80)
    got = ov.spearman_weighted(x, y, np.ones(80))[0]
    assert got == pytest.approx(spearmanr(x, y).statistic)


def test_weights_of_one_patient_scale_away():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=30), rng.normal(size=30)
    weights = np.vstack([np.ones(30), 2 * np.ones(30)])
    got = ov.spearman_weighted(x, y, weights)
    assert got[0] == pytest.approx(got[1])


@pytest.mark.parametrize(
    ("difference", "low", "high", "expected"),
    [
        (-0.06, -0.10, -0.02, "the (i) pairing is harder for the outside detector"),
        (
            0.01,
            -0.03,
            0.05,
            "the outside detector sees no difference between the two sets of normals",
        ),
        (0.12, 0.06, 0.18, "the test populations differ this much"),
        (0.05, 0.01, 0.09, "the test populations differ a little"),
    ],
)
def test_gap_reading(difference, low, high, expected):
    assert ov.gap_reading({"difference": difference, "ci_low": low, "ci_high": high}) == expected


def test_learned_reading_needs_ten_points_and_a_positive_ci():
    assert ov.learned_reading({"difference": 0.2, "ci_low": 0.1}) == (
        "most of the gap is learned in training"
    )
    assert ov.learned_reading({"difference": 0.08, "ci_low": 0.02}) == "as measured"


@pytest.mark.parametrize(
    ("difference", "low", "high", "expected"),
    [
        (-0.08, -0.15, -0.01, "(i) agrees less"),
        (0.02, -0.04, 0.08, "the same"),
        (0.14, 0.05, 0.22, "(i) ranks patients more like the outside detector"),
        (0.06, 0.01, 0.11, "small"),
    ],
)
def test_rank_reading(difference, low, high, expected):
    assert ov.rank_reading({"difference": difference, "ci_low": low, "ci_high": high}) == expected


def test_thresholds_and_seed_are_the_written_ones():
    assert ov.GAP_POINTS == 0.10 and ov.RANK_POINTS == 0.10
    assert ov.SEEDS == {"a": 20261123, "i": 20261124, "u": 20261125}
    assert ov.HANDCRAFTED_SEED == 20261127
    assert ov.ERASED_SEED == 20261131
    assert ov.OUTSIDE_ERASED_SEEDS == {"a": 20261134, "i": 20261137, "u": 20261138}
    assert ov.PHONATION_SEED == 20261136
    assert ov.N_RESAMPLES == 10_000 and ov.SVD_SPEAKERS == 1457


@pytest.mark.parametrize(
    ("difference", "low", "high", "expected"),
    [
        (
            0.15,
            0.05,
            0.25,
            "with the place direction out, (i) ranks patients more like the outside detector",
        ),
        (0.02, -0.04, 0.08, "the ranking does not change"),
        (0.06, 0.01, 0.11, "as measured"),
    ],
)
def test_erased_reading(difference, low, high, expected):
    assert ov.erased_reading({"difference": difference, "ci_low": low, "ci_high": high}) == expected
