"""Synthetic checks of the place erasure; no feature file or recording is read."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "features"))

import site_erasure as se  # noqa: E402


def test_after_the_projection_the_two_places_have_the_same_mean():
    rng = np.random.default_rng(0)
    place = rng.random(300) < 0.7
    block = rng.normal(size=(300, 40)) + 1.5 * place[:, None] * rng.normal(size=40)
    d = se.place_directions({0: block}, place, ~place)[0]
    assert np.linalg.norm(d) == pytest.approx(1.0)
    out = se.project_out(block, d)
    assert np.allclose(out[place].mean(axis=0), out[~place].mean(axis=0))
    assert np.allclose(out @ d, 0.0)


def test_the_projection_leaves_orthogonal_directions_alone():
    d = np.array([1.0, 0.0, 0.0])
    block = np.array([[2.0, 3.0, 4.0], [-1.0, 5.0, 6.0]])
    assert se.project_out(block, d).tolist() == [[0.0, 3.0, 4.0], [0.0, 5.0, 6.0]]


@pytest.mark.parametrize(
    ("difference", "low", "high", "expected"),
    [
        (0.07, 0.02, 0.12, "helps"),
        (0.02, -0.01, 0.05, "does not help"),
        (0.03, 0.01, 0.05, "small"),
        (-0.04, -0.07, -0.01, "small"),
    ],
)
def test_remedy_reading(difference, low, high, expected):
    assert se.remedy_reading({"difference": difference, "ci_low": low, "ci_high": high}) == expected


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [
        (-0.14, -0.10, "mixed training still loses"),
        (-0.03, 0.02, "mixed training no longer loses"),
        (0.01, 0.05, "mixed training wins"),
    ],
)
def test_mixed_reading(low, high, expected):
    assert se.mixed_reading({"difference": (low + high) / 2, "ci_low": low, "ci_high": high}) == (
        expected
    )


def test_gap_reading_only_speaks_when_the_gap_stays():
    assert se.gap_reading({"difference": 0.15, "ci_low": 0.08}) == "the gap stays"
    assert se.gap_reading({"difference": 0.04, "ci_low": 0.01}).startswith("the gap is smaller")


def test_ii_loss_needs_three_points_and_a_ci_below_zero():
    assert se.ii_loss({"difference": -0.04, "ci_high": -0.005})
    assert not se.ii_loss({"difference": -0.04, "ci_high": 0.01})
    assert not se.ii_loss({"difference": -0.02, "ci_high": -0.005})


def test_thresholds_and_seeds_are_the_written_ones():
    assert se.HELPS_POINTS == 0.05 and se.GAP_POINTS == 0.10 and se.II_LOSS_POINTS == 0.03
    assert se.STILL_THERE_AUC == 0.75
    assert se.SVD_SEED == 20261118 and se.AVFAD_SEED == 20261120 and se.N_SVD == 1000
    assert (se.CHECK_FOLDS, se.CHECK_REPEATS, se.CHECK_C) == (5, 10, 0.01)
