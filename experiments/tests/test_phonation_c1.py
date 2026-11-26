"""Synthetic checks of the phonation-only C1 readings; no feature file or recording is read."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "features"))

import phonation_c1 as pc  # noqa: E402


@pytest.mark.parametrize(
    ("difference", "low", "expected"),
    [
        (0.15, 0.08, "the gap stays on phonation only: not just length and silence"),
        (0.02, -0.02, "the gap is gone: length and silence are the shortcut"),
        (0.06, 0.01, "the gap shrinks"),
    ],
)
def test_gap_reading(difference, low, expected):
    assert pc.gap_reading({"difference": difference, "ci_low": low}) == expected


@pytest.mark.parametrize(
    ("difference", "low", "high", "expected"),
    [
        (-0.08, -0.12, -0.03, "taking out the silence costs the gap"),
        (-0.01, -0.04, 0.03, "the gap does not change"),
        (-0.03, -0.05, -0.01, "as measured"),
        (0.06, 0.02, 0.10, "as measured"),
    ],
)
def test_change_reading(difference, low, high, expected):
    change = {"difference": difference, "ci_low": low, "ci_high": high}
    assert pc.change_reading(change) == expected


def test_thresholds_are_the_written_ones():
    assert pc.GAP_POINTS == 0.10 and pc.CHANGE_POINTS == 0.05
    assert pc.DETECTORS == ("handcrafted", "hubert")


def test_the_handcrafted_grid_is_not_the_hubert_one():
    from arms import HUBERT_C_GRID

    assert pc.HANDCRAFTED_GRID["model__C"] == [0.01, 0.1, 1.0, 10.0]
    assert pc.HANDCRAFTED_GRID["model__C"] != list(HUBERT_C_GRID)
