"""Synthetic checks of the recording-measure R^2; no score file or recording is read."""

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

import tech_follow as tf  # noqa: E402


def test_unit_weights_give_the_least_squares_r2():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 3))
    y = x @ np.array([1.0, -0.5, 0.0]) + rng.normal(size=120)
    design = np.column_stack([np.ones(120), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    expected = 1 - ((y - design @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    assert tf.rank_r2(y, x, np.ones(120))[0] == pytest.approx(expected)


def test_integer_weights_equal_repeating_the_rows():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(40, 2)), rng.normal(size=40)
    w = rng.integers(1, 4, 40)
    repeated = tf.rank_r2(np.repeat(y, w), np.repeat(x, w, axis=0), np.ones(w.sum()))[0]
    assert tf.rank_r2(y, x, w.astype(float))[0] == pytest.approx(repeated)


@pytest.mark.parametrize(
    ("difference", "low", "high", "expected"),
    [
        (-0.08, -0.14, -0.02, "(i) follows them less"),
        (0.02, -0.03, 0.07, "the same"),
        (0.15, 0.06, 0.24, "(i) follows the recording measures more among patients"),
        (0.05, 0.01, 0.09, "small"),
    ],
)
def test_r2_reading(difference, low, high, expected):
    assert tf.r2_reading({"difference": difference, "ci_low": low, "ci_high": high}) == expected


def test_threshold_and_seed_are_the_written_ones():
    assert tf.R2_POINTS == 0.10 and tf.SEED == 20261126 and tf.N_RESAMPLES == 10_000
    assert len(tf.M_TECH) == 7
