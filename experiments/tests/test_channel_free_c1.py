"""The column split and the reading of the channel-free C1; no real data is read."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "features"))

import channel_free_c1 as cf  # noqa: E402
from extract_handcrafted import FEATURE_COLUMNS  # noqa: E402


def test_the_two_sets_split_the_78_columns():
    assert len(cf.FREE) == 40
    assert len(cf.BOUND) == 38
    assert not set(cf.FREE) & set(cf.BOUND)
    assert set(cf.FREE) | set(cf.BOUND) == set(FEATURE_COLUMNS)
    assert not any(c.startswith(("mfcc_mean_", "ltas_")) for c in cf.FREE)
    assert "level_std" in cf.FREE and "level_mean" in cf.BOUND


def test_a_fixed_filter_and_gain_leave_the_free_statistics_alone():
    rng = np.random.default_rng(0)
    frames = rng.normal(size=(200, 13))
    moved = frames + rng.normal(size=13)
    assert np.allclose(frames.std(axis=0), moved.std(axis=0))
    assert np.allclose(np.diff(frames, axis=0), np.diff(moved, axis=0))
    assert not np.allclose(frames.mean(axis=0), moved.mean(axis=0))


def test_reading():
    def gap(d, lo):
        return {"difference": d, "ci_low": lo}

    assert cf.free_reading(gap(0.20, 0.10), 0.55) == "this set is too weak to read"
    assert cf.free_reading(gap(0.12, 0.05), 0.70) == "a gap without a fixed channel or gain"
    assert cf.free_reading(gap(0.03, -0.01), 0.70) == "a fixed channel and gain explain it"
    assert cf.free_reading(gap(0.07, 0.02), 0.70) == "the gap shrinks"
