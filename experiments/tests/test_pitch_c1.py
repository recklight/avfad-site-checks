"""YIN and the pitch statistics on synthetic tones; no real data is read."""

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

import pitch_c1 as pc1  # noqa: E402

RATE = 16_000


def harmonic(f0: np.ndarray | float, seconds: float = 1.0, noise: float = 0.0) -> np.ndarray:
    t = np.arange(round(seconds * RATE)) / RATE
    f0 = np.broadcast_to(np.asarray(f0, dtype=float), t.shape)
    phase = 2.0 * np.pi * np.cumsum(f0) / RATE
    x = sum(np.sin(k * phase) / k for k in range(1, 8))
    return x + noise * np.random.default_rng(1).normal(size=t.size)


@pytest.mark.parametrize("f0", [80.0, 120.0, 220.0, 400.0])
def test_yin_finds_a_steady_tone(f0):
    track = pc1.yin(harmonic(f0), RATE)
    voiced = track[np.isfinite(track)]
    assert voiced.size >= 0.9 * track.size
    assert np.median(np.abs(voiced - f0)) < 0.005 * f0


def test_yin_holds_with_some_noise():
    track = pc1.yin(harmonic(150.0, noise=0.1), RATE)
    voiced = track[np.isfinite(track)]
    assert voiced.size >= 0.8 * track.size
    assert abs(np.median(voiced) - 150.0) < 1.5


def test_white_noise_is_not_voiced():
    x = np.random.default_rng(2).normal(size=RATE)
    track = pc1.yin(x, RATE)
    assert np.isfinite(track).mean() < 0.1


def test_a_short_piece_gives_no_frames():
    assert pc1.yin(np.zeros(100), RATE).size == 0


def test_statistics_in_semitones():
    steady = pc1.pitch_features([harmonic(100.0), harmonic(100.0, 0.5)], RATE)
    assert abs(steady["f0_median_st"]) < 0.1
    assert steady["f0_spread_st"] < 0.1
    assert steady["voiced_share"] > 0.9
    # A 5 Hz vibrato of +-1 semitone spreads the tones and makes larger steps.
    t = np.arange(RATE) / RATE
    wobbly = pc1.pitch_features([harmonic(200.0 * 2 ** (np.sin(2 * np.pi * 5 * t) / 12))], RATE)
    assert abs(wobbly["f0_median_st"] - 12.0) < 0.3
    assert 0.5 < wobbly["f0_iqr_st"] < 2.0
    assert wobbly["f0_step_st"] > steady["f0_step_st"]


def test_too_few_voiced_frames_leave_the_four_missing():
    out = pc1.pitch_features([np.random.default_rng(3).normal(size=RATE // 2)], RATE)
    assert all(np.isnan(out[name]) for name in pc1.PITCH)


def test_readings():
    assert pc1.visibility_reading({"auc": 0.70, "ci_low": 0.60, "ci_high": 0.80}).startswith(
        "the two places"
    )
    assert pc1.visibility_reading({"auc": 0.55, "ci_low": 0.45, "ci_high": 0.65}) == (
        "pitch does not show the place"
    )
    gap = {"difference": 0.12, "ci_low": 0.04}
    assert pc1.gap_reading(gap, 0.58) == "pitch is too weak to read"
    assert pc1.gap_reading(gap, 0.65) == "pitch alone carries a gap"
    assert pc1.gap_reading({"difference": 0.02, "ci_low": -0.03}, 0.65) == "pitch carries no gap"
    assert pc1.joint_reading("pitch does not show the place", "pitch carries no gap") == (
        "the cue is not in the pitch"
    )
