"""Synthetic checks of the denoising step; no recording is read."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "features"))

import denoise as dn  # noqa: E402


def test_noise_frames_match_the_source():
    assert dn.leading_noise_frames(120.0, 44_100) == 19
    assert dn.leading_noise_frames(120.0, 16_000) == 14
    assert dn.leading_noise_frames(1.0, 16_000) == 1


def _tone_in_noise(rate=16_000, seconds=2.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(round(seconds * rate)) / rate
    tone = 0.3 * np.sin(2 * np.pi * 220 * t)
    tone[: rate // 2] = 0.0  # half a second of room before the voice
    noise = 0.02 * rng.normal(size=t.size)
    return tone + noise, noise, rate


def test_denoised_file_keeps_its_length_and_loses_noise():
    x, _, rate = _tone_in_noise()
    quiet = [x[: rate // 2]]
    y = dn.denoise(x, rate, quiet)
    assert y.shape == x.shape
    assert np.std(y[: rate // 2]) < 0.5 * np.std(x[: rate // 2])


def test_prefix_is_cut_off_again():
    x, _, rate = _tone_in_noise(seed=1)
    with_prefix = dn.denoise(x, rate, [x[: rate // 4]])
    assert with_prefix.size == x.size


def test_file_without_quiet_spans_still_works():
    x, _, rate = _tone_in_noise(seed=2)
    assert dn.denoise(x, rate, []).shape == x.shape


def test_settings_are_the_written_ones():
    assert dn.PREFIX_MAX_S == 0.5 and dn.FALLBACK_NOISE_MS == 100.0 and dn.SMOOTHING == 0.98
