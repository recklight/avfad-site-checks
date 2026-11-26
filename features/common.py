"""Audio loading, span cutting and the level rule shared by the feature scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "s0_audit"))

from readers import read_wav  # noqa: E402

TARGET_RATE = 16_000
PHONATION_LEVEL_DBFS = -26.0

Spans = list[tuple[float, float]]


def to_analysis_rate(x: np.ndarray, rate: int) -> tuple[np.ndarray, int]:
    """Same resampling as the segmenter, so span times and samples line up."""
    if rate <= TARGET_RATE:
        return x, rate
    g = int(np.gcd(rate, TARGET_RATE))
    return resample_poly(x, TARGET_RATE // g, rate // g), TARGET_RATE


def load_signal(relative_wav: str) -> tuple[np.ndarray, int]:
    wav = read_wav(ROOT / relative_wav)
    return to_analysis_rate(wav.as_float()[:, 0], wav.rate)


def cut(x: np.ndarray, rate: int, spans: Spans) -> list[np.ndarray]:
    pieces = []
    for start, end in spans:
        first, last = max(0, round(start * rate)), min(x.size, round(end * rate))
        if last > first:
            pieces.append(x[first:last])
    return pieces


def rms_dbfs(x: np.ndarray) -> float:
    return float(10.0 * np.log10(np.mean(np.square(x)) + 1e-12))


def normalise_to_phonation(x: np.ndarray, rate: int, phonation: Spans) -> tuple[np.ndarray, float]:
    """Scale the whole file so its phonation samples sit at -26 dBFS.

    Returns the scaled signal and the phonation level before scaling. The file is
    scaled as one piece, so what lies outside phonation keeps its level relative
    to the voice.
    """
    pieces = cut(x, rate, phonation)
    if not pieces:
        raise ValueError("no phonation span to normalise to")
    level = rms_dbfs(np.concatenate(pieces))
    return x * 10.0 ** ((PHONATION_LEVEL_DBFS - level) / 20.0), level


def complement(spans: Spans, duration_s: float) -> Spans:
    """The parts of ``[0, duration_s]`` outside the given spans."""
    rest: Spans = []
    position = 0.0
    for start, end in sorted(spans):
        if start > position:
            rest.append((position, start))
        position = max(position, end)
    if position < duration_s:
        rest.append((position, duration_s))
    return rest
