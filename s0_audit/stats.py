"""Per-file recording-condition statistics for the S0 descriptive tables.

All levels are dB relative to digital full scale (a full-scale square wave is
0 dB). The lead and trail measures are a coarse energy threshold for
description only; the analysis intervals come from the S1 segmentation.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

FRAME_S = 0.030
HOP_S = 0.015
# A frame counts as active when it is within ACTIVE_DB of the file's
# 95th-percentile frame energy.
ACTIVE_DB = 30.0
EPS = 1e-12


def frame_energy_db(x: np.ndarray, rate: int) -> np.ndarray:
    frame = max(1, round(FRAME_S * rate))
    hop = max(1, round(HOP_S * rate))
    if x.size < frame:
        return np.array([10 * np.log10(np.mean(x**2) + EPS)]) if x.size else np.array([])
    n = 1 + (x.size - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return 10 * np.log10(np.mean(x[idx] ** 2, axis=1) + EPS)


def recording_stats(x: np.ndarray, rate: int) -> dict:
    """``x`` is float in [-1, 1); ``rate`` in Hz."""
    out: dict = {"rate": rate, "n_samples": int(x.size), "duration_s": x.size / rate}
    if x.size == 0:
        return out
    out["rms_dbfs"] = 10 * np.log10(np.mean(x**2) + EPS)
    out["peak_dbfs"] = 20 * np.log10(np.max(np.abs(x)) + EPS)
    # Samples at either int16 extreme.
    out["clip_fraction"] = float(np.mean((x >= 32767 / 32768) | (x <= -1.0)))
    out["dc_offset"] = float(np.mean(x))
    energy = frame_energy_db(x, rate)
    out["noise_floor_dbfs"] = float(np.percentile(energy, 10))
    p95 = float(np.percentile(energy, 95))
    active = energy >= p95 - ACTIVE_DB
    hop = HOP_S
    if active.any():
        first = int(np.argmax(active))
        last = int(len(active) - 1 - np.argmax(active[::-1]))
        out["lead_s"] = first * hop
        out["trail_s"] = max(0.0, out["duration_s"] - (last * hop + FRAME_S))
        out["active_fraction"] = float(np.mean(active))
    nper = int(2 ** np.ceil(np.log2(0.040 * rate)))
    if x.size >= nper:
        freqs, psd = welch(x, fs=rate, nperseg=nper)
        cum = np.cumsum(psd)
        out["rolloff99_hz"] = float(freqs[np.searchsorted(cum, 0.99 * cum[-1])])
        # Share of energy above 1 kHz; an EGG channel sits far lower than speech,
        # which is how swapped speech/EGG files show up.
        out["above_1k_fraction"] = float(psd[freqs >= 1000].sum() / (psd.sum() + EPS))
    return out
