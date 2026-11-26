"""Synthetic recordings with a known phonation layout, for tests and threshold_eval.py.

The two rooms copy what S0 measured on the AVFAD room-silence files, in dBFS and
99% rolloff: one quiet and broadband (-68 dB, 20.8 kHz), one noisier and
concentrated at low frequencies (-56 dB, 1.1 kHz). A third adds what the HSA room
recordings turned out to have and the first two lacked: a steady tone at 123 Hz
with a line at 246 Hz, 30-47 dB above the spectrum around it. The voice level
copies the /a/ files, about -40 dB over the whole file. Nothing here reads a
real recording.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfilt

ROOMS = {
    "quiet_broadband": {"level_db": -68.0, "kind": "white"},
    "noisy_lowfreq": {"level_db": -56.0, "kind": "lowfreq"},
    "noisy_tonal": {"level_db": -56.0, "kind": "tonal"},
}
FORMANTS = ((730.0, 90.0), (1090.0, 110.0), (2440.0, 170.0))
FORMANTS_I = ((270.0, 60.0), (2290.0, 100.0), (3010.0, 170.0))
FORMANTS_U = ((300.0, 60.0), (870.0, 90.0), (2240.0, 170.0))
# Mains hum: frequency in Hz and starting phase.
HUM = ((50.0, 0.3), (100.0, 1.1), (150.0, 2.0), (200.0, 0.7))


def rms_db(x: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(np.square(x)) + 1e-20))


def scaled(x: np.ndarray, level_db: float) -> np.ndarray:
    return x * 10 ** ((level_db - rms_db(x)) / 20)


def room_noise(kind: str, n: int, rate: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-level noise: white, or low-passed noise with a mains hum and its harmonics."""
    white = rng.standard_normal(n)
    if kind == "white":
        return white
    if kind not in ("lowfreq", "tonal"):
        raise ValueError(f"unknown room kind {kind!r}")
    sos = butter(2, min(400.0, 0.45 * rate), btype="low", fs=rate, output="sos")
    low = sosfilt(sos, white)
    low = low / np.sqrt(np.mean(low**2) + 1e-20)
    t = np.arange(n) / rate
    hum = sum(np.sin(2 * np.pi * f * t + p) / k for k, (f, p) in enumerate(HUM, 1))
    noise = low + 0.7 * hum
    if kind == "tonal":
        noise = (
            noise + 4.0 * np.sin(2 * np.pi * 123.0 * t) + 1.0 * np.sin(2 * np.pi * 246.0 * t + 0.4)
        )
    return noise


def vowel(
    duration_s: float,
    rate: int,
    rng: np.random.Generator,
    f0: float = 180.0,
    breathiness: float = 0.0,
    formants: tuple = FORMANTS,
    tremor_hz: float = 0.0,
    tremor_depth_db: float = 0.0,
) -> np.ndarray:
    """A sustained /a/ at unit level: harmonics through three formants, with slow F0 drift.

    ``breathiness`` is the share of the power given to noise shaped by the same
    formants. ``tremor_depth_db`` is the peak-to-trough amplitude swing.
    """
    n = round(duration_s * rate)
    t = np.arange(n) / rate
    drift = 1 + 0.01 * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 6.28))
    jitter = np.cumsum(rng.standard_normal(n)) / np.sqrt(rate) * 0.02
    phase = 2 * np.pi * np.cumsum(f0 * drift * (1 + jitter)) / rate
    top = min(5000.0, 0.45 * rate)
    harmonics = np.arange(1, int(top // f0) + 1)
    weights = np.array([_formant_gain(k * f0, formants) / k**0.5 for k in harmonics])
    voiced = np.sum(
        [w * np.sin(k * phase) for k, w in zip(harmonics, weights)],
        axis=0,
    )
    voiced = voiced / np.sqrt(np.mean(voiced**2) + 1e-20)
    if breathiness > 0:
        noise = _formant_noise(n, rate, rng, formants)
        voiced = np.sqrt(1 - breathiness) * voiced + np.sqrt(breathiness) * noise
    edge = min(round(0.04 * rate), n // 2)
    envelope = np.ones(n)
    if edge:
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(edge) / edge)
        envelope[:edge] = ramp
        envelope[-edge:] = ramp[::-1]
    if tremor_hz > 0:
        swing = 10 ** (tremor_depth_db / 40 * np.sin(2 * np.pi * tremor_hz * t))
        envelope = envelope * swing
    return voiced * envelope


def _formant_gain(f: float, formants: tuple = FORMANTS) -> float:
    return sum(1.0 / np.hypot(f - fc, bw) * bw for fc, bw in formants)


def _formant_noise(
    n: int, rate: int, rng: np.random.Generator, formants: tuple = FORMANTS
) -> np.ndarray:
    white = rng.standard_normal(n)
    out = np.zeros(n)
    for fc, bw in formants:
        lo = max(50.0, fc - 2 * bw)
        hi = min(fc + 2 * bw, 0.45 * rate)
        if hi > lo:
            out += sosfilt(butter(2, [lo, hi], btype="band", fs=rate, output="sos"), white)
    return out / np.sqrt(np.mean(out**2) + 1e-20)


@dataclass
class Recording:
    x: np.ndarray
    rate: int
    phonation: list[tuple[float, float]]  # true spans, seconds

    @property
    def duration_s(self) -> float:
        return self.x.size / self.rate

    @property
    def pauses(self) -> list[tuple[float, float]]:
        out, pos = [], 0.0
        for a, b in self.phonation:
            if a > pos:
                out.append((pos, a))
            pos = b
        if pos < self.duration_s:
            out.append((pos, self.duration_s))
        return out


# layout: seconds of (lead, [token, pause, token, ...], trail)
CASES = {
    "brief": {"lead": 0.3, "tokens": (2.0, 2.0), "pauses": (0.8,), "trail": 0.3},
    "three_reps": {"lead": 0.3, "tokens": (4.0, 4.0, 4.0), "pauses": (0.8, 0.8), "trail": 0.3},
    "short_pauses": {"lead": 0.0, "tokens": (4.0, 4.0, 4.0), "pauses": (0.35, 0.35), "trail": 0.0},
    "one_short_pause": {"lead": 0.0, "tokens": (7.0, 7.0), "pauses": (0.4,), "trail": 0.0},
    "no_pause": {"lead": 0.0, "tokens": (13.0,), "pauses": (), "trail": 0.0},
    "silence_only": {"lead": 0.0, "tokens": (), "pauses": (), "trail": 15.0},
}
VOICES = {
    "modal": {},
    "quiet": {"voice_db": -52.0},
    "breathy": {"breathiness": 0.7},
    "tremor": {"tremor_hz": 5.0, "tremor_depth_db": 14.0},
    "breathy_tremor": {"breathiness": 0.7, "tremor_hz": 5.0, "tremor_depth_db": 14.0},
    "low_f0": {"f0": 95.0},
    "vowel_i": {"formants": FORMANTS_I},
    "vowel_u": {"formants": FORMANTS_U},
    "vowel_u_low_f0": {"formants": FORMANTS_U, "f0": 95.0},
    "high_f0": {"f0": 400.0},
}


def recording(
    case: str,
    room: str,
    rate: int,
    seed: int,
    voice: str = "modal",
    voice_db: float = -37.0,
) -> Recording:
    """Tokens at ``voice_db`` (about -40 dB over a file with pauses) on top of a room's noise."""
    rng = np.random.default_rng(seed)
    layout = CASES[case]
    options = dict(VOICES[voice])
    voice_db = options.pop("voice_db", voice_db)
    chunks, spans, pos = [], [], layout["lead"]
    if layout["lead"]:
        chunks.append(np.zeros(round(layout["lead"] * rate)))
    tokens = layout["tokens"]
    for i, dur in enumerate(tokens):
        token = vowel(dur, rate, rng, **options)
        chunks.append(token * 10 ** (voice_db / 20))
        spans.append((pos, pos + dur))
        pos += dur
        if i < len(tokens) - 1:
            gap = layout["pauses"][i]
            chunks.append(np.zeros(round(gap * rate)))
            pos += gap
    if layout["trail"]:
        chunks.append(np.zeros(round(layout["trail"] * rate)))
    clean = np.concatenate(chunks)
    spec = ROOMS[room]
    noise = room_noise(spec["kind"], clean.size, rate, rng)
    x = clean + scaled(noise, spec["level_db"])
    return Recording(x=x, rate=rate, phonation=spans)
