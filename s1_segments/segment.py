"""S1: phonation and non-phonation spans per file, blind to labels and recording place.

Usage, with the Kudio venv (it has kudio.vad and librosa):
  python -I -B s1_segments/segment.py --index s1_segments/out/index.csv [--workers 10] [--tracks]

The index has exactly four columns: file_id, database, material, wav. Any other
column stops the run, so a label or a place cannot reach this script. Every
threshold is worked out from the file itself. Files already in files.csv are
skipped, so a run can be resumed or extended.

Per frame (30 ms, 10 ms hop, centred, on the signal at 16 kHz or its own rate if
lower) three things are measured:
  - level in 300-3500 Hz against the file's own floor: the 2nd percentile of
    the frame levels, leaving out digital zeros;
  - voicing, pYIN on the same band;
  - kudio.vad.

Why the band: it keeps the rooms comparable. pYIN does not look at level, and
the HSA room has a steady tone at 123 Hz with a line at 246 Hz. With a lower edge
of 150 Hz it calls 7% of the HSA room recordings voiced against 0.6% at ESSUA;
with 300 Hz, 1.0% and 0.0%. The edge was chosen on the room recordings alone
(band_scan.py), which hold no voice and no label.

The frame classes:
  phonation        within 20 dB of the file's 95th percentile: above the threshold
                   and voiced or VAD, which keeps breathy voice. Further down: pYIN
                   voiced with probability at least 0.1, and above the threshold or
                   in VAD. pYIN flags breaths near the threshold at probability
                   close to 0, and those are not voice
  non-phonation    below the threshold, outside VAD, not digital zero, and not
                   voiced. The last test applies only while the level range is
                   under 25 dB: then the floor may sit inside the voice. Above
                   that, a frame under the threshold is 25 dB or more below the
                   voice, and pYIN saying "voiced" there is a tone in the room
                   (750 Hz in some SVD sessions), since it does not look at level
  other            everything else: breathing, clicks, breathy noise the VAD
                   rejects. It belongs to neither arm.

Frames inside a bridged phonation span are never non-phonation.

A file whose level range is under 20 dB has no floor to separate from, so it holds
one class only: at less than that, frames under the threshold would be under 12 dB
below the voice, and a rough breathy voice swings that much by itself. A voice
recording is then all phonation and no non-phonation: the alternative, trusting
pYIN, turns a breathy voice without
pauses into a whole file of "room noise". Only material "silence" (the room
recordings, used as a control) is taken to be non-phonation.

Non-phonation spans shorter than 100 ms are dropped, then a guard band
is taken off every edge that touches something other than the file's start or
end. The guard is 100 ms; 50 and 200 ms are kept for the sensitivity analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

# librosa's jit cache stays in this folder rather than in the Kudio venv.
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).resolve().parent / ".numba_cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "s0_audit"))

import kudio
import librosa
import numpy as np
import pandas as pd
from kudio.core.vad import vad
from readers import read_wav
from scipy.signal import butter, resample_poly, sosfiltfilt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INDEX_COLUMNS = ["file_id", "database", "material", "wav"]

TARGET_RATE = 16000
EPS = 1e-14


@dataclass(frozen=True)
class Params:
    band_low_hz: float = 300.0
    band_high_hz: float = 3500.0
    frame_s: float = 0.030
    hop_s: float = 0.010
    floor_percentile: float = 2.0
    margin_db: float = 8.0
    min_range_db: float = 20.0
    voiced_veto_range_db: float = 25.0
    vad_only_within_db: float = 20.0
    quiet_voice_min_prob: float = 0.1
    fill_gap_s: float = 0.05
    min_nonphonation_s: float = 0.10
    phonation_bridge_s: float = 0.15
    min_phonation_s: float = 0.10
    min_kept_s: float = 0.05
    guards_s: tuple[float, ...] = (0.05, 0.10, 0.20)
    f0_min_hz: float = 60.0
    f0_max_hz: float = 800.0


PARAMS = Params()


def guard_name(guard_s: float) -> str:
    return f"g{round(guard_s * 1000):03d}"


def to_analysis_rate(x: np.ndarray, rate: int) -> tuple[np.ndarray, int]:
    if rate <= TARGET_RATE:
        return x, rate
    g = np.gcd(rate, TARGET_RATE)
    return resample_poly(x, TARGET_RATE // g, rate // g), TARGET_RATE


def zero_runs(x: np.ndarray, rate: int, min_s: float = 0.010) -> list[tuple[float, float]]:
    """Runs of exact digital zero at least ``min_s`` long, in seconds."""
    edges = np.diff(np.concatenate(([0], (x == 0).astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    keep = stops - starts >= max(1, round(min_s * rate))
    return [(a / rate, b / rate) for a, b in zip(starts[keep], stops[keep])]


def frame_times(n_samples: int, rate: int, hop_s: float) -> np.ndarray:
    hop = round(hop_s * rate)
    return np.arange(1 + n_samples // hop) * hop / rate


def band_level_db(y: np.ndarray, rate: int, p: Params = PARAMS) -> tuple[np.ndarray, np.ndarray]:
    """Band-passed signal and its level per centred frame, in dB re full scale."""
    high = min(p.band_high_hz, 0.45 * rate)
    sos = butter(4, [p.band_low_hz, high], btype="band", fs=rate, output="sos")
    band = sosfiltfilt(sos, y) if y.size > 60 else y
    cum = np.concatenate(([0.0], np.cumsum(band**2)))
    centres = np.round(frame_times(y.size, rate, p.hop_s) * rate).astype(int)
    half = round(p.frame_s * rate / 2)
    lo, hi = np.clip(centres - half, 0, y.size), np.clip(centres + half, 0, y.size)
    power = (cum[hi] - cum[lo]) / np.maximum(hi - lo, 1)
    return band, 10 * np.log10(np.maximum(power, 0) + EPS)


def spans_to_mask(spans: list[tuple[float, float]], times: np.ndarray) -> np.ndarray:
    mask = np.zeros(times.size, dtype=bool)
    for a, b in spans:
        mask |= (times >= a) & (times < b)
    return mask


def mask_to_spans(mask: np.ndarray, hop_s: float, duration_s: float) -> list[tuple[float, float]]:
    """A frame stands for the hop around its centre."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [
        (max(0.0, (a - 0.5) * hop_s), min(duration_s, (b - 0.5) * hop_s))
        for a, b in zip(starts, stops)
    ]


def close_gaps(spans: list[tuple[float, float]], gap_s: float) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in spans:
        if out and a - out[-1][1] <= gap_s:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def apply_guard(
    spans: list[tuple[float, float]], guard_s: float, duration_s: float, min_kept_s: float
) -> list[tuple[float, float]]:
    """Shrink each span by the guard, except at the very start or end of the file."""
    out = []
    for a, b in spans:
        left = a if a <= 0.0 else a + guard_s
        right = b if b >= duration_s else b - guard_s
        if right - left >= min_kept_s:
            out.append((left, right))
    return out


def total(spans: list[tuple[float, float]]) -> float:
    return float(sum(b - a for a, b in spans))


@dataclass
class Segmentation:
    summary: dict
    spans: list[dict]
    tracks: dict


def segment_signal(
    x: np.ndarray, rate: int, p: Params = PARAMS, contains_voice: bool = True
) -> Segmentation:
    """``x`` is mono float in [-1, 1); ``rate`` in Hz."""
    duration = x.size / rate
    zeros = zero_runs(x, rate)
    y, sr = to_analysis_rate(x, rate)
    band, level = band_level_db(y, sr, p)
    times = frame_times(y.size, sr, p.hop_s)
    # A frame touching a run of digital zeros says nothing about the room.
    padded = [(a - p.frame_s, b + p.frame_s) for a, b in zeros]
    zero_mask = spans_to_mask(padded, times)
    audible = level[~zero_mask]
    if audible.size == 0:
        floor = high = float("nan")
    else:
        floor = float(np.percentile(audible, p.floor_percentile))
        high = float(np.percentile(audible, 95))
    threshold = floor + p.margin_db
    range_db = high - floor
    separable = bool(range_db >= p.min_range_db)

    hop = round(p.hop_s * sr)
    frame_length = int(2 ** np.ceil(np.log2(0.060 * sr)))
    if audible.size and y.size >= frame_length:
        _, flag, prob = librosa.pyin(
            band,
            fmin=p.f0_min_hz,
            fmax=p.f0_max_hz,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop,
        )
        voiced = np.resize(flag, times.size).astype(bool)
        prob = np.resize(prob, times.size)
    else:
        voiced = np.zeros(times.size, dtype=bool)
        prob = np.zeros(times.size)
    voiced &= ~zero_mask
    voiced_fraction = float(voiced[~zero_mask].mean()) if audible.size else 0.0

    # kudio.vad raises on clips under 0.43 s: its 41-frame smoothing window comes out
    # longer than the frames. One SVD a_n is that short.
    vad_spans = vad(y, sr) if audible.size and y.size >= 0.45 * sr else []
    in_vad = spans_to_mask(vad_spans, times)

    if not audible.size:
        phonation_mask = np.zeros(times.size, dtype=bool)
        quiet = np.zeros(times.size, dtype=bool)
    elif separable:
        above = level >= threshold
        loud = level >= high - p.vad_only_within_db
        phonation_mask = loud & above & (voiced | in_vad)
        sure = voiced & (prob >= p.quiet_voice_min_prob)
        phonation_mask |= sure & (above | in_vad)
        quiet = ~above & ~in_vad
        if range_db < p.voiced_veto_range_db:
            quiet &= ~voiced
    elif contains_voice:
        phonation_mask = np.ones(times.size, dtype=bool)
        quiet = np.zeros(times.size, dtype=bool)
    else:
        phonation_mask = np.zeros(times.size, dtype=bool)
        quiet = ~voiced
    phonation_mask &= ~zero_mask
    quiet &= ~zero_mask
    other = ~(phonation_mask | quiet | zero_mask)
    low_voiced = voiced & ~in_vad & ~phonation_mask

    phon = close_gaps(mask_to_spans(phonation_mask, p.hop_s, duration), p.phonation_bridge_s)
    phon = [s for s in phon if s[1] - s[0] >= p.min_phonation_s]
    # Bridging can cover frames that are below the threshold, as when pYIN flickers
    # around it. They go with the phonation span.
    quiet &= ~spans_to_mask(phon, times)
    raw = close_gaps(mask_to_spans(quiet, p.hop_s, duration), p.fill_gap_s)
    raw = [s for s in raw if s[1] - s[0] >= p.min_nonphonation_s]

    guard_spans = {guard_name(g): apply_guard(raw, g, duration, p.min_kept_s) for g in p.guards_s}
    inside = spans_to_mask(guard_spans[guard_name(0.10)], times)
    summary = {
        "rate": rate,
        "analysis_rate": sr,
        "duration_s": duration,
        "floor_db": floor,
        "threshold_db": threshold,
        "margin_db": p.margin_db,
        "floor_percentile": p.floor_percentile,
        "band_low_hz": p.band_low_hz,
        "band_high_hz": min(p.band_high_hz, 0.45 * sr),
        "level_p95_db": high,
        "range_db": range_db,
        "separable": separable,
        "below_threshold_fraction": float(np.mean(level[~zero_mask] < threshold))
        if audible.size
        else 0.0,
        "voiced_fraction": voiced_fraction,
        "digital_zero_s": total(zeros),
        "phonation_s": total(phon),
        "n_phonation": len(phon),
        "vad_s": total(vad_spans),
        "n_vad": len(vad_spans),
        "other_s": float(other.sum() * p.hop_s),
        "low_level_voiced_s": float(low_voiced.sum() * p.hop_s),
        "nonphonation_raw_s": total(raw),
        "n_nonphonation_raw": len(raw),
        "nonphonation_voiced_prob": float(prob[inside].mean()) if inside.any() else float("nan"),
        "nonphonation_voiced_fraction": float(voiced[inside].mean())
        if inside.any()
        else float("nan"),
    }
    for name, spans in guard_spans.items():
        summary[f"nonphonation_{name}_s"] = total(spans)
        summary[f"n_nonphonation_{name}"] = len(spans)
        summary[f"longest_nonphonation_{name}_s"] = max((b - a for a, b in spans), default=0.0)
    rows = [{"kind": "phonation", "start_s": a, "end_s": b} for a, b in phon]
    rows += [{"kind": "vad", "start_s": a, "end_s": b} for a, b in vad_spans]
    rows += [{"kind": "digital_zero", "start_s": a, "end_s": b} for a, b in zeros]
    rows += [{"kind": "nonphonation_raw", "start_s": a, "end_s": b} for a, b in raw]
    for name, spans in guard_spans.items():
        rows += [{"kind": f"nonphonation_{name}", "start_s": a, "end_s": b} for a, b in spans]
    tracks = {
        "times": times.astype(np.float32),
        "level_db": level.astype(np.float32),
        "voiced": voiced,
        "voiced_prob": prob.astype(np.float32),
        "in_vad": in_vad,
    }
    return Segmentation(summary=summary, spans=rows, tracks=tracks)


def segment_row(row: dict, tracks_dir: str | None = None) -> tuple[dict, list[dict]]:
    identity = {k: row[k] for k in INDEX_COLUMNS}
    try:
        wav = read_wav(ROOT / row["wav"])
        if wav.n_frames == 0:
            return {**identity, "status": "empty"}, []
        contains_voice = row["material"] != "silence"
        result = segment_signal(wav.as_float()[:, 0], wav.rate, contains_voice=contains_voice)
    except Exception as err:  # one unreadable file must not stop a long run
        return {**identity, "status": f"error: {err}"}, []
    if tracks_dir:
        np.savez_compressed(Path(tracks_dir) / f"{row['file_id']}.npz", **result.tracks)
    spans = [{"file_id": row["file_id"], **s} for s in result.spans]
    return {**identity, "status": "ok", **result.summary}, spans


def load_index(path: Path) -> pd.DataFrame:
    index = pd.read_csv(path, dtype=str)
    if list(index.columns) != INDEX_COLUMNS:
        got = list(index.columns)
        sys.exit(f"the index must have exactly the columns {INDEX_COLUMNS}, got {got}")
    if index.file_id.duplicated().any():
        sys.exit("file_id is not unique in the index")
    return index


def write_tables(files: list[dict], spans: list[dict], out: Path) -> None:
    pd.DataFrame(files).to_csv(out / "files.csv", index=False)
    table = pd.DataFrame(spans, columns=["file_id", "kind", "start_s", "end_s"])
    table.to_csv(out / "spans.csv", index=False)


def warm_jit_cache(p: Params = PARAMS) -> None:
    """Compile pYIN's kernels here first.

    Workers that all start from an empty numba cache write to it at once, and
    a cache broken that way crashes every later run with an access violation.
    """
    noise = np.random.default_rng(0).standard_normal(TARGET_RATE)
    librosa.pyin(
        noise,
        fmin=p.f0_min_hz,
        fmax=p.f0_max_hz,
        sr=TARGET_RATE,
        frame_length=1024,
        hop_length=round(p.hop_s * TARGET_RATE),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=HERE / "out" / "index.csv")
    parser.add_argument("--out", type=Path, default=HERE / "out")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--tracks", action="store_true", help="keep every file's frame tracks")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    record = {
        **asdict(PARAMS),
        "target_rate": TARGET_RATE,
        "versions": {
            "python": sys.version.split()[0],
            "kudio": kudio.__version__,
            "librosa": librosa.__version__,
            "numpy": np.__version__,
        },
        "segment_py_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.out / "params.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    index = load_index(args.index)
    files: list[dict] = []
    spans: list[dict] = []
    if (args.out / "files.csv").exists():
        # A file that raised gets another try; one that was empty stays as it was.
        kept = pd.read_csv(args.out / "files.csv")
        files = kept[~kept.status.str.startswith("error")].to_dict("records")
        spans = pd.read_csv(args.out / "spans.csv").to_dict("records")
    done = {f["file_id"] for f in files}
    todo = index[~index.file_id.isin(done)].to_dict("records")

    tracks_dir = args.out / "tracks"
    if args.tracks:
        tracks_dir.mkdir(exist_ok=True)
    targets = [str(tracks_dir) if args.tracks else None] * len(todo)

    warm_jit_cache()
    with ProcessPoolExecutor(args.workers) as pool:
        for i, (summary, rows) in enumerate(pool.map(segment_row, todo, targets, chunksize=4), 1):
            files.append(summary)
            spans.extend(rows)
            if i % 200 == 0:
                write_tables(files, spans, args.out)
                print(f"{i}/{len(todo)}", flush=True)
    write_tables(files, spans, args.out)
    failed = sum(f["status"] != "ok" for f in files)
    print(f"{len(files)} files, {failed} not segmented")


if __name__ == "__main__":
    main()
