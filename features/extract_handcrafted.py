"""Handcrafted features of each condition, one CSV per database, material and condition.

Frames never cross a span: every span is framed by itself and the statistics pool the
frames of all spans of a file. A span shorter than one frame of a feature adds nothing
to that feature. The parameters are in section 4 of analysis_plan.md.

Run from the paper's own venv, for example:

    OMP_NUM_THREADS=1 .venv/Scripts/python.exe -I -B features/extract_handcrafted.py \
        --database avfad --material a --jobs 16
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import speechdsp  # noqa: E402
from common import (  # noqa: E402
    ROOT,
    Spans,
    complement,
    cut,
    load_signal,
    normalise_to_phonation,
    rms_dbfs,
)

S1_OUT = ROOT / "s1_segments" / "out"
OUT = HERE / "out"

N_MFCC = 13
LTAS_WINDOW_S = 0.032
LTAS_HOP_S = 0.016
LTAS_N_FFT = 1024
# Base-ten third-octave centres, 63 Hz to 6.3 kHz. The top band ends at 7.08 kHz, below the
# 8 kHz Nyquist limit of the analysis rate; the bottom ones hold the low room tones.
BAND_CENTRES_HZ = 1000.0 * 10.0 ** (np.arange(-12, 9) / 10.0)
LEVEL_FRAME_S = 0.030
LEVEL_HOP_S = 0.010
# -120 dB: exact digital zero must not give minus infinity.
POWER_FLOOR = 1e-12

MFCC_COLUMNS = [
    f"{name}_{stat}_{i:02d}"
    for name in ("mfcc", "dmfcc")
    for stat in ("mean", "std")
    for i in range(N_MFCC)
]
LTAS_COLUMNS = [f"ltas_{round(c):04d}" for c in BAND_CENTRES_HZ]
LEVEL_COLUMNS = ["level_mean", "level_std", "level_p05", "level_p50", "level_p95"]
FEATURE_COLUMNS = MFCC_COLUMNS + LTAS_COLUMNS + LEVEL_COLUMNS


@dataclass(frozen=True)
class Condition:
    """Which samples a condition keeps and whether the file is scaled first."""

    kind: str  # full, phonation, phonation_breath or a nonphonation_g* span kind
    normalise: bool


CONDITIONS = {
    "full_original": Condition("full", False),
    "full_norm": Condition("full", True),
    "phonation_norm": Condition("phonation", True),
    "phonation_breath_norm": Condition("phonation_breath", True),
    "nonphon_g100": Condition("nonphonation_g100", False),
    "nonphon_g050": Condition("nonphonation_g050", False),
    "nonphon_g200": Condition("nonphonation_g200", False),
}
# What each material needs for the confirmatory tests and their sensitivity analyses.
PLANNED = {
    ("avfad", "a"): [
        "full_original",
        "phonation_norm",
        "nonphon_g100",
        "nonphon_g050",
        "nonphon_g200",
    ],
    ("svd", "iau"): [
        "full_norm",
        "phonation_norm",
        "phonation_breath_norm",
        "nonphon_g100",
        "nonphon_g050",
        "nonphon_g200",
    ],
    ("svd", "a_n"): ["full_norm"],
}


def frame_starts(n: int, frame: int, hop: int) -> np.ndarray:
    if n < frame:
        return np.empty(0, dtype=np.intp)
    return np.arange(0, n - frame + 1, hop)


def frames_of(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    starts = frame_starts(x.size, frame, hop)
    return x[starts[:, None] + np.arange(frame)] if starts.size else np.empty((0, frame))


def band_bins(rate: int) -> list[np.ndarray]:
    """FFT bins of every third-octave band; empty above the Nyquist limit of ``rate``."""
    freqs = np.fft.rfftfreq(LTAS_N_FFT, 1.0 / rate)
    bands = []
    for centre in BAND_CENTRES_HZ:
        low, high = centre * 10.0**-0.05, centre * 10.0**0.05
        inside = (freqs >= low) & (freqs < high) & (high <= rate / 2.0)
        bands.append(np.flatnonzero(inside))
    return bands


def ltas_db(pieces: list[np.ndarray], rate: int) -> np.ndarray:
    """Power averaged over all frames first, then gathered in bands, then dB."""
    frame = round(LTAS_WINDOW_S * rate)
    hop = round(LTAS_HOP_S * rate)
    window = np.hanning(frame)
    total = np.zeros(LTAS_N_FFT // 2 + 1)
    count = 0
    for piece in pieces:
        frames = frames_of(piece, frame, hop)
        if frames.shape[0]:
            spectrum = np.fft.rfft(frames * window, n=LTAS_N_FFT, axis=1)
            total += np.sum(np.abs(spectrum) ** 2, axis=0) / np.sum(window**2)
            count += frames.shape[0]
    if count == 0:
        return np.full(BAND_CENTRES_HZ.size, np.nan)
    mean_power = total / count
    return np.array(
        [
            10.0 * np.log10(mean_power[bins].sum() + POWER_FLOOR) if bins.size else np.nan
            for bins in band_bins(rate)
        ]
    )


def level_statistics(pieces: list[np.ndarray], rate: int) -> np.ndarray:
    frame = round(LEVEL_FRAME_S * rate)
    hop = round(LEVEL_HOP_S * rate)
    levels = []
    for piece in pieces:
        frames = frames_of(piece, frame, hop)
        if frames.shape[0]:
            levels.append(10.0 * np.log10(np.mean(frames**2, axis=1) + POWER_FLOOR))
    if not levels:
        return np.full(len(LEVEL_COLUMNS), np.nan)
    values = np.concatenate(levels)
    p05, p50, p95 = np.percentile(values, [5, 50, 95])
    return np.array([values.mean(), values.std(ddof=0), p05, p50, p95])


def mfcc_statistics(pieces: list[np.ndarray], rate: int) -> np.ndarray:
    static, moving = [], []
    for piece in pieces:
        coefficients = speechdsp.mfcc(piece, rate)
        if coefficients.shape[0]:
            static.append(coefficients)
            # Per span, so the regression window never reaches over a gap.
            moving.append(speechdsp.delta(coefficients))
    if not static:
        return np.full(len(MFCC_COLUMNS), np.nan)
    c, d = np.concatenate(static), np.concatenate(moving)
    return np.concatenate([c.mean(0), c.std(0, ddof=0), d.mean(0), d.std(0, ddof=0)])


def features_of(pieces: list[np.ndarray], rate: int) -> np.ndarray:
    return np.concatenate(
        [mfcc_statistics(pieces, rate), ltas_db(pieces, rate), level_statistics(pieces, rate)]
    )


def spans_for(kind: str, duration_s: float, spans_by_kind: dict[str, Spans]) -> Spans:
    if kind == "full":
        return [(0.0, duration_s)]
    if kind == "phonation_breath":
        # Everything that is neither quiet room nor digital zero: phonation plus "other".
        quiet = spans_by_kind.get("nonphonation_raw", []) + spans_by_kind.get("digital_zero", [])
        return complement(quiet, duration_s)
    return spans_by_kind.get(kind, [])


def extract_file(
    job: tuple[str, str, float, dict[str, Spans], list[str]],
) -> dict[str, tuple[np.ndarray | None, dict[str, float]]]:
    _, wav, duration_s, spans_by_kind, conditions = job
    x, rate = load_signal(wav)
    phonation = spans_by_kind.get("phonation", [])
    scaled, phonation_level = (None, np.nan)
    if phonation:
        scaled, phonation_level = normalise_to_phonation(x, rate, phonation)
    results: dict[str, tuple[np.ndarray | None, dict[str, float]]] = {}
    for name in conditions:
        condition = CONDITIONS[name]
        signal = scaled if condition.normalise else x
        spans = spans_for(condition.kind, duration_s, spans_by_kind)
        info = {
            "phonation_rms_dbfs": phonation_level,
            "kept_s": float(sum(end - start for start, end in spans)),
            "n_spans": float(len(spans)),
        }
        if signal is None or not spans:
            results[name] = (None, info)
            continue
        pieces = cut(signal, rate, spans)
        info["kept_rms_dbfs"] = rms_dbfs(np.concatenate(pieces)) if pieces else np.nan
        results[name] = (features_of(pieces, rate), info)
    return results


def jobs_for(database: str, material: str, conditions: list[str]) -> list[tuple]:
    files = pd.read_csv(S1_OUT / "files.csv", dtype={"file_id": str})
    chosen = files[
        (files["database"] == database)
        & (files["material"] == material)
        & (files["status"] == "ok")
    ].sort_values("file_id")
    spans = pd.read_csv(S1_OUT / "spans.csv")
    spans = spans[spans["file_id"].isin(chosen["file_id"])]
    grouped: dict[str, dict[str, Spans]] = {}
    for (file_id, kind), part in spans.groupby(["file_id", "kind"], sort=False):
        grouped.setdefault(file_id, {})[kind] = list(
            zip(part["start_s"].astype(float), part["end_s"].astype(float), strict=True)
        )
    return [
        (row.file_id, row.wav, float(row.duration_s), grouped.get(row.file_id, {}), conditions)
        for row in chosen.itertuples()
    ]


def write_tables(
    database: str, material: str, conditions: list[str], jobs: list[tuple], outputs: list[dict]
) -> None:
    OUT.mkdir(exist_ok=True)
    for name in conditions:
        rows = []
        skipped = 0
        for job, output in zip(jobs, outputs, strict=True):
            values, info = output[name]
            if values is None:
                skipped += 1
                continue
            rows.append(
                {"file_id": job[0], **info, **dict(zip(FEATURE_COLUMNS, values, strict=True))}
            )
        table = pd.DataFrame(rows)
        path = OUT / f"{database}_{material}_{name}.csv"
        table.to_csv(path, index=False, lineterminator="\n")
        print(f"{path.name}: {len(table)} files, {skipped} with nothing to keep", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", required=True)
    parser.add_argument("--material", required=True)
    parser.add_argument("--conditions", nargs="*", help="default: the planned ones")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--limit", type=int, help="first N files, to measure speed; writes nothing")
    args = parser.parse_args()
    conditions = args.conditions or PLANNED[(args.database, args.material)]
    if args.database == "avfad":
        sys.path.insert(0, str(ROOT / "s0_audit"))
        from avfad_extract import require_access

        require_access()
    jobs = jobs_for(args.database, args.material, conditions)
    if args.limit:
        jobs = jobs[: args.limit]
    started = time.perf_counter()
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as pool:
            outputs = list(pool.map(extract_file, jobs, chunksize=4))
    else:
        outputs = [extract_file(job) for job in jobs]
    elapsed = time.perf_counter() - started
    print(f"{len(jobs)} files in {elapsed:.1f} s with {args.jobs} workers", flush=True)
    if not args.limit:
        write_tables(args.database, args.material, conditions, jobs, outputs)


if __name__ == "__main__":
    main()
