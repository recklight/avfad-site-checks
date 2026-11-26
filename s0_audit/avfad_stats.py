"""Recording statistics for AVFAD WAV files in the inventory.

Usage: python -I s0_audit/avfad_stats.py [--tasks 002 ...] [--workers 10]

Reads s0_audit/out/avfad_files.csv and writes s0_audit/out/avfad_stats.csv.
Speaker files are named <code><task>.wav; task 002 is the sustained /a/,
which is all this study uses, so it is the default. The room-silence files
in the README folder are always included.
The statistics are taken on the first channel. Files with more channels also
get the RMS of each channel and the largest difference between the first two,
which shows whether the second channel is silent, a copy, or another signal.
"""

from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from avfad_extract import OUT, ROOT, require_access
from readers import read_wav
from stats import EPS, recording_stats


def one(path: str) -> dict:
    wav = read_wav(ROOT / path)
    x = wav.as_float()
    first = x[:, 0]
    stats = recording_stats(first, wav.rate)
    # Exact digital zeros at the start and end, separate from low-level noise.
    nz = np.flatnonzero(first)
    stats["lead_zero_s"] = (nz[0] if nz.size else first.size) / wav.rate
    stats["trail_zero_s"] = (first.size - 1 - nz[-1] if nz.size else first.size) / wav.rate
    if wav.channels > 1:
        rms = 10 * np.log10(np.mean(x**2, axis=0) + EPS)
        stats["rms_dbfs_by_channel"] = " ".join(f"{v:.1f}" for v in rms)
        stats["max_abs_diff_ch1_ch2"] = float(np.max(np.abs(x[:, 0] - x[:, 1]), initial=0.0))
    return {"path": path, **stats}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["002"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args(argv)
    require_access()
    files = pd.read_csv(OUT / "avfad_files.csv")
    task = files.path.str.extract(r"/[A-Za-z]{3}(\d{3})\.wav$", flags=re.IGNORECASE)[0]
    silence = files.path.str.contains("_silence", case=False)
    wanted = (files.ext == ".wav") & files.rate.notna() & (task.isin(args.tasks) | silence)
    paths = files.loc[wanted, "path"].tolist()
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(one, paths, chunksize=16))
    pd.DataFrame(rows).to_csv(OUT / "avfad_stats.csv", index=False)
    print(f"{len(rows)} files")


if __name__ == "__main__":
    main()
