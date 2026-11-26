"""Recording statistics for every extracted SVD speech file.

Usage: python -I s0_audit/svd_stats.py [--workers 10]

Reads s0_audit/out/svd_files.csv and writes s0_audit/out/svd_stats.csv.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats import recording_stats

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "out"


def one(args: tuple[int, str, str]) -> dict:
    session, material, wav = args
    rate, x = wavfile.read(ROOT / wav)
    stats = recording_stats(x.astype(np.float64) / 32768.0, rate)
    # Exact digital zeros at the start and end, separate from low-level noise.
    nz = np.flatnonzero(x)
    stats["lead_zero_s"] = (nz[0] if nz.size else x.size) / rate
    stats["trail_zero_s"] = (x.size - 1 - nz[-1] if nz.size else x.size) / rate
    return {"session": session, "material": material, **stats}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    files = pd.read_csv(OUT / "svd_files.csv")
    files = files[files.wav.notna()]
    jobs = list(zip(files.session, files.material, files.wav))
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(one, jobs, chunksize=16))
    pd.DataFrame(rows).to_csv(OUT / "svd_stats.csv", index=False)
    print(f"{len(rows)} files")


if __name__ == "__main__":
    main()
