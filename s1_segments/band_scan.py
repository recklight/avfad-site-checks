"""Pick the lower band edge on the room recordings only: they hold no voice and no label.

Usage, with the Kudio venv: python -I -B s1_segments/band_scan.py [--workers 14]

Runs the segmenter on the first 30 s of each AVFAD room recording at several
lower edges and reports, per place, how much comes out as non-phonation and how
much pYIN calls voiced. Writes s1_segments/out/band_scan.csv.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "s0_audit"))

import pandas as pd
import segment as sg
from readers import read_wav

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EDGES_HZ = (150.0, 200.0, 250.0, 300.0, 400.0, 500.0)
SECONDS = 30


def one(job: tuple[str, str, float]) -> dict:
    wav_path, place, edge = job
    wav = read_wav(ROOT / wav_path)
    x = wav.as_float()[: SECONDS * wav.rate, 0]
    params = dataclasses.replace(sg.PARAMS, band_low_hz=edge)
    s = sg.segment_signal(x, wav.rate, params, contains_voice=False).summary
    return {
        "wav": wav_path,
        "place": place,
        "band_low_hz": edge,
        "nonphonation_share": s["nonphonation_raw_s"] / s["duration_s"],
        "voiced_fraction": s["voiced_fraction"],
        "floor_db": s["floor_db"],
        "separable": s["separable"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    rooms = pd.read_csv(
        HERE.parent / "s0_audit" / "out" / "avfad_silence.csv", usecols=["path", "place"]
    )
    jobs = [(r.path, r.place, e) for r in rooms.itertuples() for e in EDGES_HZ]
    sg.warm_jit_cache()
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(one, jobs, chunksize=4))
    table = pd.DataFrame(rows)
    table.to_csv(HERE / "out" / "band_scan.csv", index=False)
    summary = (
        table[table.place.isin(["ESSUA", "HSA"])]
        .groupby(["band_low_hz", "place"])
        .agg(
            share_median=("nonphonation_share", "median"),
            share_p10=("nonphonation_share", lambda v: v.quantile(0.1)),
            voiced_median=("voiced_fraction", "median"),
            voiced_p90=("voiced_fraction", lambda v: v.quantile(0.9)),
            floor_median=("floor_db", "median"),
        )
    )
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
