"""Look for speech/EGG swaps by comparing each speech file with its EGG partner.

Usage: python -I s0_audit/svd_eggcheck.py [data.zip|healthy.zip] [--workers 10]

The EGG files are read from the zip in memory and never written to disk.

The feature is the energy in the F1 region (500-1500 Hz) relative to the F0
region (60-400 Hz), in dB, over the loudest 30% of 30 ms frames. A microphone
signal of /a/ or running speech has strong F1 energy; an EGG falls off from
F0 upwards. A pair where the ``.nsp`` member scores lower than its ``.egg``
partner is a swap candidate. The share of energy above 1 kHz does not work
here: many EGG channels carry enough broadband noise to beat the speech.
On the documented swaps (sessions 139, 141; 1697 phrase and iau; 980 phrase)
the speech file sits 15-25 dB below its partner, on ordinary sessions
10-20 dB above. Writes s0_audit/out/svd_eggcheck_<zip stem>.csv.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from readers import parse_nsp
from svd_extract import OUT, SPEECH, SVD


def f1_to_f0_db(samples: np.ndarray, rate: int) -> float:
    x = samples.astype(np.float64) / 32768.0
    frame = int(0.030 * rate)
    hop = frame // 2
    n = 1 + (x.size - frame) // hop if x.size >= frame else 0
    if n < 3:
        return float("nan")
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx]
    energy = np.mean(frames**2, axis=1)
    loud = frames[energy >= np.percentile(energy, 70)] * np.hanning(frame)
    nfft = 4096
    power = np.mean(np.abs(np.fft.rfft(loud, n=nfft, axis=1)) ** 2, axis=0)
    freqs = np.fft.rfftfreq(nfft, 1 / rate)
    f1 = power[(freqs >= 500) & (freqs < 1500)].sum()
    f0 = power[(freqs >= 60) & (freqs < 400)].sum()
    return float(10 * np.log10((f1 + 1e-30) / (f0 + 1e-30)))


def check_pairs(args: tuple[str, list[tuple[str, str]]]) -> list[dict]:
    zip_path, pairs = args
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        for speech, egg in pairs:
            row = {"member": speech, "egg_member": egg}
            for key, member in (("nsp", speech), ("egg", egg)):
                try:
                    f = parse_nsp(zf.read(member))
                    row[f"{key}_f1_f0_db"] = f1_to_f0_db(f.samples, f.rate)
                    row[f"{key}_n"] = int(f.samples.size)
                except (KeyError, ValueError) as err:
                    row[f"{key}_error"] = str(err)
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip", nargs="?", default="data.zip")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    zip_path = SVD / args.zip
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    pairs = []
    for name in sorted(names):
        p = PurePosixPath(name)
        if SPEECH.match(p.name):
            egg = str(p.with_name(p.stem + "-egg.egg"))
            pairs.append((name, egg if egg in names else ""))
    chunks = [pairs[i :: args.workers] for i in range(args.workers)]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = [
            r for part in pool.map(check_pairs, [(str(zip_path), c) for c in chunks]) for r in part
        ]
    df = pd.DataFrame(rows)
    m = df.member.str.extract(r"/(?P<session>\d+)-(?P<material>a_n|iau|phrase)\.nsp$")
    df.insert(0, "material", m.material)
    df.insert(0, "session", m.session.astype(int))
    df["margin_db"] = df.nsp_f1_f0_db - df.egg_f1_f0_db
    df["swap_candidate"] = df.margin_db < 0
    df = df.sort_values(["session", "material"])
    out = OUT / f"svd_eggcheck_{Path(args.zip).stem}.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} pairs -> {out.name}")
    print(df.margin_db.describe(percentiles=[0.005, 0.01, 0.05, 0.5]).to_string())
    cols = ["session", "material", "nsp_f1_f0_db", "egg_f1_f0_db", "margin_db"]
    print(df[df.margin_db < 3][cols].to_string())


if __name__ == "__main__":
    main()
