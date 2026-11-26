"""Spectrograms of the five SVD files the speech/EGG screen left on the border.

Usage: python -I s1_segments/swap_screen_figures.py

These six files stay excluded whatever the pictures show (decisions.csv, S10); the
figures only go on record. 1697 a_n was a clear swap and is not drawn.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "s0_audit"))
from readers import read_wav

ROOT = Path(__file__).resolve().parents[1]
FILES = [(859, "phrase"), (1678, "phrase"), (1624, "iau"), (920, "a_n"), (1909, "a_n")]


def main() -> None:
    out = ROOT / "s1_segments" / "out" / "figures_swap_screen"
    out.mkdir(parents=True, exist_ok=True)
    for session, material in FILES:
        wav = read_wav(
            ROOT / "data" / "derived" / "svd" / str(session) / f"{session}-{material}.wav"
        )
        x = wav.as_float()[:, 0]
        fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True, constrained_layout=True)
        axes[0].plot(np.arange(x.size)[::20] / wav.rate, x[::20], lw=0.3, color="k")
        axes[0].set_title(f"{session}-{material}")
        axes[1].specgram(
            x, NFFT=int(0.03 * wav.rate), Fs=wav.rate, noverlap=int(0.02 * wav.rate), cmap="magma"
        )
        axes[1].set_ylim(0, 8000)
        axes[1].set_ylabel("Hz")
        axes[1].set_xlabel("s")
        fig.savefig(out / f"{session}-{material}.png", dpi=90)
        plt.close(fig)
        print(session, material, round(x.size / wav.rate, 2))


if __name__ == "__main__":
    main()
