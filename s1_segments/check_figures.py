"""S1 acceptance: figures to judge by eye, with nothing on them that gives the group away.

Usage, with the paper venv (it has matplotlib):
  python -I s1_segments/check_figures.py sample    # before the run: picks the files
  python -I s1_segments/check_figures.py draw      # after segment.py --tracks-for check_ids.txt

The sample is 30 files per database. Strata: VOICED by label; SVD by material and
label; AVFAD ESSUA, HSA normal and HSA pathological, because the noisy room with
pathological voices is where the segmenter is most likely to go wrong. The
strata go to check_key.csv and only the shuffled figure numbers go on the
figures. check_key.csv is opened after check_sheet.csv is filled in.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "s0_audit"))
from readers import read_wav

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
S0_OUT = HERE.parent / "s0_audit" / "out"
SEED = 20261104
COLOURS = {"phonation": "tab:green", "nonphonation_g100": "tab:blue", "digital_zero": "tab:gray"}


def strata(index: pd.DataFrame, s0_out: Path) -> pd.Series:
    """A stratum name per row of the index; rows left as NaN are not sampled."""
    out = pd.Series(np.nan, index=index.index, dtype=object)
    voiced = pd.read_csv(s0_out / "voiced_meta.csv", usecols=["record", "label_header"], dtype=str)
    labels = dict(zip("voiced__" + voiced.record, voiced.label_header))
    is_voiced = index.database == "voiced"
    out[is_voiced] = "voiced_" + index.file_id[is_voiced].map(labels)

    svd = pd.read_csv(
        s0_out / "svd_file_status.csv", usecols=["session", "material", "label"], dtype=str
    )
    svd_key = "svd__" + svd.session + "-" + svd.material
    svd_labels = dict(zip(svd_key, svd.label))
    is_svd = index.database == "svd"
    out[is_svd] = "svd_" + index.material[is_svd] + "_" + index.file_id[is_svd].map(svd_labels)

    speakers = pd.read_csv(
        s0_out / "avfad_speakers.csv", usecols=["speaker", "place", "label"], dtype=str
    )
    key = "avfad__" + speakers.speaker
    place = dict(zip(key, speakers.place))
    label = dict(zip(key, speakers.label))
    is_a = (index.database == "avfad") & (index.material == "a")
    where = index.file_id[is_a].map(place)
    what = index.file_id[is_a].map(label)
    stratum = pd.Series(np.nan, index=where.index, dtype=object)
    stratum[where == "ESSUA"] = "avfad_essua"
    stratum[where == "HSA"] = "avfad_hsa_" + what[where == "HSA"]
    out[is_a] = stratum
    return out


def choose(
    index: pd.DataFrame, names: pd.Series, per_database: int = 30, seed: int = SEED
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    picked = []
    table = index.assign(stratum=names).dropna(subset=["stratum"])
    for _, group in table.groupby("database"):
        cells = sorted(group.stratum.unique())
        share = per_database // len(cells)
        extra = per_database - share * len(cells)
        for i, cell in enumerate(cells):
            members = group[group.stratum == cell]
            n = min(len(members), share + (1 if i < extra else 0))
            chosen = rng.choice(members.index.to_numpy(), size=n, replace=False)
            picked.append(table.loc[chosen])
    sample = pd.concat(picked, ignore_index=True)
    order = rng.permutation(len(sample))
    sample = sample.iloc[order].reset_index(drop=True)
    sample["figure"] = [f"check_{i + 1:03d}" for i in range(len(sample))]
    return sample


def sample_step(
    index_path: Path, s0_out: Path, out: Path, seed: int = SEED, only: str | None = None
) -> pd.DataFrame:
    """``only`` keeps the strata that start with one of these comma-separated prefixes."""
    index = pd.read_csv(index_path, dtype=str)
    names = strata(index, s0_out)
    if only:
        wanted = tuple(only.split(","))
        names = names.where(names.fillna("").str.startswith(wanted))
    sample = choose(index, names, seed=seed)
    out.mkdir(parents=True, exist_ok=True)
    sample[["figure", "file_id", "stratum"]].to_csv(out / "check_key.csv", index=False)
    (out / "check_ids.txt").write_text("\n".join(sample.file_id) + "\n", encoding="utf-8")
    sheet = sample[["figure"]].assign(correct="", note="")
    sheet.sort_values("figure").to_csv(out / "check_sheet.csv", index=False)
    return sample


def draw_one(
    figure: str,
    file_row: dict,
    spans: pd.DataFrame,
    tracks: dict,
    x: np.ndarray,
    rate: int,
    path: Path,
) -> None:
    """The title carries the figure number only."""
    t = np.arange(x.size) / rate
    fig, axes = plt.subplots(4, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    axes[0].plot(t[:: max(1, x.size // 20000)], x[:: max(1, x.size // 20000)], lw=0.3, color="k")
    axes[0].set_ylabel("waveform")
    axes[1].specgram(x, NFFT=int(0.03 * rate), Fs=rate, noverlap=int(0.02 * rate), cmap="magma")
    axes[1].set_ylim(0, 5000)
    axes[1].set_ylabel("Hz")
    axes[2].plot(tracks["times"], tracks["level_db"], lw=0.6, color="k")
    axes[2].axhline(file_row["floor_db"], color="tab:gray", ls=":", lw=0.8)
    axes[2].axhline(file_row["threshold_db"], color="tab:red", ls="--", lw=0.8)
    axes[2].set_ylabel("band level, dB")
    axes[3].plot(tracks["times"], tracks["voiced_prob"], lw=0.6, color="k")
    voiced = tracks["voiced"].astype(float)
    axes[3].fill_between(tracks["times"], 0, voiced, alpha=0.2, color="tab:orange")
    axes[3].set_ylabel("voiced")
    axes[3].set_xlabel("s")
    for kind, colour in COLOURS.items():
        for row in spans[spans.kind == kind].itertuples():
            for ax in axes:
                ax.axvspan(row.start_s, row.end_s, color=colour, alpha=0.18, lw=0)
    legend = "green phonation, blue non-phonation (100 ms guard), grey digital zero"
    axes[0].set_title(f"{figure}  {legend}")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def draw_step(s1_out: Path, check_dir: Path | None = None) -> int:
    check_dir = check_dir or s1_out
    key = pd.read_csv(check_dir / "check_key.csv", dtype=str)
    files = pd.read_csv(s1_out / "files.csv").set_index("file_id")
    spans = pd.read_csv(s1_out / "spans.csv")
    figures = check_dir / "figures"
    figures.mkdir(exist_ok=True)
    drawn = 0
    for row in key.itertuples():
        info = files.loc[row.file_id]
        if info.status != "ok":
            continue
        wav = read_wav(ROOT / info.wav)
        draw_one(
            row.figure,
            info.to_dict(),
            spans[spans.file_id == row.file_id],
            dict(np.load(s1_out / "tracks" / f"{row.file_id}.npz")),
            wav.as_float()[:, 0],
            wav.rate,
            figures / f"{row.figure}.png",
        )
        drawn += 1
    return drawn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["sample", "draw"])
    parser.add_argument("--index", type=Path, default=HERE / "out" / "index.csv")
    parser.add_argument("--s1-out", type=Path, default=HERE / "out")
    parser.add_argument("--s0-out", type=Path, default=S0_OUT)
    parser.add_argument("--check-dir", type=Path, help="where the sample and figures go")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--only", help="comma-separated stratum prefixes")
    args = parser.parse_args()
    if args.step == "sample":
        out = args.check_dir or args.s1_out
        print(len(sample_step(args.index, args.s0_out, out, args.seed, args.only)), "files chosen")
    else:
        print(draw_step(args.s1_out, args.check_dir), "figures drawn")


if __name__ == "__main__":
    main()
