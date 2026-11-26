"""What each rule change did, from the saved frame tracks. No label is read.

Usage, with the Kudio venv: python -I -B s1_segments/rule_compare.py [--workers 8]

segment.py wrote the level, pYIN and VAD of every frame, so the classification can
be replayed without pYIN. Two changes came out of the visual check of the second
pass: a frame inside a bridged phonation span can no longer be non-phonation, and
a frame more than 20 dB below the file's 95th percentile needs pYIN probability at
least 0.1 to count as phonation. The 0.1 was picked after looking at the figures,
so 0.05 and 0.2 are replayed as well.

Groups: database and material, AVFAD by place, SVD by recording period. Writes
s1_segments/out/rule_compare.csv and rule_compare_files.csv.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import segment as sg

HERE = Path(__file__).resolve().parent
S0_OUT = HERE.parent / "s0_audit" / "out"
VARIANTS = {
    "second_pass_rules": {"prob": None, "no_overlap": False},
    "overlap_fix_only": {"prob": None, "no_overlap": True},
    "prob_0.1_only": {"prob": 0.1, "no_overlap": False},
    "final_prob_0.1": {"prob": 0.1, "no_overlap": True},
    "final_prob_0.05": {"prob": 0.05, "no_overlap": True},
    "final_prob_0.2": {"prob": 0.2, "no_overlap": True},
}
YEAR_BANDS = [1997, 1999, 2001, 2005]
YEAR_NAMES = ["1997-98", "1999-2000", "2001+"]


def replay(
    tracks: dict,
    floor: float,
    high: float,
    duration: float,
    zeros: list[tuple[float, float]],
    contains_voice: bool,
    prob: float | None,
    no_overlap: bool,
    p: sg.Params = sg.PARAMS,
) -> dict:
    """The classification part of segment.segment_signal, with the two changes optional."""
    times = tracks["times"].astype(np.float64)
    level = tracks["level_db"]
    voiced, voiced_prob, in_vad = tracks["voiced"], tracks["voiced_prob"], tracks["in_vad"]
    zero_mask = sg.spans_to_mask([(a - p.frame_s, b + p.frame_s) for a, b in zeros], times)
    threshold = floor + p.margin_db
    range_db = high - floor
    if not np.isfinite(range_db):
        phonation = quiet = np.zeros(times.size, dtype=bool)
    elif range_db >= p.min_range_db:
        above = level >= threshold
        loud = level >= high - p.vad_only_within_db
        if prob is None:
            phonation = (above & voiced) | (voiced & in_vad) | (above & in_vad & loud)
        else:
            sure = voiced & (voiced_prob >= prob)
            phonation = (loud & above & (voiced | in_vad)) | (sure & (above | in_vad))
        quiet = ~above & ~in_vad
        if range_db < p.voiced_veto_range_db:
            quiet = quiet & ~voiced
    elif contains_voice:
        phonation = np.ones(times.size, dtype=bool)
        quiet = np.zeros(times.size, dtype=bool)
    else:
        phonation = np.zeros(times.size, dtype=bool)
        quiet = ~voiced
    phonation = phonation & ~zero_mask
    quiet = quiet & ~zero_mask
    phon = sg.close_gaps(sg.mask_to_spans(phonation, p.hop_s, duration), p.phonation_bridge_s)
    phon = [s for s in phon if s[1] - s[0] >= p.min_phonation_s]
    if no_overlap:
        quiet = quiet & ~sg.spans_to_mask(phon, times)
    raw = sg.close_gaps(sg.mask_to_spans(quiet, p.hop_s, duration), p.fill_gap_s)
    raw = [s for s in raw if s[1] - s[0] >= p.min_nonphonation_s]
    guarded = sg.apply_guard(raw, 0.10, duration, p.min_kept_s)
    return {
        "phonation_s": sg.total(phon),
        "nonphonation_raw_s": sg.total(raw),
        "nonphonation_g100_s": sg.total(guarded),
        "n_nonphonation_g100": len(guarded),
    }


def one_file(job: tuple) -> list[dict]:
    file_id, material, floor, high, duration, zeros, tracks_dir = job
    with np.load(Path(tracks_dir) / f"{file_id}.npz") as saved:
        tracks = {k: saved[k] for k in saved.files}
    rows = []
    for name, options in VARIANTS.items():
        row = replay(tracks, floor, high, duration, zeros, material != "silence", **options)
        rows.append({"file_id": file_id, "variant": name, "duration_s": duration, **row})
    return rows


def group_table(s0_out: Path) -> pd.DataFrame:
    """Place for AVFAD, recording period for SVD. Only those columns are read."""
    speakers = pd.read_csv(s0_out / "avfad_speakers.csv", usecols=["speaker", "place"], dtype=str)
    avfad = pd.DataFrame({"file_id": "avfad__" + speakers.speaker, "group": speakers.place})
    sessions = pd.read_csv(s0_out / "svd_sessions.csv", usecols=["session", "year"])
    sessions = sessions.dropna(subset=["year"])
    period = pd.cut(sessions.year, YEAR_BANDS, right=False, labels=YEAR_NAMES).astype(str)
    svd = pd.concat(
        [
            pd.DataFrame(
                {
                    "file_id": "svd__" + sessions.session.astype(int).astype(str) + "-" + m,
                    "group": period,
                }
            )
            for m in ("a_n", "iau")
        ]
    )
    return pd.concat([avfad, svd], ignore_index=True)


def summarise(table: pd.DataFrame) -> pd.DataFrame:
    table = table.assign(phonation_share=table.phonation_s / table.duration_s)
    rows = []
    keys = ["database", "material", "group", "variant"]
    for key, g in table.groupby(keys, dropna=False):
        rows.append(
            {
                **dict(zip(keys, key)),
                "n_files": len(g),
                "share_at_least_1s": float((g.nonphonation_g100_s >= 1.0).mean()),
                "median_nonphonation_s": g.nonphonation_g100_s.median(),
                "median_phonation_share": g.phonation_share.median(),
                "mean_phonation_s": g.phonation_s.mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1-out", type=Path, default=HERE / "out")
    parser.add_argument("--s0-out", type=Path, default=S0_OUT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    files = pd.read_csv(args.s1_out / "files.csv")
    files = files[files.status == "ok"]
    spans = pd.read_csv(args.s1_out / "spans.csv")
    zero_rows = spans[spans.kind == "digital_zero"]
    zeros = {k: list(zip(g.start_s, g.end_s)) for k, g in zero_rows.groupby("file_id")}
    jobs = [
        (
            r.file_id,
            r.material,
            r.floor_db,
            r.level_p95_db,
            r.duration_s,
            zeros.get(r.file_id, []),
            str(args.s1_out / "tracks"),
        )
        for r in files.itertuples()
    ]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = [row for batch in pool.map(one_file, jobs, chunksize=16) for row in batch]
    table = pd.DataFrame(rows).merge(
        files[["file_id", "database", "material"]], on="file_id", how="left"
    )
    table = table.merge(group_table(args.s0_out), on="file_id", how="left")
    table.to_csv(args.s1_out / "rule_compare_files.csv", index=False)
    summary = summarise(table)
    summary.round(4).to_csv(args.s1_out / "rule_compare.csv", index=False)

    final = table[table.variant == "final_prob_0.1"].set_index("file_id")
    actual = files.set_index("file_id")
    gap_np = (final.nonphonation_g100_s - actual.nonphonation_g100_s.reindex(final.index)).abs()
    gap_ph = (final.phonation_s - actual.phonation_s.reindex(final.index)).abs()
    print("replay against files.csv, largest difference in s:", gap_np.max(), gap_ph.max())
    wide = summary.pivot_table(
        index=["database", "material", "group"],
        columns="variant",
        values=["share_at_least_1s", "median_phonation_share"],
        dropna=False,
    )
    with pd.option_context("display.width", 250, "display.max_rows", 100):
        print(wide.round(3).to_string())


if __name__ == "__main__":
    main()
