"""After S1: how much non-phonation each database and each AVFAD place actually has.

Usage: python -I s1_segments/availability.py [--min-total-s 1.0] [--min-share 0.8]

Reads s1_segments/out/files.csv and, for AVFAD, only the speaker and place
columns of avfad_speakers.csv. The label is not read: whether H1-AVFAD stays
confirmatory is judged per place, before anyone sees the arms split by label.

The rule, in analysis_plan: at the 100 ms guard, at least ``min_share`` of
the /a/ files in ESSUA and in HSA, each on its own, must hold at least
``min_total_s`` of non-phonation. Otherwise H1-AVFAD becomes exploratory, as V5
does for VOICED.

The room recordings are the control: they hold no voice, so nearly every frame
should come out as non-phonation in every place, and pYIN should find little
voicing there. A gap between places here is the segmenter's own doing.

Writes availability_by_database.csv, availability_avfad_by_place.csv,
availability_silence_by_place.csv and availability_rule.json in s1_segments/out/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
S0_OUT = HERE.parent / "s0_audit" / "out"
LENGTHS_S = (0.2, 0.5, 1.0, 2.0, 5.0)
GUARDS = ("g050", "g100", "g200")
RULE_PLACES = ("ESSUA", "HSA")


def availability(files: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """One row per group and guard: file counts, shares at each length, medians, thresholds."""
    rows = []
    for key, g in files.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        for guard in GUARDS:
            length = g[f"nonphonation_{guard}_s"]
            row = dict(zip(by, key))
            row.update(
                {
                    "guard": guard,
                    "n_files": len(g),
                    "median_total_s": length.median(),
                    "median_segments": g[f"n_nonphonation_{guard}"].median(),
                    "median_longest_s": g[f"longest_nonphonation_{guard}_s"].median(),
                    "median_floor_db": g.floor_db.median(),
                    "median_threshold_db": g.threshold_db.median(),
                    "share_separable": g.separable.astype(bool).mean(),
                    "median_voiced_fraction": g.voiced_fraction.median(),
                    "median_low_level_voiced_s": g.low_level_voiced_s.median(),
                    "median_other_s": g.other_s.median(),
                }
            )
            for seconds in LENGTHS_S:
                row[f"share_at_least_{seconds:g}s"] = float((length >= seconds).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def h1_avfad_rule(
    a_files: pd.DataFrame, min_total_s: float, min_share: float, guard: str = "g100"
) -> dict:
    """Does each rule place hold enough files with enough non-phonation?"""
    length = f"nonphonation_{guard}_s"
    places = {}
    for place in RULE_PLACES:
        g = a_files[a_files.place == place]
        share = float((g[length] >= min_total_s).mean()) if len(g) else 0.0
        places[place] = {"n_files": len(g), "share": share, "meets": share >= min_share}
    return {
        "guard": guard,
        "min_total_s": min_total_s,
        "min_share": min_share,
        "places": places,
        "h1_avfad_stays_confirmatory": all(p["meets"] for p in places.values()),
    }


def load_places(s0_out: Path) -> pd.DataFrame:
    speakers = pd.read_csv(s0_out / "avfad_speakers.csv", usecols=["speaker", "place"], dtype=str)
    speakers["file_id"] = "avfad__" + speakers.speaker
    return speakers[["file_id", "place"]]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1-out", type=Path, default=HERE / "out")
    parser.add_argument("--s0-out", type=Path, default=S0_OUT)
    parser.add_argument("--min-total-s", type=float, default=1.0)
    parser.add_argument("--min-share", type=float, default=0.8)
    args = parser.parse_args(argv)

    files = pd.read_csv(args.s1_out / "files.csv")
    ok = files[files.status == "ok"]
    status = files.groupby(["database", "material"]).status.agg(
        n_files="size", n_not_segmented=lambda s: int((s != "ok").sum())
    )
    by_db = availability(ok, ["database", "material"]).merge(
        status.reset_index(), on=["database", "material"], suffixes=("", "_all")
    )
    by_db.to_csv(args.s1_out / "availability_by_database.csv", index=False)

    a_files = ok[(ok.database == "avfad") & (ok.material == "a")].merge(
        load_places(args.s0_out), on="file_id", how="left"
    )
    availability(a_files, ["place"]).to_csv(
        args.s1_out / "availability_avfad_by_place.csv", index=False
    )

    room = ok[ok.material == "silence"].copy()
    silence_places = pd.read_csv(args.s0_out / "avfad_silence.csv", usecols=["path", "place"])
    room = room.merge(silence_places, left_on="wav", right_on="path", how="left")
    room["nonphonation_share"] = room.nonphonation_raw_s / room.duration_s
    control = room.groupby("place").agg(
        n_files=("file_id", "size"),
        median_nonphonation_share=("nonphonation_share", "median"),
        min_nonphonation_share=("nonphonation_share", "min"),
        median_voiced_fraction=("voiced_fraction", "median"),
        max_voiced_fraction=("voiced_fraction", "max"),
        median_floor_db=("floor_db", "median"),
    )
    control.to_csv(args.s1_out / "availability_silence_by_place.csv")

    rule = h1_avfad_rule(a_files, args.min_total_s, args.min_share)
    (args.s1_out / "availability_rule.json").write_text(
        json.dumps(rule, indent=2), encoding="utf-8"
    )
    print(json.dumps(rule, indent=2))
    print(control.round(3).to_string())


if __name__ == "__main__":
    main()
