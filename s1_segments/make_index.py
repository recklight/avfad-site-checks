"""Build the S1 index: which files to segment, with nothing about who is in them.

Usage: python -I s1_segments/make_index.py [--svd-scope primary|eligible] [--out ...]

Reads the S0 tables and writes file_id, database, material, wav. Only the columns
needed to pick the files are read, so no label, place, age or diagnosis passes
through here. SVD's a_n and iau files with no file problem come from the primary
set, or from every eligible session for the sensitivity analysis. AVFAD gives its
/a/ files and, as a control, its room-silence recordings.

Which SVD files are in the primary set follows decisions.csv, so run this after
the sign-off.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "s0_audit"))
from avfad_extract import require_access

HERE = Path(__file__).resolve().parent
S0_OUT = HERE.parent / "s0_audit" / "out"
COLUMNS = ["file_id", "database", "material", "wav"]


def voiced_rows(out: Path) -> pd.DataFrame:
    t = pd.read_csv(out / "voiced_files.csv", usecols=["record", "wav"], dtype=str)
    return pd.DataFrame(
        {
            "file_id": "voiced__" + t.record,
            "database": "voiced",
            "material": "a",
            "wav": t.wav,
        }
    )


def svd_rows(out: Path, scope: str) -> pd.DataFrame:
    cols = ["session", "material", "wav", "file_problem", "eligible", "primary"]
    t = pd.read_csv(out / "svd_file_status.csv", usecols=cols, dtype=str, keep_default_na=False)
    chosen = t.material.isin(["a_n", "iau"]) & (t.file_problem == "") & (t.wav != "")
    chosen &= t[scope].str.lower() == "true"
    t = t[chosen]
    return pd.DataFrame(
        {
            "file_id": "svd__" + t.session + "-" + t.material,
            "database": "svd",
            "material": t.material,
            "wav": t.wav,
        }
    )


def avfad_rows(out: Path) -> pd.DataFrame:
    require_access()
    t = pd.read_csv(out / "avfad_file_status.csv", usecols=["path", "speaker", "task"], dtype=str)
    a = t[t.task == "002"]
    speakers = pd.DataFrame(
        {
            "file_id": "avfad__" + a.speaker,
            "database": "avfad",
            "material": "a",
            "wav": a.path,
        }
    )
    room = pd.read_csv(out / "avfad_silence.csv", usecols=["path"], dtype=str).sort_values("path")
    silence = pd.DataFrame(
        {
            "file_id": [f"avfad_silence__{i:04d}" for i in range(1, len(room) + 1)],
            "database": "avfad",
            "material": "silence",
            "wav": room.path.to_numpy(),
        }
    )
    return pd.concat([speakers, silence], ignore_index=True)


def build_index(out: Path = S0_OUT, svd_scope: str = "primary") -> pd.DataFrame:
    index = pd.concat(
        [voiced_rows(out), svd_rows(out, svd_scope), avfad_rows(out)], ignore_index=True
    )
    return index[COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--svd-scope", choices=["primary", "eligible"], default="primary")
    parser.add_argument("--out", type=Path, default=HERE / "out" / "index.csv")
    args = parser.parse_args()
    index = build_index(svd_scope=args.svd_scope)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(args.out, index=False)
    print(index.groupby(["database", "material"]).size().to_string())


if __name__ == "__main__":
    main()
