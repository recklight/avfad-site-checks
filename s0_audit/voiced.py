"""S0 for VOICED: read every record, check it, write a 16-bit WAV copy, and
collect metadata and recording statistics.

Usage: python -I s0_audit/voiced.py

Writes data/derived/voiced/<record>.wav, s0_audit/out/voiced_files.csv,
s0_audit/out/voiced_meta.csv and s0_audit/out/voiced_reader_check.json.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from readers import check_wfdb, read_wfdb32, to_int16_grid, write_wav16
from stats import recording_stats

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "voiced"
DERIVED = ROOT / "data" / "derived" / "voiced"
OUT = Path(__file__).resolve().parent / "out"

_HEADER_COMMENT = re.compile(
    r"<age>:\s*(?P<age>\d+)\s*<sex>:\s*(?P<sex>\w)\s*<diagnoses>:\s*(?P<diag>.*?)\s*"
    r"<medications>:\s*(?P<med>.*)$"
)


def read_info(path: Path) -> dict:
    fields = {}
    for line in path.read_text(encoding="latin-1").splitlines():
        if "\t" not in line:
            continue
        key, value = line.split("\t", 1)
        key = key.strip().rstrip(":").strip()
        if key:
            fields[key] = value.strip()
    return fields


def diagnosis_class(text: str) -> str:
    """Three pathology classes plus healthy, from the start of the diagnosis text.

    One header spells it "hyperkineti dysphonia", hence the prefix match.
    """
    t = text.strip().lower()
    if t == "healthy":
        return "healthy"
    if t.startswith("hyperkineti"):
        return "hyperkinetic"
    if t.startswith("hypokinetic"):
        return "hypokinetic"
    if t.startswith("reflux"):
        return "reflux"
    return "unknown"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = (RAW / "RECORDS").read_text().split()
    file_rows, meta_rows, reader_check = [], [], {}
    for rid in records:
        rec = read_wfdb32(RAW / f"{rid}.hea")
        problems = check_wfdb(rec)
        x = rec.physical
        pcm = to_int16_grid(x)
        txt = RAW / f"{rid}.txt"
        if txt.exists():
            ref = np.loadtxt(txt)
            reader_check[rid] = {
                "n_txt": int(ref.size),
                "n_dat": int(x.size),
                "max_abs_diff": float(np.max(np.abs(ref - x))) if ref.size == x.size else None,
            }
        wav = DERIVED / f"{rid}.wav"
        write_wav16(wav, rec.rate, pcm)
        stats = recording_stats(pcm.astype(np.float64) / 32768.0, rec.rate)
        file_rows.append(
            {
                "record": rid,
                "wav": wav.relative_to(ROOT).as_posix(),
                "wav_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
                "reader_problems": "; ".join(problems),
                **stats,
            }
        )
        info = read_info(RAW / f"{rid}-info.txt")
        m = _HEADER_COMMENT.search(rec.comments[0]) if rec.comments else None
        if m is None:
            raise ValueError(f"{rid}: cannot parse header comment {rec.comments!r}")
        meta_rows.append(
            {
                "record": rid,
                "age_info": int(info["Age"]),
                "sex_info": info["Gender"].lower(),
                "diagnosis_info": info["Diagnosis"],
                "age_header": int(m["age"]),
                "sex_header": m["sex"].lower(),
                "diagnosis_header": m["diag"],
                "medications_header": m["med"],
                "class_info": diagnosis_class(info["Diagnosis"]),
                "class_header": diagnosis_class(m["diag"]),
            }
        )
    files = pd.DataFrame(file_rows)
    meta = pd.DataFrame(meta_rows)
    meta["label_info"] = np.where(meta.class_info == "healthy", "healthy", "pathological")
    meta["label_header"] = np.where(meta.class_header == "healthy", "healthy", "pathological")
    meta["label_conflict"] = meta.label_info != meta.label_header
    meta["class_conflict"] = meta.class_info != meta.class_header
    meta["age_conflict"] = meta.age_info != meta.age_header
    meta["sex_conflict"] = meta.sex_info != meta.sex_header
    files.to_csv(OUT / "voiced_files.csv", index=False)
    meta.to_csv(OUT / "voiced_meta.csv", index=False)
    (OUT / "voiced_reader_check.json").write_text(json.dumps(reader_check, indent=2))
    print(f"{len(files)} records; reader problems: {int((files.reader_problems != '').sum())}")
    print("txt check:", {k: v["max_abs_diff"] for k, v in reader_check.items()})
    print(meta.label_header.value_counts().to_dict(), meta.label_info.value_counts().to_dict())
    print(
        "conflicts:",
        meta.loc[
            meta[["label_conflict", "class_conflict", "age_conflict", "sex_conflict"]].any(axis=1),
            [
                "record",
                "class_info",
                "class_header",
                "age_info",
                "age_header",
                "sex_info",
                "sex_header",
            ],
        ].to_string(index=False),
    )


if __name__ == "__main__":
    main()
