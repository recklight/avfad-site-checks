"""Descriptive tables by label: median [IQR] per recording-condition column,
plus counts, and a Mann-Whitney p as a rough guide (not a confirmatory test).

Usage: python -I s0_audit/describe.py voiced|svd
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

OUT = Path(__file__).resolve().parent / "out"

NUMERIC = [
    "duration_s",
    "lead_s",
    "trail_s",
    "lead_zero_s",
    "trail_zero_s",
    "active_fraction",
    "rms_dbfs",
    "peak_dbfs",
    "noise_floor_dbfs",
    "clip_fraction",
    "dc_offset",
    "rolloff99_hz",
    "above_1k_fraction",
    "age",
    "year",
]


def table(df: pd.DataFrame, by: str, columns: list[str]) -> pd.DataFrame:
    groups = sorted(df[by].dropna().unique())
    rows = []
    for col in columns:
        if col not in df:
            continue
        row = {"variable": col}
        samples = []
        for g in groups:
            v = df.loc[df[by] == g, col].dropna().to_numpy(dtype=float)
            samples.append(v)
            if v.size:
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                row[str(g)] = f"{med:.4g} [{q1:.4g}, {q3:.4g}] (n={v.size})"
            else:
                row[str(g)] = "n=0"
        if len(samples) == 2 and all(s.size for s in samples):
            row["mannwhitney_p"] = mannwhitneyu(*samples).pvalue
        rows.append(row)
    return pd.DataFrame(rows)


def voiced() -> None:
    files = pd.read_csv(OUT / "voiced_files.csv")
    meta = pd.read_csv(OUT / "voiced_meta.csv")
    from scipy.io import wavfile

    root = Path(__file__).resolve().parents[1]
    zeros = []
    for wav in files.wav:
        rate, x = wavfile.read(root / wav)
        nz = np.flatnonzero(x)
        zeros.append((nz[0] / rate, (x.size - 1 - nz[-1]) / rate))
    files[["lead_zero_s", "trail_zero_s"]] = zeros
    df = files.merge(meta, on="record")
    conflict = df.age_conflict | df.sex_conflict
    df["age"] = df.age_info.where(~conflict)
    t = table(df, "label_header", NUMERIC)
    sex = pd.crosstab(df.label_header, df.sex_info.where(~conflict).fillna("missing"))
    clipped = (df.clip_fraction > 0).groupby(df.label_header).agg(["sum", "count"])
    t.to_csv(OUT / "voiced_describe.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(t.to_string(index=False))
        print(sex)
        print("files with any clipping:\n", clipped)


if __name__ == "__main__":
    {"voiced": voiced}[sys.argv[1]]()
