"""Acceptance of the feature tables: row counts, no NaN, and five files extracted again.

Writes features/out/check.json. Stops with an error at the first failed condition, after
writing what it found.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from extract_handcrafted import (  # noqa: E402
    FEATURE_COLUMNS,
    OUT,
    PLANNED,
    S1_OUT,
    extract_file,
    jobs_for,
)

RECHECK_FILES = 5
RECHECK_TOLERANCE = 1e-5
RECHECK_SEED = 20261104
MIN_NONPHONATION_S = 1.0


def check_table(database: str, material: str, condition: str, n_files: int) -> dict:
    path = OUT / f"{database}_{material}_{condition}.csv"
    table = pd.read_csv(path)
    values = table[FEATURE_COLUMNS].to_numpy()
    jobs = {job[0]: job for job in jobs_for(database, material, [condition])}
    rng = np.random.default_rng(RECHECK_SEED)
    chosen = rng.choice(
        table["file_id"].to_numpy(), size=min(RECHECK_FILES, len(table)), replace=False
    )
    worst = 0.0
    for file_id in chosen:
        again, _ = extract_file(jobs[file_id])[condition]
        stored = table.loc[table["file_id"] == file_id, FEATURE_COLUMNS].to_numpy().ravel()
        worst = max(worst, float(np.max(np.abs(again - stored))))
    return {
        "table": path.name,
        "files_in_s1": n_files,
        "rows": len(table),
        "duplicate_ids": int(table["file_id"].duplicated().sum()),
        "columns": len(FEATURE_COLUMNS),
        "nan_cells": int(np.isnan(values).sum()),
        "inf_cells": int(np.isinf(values).sum()),
        "rows_with_nan": table.loc[np.isnan(values).any(axis=1), "file_id"].tolist(),
        "recheck_files": [str(f) for f in chosen],
        "recheck_worst_abs_diff": worst,
    }


def main() -> None:
    files = pd.read_csv(S1_OUT / "files.csv")
    report = []
    problems = []
    for (database, material), conditions in PLANNED.items():
        chosen = files[
            (files["database"] == database)
            & (files["material"] == material)
            & (files["status"] == "ok")
        ]
        for condition in conditions:
            if not (OUT / f"{database}_{material}_{condition}.csv").exists():
                problems.append(f"{database}_{material}_{condition}.csv is missing")
                continue
            row = check_table(database, material, condition, len(chosen))
            report.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "recheck_files"}), flush=True)
            if row["duplicate_ids"] or row["inf_cells"]:
                problems.append(f"{row['table']}: duplicates or infinities")
            if row["recheck_worst_abs_diff"] > RECHECK_TOLERANCE:
                problems.append(f"{row['table']}: second extraction differs")
            if row["rows"] > row["files_in_s1"]:
                problems.append(f"{row['table']}: more rows than files")
            if not condition.startswith("nonphon") and row["rows"] != row["files_in_s1"]:
                problems.append(
                    f"{row['table']}: {row['rows']} rows for {row['files_in_s1']} files"
                )
    (OUT / "check.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    if problems:
        sys.exit("feature check failed:\n" + "\n".join(problems))
    print("feature check passed", flush=True)


if __name__ == "__main__":
    main()
