"""Day-grouped folds of C1 on a synthetic speaker list; no real data is read."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "features"))

import c1_by_day as cbd  # noqa: E402


def test_one_day_never_spans_two_folds(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    speakers = [f"S{i:03d}" for i in range(200)]
    days = rng.choice([f"2014-05-{d:02d}" for d in range(1, 21)], size=200)
    pd.DataFrame({"speaker": speakers, "visit_date": days}).to_csv(
        tmp_path / "days.csv", index=False
    )
    monkeypatch.setattr(cbd, "DAYS", tmp_path / "days.csv")
    table = pd.DataFrame(
        {"speaker": speakers, "label": np.where(np.arange(200) % 3 == 0, "normal", "x")}
    )
    folds = cbd.day_folds(table, 0, 5)
    assert set(folds) == set(range(5))
    for day in np.unique(days):
        assert len(set(folds[days == day])) == 1


def test_reading():
    assert cbd.reading({"difference": 0.20, "ci_low": 0.12}).startswith("grouped by recording day")
    assert cbd.reading({"difference": 0.03, "ci_low": -0.02}) == (
        "C1 rests on the same-day recordings"
    )
    assert cbd.reading({"difference": 0.07, "ci_low": 0.02}) == "the gap shrinks"
