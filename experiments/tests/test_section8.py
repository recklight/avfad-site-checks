"""Synthetic checks of the remaining section 8 items; no recording is read."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "features"))

import section8_rest as s8  # noqa: E402


def test_m_tech_has_the_seven_columns_of_d4():
    assert len(s8.M_TECH) == 7
    assert "duration_s" not in s8.M_TECH_NO_LENGTH and len(s8.M_TECH_NO_LENGTH) == 6


def test_lead_and_trail_come_from_the_phonation_spans():
    spans = pd.DataFrame(
        {
            "file_id": ["a", "a", "a", "b"],
            "kind": ["phonation", "phonation", "vad", "phonation"],
            "start_s": [1.0, 3.0, 0.0, 0.5],
            "end_s": [2.0, 4.5, 9.0, 1.0],
        }
    )
    out = s8.head_and_tail(pd.Series(["a", "b"]), pd.Series([6.0, 2.0]), spans)
    assert out["lead_s"].tolist() == [1.0, 0.5]
    assert out["trail_s"].tolist() == [1.5, 1.0]


def test_a_file_without_phonation_stops():
    spans = pd.DataFrame({"file_id": ["a"], "kind": ["vad"], "start_s": [0.0], "end_s": [1.0]})
    with pytest.raises(ValueError, match="no phonation"):
        s8.head_and_tail(pd.Series(["a"]), pd.Series([1.0]), spans)


def test_rank_biserial_ends_and_middle():
    sick = np.array([True, True, False, False])
    assert s8.rank_biserial(np.array([3.0, 4.0, 1.0, 2.0]), sick) == pytest.approx(1.0)
    assert s8.rank_biserial(np.array([1.0, 2.0, 3.0, 4.0]), sick) == pytest.approx(-1.0)
    assert s8.rank_biserial(np.array([1.0, 1.0, 1.0, 1.0]), sick) == pytest.approx(0.0)


def test_stratified_rank_biserial_removes_what_only_the_strata_explain():
    # Patients look larger overall only because they sit in the stratum with larger values.
    values = np.array([1.0, 2.0, 1.5, 11.0, 12.0, 11.5])
    sick = np.array([False, False, True, True, True, False])
    cells = np.array(["a", "a", "a", "b", "b", "b"])
    overall = s8.rank_biserial(values, sick)
    inside = s8.stratified_rank_biserial(values, sick, cells)
    assert overall > 0.3
    assert abs(inside) < 0.01


def test_stratified_difference_recovers_a_known_gap():
    rng = np.random.default_rng(0)
    n = 400
    cells = np.array(["a", "b"] * (n // 2))
    sick = rng.random(n) < 0.5
    values = 0.4 * sick + 5.0 * (cells == "b") + rng.normal(0, 0.1, n)
    out = s8.stratified_difference(values, sick, cells, 0)
    assert out["estimate"] == pytest.approx(0.4, abs=0.03)
    assert out["ci"][0] < out["estimate"] < out["ci"][1]


def test_demographic_cell_ignores_the_year():
    table = pd.DataFrame(
        {"age": [20.0, 20.0, 45.0], "sex": ["m", "m", "m"], "year": [1997, 2003, 1997]}
    )
    cells = s8.demographic_cell(table)
    assert cells[0] == cells[1] and cells[0] != cells[2]


def test_room_bands_are_real_feature_columns():
    assert set(s8.ROOM_BANDS) <= set(s8.LTAS_COLUMNS)


def test_voiced_labels_read_the_two_classes():
    labels = s8.voiced_labels()
    assert len(labels) == 208 and labels["speaker"].is_unique
    assert labels["label"].value_counts().to_dict() == {"pathological": 151, "healthy": 57}
