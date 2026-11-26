"""Counting helpers of field_count.py on made-up tables; no AVFAD file is read."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

import field_count as fc  # noqa: E402


def test_wilson_matches_textbook_values():
    lower, upper = fc.wilson(0, 10)
    assert lower == pytest.approx(0.0, abs=1e-12)
    assert upper == pytest.approx(0.2775, abs=1e-4)
    lower, upper = fc.wilson(81, 263)
    assert (lower, upper) == pytest.approx((0.2553, 0.3662), abs=1e-4)


def test_newcombe_reproduces_his_worked_example():
    # Newcombe (1998), Statistics in Medicine 17:873, Table II, method 10.
    d, lower, upper = fc.newcombe(56, 70, 48, 80)
    assert d == pytest.approx(0.2, abs=1e-12)
    assert (lower, upper) == pytest.approx((0.0524, 0.3339), abs=1e-4)
    _, lower, upper = fc.newcombe(9, 10, 3, 10)
    assert (lower, upper) == pytest.approx((0.1705, 0.8090), abs=1e-4)


def people(rows):
    table = pd.DataFrame(rows, columns=["age", "sex", "smoking", "laryngeal_surgery"])
    return fc.prepare(table)


def test_standardization_reweights_by_the_target_cells():
    target = people([(70, "F", 1, 0)] * 3 + [(30, "F", 0, 0)])
    # The other group: young people never smoked, old ones all quit.
    other = people([(30, "F", 0, 0)] * 10 + [(70, "F", 1, 0)] * 2)
    std = fc.standardized(target, other, "former")
    assert std["share"] == pytest.approx(0.75)
    assert std["n_target_used"] == 4
    assert other["former"].mean() == pytest.approx(2 / 12)


def test_cells_missing_in_the_other_group_are_left_out():
    target = people([(70, "M", 1, 0), (30, "F", 0, 0)])
    other = people([(30, "F", 1, 0)] * 4)
    std = fc.standardized(target, other, "former")
    assert std["share"] == pytest.approx(1.0)
    assert std["n_target_used"] == 1


def test_age_bands_follow_the_chapter():
    bands = fc.age_band(pd.Series([18, 45, 46, 65, 66, 93]))
    assert list(bands) == ["18-45", "18-45", "46-65", "46-65", "66+", "66+"]


def test_unknown_smoking_code_stops():
    with pytest.raises(ValueError, match="unexpected smoking codes"):
        people([(30, "F", 3, 0)])


@pytest.mark.parametrize(
    ("difference", "ci", "std", "reading"),
    [
        (0.15, (0.05, 0.25), 0.12, "HSA 正常人的比例比 ESSUA 高"),
        (-0.15, (-0.25, -0.05), -0.12, "比 ESSUA 低"),
        (0.15, (0.05, 0.25), 0.04, "差別大多來自年齡和性別"),
        (0.15, (-0.01, 0.25), 0.12, "看不出兩地正常人有差"),
        (0.05, (-0.05, 0.15), 0.05, "看不出兩地正常人有差"),
    ],
)
def test_reading_against_essua(difference, ci, std, reading):
    c = {"difference": difference, "ci": list(ci), "standardized_difference": std}
    assert fc.read_vs_essua(c) == reading


@pytest.mark.parametrize(
    ("difference", "ci", "reading"),
    [
        (0.03, (-0.05, 0.11), "和同院病人相近"),
        (-0.20, (-0.30, -0.10), "和同院病人不同"),
        (0.05, (0.01, 0.09), "照實"),
    ],
)
def test_reading_against_patients(difference, ci, reading):
    assert fc.read_vs_patients({"difference": difference, "ci": list(ci)}) == reading
