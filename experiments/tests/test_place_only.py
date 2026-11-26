"""The place-only detector never learns from a patient; no real data is read."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "features"))

import place_only as po  # noqa: E402


def synthetic(rng: np.random.Generator):
    place = np.array(["ESSUA"] * 60 + ["HSA"] * 20 + ["HSA"] * 40 + ["ULSM"] * 4)
    normal = np.array([True] * 80 + [False] * 40 + [True, False, True, False])
    values = rng.normal(size=(len(place), 3))
    values[place == "HSA", 0] += 2.0
    return values, place, normal


def test_patients_outside_the_test_part_change_nothing():
    rng = np.random.default_rng(0)
    values, place, normal = synthetic(rng)
    test = np.zeros(len(place), dtype=bool)
    test[::5] = True
    before = po.place_scores(values, place, normal, test)
    moved = values.copy()
    moved[~test & ~normal] = rng.normal(scale=50.0, size=(int((~test & ~normal).sum()), 3))
    after = po.place_scores(moved, place, normal, test)
    assert np.allclose(before, after)


def test_hsa_rows_score_higher():
    rng = np.random.default_rng(1)
    values, place, normal = synthetic(rng)
    test = np.zeros(len(place), dtype=bool)
    test[::3] = True
    score = po.place_scores(values, place, normal, test)
    assert score[place[test] == "HSA"].mean() > score[place[test] == "ESSUA"].mean()


def test_reading():
    assert po.reading({"auc": 0.84, "ci_low": 0.80}).startswith("a detector that never saw")
    assert po.reading({"auc": 0.81, "ci_low": 0.74}) == "as measured"
    assert po.reading({"auc": 0.60, "ci_low": 0.55}) == "place alone does not get the full score"


def test_every_variant_ignores_patients_and_ranks_hsa_higher():
    rng = np.random.default_rng(4)
    values, place, normal = synthetic(rng)
    test = np.zeros(len(place), dtype=bool)
    test[::4] = True
    moved = values.copy()
    moved[~test & ~normal] = 99.0
    for variant in ("logistic", "c0.01", "c100", "mean"):
        score = po.place_scores(values, place, normal, test, variant)
        assert np.allclose(score, po.place_scores(moved, place, normal, test, variant))
        assert score[place[test] == "HSA"].mean() > score[place[test] == "ESSUA"].mean()


def test_agreement_reading():
    assert po.agreement_reading([0.88, 0.75]) == "the full detector's ranking is mostly place"
    assert po.agreement_reading([0.30, 0.20]) == "the rankings differ"
    assert po.agreement_reading([0.88, 0.50]) == "as measured"


def test_dose_reading():
    assert po.dose_reading(0.46, 0.40) == "the place detector draws the dose curve too"
    assert po.dose_reading(0.05, 0.40) == "it does not draw the dose curve"
    assert po.dose_reading(0.20, 0.40) == "as measured"


def test_few_reading():
    assert po.few_reading([0.82, 0.85], [0.85, 0.90]) == "ten normals per place are enough"
    assert po.few_reading([0.70, 0.72], [0.78, 0.85]) == "it needs many normals"
    assert po.few_reading([0.70, 0.85], [0.84, 0.90]) == "as measured"


def test_full_uar_references_match_the_summary_numbers():
    assert po.REAL_FULL_UAR["compare"] == max(po.REAL_FULL_UAR.values())
    assert po.REAL_UAR["hubert"] == po.REAL_FULL_UAR["hubert"]


def test_cross_reading():
    assert po.cross_reading([0.82, 0.88]).startswith("the place cue follows")
    assert po.cross_reading([0.60, 0.62]) == "it has to be learned on the same material"
    assert po.cross_reading([0.78, 0.88]) == "as measured"


def test_paired_reading():
    assert po.paired_reading({"ci_low": -0.01, "ci_high": 0.02}).startswith("no better")
    assert po.paired_reading({"ci_low": -0.04, "ci_high": -0.01}) == "below the full detector"
    assert po.paired_reading({"ci_low": 0.01, "ci_high": 0.03}) == "above the full detector"


def test_inside_reading():
    assert po.inside_reading({"ci_low": -0.03, "ci_high": 0.05}).startswith(
        "the ESSUA normals do not"
    )
    assert po.inside_reading({"ci_low": -0.08, "ci_high": -0.01}).endswith("worse")
    assert po.inside_reading({"ci_low": 0.01, "ci_high": 0.05}) == "the ESSUA normals help"


def test_day_reading():
    assert po.day_reading([0.85, 0.90]) == "the place cue holds across recording days"
    assert po.day_reading([0.60, 0.62]) == "mostly the conditions of the same day"
    assert po.day_reading([0.75, 0.90]) == "as measured"
