"""Synthetic checks of the per-chapter breakdown; no score file or recording is read."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "features"))

import diagnosis_breakdown as db  # noqa: E402


def test_chapters_follow_the_first_digit_of_the_code():
    assert db.chapter_group(1220) == "structural"
    assert db.chapter_group(2300) == "inflammatory"
    assert db.chapter_group(7120) == "neurological"
    assert [db.chapter_group(c) for c in (3110, 6051, 8010, 9000)] == ["other"] * 4


def test_unit_weights_give_the_ordinary_auc_with_ties():
    rng = np.random.default_rng(0)
    pos = np.round(rng.normal(0.5, 1, 40), 1)
    neg = np.round(rng.normal(0.0, 1, 30), 1)
    kernel = db.comparisons(pos, neg)
    expected = roc_auc_score(np.r_[np.ones(40), np.zeros(30)], np.r_[pos, neg])
    assert db.weighted_auc(kernel, np.ones(40), np.ones(30))[0] == pytest.approx(expected)
    assert kernel.mean() == pytest.approx(expected)


def test_integer_weights_equal_repeating_the_speakers():
    rng = np.random.default_rng(1)
    pos, neg = rng.normal(1, 1, 12), rng.normal(0, 1, 9)
    w_pos, w_neg = rng.integers(0, 4, 12), rng.integers(1, 4, 9)
    w_pos[0] = 1
    expected = roc_auc_score(
        np.r_[np.ones(w_pos.sum()), np.zeros(w_neg.sum())],
        np.r_[np.repeat(pos, w_pos), np.repeat(neg, w_neg)],
    )
    got = db.weighted_auc(db.comparisons(pos, neg), w_pos.astype(float), w_neg.astype(float))
    assert got[0] == pytest.approx(expected)


def test_resample_counts_keep_the_group_size():
    counts = db.resample_counts(17, 50, np.random.default_rng(2))
    assert counts.shape == (50, 17)
    assert (counts.sum(axis=1) == 17).all()


def _cell(auc, low=None):
    return {
        "auc": auc,
        "ci_low": auc - 0.06 if low is None else low,
        "ci_high": auc + 0.06,
    }


@pytest.mark.parametrize(
    ("aucs", "expected"),
    [
        ((0.80, 0.58, 0.70), "case mix in part"),
        ((0.66, 0.60, 0.70), "not case mix"),
        ((0.74, 0.60, 0.70), "unclear"),
        ((0.80, 0.68, 0.70), "unclear"),
    ],
)
def test_case_mix_reading(aucs, expected):
    cells = dict(zip(db.MAIN, (_cell(a) for a in aucs), strict=True))
    assert db.case_mix_reading(cells) == expected


def test_case_mix_needs_the_best_ci_above_the_floor():
    cells = {
        "structural": _cell(0.78, 0.64),
        "inflammatory": _cell(0.6),
        "neurological": _cell(0.7),
    }
    assert db.case_mix_reading(cells) == "unclear"


def test_gains_everywhere_needs_ten_points_and_a_positive_ci():
    good = {"difference": 0.15, "ci_low": 0.05}
    assert db.gains_everywhere({g: good for g in db.MAIN})
    assert not db.gains_everywhere(
        {
            "structural": good,
            "inflammatory": good,
            "neurological": {"difference": 0.08, "ci_low": 0.01},
        }
    )


@pytest.mark.parametrize(
    ("difference", "low", "high", "expected"),
    [
        (0.05, -0.03, 0.12, "the least audible chapter gains no less"),
        (-0.12, -0.20, -0.04, "the gain follows how audible the disorder is"),
        (-0.04, -0.09, -0.01, "unclear"),
        (-0.02, -0.10, 0.06, "unclear"),
    ],
)
def test_contrast_reading(difference, low, high, expected):
    contrast = {"difference": difference, "ci_low": low, "ci_high": high}
    assert db.contrast_reading(contrast) == expected


def test_thresholds_and_seed_are_the_written_ones():
    assert (db.BEST_AUC, db.BEST_LOW, db.WORST_AUC, db.NONE_ABOVE) == (
        0.75,
        0.65,
        0.65,
        0.70,
    )
    assert db.GAIN_POINTS == 0.10 and db.CONTRAST_POINTS == 0.05
    assert db.SEED == 20261121 and db.N_RESAMPLES == 10_000
    assert db.VOWEL_SEEDS == {"i": 20261129, "u": 20261130}
