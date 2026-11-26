"""Synthetic checks of the noise augmentation; no recording is read."""

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

import noise_augmentation as na  # noqa: E402
from extract_handcrafted import FEATURE_COLUMNS  # noqa: E402


def test_copy_plan_gives_each_file_one_copy_per_place_within_six_db():
    plan = na.copy_plan(["a", "b", "c"], {"HSA": 5, "ESSUA": 7})
    assert [(f, p) for f, p, _, _ in plan] == [
        (f, p) for f in ("a", "b", "c") for p in ("HSA", "ESSUA")
    ]
    for _, place, room, offset in plan:
        assert 0 <= room < {"HSA": 5, "ESSUA": 7}[place]
        assert -6.0 <= offset <= 6.0
    assert plan == na.copy_plan(["a", "b", "c"], {"HSA": 5, "ESSUA": 7})


def test_training_rows_take_every_block_of_the_training_speakers():
    rows = na.training_rows(np.array([0, 2]), n=4, blocks=3)
    assert rows.tolist() == [0, 2, 4, 6, 8, 10]


def _tables(n=6):
    table = pd.DataFrame(
        {
            "file_id": [f"avfad__{i}" for i in range(n)],
            "speaker": [str(i) for i in range(n)],
            "label": ["normal", "pathological"] * (n // 2),
        }
    )
    clean = pd.DataFrame({"file_id": table["file_id"]})
    for k, column in enumerate(FEATURE_COLUMNS):
        clean[column] = np.arange(n) + k
    copies = pd.concat(
        [
            clean.assign(copy=place, **{c: clean[c] + 1000 * (j + 1) for c in FEATURE_COLUMNS})
            for j, place in enumerate(("HSA", "ESSUA"))
        ]
    )
    return table, clean, copies


def test_stacked_keeps_clean_rows_first_and_copies_in_the_same_order():
    table, clean, copies = _tables()
    X, y, groups = na.stacked(table, clean, copies)
    assert X.shape == (18, len(FEATURE_COLUMNS))
    assert X[0, 0] == 0 and X[6, 0] == 1000 and X[12, 0] == 2000
    assert groups.tolist() == table["speaker"].tolist() * 3
    assert y.tolist() == table["label"].tolist() * 3


@pytest.mark.parametrize(
    ("flip_aug", "flip_base", "flip_ci", "gap_change", "gap_ci", "ii_change", "verdict"),
    [
        (0.20, 0.48, [-0.35, -0.20], -0.08, [-0.12, -0.03], 0.00, "works"),
        (0.20, 0.48, [-0.35, -0.20], -0.08, [-0.12, -0.03], -0.03, "partly"),
        (0.30, 0.48, [-0.25, -0.10], -0.08, [-0.12, -0.03], 0.01, "partly"),
        (0.30, 0.48, [-0.25, -0.10], -0.02, [-0.05, 0.01], 0.01, "does not work"),
        (0.20, 0.48, [-0.35, 0.01], -0.02, [-0.05, 0.01], 0.01, "does not work"),
    ],
)
def test_reading(flip_aug, flip_base, flip_ci, gap_change, gap_ci, ii_change, verdict):
    assert (
        na.reading(flip_aug, flip_base, flip_ci, gap_change, gap_ci, ii_change)["verdict"]
        == verdict
    )


def test_reading_marks_a_better_inside_place_score():
    record = na.reading(0.4, 0.48, [-0.1, -0.01], 0.0, [-0.02, 0.02], 0.06)
    assert record["ii_better"] and record["ii_not_worse"]


def test_paired_ci_brackets_the_mean():
    values = np.random.default_rng(0).normal(-0.2, 0.1, 300)
    low, high = na.paired_mean_ci(values, seed=1, n=2000)
    assert low < values.mean() < high < 0


def test_thresholds_are_the_written_ones():
    assert na.FLIP_FRACTION == 0.5 and na.GAP_SHRINK == 0.05
    assert na.II_TOLERANCE == 0.02 and na.II_BETTER == 0.05
    assert na.LEVEL_SPREAD_DB == 6.0 and na.AUG_SEED == 20261112
    assert na.MAIN_CASE == "ESSUA_with_HSA_+0"
