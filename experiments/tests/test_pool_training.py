"""Synthetic checks of the pooled training; no feature file or recording is read."""

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

import pool_training as pt  # noqa: E402


def test_every_row_is_tested_once_per_repeat():
    from harness import load_plan, make_config

    config = make_config(load_plan(), "pathological")
    table = pd.DataFrame(
        {
            "speaker": [f"S{i:03d}" for i in range(60)],
            "label": np.where(np.arange(60) % 5 == 0, "normal", "pathological"),
        }
    )
    folds = pt.fold_ids(table, config)
    assert folds.shape == (60, config.evaluation.n_repeats)
    assert set(np.unique(folds)) == set(range(config.evaluation.n_splits))
    normals = table["label"].to_numpy() == "normal"
    for r in range(folds.shape[1]):
        counts = np.bincount(folds[normals, r], minlength=config.evaluation.n_splits)
        assert counts.max() - counts.min() <= 1


@pytest.mark.parametrize(
    ("difference", "low", "high", "expected"),
    [
        (0.07, 0.02, 0.12, "helps"),
        (0.02, -0.01, 0.05, "does not help"),
        (-0.06, -0.10, -0.02, "worse"),
        (0.03, 0.01, 0.05, "small"),
    ],
)
def test_help_reading(difference, low, high, expected):
    assert pt.help_reading({"difference": difference, "ci_low": low, "ci_high": high}) == expected


def test_the_added_sets_are_the_written_ones():
    assert pt.ARMS == ("ii", "ii_essua", "ii_svd")
    assert pt.added_members("ii") is None
    with pytest.raises(ValueError, match="other"):
        pt.added_members("other")


def test_thresholds_and_seed_are_the_written_ones():
    assert pt.HELPS_POINTS == 0.05 and pt.REACHES == 0.70 and pt.SAME_PATH == 1e-6
    assert pt.SEED == 20261122
