"""Synthetic checks of the AUC inside the strata; no recording is read."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import within_strata_auc as w


def test_auc_ends_and_ties():
    assert w.mann_whitney_auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert w.mann_whitney_auc(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 0.0
    assert w.mann_whitney_auc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5


def test_a_score_that_only_knows_the_stratum_is_chance_inside():
    # The score separates the strata, and the labels follow the strata, so the overall AUC is
    # high; inside each stratum it knows nothing.
    rng = np.random.default_rng(0)
    n = 4000
    cells = np.array(["young"] * (n // 2) + ["old"] * (n // 2))
    sick = rng.random(n) < np.where(cells == "old", 0.8, 0.2)
    scores = (cells == "old").astype(float) + rng.normal(0, 0.01, n)
    overall = w.mann_whitney_auc(scores[sick], scores[~sick])
    assert overall > 0.75
    assert w.pooled_auc(scores, sick, cells) == pytest.approx(0.5, abs=0.03)


def test_a_score_with_real_signal_stays_high_inside():
    rng = np.random.default_rng(1)
    n = 2000
    cells = np.array(["a", "b"] * (n // 2))
    sick = rng.random(n) < 0.5
    scores = 2.0 * sick + rng.normal(0, 0.5, n)
    assert w.pooled_auc(scores, sick, cells) > 0.95


def test_a_stratum_with_one_label_is_left_out():
    scores = np.array([2.0, 1.0, 9.0, 9.0])
    sick = np.array([True, False, True, True])
    cells = np.array(["a", "a", "b", "b"])
    assert w.pooled_auc(scores, sick, cells) == 1.0


def test_interval_brackets_the_estimate(monkeypatch):
    monkeypatch.setattr(w, "N_RESAMPLES", 100)
    rng = np.random.default_rng(2)
    n = 300
    cells = np.array(["a", "b", "c"] * (n // 3))
    sick = rng.random(n) < 0.5
    matrix = np.column_stack([sick + rng.normal(0, 1, n) for _ in range(3)])
    out = w.within_strata(matrix, sick, cells, 0)
    assert out["ci"][0] < out["auc"] < out["ci"][1]
    assert out["strata_with_both_labels"] == 3 and out["speakers"] == n
