"""Synthetic checks of the CORAL alignment and the place-visibility scores; no real data is read."""

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

import coral_c1 as cc  # noqa: E402
import harmonize_c1 as hc  # noqa: E402

COLUMNS = ["f0", "f1", "f2"]


def _correlated(rng, n, mean, cov):
    return rng.multivariate_normal(mean, cov, size=n)


def test_coral_matches_mean_exactly_and_covariance_closely():
    rng = np.random.default_rng(0)
    cov_e = np.array([[1.0, 0.8, 0.0], [0.8, 1.0, 0.3], [0.0, 0.3, 2.0]])
    cov_h = np.array([[2.0, -0.5, 0.2], [-0.5, 1.0, 0.0], [0.2, 0.0, 0.5]])
    essua = _correlated(rng, 20_000, [3.0, -1.0, 0.5], cov_e)
    hsa = _correlated(rng, 20_000, [0.0, 0.0, 0.0], cov_h)
    moved = cc.coral_apply(essua, cc.coral_from(essua, hsa))
    assert np.allclose(moved.mean(axis=0), hsa.mean(axis=0))
    assert np.allclose(np.cov(moved, rowvar=False), np.cov(hsa, rowvar=False), atol=0.03)


def test_coral_is_finite_when_dimensions_outnumber_rows():
    rng = np.random.default_rng(1)
    essua = rng.normal(size=(40, 78)) * 2.0 + 1.0
    hsa = rng.normal(size=(30, 78))
    moved = cc.coral_apply(essua, cc.coral_from(essua, hsa))
    assert np.isfinite(moved).all()
    assert np.allclose(moved.mean(axis=0), hsa.mean(axis=0))


def _normals(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    place = np.where(np.arange(n) % 3 == 0, "HSA", "ESSUA")
    values = rng.normal(size=(n, len(COLUMNS)))
    essua = place == "ESSUA"
    values[essua] = values[essua] * 0.5 + 2.0
    table = pd.DataFrame(values, columns=COLUMNS)
    table.insert(0, "place", place)
    table.insert(0, "speaker", [f"S{i:02d}" for i in range(n)])
    return table


def test_coral_table_leaves_hsa_rows_and_the_test_fold_out():
    normals = _normals()
    test = {"S01", "S02", "S03", "S04"}
    out = cc.coral_table(normals, COLUMNS, normals, test)
    hsa = (normals["place"] == "HSA").to_numpy()
    assert np.array_equal(out.loc[hsa, COLUMNS].to_numpy(), normals.loc[hsa, COLUMNS].to_numpy())
    kept = ~normals["speaker"].isin(test).to_numpy()
    essua_kept = kept & ~hsa
    assert np.allclose(
        out.loc[essua_kept, COLUMNS].mean().to_numpy(),
        normals.loc[kept & hsa, COLUMNS].mean().to_numpy(),
    )
    with pytest.raises(RuntimeError, match="no normal speaker"):
        cc.coral_table(normals, COLUMNS, normals, set(normals.loc[hsa, "speaker"]))


def test_numpy_location_and_scale_match_harmonize():
    normals = _normals()
    values = normals[COLUMNS].to_numpy(dtype=float)
    essua = (normals["place"] == "ESSUA").to_numpy()
    fit = np.ones(len(normals), dtype=bool)
    moments = hc.place_moments(normals, COLUMNS, set())
    for kind in ("location", "scale"):
        mine = cc.align_values(values, essua, fit, [np.arange(3)], kind)
        theirs = hc.aligned(normals, COLUMNS, moments, kind)[COLUMNS].to_numpy()
        assert np.allclose(mine, theirs)
    with pytest.raises(ValueError):
        cc.align_values(values, essua, fit, [np.arange(3)], "rank")


def test_location_hides_a_mean_shift_from_a_linear_place_detector():
    rng = np.random.default_rng(3)
    n = 300
    essua = np.arange(n) % 3 != 0
    values = rng.normal(size=(n, 4))
    values[essua, 0] += 2.0
    parts = np.arange(n) % 5
    groups = [np.arange(4)]
    from sklearn.metrics import roc_auc_score

    before = roc_auc_score(essua, cc.place_scores(values, essua, parts, groups, "none", "logistic"))
    after = roc_auc_score(
        essua, cc.place_scores(values, essua, parts, groups, "location", "logistic")
    )
    assert before > 0.85
    assert after < 0.65


def test_coral_reading():
    assert (
        cc.coral_reading({"difference": -0.04, "ci_low": -0.06, "ci_high": -0.01})
        == "the correlations carry part of the gap"
    )
    assert (
        cc.coral_reading({"difference": 0.0, "ci_low": -0.02, "ci_high": 0.02})
        == "the correlations carry nothing more"
    )
    assert cc.coral_reading({"difference": 0.01, "ci_low": 0.005, "ci_high": 0.02}) == (
        "as measured"
    )
