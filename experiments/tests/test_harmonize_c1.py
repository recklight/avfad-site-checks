"""Synthetic checks of the per-dimension alignment; no feature file or recording is read."""

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

import harmonize_c1 as hc  # noqa: E402

COLUMNS = ["f0", "f1", "f2"]


def _normals(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    place = np.where(np.arange(n) % 3 == 0, "HSA", "ESSUA")
    values = rng.normal(size=(n, len(COLUMNS)))
    # ESSUA: shifted and tighter; the last column is constant there.
    essua = place == "ESSUA"
    values[essua] = values[essua] * 0.5 + 2.0
    values[essua, 2] = 7.0
    table = pd.DataFrame(values, columns=COLUMNS)
    table.insert(0, "place", place)
    table.insert(0, "speaker", [f"S{i:02d}" for i in range(n)])
    return table


def test_moments_leave_out_the_test_fold():
    normals = _normals()
    test = {"S01", "S02", "S03"}
    moments = hc.place_moments(normals, COLUMNS, test)
    kept = normals[~normals["speaker"].isin(test)]
    expected = kept.loc[kept["place"] == "ESSUA", COLUMNS].to_numpy().mean(axis=0)
    assert np.allclose(moments["mu_e"], expected)


def test_moments_need_both_places():
    normals = _normals()
    hsa = set(normals.loc[normals["place"] == "HSA", "speaker"])
    with pytest.raises(RuntimeError, match="no normal speaker"):
        hc.place_moments(normals, COLUMNS, hsa)


def test_location_matches_the_means_and_keeps_hsa():
    normals = _normals()
    moments = hc.place_moments(normals, COLUMNS, set())
    out = hc.aligned(normals, COLUMNS, moments, "location")
    essua = (normals["place"] == "ESSUA").to_numpy()
    assert np.allclose(out.loc[essua, COLUMNS].mean(), out.loc[~essua, COLUMNS].mean())
    assert np.array_equal(out.loc[~essua, COLUMNS].to_numpy(), normals.loc[~essua, COLUMNS])


def test_scale_matches_the_spread_except_constant_dimensions():
    normals = _normals()
    moments = hc.place_moments(normals, COLUMNS, set())
    out = hc.aligned(normals, COLUMNS, moments, "scale")
    essua = (normals["place"] == "ESSUA").to_numpy()
    sd_e = out.loc[essua, COLUMNS[:2]].std(ddof=1).to_numpy()
    sd_h = out.loc[~essua, COLUMNS[:2]].std(ddof=1).to_numpy()
    assert np.allclose(sd_e, sd_h)
    # The constant column only moves to the HSA mean.
    assert np.allclose(out.loc[essua, "f2"], moments["mu_h"][2])
    assert np.isfinite(out[COLUMNS].to_numpy()).all()


def test_unknown_kind():
    normals = _normals()
    with pytest.raises(ValueError):
        hc.aligned(normals, COLUMNS, hc.place_moments(normals, COLUMNS, set()), "rank")


def test_readings():
    assert hc.gap_reading({"difference": 0.12, "ci_low": 0.02}) == "the gap stays"
    assert hc.gap_reading({"difference": 0.04, "ci_low": -0.01}) == "the gap is gone"
    assert hc.gap_reading({"difference": 0.06, "ci_low": 0.01}) == "the gap shrinks"
    assert (
        hc.scale_reading({"difference": -0.04, "ci_low": -0.06, "ci_high": -0.01})
        == "the spread carries part of the gap"
    )
    assert (
        hc.scale_reading({"difference": -0.01, "ci_low": -0.03, "ci_high": 0.01})
        == "the spread carries nothing more"
    )
    assert (
        hc.scale_reading({"difference": -0.02, "ci_low": -0.03, "ci_high": -0.005}) == "as measured"
    )


def test_paired_difference_uses_the_same_resamples():
    a = ({"difference": 0.10}, np.array([0.1, 0.2, np.nan, 0.3]))
    b = ({"difference": 0.04}, np.array([0.0, 0.1, 0.1, 0.2]))
    out = hc.paired(a, b)
    assert out["difference"] == pytest.approx(0.06)
    assert out["ci_low"] == pytest.approx(0.1)
    assert out["ci_high"] == pytest.approx(0.1)
