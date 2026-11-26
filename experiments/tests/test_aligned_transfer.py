"""Synthetic checks of the aligned mixed-place training; no feature file or recording is read."""

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

import aligned_transfer as at  # noqa: E402

COLUMNS = ["a0", "a1", "b0", "b1"]


def _speakers(rng, place, label, n, shift):
    values = rng.normal(size=(n, len(COLUMNS))) + shift
    table = pd.DataFrame(values, columns=COLUMNS)
    table.insert(0, "label", label)
    table.insert(0, "place", place)
    return table


def test_only_essua_rows_move_and_land_on_the_hsa_mean():
    rng = np.random.default_rng(0)
    normals = pd.concat(
        [_speakers(rng, "ESSUA", "normal", 200, 3.0), _speakers(rng, "HSA", "normal", 80, 0.0)],
        ignore_index=True,
    )
    train = pd.concat([normals, _speakers(rng, "HSA", "pathological", 50, 1.0)], ignore_index=True)
    train.loc[len(train)] = ["other", "normal", 9.0, 9.0, 9.0, 9.0]
    groups = at.groups_of(COLUMNS, {"l0": ["a0", "a1"], "l1": ["b0", "b1"]})
    for kind in ("location", "scale", "coral"):
        out = at.moved_training(train, normals, COLUMNS, groups, kind)
        essua = (train["place"] == "ESSUA").to_numpy()
        assert np.array_equal(
            out.loc[~essua, COLUMNS].to_numpy(), train.loc[~essua, COLUMNS].to_numpy()
        )
        hsa_normals = normals.loc[normals["place"] == "HSA", COLUMNS].mean().to_numpy()
        assert np.allclose(out.loc[essua, COLUMNS].mean().to_numpy(), hsa_normals)


def test_groups_follow_the_blocks():
    groups = at.groups_of(COLUMNS, {"l0": ["a0", "a1"], "l1": ["b0", "b1"]})
    assert [g.tolist() for g in groups] == [[0, 1], [2, 3]]
    assert [g.tolist() for g in at.groups_of(COLUMNS, None)] == [[0, 1, 2, 3]]


def test_paired_difference_and_readings():
    first = (0.70, np.array([0.70, 0.72, 0.68, 0.71]))
    second = (0.60, np.array([0.60, 0.61, 0.59, 0.62]))
    change = at.difference(first, second)
    assert np.isclose(change["difference"], 0.10)
    assert change["ci_low"] > 0
    assert at.change_reading(change) == "the alignment makes mixed training better outside"
    assert at.change_reading({"difference": -0.05, "ci_low": -0.08, "ci_high": -0.02}) == (
        "the alignment makes it worse"
    )
    assert at.change_reading({"difference": 0.01, "ci_low": -0.01, "ci_high": 0.03}) == (
        "the alignment does not change the outside score"
    )
    assert at.versus_ii_reading({"ci_low": -0.02, "ci_high": 0.01}) == (
        "catches up with training inside HSA"
    )
    assert at.versus_ii_reading({"ci_low": -0.09, "ci_high": -0.01}) == (
        "still below training inside HSA"
    )
