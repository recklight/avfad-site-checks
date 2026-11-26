"""Synthetic checks of the placebo C1 sets; no SVD file is read."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "features"))

import placebo_c1 as pc  # noqa: E402


def _meta(n_patients=500, n_normals=400):
    rng = np.random.default_rng(0)
    n = n_patients + n_normals
    return pd.DataFrame(
        {
            "speaker": [f"svd{i:04d}" for i in range(n)],
            "file_id": [f"svd__{i}-a_n" for i in range(n)],
            "label": ["pathological"] * n_patients + ["normal"] * n_normals,
            "age": rng.uniform(20, 60, n),
            "sex": rng.choice(["m", "w"], n),
        }
    )


def test_sets_share_the_patients_and_match_the_cells():
    sets = pc.placebo_sets(_meta(), 0, 5, 1)
    assert [n for n, _ in sets] == ["ii", "i_00", "i_01", "i_02", "i_03", "i_04"]
    ii = sets[0][1]
    patients = set(ii.loc[ii["label"] == "pathological", "speaker"])
    assert len(patients) == pc.N_PATIENTS
    assert (ii["label"] == "normal").sum() == pc.N_NORMALS
    for _, members in sets[1:]:
        assert set(members.loc[members["label"] == "pathological", "speaker"]) == patients
        assert (members["label"] == "normal").sum() == pc.N_NORMALS
        assert members["speaker"].is_unique


def test_splits_differ_and_repeat():
    a = pc.placebo_sets(_meta(), 0, 1, 1)[0][1]["speaker"].tolist()
    b = pc.placebo_sets(_meta(), 1, 1, 1)[0][1]["speaker"].tolist()
    assert a != b
    assert a == pc.placebo_sets(_meta(), 0, 1, 1)[0][1]["speaker"].tolist()


def test_reading():
    assert pc.reading({"difference": 0.01, "ci_low": -0.02, "ci_high": 0.04}) == "no place, no gap"
    assert pc.reading({"difference": 0.05, "ci_low": 0.01, "ci_high": 0.09}) == (
        "the design makes a gap by itself"
    )
    assert pc.reading({"difference": -0.05, "ci_low": -0.09, "ci_high": -0.01}) == (
        "the design puts (i) lower by itself"
    )
    assert pc.reading({"difference": 0.02, "ci_low": 0.005, "ci_high": 0.04}) == "as measured"


def test_spread_reading():
    values = [-0.04, 0.0, 0.04]
    assert pc.spread_reading(values, [0.01, 0.03]) == (
        "what is left after the alignment is within the spread without place"
    )
    assert (
        pc.spread_reading(values, [0.05, 0.06])
        == "what is left is larger than the spread without place"
    )
    assert pc.spread_reading(values, [0.03, 0.06]) == "as measured"
