"""Synthetic checks of the within-place correlations; no real score file is read."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "features"))

import within_place_axis as wa  # noqa: E402


def test_per_speaker_averages_repeats_then_the_draws_a_speaker_is_in(tmp_path):
    rows = [
        ("ii", "A", 0, 0.2),
        ("ii", "A", 1, 0.4),
        ("i_00", "A", 0, 0.6),
        ("i_00", "A", 1, 0.8),
        ("i_01", "A", 0, 0.1),
        ("i_01", "A", 1, 0.3),
        ("i_01", "B", 0, 0.9),
        ("i_01", "B", 1, 0.7),
    ]
    path = tmp_path / "scores.csv"
    pd.DataFrame(rows, columns=["arm", "speaker", "repeat", "score"]).to_csv(path, index=False)
    got = wa.per_speaker(path)
    assert np.isclose(got["ii"]["A"], 0.3)
    assert np.isclose(got["i"]["A"], (0.7 + 0.2) / 2)
    assert np.isclose(got["i"]["B"], 0.8)


def test_correlations_and_reading():
    index = pd.Index([f"S{i}" for i in range(100)])
    projection = pd.Series(np.arange(100.0), index=index)
    scores = {
        "i": pd.Series(-np.arange(100.0), index=index),
        "ii": pd.Series(np.zeros(100), index=index),
    }
    scores["ii"].iloc[::2] = 1.0
    got = wa.correlations(projection, scores, np.ones((3, 100)))
    assert np.isclose(got["i"][0], -1.0)
    assert wa.follow_reading({"difference": -0.3, "ci_low": -0.5, "ci_high": -0.1}) == (
        "inside one place, the (i) score follows the place direction too"
    )
    assert wa.follow_reading({"difference": -0.05, "ci_low": -0.2, "ci_high": 0.1}) == (
        "not visible inside one place"
    )
    assert wa.follow_reading({"difference": -0.08, "ci_low": -0.15, "ci_high": -0.01}) == (
        "as measured"
    )
