"""Synthetic checks of the fold-wise place erasure; no feature file or recording is read."""

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

import erasure_folds as ef  # noqa: E402
from arms import layer_columns  # noqa: E402
from sensitivity_avfad import CHOSEN_LAYERS  # noqa: E402


def _normals(n: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    columns = [c for layer in CHOSEN_LAYERS for c in layer_columns(layer)]
    table = pd.DataFrame(rng.normal(size=(n, len(columns))), columns=columns)
    table.insert(0, "place", np.where(np.arange(n) % 3 == 0, "HSA", "ESSUA"))
    table.insert(0, "speaker", [f"S{i:02d}" for i in range(n)])
    return table


def test_the_test_fold_does_not_move_the_direction():
    normals = _normals()
    held = {"S00", "S01", "S02"}
    first = ef.held_out_directions(normals, held)
    changed = normals.copy()
    rows = changed["speaker"].isin(held)
    changed.loc[rows, changed.columns[2:]] = 100.0
    second = ef.held_out_directions(changed, held)
    for layer in CHOSEN_LAYERS:
        assert np.allclose(first[layer], second[layer])
    assert not np.allclose(first[0], ef.held_out_directions(normals, set())[0])


def test_a_place_with_no_one_left_is_refused():
    normals = _normals(6)
    hsa = set(normals.loc[normals["place"] == "HSA", "speaker"])
    with pytest.raises(RuntimeError, match="no normal speaker"):
        ef.held_out_directions(normals, hsa)


@pytest.mark.parametrize(
    ("difference", "low", "expected"),
    [
        (0.15, 0.08, "the gap stays: one direction is not enough"),
        (0.02, -0.02, "the gap is gone: it sits on this one direction"),
        (0.05, 0.01, "the gap shrinks"),
    ],
)
def test_gap_reading(difference, low, expected):
    assert ef.gap_reading({"difference": difference, "ci_low": low}) == expected


def test_thresholds_and_seed_are_the_written_ones():
    assert ef.GAP_POINTS == 0.10 and ef.II_LOSS_POINTS == 0.03 and ef.PAIRED_SEED == 20261128
    assert ef.VOWEL_PAIRED_SEEDS == {"i": 20261132, "u": 20261133}
    assert [ef.suffix(v) for v in "aiu"] == ["", "_i", "_u"]
    assert ef.suffix("a", "phonation") == "_phonation" and ef.PHONATION_PAIRED_SEED == 20261135
