"""Synthetic checks of the ESSUA-share steps; no feature file or recording is read."""

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

import dose_response as dr  # noqa: E402


def _people(prefix: str, ages, sexes, place: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "speaker": [f"{prefix}{i:02d}" for i in range(len(ages))],
            "age": ages,
            "sex": sexes,
            "place": place,
            "label": "normal",
        }
    )


def test_swaps_keep_the_size_and_the_cell():
    hsa = _people("H", [40, 41, 52, 63, 70, 71], ["F", "F", "M", "F", "M", "M"], "HSA")
    essua = _people("E", [40, 42, 53, 64, 60, 44, 43], ["F", "F", "M", "F", "F", "F", "F"], "ESSUA")
    out = dr.swapped_normals(hsa, essua, 6, np.random.default_rng(0))
    assert len(out) == 6 and not out["speaker"].duplicated().any()
    swapped = out[out["place"] == "ESSUA"]
    kept = out[out["place"] == "HSA"]
    # The two men of 70 and 71 have no ESSUA match in their cell and stay.
    assert set(kept["speaker"]) == {"H04", "H05"}
    assert len(swapped) == 4
    cells = sorted(dr.cell_of(a, s) for a, s in zip(swapped["age"], swapped["sex"], strict=True))
    assert cells == sorted(["8_F", "8_F", "10_M", "12_F"])


def test_no_swap_is_the_hsa_group():
    hsa = _people("H", [40, 50], ["F", "M"], "HSA")
    essua = _people("E", [40, 50], ["F", "M"], "ESSUA")
    out = dr.swapped_normals(hsa, essua, 0, np.random.default_rng(1))
    assert sorted(out["speaker"]) == ["H00", "H01"]


def test_the_slope_of_a_line_and_its_interval():
    levels = {k: [(k / 82, 0.6 + 0.2 * k / 82)] * 3 for k in dr.LEVELS}
    out = dr.slope_interval(levels, (0.0, 0.6))
    assert out["slope"] == pytest.approx(0.2)
    assert out["ci_low"] == pytest.approx(0.2) and out["ci_high"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("slope", "low", "high", "rise", "outside"),
    [
        (0.25, 0.15, 0.35, "the score rises with the ESSUA share", "as measured"),
        (0.05, 0.01, 0.09, "as measured", "as measured"),
        (0.01, -0.04, 0.06, "as measured", "flat for the outside detector"),
        (-0.06, -0.10, -0.02, "as measured", "the outside detector goes down"),
    ],
)
def test_readings(slope, low, high, rise, outside):
    s = {"slope": slope, "ci_low": low, "ci_high": high}
    assert dr.rise_reading(s) == rise
    assert dr.outside_reading(s) == outside


def test_constants_are_the_written_ones():
    assert dr.LEVELS == (20, 41, 61, 82) and dr.DRAWS == 10 and dr.NORMALS == 82
    assert dr.SEED == 20261139 and dr.N_SLOPE == 1000 and dr.SLOPE_POINTS == 0.10
    assert dr.HANDCRAFTED_GRID["model__C"] == [0.01, 0.1, 1.0, 10.0]


@pytest.mark.parametrize(
    ("slope", "low", "expected"),
    [
        (0.08, 0.05, "with the direction out, most of the dose response is gone"),
        (0.20, 0.15, "with one direction out, the dose response stays"),
        (0.12, 0.08, "as measured"),
    ],
)
def test_erased_reading(slope, low, expected):
    assert dr.erased_reading({"slope": slope, "ci_low": low, "ci_high": slope + 0.03}) == expected
