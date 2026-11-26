"""Readings of the non-linear C1; no feature file or recording is read."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "features"))

import nonlinear_c1 as nl  # noqa: E402


def test_plain_reading():
    assert nl.plain_reading({"difference": 0.15, "ci_low": 0.05}) == (
        "the non-linear detector has the gap too"
    )
    assert nl.plain_reading({"difference": 0.03, "ci_low": -0.02}) == (
        "the non-linear detector has no gap"
    )
    assert nl.plain_reading({"difference": 0.07, "ci_low": 0.01}) == "a smaller gap"


def test_versus_reading():
    assert nl.versus_reading({"difference": 0.05, "ci_low": 0.01, "ci_high": 0.09}) == (
        "learns more than logistic"
    )
    assert nl.versus_reading({"difference": -0.05, "ci_low": -0.09, "ci_high": -0.01}) == (
        "learns less than logistic"
    )
    assert nl.versus_reading({"difference": 0.01, "ci_low": -0.02, "ci_high": 0.04}) == (
        "about the same as logistic"
    )
    assert nl.versus_reading({"difference": 0.02, "ci_low": 0.005, "ci_high": 0.04}) == (
        "as measured"
    )


def test_aligned_versus_reading():
    assert nl.aligned_versus_reading({"difference": 0.06, "ci_low": 0.02, "ci_high": 0.1}) == (
        "with both moments aligned, the non-linear detector still finds more place"
    )
    assert nl.aligned_versus_reading({"difference": 0.0, "ci_low": -0.03, "ci_high": 0.03}) == (
        "aligned, the same as logistic"
    )
    assert nl.aligned_versus_reading({"difference": -0.05, "ci_low": -0.08, "ci_high": -0.02}) == (
        "as measured"
    )


def test_output_names():
    assert nl.path_of("svc", "scale").name == "c1_predictions_svc_scale.csv.gz"
