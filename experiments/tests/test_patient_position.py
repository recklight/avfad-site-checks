"""Synthetic checks of the patients' position on the place direction; no real data is read."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "features"))

import patient_position as pp  # noqa: E402


def test_parts_are_dealt_inside_each_place():
    place = np.array(["ESSUA"] * 23 + ["HSA"] * 12)
    parts = pp.parts_within_place(place, 5, np.random.default_rng(0))
    for value in ("ESSUA", "HSA"):
        counts = np.bincount(parts[place == value], minlength=5)
        assert counts.max() - counts.min() <= 1


def _blocks(rng, n, shift):
    return {0: rng.normal(size=(n, 16)) + shift, 3: rng.normal(size=(n, 16)) + shift}


def test_patients_like_hsa_normals_sit_near_zero():
    rng = np.random.default_rng(3)
    shift = np.zeros(16)
    shift[0] = 3.0
    essua_n = _blocks(rng, 200, shift)
    hsa_n = _blocks(rng, 80, 0.0)
    patients = _blocks(rng, 300, 0.0)
    normals = {k: np.vstack([essua_n[k], hsa_n[k]]) for k in essua_n}
    essua = np.array([True] * 200 + [False] * 80)
    parts = pp.parts_within_place(np.where(essua, "E", "H"), 5, np.random.default_rng(1))
    normal_proj, patient_proj = pp.cross_fitted(normals, patients, essua, parts)
    assert not np.isnan(normal_proj[0]).any()
    p = pp.position_interval(patient_proj[0], normal_proj[0][~essua], normal_proj[0][essua], 500, 2)
    assert abs(p["position"]) < 0.15
    assert pp.reading(p) == "the patients sit with the HSA normals: the direction is place"


def test_patients_further_out_are_flagged():
    rng = np.random.default_rng(4)
    hsa = rng.normal(size=80)
    essua = rng.normal(size=200) + 2.0
    patients = rng.normal(size=300) - 1.5
    p = pp.position_interval(patients, hsa, essua, 500, 0)
    assert p["position"] < -0.5
    assert pp.reading(p) == "the patients sit further out: the direction runs with pathology"


def test_reading_towards_essua_and_unclear():
    assert pp.reading({"position": 0.6, "ci_low": 0.3, "ci_high": 0.9}) == (
        "the patients lean towards ESSUA"
    )
    assert pp.reading({"position": 0.2, "ci_low": -0.6, "ci_high": 0.9}) == "as measured"
