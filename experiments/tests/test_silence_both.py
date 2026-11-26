"""Shares and readings of the two-sided silence variant; no real data is read."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "features"))

import silence_both as sb  # noqa: E402


def test_each_site_takes_its_own_place_share():
    measured = {"silence_share": {"ESSUA": 0.304, "HSA": 0.244, "difference": 0.06}}
    assert sb.shares(measured) == {"A": 0.244, "B": 0.304}


def test_readings():
    assert sb.reading({"difference": 0.12, "ci_low": 0.05, "ci_high": 0.2}, "handcrafted") == (
        "the measured silence difference alone reaches half of AVFAD's gap or more"
    )
    assert sb.reading({"difference": 0.04, "ci_low": 0.01, "ci_high": 0.08}, "hubert") == (
        "a gap, less than half"
    )
    assert sb.reading({"difference": 0.0, "ci_low": -0.04, "ci_high": 0.04}, "hubert") == (
        "with silence on both sides, the measured difference makes no gap"
    )
