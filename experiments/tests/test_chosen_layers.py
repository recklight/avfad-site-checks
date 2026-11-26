"""Synthetic checks of the chosen-layer count; no result file is read."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "features"))

import chosen_layers as cl  # noqa: E402


def test_shares_add_the_folds_of_every_arm():
    records = [
        '{"C": {"0.01": 4}, "layer": {"l00": 3, "l09": 1}}',
        '{"C": {"0.01": 4}, "layer": {"l00": 1, "l12": 3}}',
    ]
    out = cl.shares(records)
    assert out == pytest.approx({"l00": 0.5, "l03": 0.0, "l06": 0.0, "l09": 0.125, "l12": 0.375})


@pytest.mark.parametrize(
    ("ii0", "i0", "expected"),
    [
        (0.1, 0.4, "(i) leans to the shallowest layer"),
        (0.5, 0.2, "(i) leans to deeper layers"),
        (0.3, 0.4, "both choose about the same layers"),
    ],
)
def test_reading(ii0, i0, expected):
    assert cl.reading({"l00": ii0}, {"l00": i0}) == expected


def test_threshold_is_the_written_one():
    assert cl.SHIFT == 0.20
