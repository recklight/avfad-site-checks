"""The timing script's helpers; nothing is trained and no recording is read."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

import finetune_timing as timing  # noqa: E402


def test_peak_memory_is_read():
    assert timing.peak_memory_gb() > 0.0


@pytest.mark.parametrize("output", [("tensor", "attention"), "tensor"])
def test_through_takes_a_tuple_or_the_tensor(output):
    assert timing.through(lambda _hidden: output, None) == "tensor"


def test_settings_are_mode_layers_batch_seconds_threads():
    for mode, layers, batch, seconds, threads in timing.SETTINGS + timing.EXTRA_SETTINGS:
        assert mode in ("cached", "end_to_end")
        assert 1 <= layers <= 12
        assert batch in (4, 8)
        assert seconds in (4.0, 10.0)
        assert threads <= 8
