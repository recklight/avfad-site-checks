"""Synthetic checks of the redone C2; no recording is read."""

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

import redo_c2 as redo  # noqa: E402
from common import rms_dbfs  # noqa: E402
from extract_handcrafted import FEATURE_COLUMNS, level_statistics  # noqa: E402


def test_conditions_are_the_ones_written_down():
    by_name = {c.name: c for c in redo.CONDITIONS}
    assert by_name["B"].who == ("ESSUA",) and by_name["B"].room == "HSA"
    assert len(by_name["B"].seeds) == 3
    assert by_name["C"].who == ("ESSUA",) and by_name["C"].room == "ESSUA"
    assert by_name["D"].who == ("ESSUA", "HSA") and by_name["D"].extra_noise_db == 6.0


def test_gain_puts_the_noise_at_the_target_snr():
    rng = np.random.default_rng(0)
    noise = 0.001 * rng.normal(size=16_000)
    level = float(level_statistics([noise], 16_000)[0])
    gain = redo.added_noise_gain_db(-40.0, 15.0, level)
    scaled = noise * 10.0 ** (gain / 20.0)
    assert float(level_statistics([scaled], 16_000)[0]) == pytest.approx(-55.0, abs=1e-3)


def test_snr_follows_a_louder_voice():
    assert redo.added_noise_gain_db(-30.0, 15.0, -60.0) - redo.added_noise_gain_db(
        -40.0, 15.0, -60.0
    ) == pytest.approx(10.0)


def test_a_speaker_noisier_than_the_target_gets_nothing():
    assert redo.needs_noise(25.0, 15.0)
    assert not redo.needs_noise(10.0, 15.0)
    assert redo.needs_noise(float("nan"), 15.0)


def test_noise_at_a_chosen_snr_measures_that_snr():
    rng = np.random.default_rng(1)
    voice = 0.05 * np.sin(2 * np.pi * 200 * np.arange(16_000) / 16_000)
    noise = 0.001 * rng.normal(size=16_000)
    level = float(level_statistics([noise], 16_000)[0])
    phonation_db = rms_dbfs(voice)
    gain = redo.added_noise_gain_db(phonation_db, 12.0, level)
    scaled = noise * 10.0 ** (gain / 20.0)
    assert phonation_db - float(level_statistics([scaled], 16_000)[0]) == pytest.approx(
        12.0, abs=1e-3
    )


def test_replaced_swaps_only_the_changed_speakers():
    base = pd.DataFrame({"file_id": ["a", "b", "c"], "x": [1.0, 2.0, 3.0]})
    new = pd.DataFrame({"file_id": ["b"], "x": [9.0], "extra": [0.0]})
    out = redo.replaced(base, new).set_index("file_id")["x"]
    assert out.to_dict() == {"a": 1.0, "b": 9.0, "c": 3.0}


def test_set_names_are_stable():
    assert redo.set_name("B", 2) == "B_s2"


class _FakeArm:
    def __init__(self, name, groups, y, predictions, uar):
        self.name, self.groups, self.y_true = name, groups, y
        self.predictions, self.uar, self.roc_auc = predictions, uar, float("nan")


def _arm(name, predictions, uar):
    y = np.array(["ESSUA"] * 20 + ["HSA"] * 20)
    groups = np.array([f"s{i}" for i in range(40)])
    return _FakeArm(name, groups, y, predictions, uar)


def test_paired_row_reports_the_drop_from_a():
    y = np.array(["ESSUA"] * 20 + ["HSA"] * 20)
    perfect = _arm("A", y[:, None], 1.0)
    flipped = _arm("B", np.where(y == "ESSUA", "HSA", "ESSUA")[:, None], 0.0)
    row = redo.paired_row(flipped, perfect, 0)
    assert row["drop_from_a"] == pytest.approx(1.0)
    assert row["drop_ci"][0] == pytest.approx(1.0)


def test_paired_row_refuses_other_speakers():
    y = np.array(["ESSUA"] * 20 + ["HSA"] * 20)
    a = _arm("A", y[:, None], 1.0)
    b = _arm("B", y[:, None], 1.0)
    b.groups = b.groups[::-1]
    with pytest.raises(ValueError, match="speakers differ"):
        redo.paired_row(b, a, 0)


def test_arm_meta_leaves_no_column_that_clashes_with_the_features():
    table = pd.DataFrame(
        {
            "file_id": ["a", "b", "c", "d"],
            "speaker": list("abcd"),
            "place": ["HSA", "HSA", "ESSUA", "ESSUA"],
            "label": ["normal"] * 4,
            "age": [30, 40, 50, 60],
            "sex": list("FMFM"),
            "level_mean": [-50.0] * 4,
            "phonation_rms_dbfs": [-30.0] * 4,
            "kept_s": [2.0] * 4,
        }
    )
    kept = redo.arm_meta(table)
    assert not set(kept.columns) & set(FEATURE_COLUMNS)
    assert kept["age_group"].tolist() == ["younger", "younger", "older", "older"]
