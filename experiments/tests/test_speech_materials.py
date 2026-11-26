"""Synthetic checks of C1 on other materials; no recording is read."""

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

import speech_materials as sm  # noqa: E402
from extract_hubert import pool_frames  # noqa: E402


def test_sets_are_the_written_ones():
    assert sm.SETS["cape_v"] == ("004", "005", "006", "007", "008", "009")
    assert sm.SETS["reading"] == ("010",)
    assert sm.SETS["connected"] == (*sm.SETS["cape_v"], "010")
    assert sm.SETS["s004"] == ("004",)
    assert "011" not in {task for tasks in sm.SETS.values() for task in tasks}
    assert "002" not in {task for tasks in sm.SETS.values() for task in tasks}


@pytest.mark.parametrize(
    ("task", "duration", "reason"),
    [
        ("004", 0.09, "sentence_shorter_than_0.5s"),
        ("007", 0.0, "sentence_shorter_than_0.5s"),
        ("008", 277.0, "sentence_longer_than_60s"),
        ("009", 76.7, "sentence_longer_than_60s"),
        ("010", 1.27, "reading_shorter_than_20s"),
        ("004", 42.5, ""),
        ("010", 160.6, ""),
        ("010", 28.2, ""),
    ],
)
def test_exclusions_depend_on_task_and_length_only(task, duration, reason):
    assert sm.excluded_reason(task, duration) == reason


def test_combined_pooling_equals_pooling_all_frames():
    rng = np.random.default_rng(0)
    first = rng.normal(0.5, 1.0, size=(13, 40, 768))
    second = rng.normal(-1.0, 2.0, size=(13, 90, 768))
    together, n = pool_frames([first, second])
    one, n1 = pool_frames([first])
    two, n2 = pool_frames([second])
    combined = sm.combine_pooled(np.stack([one, two]), np.array([n1, n2]))
    assert n == 130
    assert np.allclose(combined, together, atol=1e-9)


def test_combined_pooling_of_one_file_changes_nothing():
    rng = np.random.default_rng(1)
    pooled, n = pool_frames([rng.normal(size=(13, 30, 768))])
    assert np.allclose(sm.combine_pooled(pooled[None], np.array([n])), pooled, atol=1e-9)


def test_fusion_uses_the_speakers_all_materials_share():
    a = pd.DataFrame([[0.5], [0.5], [0.9]], index=["x", "y", "z"], columns=[0])
    b = pd.DataFrame([[0.5], [0.9]], index=["y", "z"], columns=[0])
    fused = sm.fuse([a, b])
    assert fused.index.tolist() == ["y", "z"]
    assert fused.loc["y", 0] == pytest.approx(0.5)
    assert fused.loc["z", 0] == pytest.approx(0.9)


def test_fusion_averages_logits_not_probabilities():
    a = pd.DataFrame([[0.9]], index=["x"], columns=[0])
    b = pd.DataFrame([[0.1]], index=["x"], columns=[0])
    assert sm.fuse([a, b]).loc["x", 0] == pytest.approx(0.5)


def test_verdict_thresholds():
    assert sm.verdict(0.80, 0.20, 0.10) == "shrinks to /a/"
    assert sm.verdict(0.65, 0.04, 0.01) == "shrinks to /a/"
    assert sm.verdict(0.65, 0.20, -0.01) == "shrinks to /a/"
    assert sm.verdict(0.69, 0.10, 0.02) == "not only /a/"
    assert sm.verdict(0.70, 0.20, 0.10) == "middle"
    assert sm.verdict(0.65, 0.07, 0.01) == "middle"


def test_quiet_verdict():
    assert sm.quiet_verdict(0.80, 0.50) == "not only /a/"
    assert sm.quiet_verdict(0.80, 0.65) == "pauses carry the label inside HSA"
    assert sm.quiet_verdict(0.70, 0.55) == "place cue weaker here"


def test_overall_is_strict_on_both_sides():
    assert sm.overall({"a": "not only /a/", "b": "not only /a/"}) == "not only /a/"
    assert sm.overall({"a": "not only /a/", "b": "middle"}) == "middle"
    assert sm.overall({"a": "not only /a/", "b": "shrinks to /a/"}) == "shrinks to /a/"


def test_thresholds_are_the_written_ones():
    assert sm.SHRINK_UAR == 0.80 and sm.MIDDLE_UAR == 0.70
    assert sm.GAP_HOLDS == 0.10 and sm.GAP_GONE == 0.05


def test_logit_handles_the_ends():
    assert np.isfinite(sm.logit(np.array([0.0, 1.0]))).all()
