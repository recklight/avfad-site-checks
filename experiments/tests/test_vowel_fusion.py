"""Synthetic checks of the three-vowel C1; no feature file or recording is read."""

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

import vowel_fusion as vf  # noqa: E402


def _arm(name, scores, labels, groups=None):
    scores = np.asarray(scores, dtype=float)
    groups = np.array([f"S{i:03d}" for i in range(scores.shape[0])] if groups is None else groups)
    return vf.make_arm(name, groups, np.asarray(labels), scores)


def test_file_ids_match_the_feature_files():
    assert vf.file_id("AAC", "a") == "avfad__AAC"
    assert vf.file_id("AAC", "i") == "avfad__AAC-i"
    assert vf.file_id("AAC", "u") == "avfad__AAC-u"


def test_a_shuffle_gives_each_speaker_one_label_whatever_the_row_order():
    members = pd.DataFrame(
        {"speaker": [f"S{i}" for i in range(20)], "label": ["pathological"] * 15 + ["normal"] * 5}
    )
    first = vf.shuffled_labels(members, 7).set_index("speaker")["label"]
    second = vf.shuffled_labels(members.sample(frac=1, random_state=3), 7)
    second = second.set_index("speaker")["label"]
    assert first.equals(second.loc[first.index])
    assert (first == "normal").sum() == 5
    assert not first.equals(vf.shuffled_labels(members, 8).set_index("speaker")["label"])


def test_the_average_arm_cuts_the_mean_score_at_one_half():
    labels = ["pathological", "pathological", "normal", "normal"]
    a = _arm("ii", [[0.9], [0.4], [0.2], [0.6]], labels)
    i = _arm("ii", [[0.7], [0.4], [0.2], [0.2]], labels)
    u = _arm("ii", [[0.2], [0.9], [0.5], [0.4]], labels)
    fused = vf.average_arm("ii", [a, i, u])
    assert fused.scores[:, 0] == pytest.approx([0.6, 0.5666667, 0.3, 0.4])
    assert fused.predictions[:, 0].tolist() == ["pathological", "pathological", "normal", "normal"]
    assert fused.uar == pytest.approx(1.0)
    assert a.uar == pytest.approx(0.5)


def test_the_average_refuses_different_speakers():
    labels = ["pathological", "normal"]
    with pytest.raises(ValueError, match="differ"):
        vf.average_arm(
            "ii",
            [
                _arm("ii", [[0.9], [0.1]], labels, ["A", "B"]),
                _arm("ii", [[0.9], [0.1]], labels, ["A", "C"]),
            ],
        )


def test_a_paired_difference_of_an_arm_with_itself_is_zero():
    rng = np.random.default_rng(0)
    labels = np.where(rng.random(60) < 0.5, "pathological", "normal")
    arm = _arm("ii", rng.random((60, 3)), labels)
    for metric in ("uar", "roc_auc"):
        out = vf.paired_difference(arm, arm, metric, seed=1)
        assert out["difference"] == 0.0
        assert out["ci_low"] == 0.0 and out["ci_high"] == 0.0


def test_the_same_speakers_and_labels_get_the_same_folds_in_every_vowel():
    from harness import load_plan, make_config

    config = make_config(load_plan(), "pathological")
    rng = np.random.default_rng(2)
    labels = np.where(np.arange(80) % 4 == 0, "normal", "pathological")
    arm = _arm("ii", rng.random((80, 1)), labels)
    assert vf.same_folds([("a", arm), ("i", arm), ("u", arm)], config)


@pytest.mark.parametrize(
    ("uar_ii", "difference", "low", "expected"),
    [
        (0.62, 0.18, 0.13, "not only /a/"),
        (0.72, 0.18, 0.13, "does not meet the C1 materials thresholds"),
        (0.62, 0.08, 0.02, "does not meet the C1 materials thresholds"),
        (0.62, 0.12, -0.01, "does not meet the C1 materials thresholds"),
    ],
)
def test_material_reading(uar_ii, difference, low, expected):
    gap = {"difference": difference, "ci_low": low}
    assert vf.material_reading(uar_ii, gap) == expected


@pytest.mark.parametrize(
    ("difference", "low", "high", "expected"),
    [
        (0.06, 0.01, 0.11, "more material helps"),
        (0.03, -0.01, 0.07, "more material does not help"),
        (0.03, 0.005, 0.06, "small"),
        (-0.04, -0.08, -0.01, "small"),
    ],
)
def test_helps_reading(difference, low, high, expected):
    assert vf.helps_reading({"difference": difference, "ci_low": low, "ci_high": high}) == expected


@pytest.mark.parametrize(
    ("difference", "low", "expected"),
    [
        (0.15, 0.10, "the gap stays"),
        (0.06, 0.01, "the gap shrinks"),
        (0.02, -0.02, "the gap is gone"),
    ],
)
def test_gap_reading(difference, low, expected):
    assert vf.gap_reading({"difference": difference, "ci_low": low}) == expected


def test_shuffle_band_reads_the_mean_of_five():
    assert vf.shuffle_passes([0.44, 0.52, 0.50, 0.51, 0.49])
    assert not vf.shuffle_passes([0.55, 0.56, 0.52, 0.54, 0.53])


def test_thresholds_are_the_written_ones():
    assert vf.NOT_ONLY_A_II == 0.70 and vf.GAP_POINTS == 0.10 and vf.HELPS_POINTS == 0.05
    assert vf.SHUFFLE_BAND == (0.47, 0.53)
    assert vf.PAIRED_SEED == 20261117 and vf.N_RESAMPLES == 10_000 and vf.N_SHUFFLES == 5
