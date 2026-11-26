"""Synthetic checks of the published-split analysis; no list or recording is read."""

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

import published_splits as ps  # noqa: E402


def test_list_parsing_takes_the_stem_and_maps_the_labels():
    text = (
        '{"labels": {"HEALTH": 0, "PATH": 1}, "meta_data": ['
        '{"path": "audio_16khz/AAC/AAC002.wav", "label": "PATH", "speaker": "AAC"}, '
        '{"path": "audio_16khz/ABR/ABR004.WAV", "label": "HEALTH", "speaker": "ABR"}]}'
    )
    rows = ps.parse_list(text)
    assert rows[0] == {"stem": "AAC002", "speaker": "AAC", "label": "pathological"}
    assert rows[1] == {"stem": "ABR004", "speaker": "ABR", "label": "normal"}


def test_prediction_parsing_keeps_truth_and_prediction_apart():
    rows = ps.parse_predictions("AAC002 PATH PATH\nAFB002 HEALTH PATH\n\n")
    assert rows == [
        {"stem": "AAC002", "label": "pathological", "predicted": "pathological"},
        {"stem": "AFB002", "label": "normal", "predicted": "pathological"},
    ]


def _table(rows):
    return pd.DataFrame(rows, columns=["speaker", "label", "place"])


def test_a_speaker_with_many_files_counts_once():
    table = _table(
        [
            ("A", "normal", "ESSUA"),
            ("A", "normal", "ESSUA"),
            ("B", "normal", "HSA"),
            ("C", "pathological", "HSA"),
            ("D", "pathological", "HSJ"),
        ]
    )
    record = ps.composition_of(table)
    assert record["files"] == 5
    assert record["speakers"] == 4
    assert record["normal"] == {"ESSUA": 1, "HSA": 1, "other": 0}
    assert record["patients"] == {"ESSUA": 0, "HSA": 1, "other": 1}
    assert record["essua_share_of_normals"] == pytest.approx(0.5)


def test_overlap_is_counted_per_fold():
    lists = pd.DataFrame(
        [
            (1, "train", "A", "normal", "ESSUA"),
            (1, "test", "A", "normal", "ESSUA"),
            (1, "test", "B", "normal", "HSA"),
            (2, "train", "B", "normal", "HSA"),
            (2, "test", "A", "normal", "ESSUA"),
        ],
        columns=["fold", "side", "speaker", "label", "place"],
    )
    folds = ps.fold_table_composition(lists)
    assert [f["speakers_on_both_sides"] for f in folds] == [1, 0]


def _fold(train_essua, train_hsa, test_essua, test_hsa):
    return {
        "train": {"normal": {"ESSUA": train_essua, "HSA": train_hsa, "other": 0}},
        "test": {"normal": {"ESSUA": test_essua, "HSA": test_hsa, "other": 0}},
    }


def test_mixed_needs_both_places_on_every_side_of_every_fold():
    assert ps.is_mixed([_fold(10, 2, 3, 1), _fold(10, 2, 3, 1)])
    assert not ps.is_mixed([_fold(10, 2, 3, 1), _fold(10, 2, 3, 0)])
    assert not ps.is_mixed([_fold(0, 2, 3, 1)])


def test_expected_share_brackets_the_corpus_share():
    low, high = ps.expected_share(73, 363, 274)
    assert low < 274 / 363 < high
    assert 0.6 < low and high < 0.9


def test_place_split_reads_each_place_on_its_own():
    rows = []
    # ESSUA normals all right; HSA normals all called patients; HSA patients all right.
    for i in range(40):
        rows.append((f"e{i}", "normal", "normal", "ESSUA"))
    for i in range(20):
        rows.append((f"h{i}", "normal", "pathological", "HSA"))
    for i in range(40):
        rows.append((f"p{i}", "pathological", "pathological", "HSA"))
    table = pd.DataFrame(rows, columns=["speaker", "label", "predicted", "place"])
    record = ps.split_by_place(table, seed=0, n=200)
    assert record["uar_hsa"] == pytest.approx(0.5)
    assert record["uar_all"] == pytest.approx((40 / 60 + 1.0) / 2)
    assert record["drop"] == pytest.approx(record["uar_all"] - 0.5)
    assert record["uar_essua_normals_vs_hsa_patients"] == pytest.approx(1.0)
    assert record["specificity_essua"]["rate"] == 1.0
    assert record["specificity_hsa"]["rate"] == 0.0
    assert record["sensitivity_hsa"]["rate"] == 1.0
    assert record["drop_ci"][0] > 0


def test_files_of_a_speaker_are_resampled_together():
    rows = [("a", "normal", "normal", "HSA")] * 6 + [("b", "pathological", "normal", "HSA")] * 6
    rows += [("c", "normal", "normal", "HSA")] * 6 + [
        ("d", "pathological", "pathological", "HSA")
    ] * 6
    table = pd.DataFrame(rows, columns=["speaker", "label", "predicted", "place"])
    everyone = sorted(table["speaker"].unique())
    position = {s: i for i, s in enumerate(everyone)}
    weights = np.array([[2.0, 1.0, 0.0, 1.0]])
    assert ps.weighted_uar(table, weights, position)[0] == pytest.approx(0.75)


def test_fixing_speakers_puts_all_sentences_in_one_fold():
    table = pd.DataFrame(
        {
            "speaker": ["A"] * 3 + ["B"] * 2,
            "task": ["005", "004", "006", "004", "005"],
            "fold": [2, 1, 3, 4, 5],
        }
    )
    fixed = ps.fix_speakers(table)
    assert fixed["fold"].tolist() == [1, 1, 1, 4, 4]


def test_listed_folds_give_the_same_predictions_as_the_library_on_the_same_folds():
    from voice_severity_bench.config import Config
    from voice_severity_bench.dataset import feature_set_from_table
    from voice_severity_bench.splitting import repeated_split_indices

    rng = np.random.default_rng(3)
    n = 60
    labels = np.array(["normal", "pathological"] * (n // 2))
    columns = ["f0", "f1"]
    values = rng.normal(size=(n, 2)) + (labels == "pathological")[:, None] * 1.5
    table = pd.DataFrame(values, columns=columns)
    table["label"] = labels
    table["speaker"] = [f"s{i:02d}" for i in range(n)]
    table["path"] = table["speaker"]
    table["group"] = table["speaker"]
    config = Config().override(
        {
            "seed": 1,
            "data.task": "classification",
            "data.target": "label",
            "model.name": "logistic",
            "model.tune": True,
            "model.class_weight": "balanced",
            "evaluation.n_splits": 3,
            "evaluation.n_inner_splits": 3,
            "evaluation.n_repeats": 1,
            "evaluation.positive_class": "pathological",
        }
    )
    features = feature_set_from_table(
        table[["path", "group", "label", *columns]],
        target="label",
        feature_columns=columns,
        task="classification",
    )
    y = np.asarray(features.y).astype(str)
    groups = np.asarray(features.groups).astype(str)
    folds = repeated_split_indices(
        y, groups, task="classification", n_splits=3, n_repeats=1, seed=1
    )[0]
    table["fold"] = 0
    for k, (_, test) in enumerate(folds, start=1):
        table.loc[test, "fold"] = k
    from voice_severity_bench.comparison import compare_feature_sets

    expected = compare_feature_sets({"x": features}, config, n_jobs=1).prediction_matrix("x")[:, 0]
    arm = ps.arm_on_folds("x", table, columns, None, config)
    assert arm.predictions[:, 0].tolist() == expected.tolist()


def test_subset_arm_keeps_the_predictions_and_refits_nothing():
    from harness import Arm

    arm = Arm(
        name="x",
        y_true=np.array(["normal", "pathological", "normal", "pathological"]),
        groups=np.array(["a", "b", "c", "d"]),
        predictions=np.array([["normal"], ["pathological"], ["pathological"], ["pathological"]]),
        scores=np.array([[0.1], [0.9], [0.8], [0.7]]),
        uar=0.75,
        roc_auc=1.0,
        sensitivity=1.0,
        specificity=0.5,
        worst_share_deviation=0.0,
        chosen_c={},
    )
    part = ps.subset_arm("y", arm, np.array([False, False, True, True]))
    assert part.uar == pytest.approx(0.5)
    assert part.groups.tolist() == ["c", "d"]


@pytest.mark.parametrize(
    ("drop", "low", "expected"),
    [
        (0.10, 0.01, "falls past the threshold inside one place"),
        (0.04, 0.01, "no clear reliance on place"),
        (0.20, -0.01, "no clear reliance on place"),
        (0.07, 0.01, "in between"),
    ],
)
def test_published_verdict(drop, low, expected):
    assert ps.published_verdict(drop, low) == expected


def test_my_verdict_uses_c1s_own_threshold():
    assert ps.my_verdict(0.69, 0.10) == "same as C1"
    assert ps.my_verdict(0.70, 0.20) == "differs from C1"
    assert ps.my_verdict(0.55, 0.09) == "differs from C1"


def test_leak_verdict():
    assert ps.leak_verdict(0.80, 0.75) == "overlap raises the score"
    assert ps.leak_verdict(0.80, 0.76) == "overlap matters little"


def test_thresholds_are_the_written_ones():
    assert ps.DROP_HOLDS == 0.10 and ps.DROP_GONE == 0.05
    assert ps.HSA_SEPARATES == 0.80 and ps.II_BELOW == 0.70 and ps.LEAK_MATTERS == 0.05
    assert ps.PAPER_UAR == {"a_n": 86.70, "phrase": 87.03}
