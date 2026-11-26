"""Synthetic checks of the Ribas-method run; no feature file or recording is read."""

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

import extract_compare  # noqa: E402
import ribas_method as rm  # noqa: E402


def test_their_zscore_uses_the_training_statistics_only():
    rng = np.random.default_rng(0)
    x_train = rng.normal(5.0, 2.0, size=(40, 3))
    x_test = rng.normal(5.0, 2.0, size=(10, 3))
    y = np.array([0, 1] * 20)
    predicted, scores = rm.fit_predict(x_train, y, x_test)
    assert predicted.shape == (10,) and set(predicted) <= {0, 1}
    assert np.all((scores > 0) & (scores < 1))


def test_a_constant_column_does_not_break_the_zscore():
    x = np.column_stack([np.ones(20), np.r_[np.zeros(10), np.ones(10)]])
    y = np.r_[np.zeros(10, dtype=int), np.ones(10, dtype=int)]
    predicted, _ = rm.fit_predict(x, y, x)
    assert np.mean(predicted == y) == 1.0


def test_score_and_prediction_agree_at_one_half():
    rng = np.random.default_rng(1)
    x = np.r_[rng.normal(-1, 1, size=(30, 4)), rng.normal(1, 1, size=(30, 4))]
    y = np.r_[np.zeros(30, dtype=int), np.ones(30, dtype=int)]
    predicted, scores = rm.fit_predict(x, y, x)
    assert np.array_equal(predicted == 1, scores > 0.5)


def test_folds_test_every_item_once():
    rng = np.random.default_rng(2)
    table = pd.DataFrame(
        {
            "stem": [f"s{i:02d}" for i in range(30)],
            "speaker": [f"p{i:02d}" for i in range(30)],
            "label": ["normal", "pathological"] * 15,
            "fold": np.repeat([1, 2, 3], 10),
        }
    )
    features = pd.DataFrame(rng.normal(size=(30, 5)), index=table["stem"])
    arm = rm.on_folds("x", table, features)
    assert arm.predictions.shape == (30, 1)
    assert 0.0 <= arm.uar <= 1.0


def test_arm_from_reads_one_as_pathological():
    table = pd.DataFrame({"label": ["normal", "pathological"], "speaker": ["a", "b"]})
    arm = rm.arm_from("x", table, np.array([[0], [1]]), np.array([[0.2], [0.8]]))
    assert arm.uar == pytest.approx(1.0)
    assert arm.predictions[:, 0].tolist() == ["normal", "pathological"]


def test_draws_bring_only_normal_speakers():
    items = pd.DataFrame(
        {
            "stem": ["A004", "A005", "B004", "C004", "D004"],
            "speaker": ["A", "A", "B", "C", "D"],
            "place": ["HSA", "HSA", "HSA", "ESSUA", "HSA"],
            "label": ["pathological", "pathological", "normal", "normal", "pathological"],
        }
    )
    tables = rm.c1_tables(items)
    assert set(tables["ii"]["speaker"]) == {"A", "B", "D"}
    for name, table in tables.items():
        if name.startswith("i_"):
            normal = table[table["label"] == "normal"]
            assert set(normal["speaker"]) <= {"B", "C"}
            assert {"A", "D"} <= set(table["speaker"])


@pytest.mark.parametrize(
    ("a", "sentences", "expected"),
    [
        (0.80, 0.60, "their method separates inside one place"),
        (0.60, 0.81, "their method separates inside one place"),
        (0.69, 0.69, "their method does not separate inside one place either"),
        (0.72, 0.65, "middle"),
    ],
)
def test_primary_reading(a, sentences, expected):
    assert rm.primary_reading(a, sentences) == expected


def test_c1_reading():
    assert rm.c1_reading(0.69, 0.10) == "same as C1"
    assert rm.c1_reading(0.70, 0.20) == "differs from C1"


def test_thresholds_are_the_written_ones():
    assert rm.SEPARATES == 0.80 and rm.FAILS == 0.70 and rm.DROP_HOLDS == 0.10
    assert rm.REPRODUCE_POINTS == 2.0 and rm.REPRODUCE_AGREEMENT == 0.90


def test_extractor_job_list_is_read_back(tmp_path):
    path = tmp_path / "jobs.csv"
    pd.DataFrame({"stem": ["AAC002"], "path": ["data/x/AAC002.wav"]}).to_csv(path, index=False)
    assert extract_compare.read_jobs(path) == [("AAC002", "data/x/AAC002.wav")]
