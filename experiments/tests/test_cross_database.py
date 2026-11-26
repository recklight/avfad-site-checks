"""Synthetic checks of the cross-database run; no feature file or recording is read."""

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

import cross_database as cd  # noqa: E402


def _frame(rng, n=200, shift=1.0):
    sick = rng.random(n) < 0.5
    frame = pd.DataFrame(
        {
            "file_id": [f"svd__{i}-a_n" for i in range(n)],
            "speaker": [f"svd{i}" for i in range(n)],
            "label": np.where(sick, "pathological", "normal"),
            "age": rng.integers(20, 69, n),
            "sex": rng.choice(["m", "w"], n),
            "year": rng.choice([1998, 2003], n),
        }
    )
    frame["ii"] = rng.normal(size=n)
    frame["all"] = rng.normal(size=n)
    frame["essua_normals_hsa_patients"] = rng.normal(size=n)
    for d in range(3):
        frame[f"i_{d:02d}"] = rng.normal(size=n) + shift * sick
    return frame


def test_evaluate_finds_a_difference_that_is_there():
    record = cd.evaluate(_frame(np.random.default_rng(0)), n=200, seed=1)
    assert record["i"]["auc"] > 0.7
    assert abs(record["ii"]["auc"] - 0.5) < 0.12
    assert record["d_ext"]["auc_ci"][0] > 0
    assert record["i"]["draws"] == 3


def test_evaluate_without_a_difference_keeps_zero_inside():
    record = cd.evaluate(_frame(np.random.default_rng(2), shift=0.0), n=200, seed=1)
    low, high = record["d_ext"]["auc_ci"]
    assert low < 0 < high


def test_resamples_keep_the_size_of_every_stratum_and_label():
    sick = np.array([True, True, False, False, True, False])
    cells = np.array(["a", "a", "a", "b", "b", "b"])
    for chosen in cd.resample_indices(sick, cells, 5, 0):
        assert chosen.size == 6
        assert sick[chosen].sum() == 3


def test_uar_at_half():
    scores = np.array([0.9, 0.2, 0.4, 0.6])
    sick = np.array([True, True, False, False])
    assert cd.uar_at_half(scores, sick) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("d_ext", "low", "expected"),
    [
        (0.04, 0.01, "does not travel: consistent with a place shortcut"),
        (0.12, -0.01, "does not travel: consistent with a place shortcut"),
        (0.10, 0.02, "part of it travels"),
        (0.07, 0.02, "middle"),
    ],
)
def test_reading(d_ext, low, expected):
    assert cd.reading(d_ext, low) == expected


def test_scoring_trains_on_avfad_rows_and_scores_only_svd_rows():
    from harness import load_plan, make_config

    rng = np.random.default_rng(4)
    columns = ["f0", "f1"]

    def table(n, prefix):
        sick = np.arange(n) % 2 == 1
        values = rng.normal(size=(n, 2)) + 2.0 * sick[:, None]
        frame = pd.DataFrame(values, columns=columns)
        frame["file_id"] = [f"{prefix}{i}" for i in range(n)]
        frame["speaker"] = frame["file_id"]
        frame["label"] = np.where(sick, "pathological", "normal")
        return frame

    config = make_config(load_plan(), "pathological")
    scores = cd.score_on_svd(table(60, "a"), table(40, "s"), columns, None, config)
    assert scores.shape == (40,)
    assert np.all(np.isfinite(scores))
    sick = np.arange(40) % 2 == 1
    assert cd.auc(scores, sick) > 0.9


def test_thresholds_are_the_written_ones():
    assert cd.TRAVELS == 0.10 and cd.STAYS == 0.05
    assert cd.N_RESAMPLES == 1000 and cd.SEED == 20261111


def _avfad_frame(rng, separation=1.0):
    rows = []
    for place, label, n in (
        ("HSA", "pathological", 120),
        ("HSA", "normal", 40),
        ("ESSUA", "normal", 100),
        ("HSJ", "pathological", 5),
    ):
        for i in range(n):
            rows.append(
                {
                    "speaker": f"{place}{label}{i}",
                    "place": place,
                    "label": label,
                    "age": int(rng.integers(20, 70)),
                    "sex": rng.choice(["F", "M"]),
                    "score": rng.normal() + separation * (label == "pathological"),
                }
            )
    return pd.DataFrame(rows)


def test_reverse_measures_find_separation_inside_hsa():
    record = cd.evaluate_reverse(_avfad_frame(np.random.default_rng(5)), n=200, seed=1)
    assert record["hsa_patients_vs_hsa_normals"]["auc"] > 0.7
    assert abs(record["hsa_normals_vs_essua_normals"]["auc"] - 0.5) < 0.12
    assert record["counts"] == {
        "hsa_patients": 120,
        "hsa_normals": 40,
        "essua_normals": 100,
        "others": 5,
    }


def test_reverse_reading():
    record = {
        "hsa_patients_vs_hsa_normals": {"auc": 0.76},
        "hsa_normals_vs_essua_normals": {"auc": 0.62, "ci": [0.55, 0.69]},
    }
    assert cd.reverse_reading(record) == {
        "inside_hsa": "an outside model separates inside HSA",
        "hsa_normals": "HSA normals look sicker to an outside model",
    }
    record["hsa_patients_vs_hsa_normals"]["auc"] = 0.60
    record["hsa_normals_vs_essua_normals"] = {"auc": 0.52, "ci": [0.45, 0.59]}
    assert cd.reverse_reading(record) == {
        "inside_hsa": "hard inside HSA for an outside model too",
        "hsa_normals": "no sign that HSA normals look sicker",
    }
