"""Synthetic checks of the fine-tuning run; no model weights and no recording are read."""

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

import finetune_hubert as ft  # noqa: E402


def test_schedule_matches_hf_linear_with_warmup():
    assert ft.linear_schedule(0, 300, 3) == 0.0
    assert ft.linear_schedule(1, 300, 3) == pytest.approx(1 / 3)
    assert ft.linear_schedule(3, 300, 3) == pytest.approx(1.0)
    assert ft.linear_schedule(150, 300, 3) == pytest.approx(150 / 297)
    assert ft.linear_schedule(300, 300, 3) == 0.0


def test_class_weights_are_balanced():
    weights = ft.class_weights(np.array([0] * 82 + [1] * 329))
    assert weights[0] == pytest.approx(411 / (2 * 82))
    assert weights[1] == pytest.approx(411 / (2 * 329))


def test_crop_takes_one_length_for_the_whole_batch():
    rng = np.random.default_rng(0)
    items = [[np.zeros((900, 4))], [np.zeros((120, 4)), np.zeros((30, 4))], [np.zeros((700, 4))]]
    batch = ft.crop_batch(items, rng, crop=500)
    assert batch.shape[0] == 3 and batch.shape[2] == 4
    assert batch.shape[1] in (30, 120)


def test_crop_stops_at_the_crop_length():
    rng = np.random.default_rng(1)
    batch = ft.crop_batch([[np.ones((900, 2))], [np.ones((600, 2))]], rng, crop=500)
    assert batch.shape == (2, 500, 2)


def _layers(size):
    import torch

    class Wrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = torch.nn.Linear(size, size)

        def forward(self, hidden):
            return (hidden + self.inner(hidden),)

    return torch.nn.ModuleList([Wrapped(), Wrapped()])


def test_training_learns_a_separable_toy_problem():
    rng = np.random.default_rng(2)
    size = 8
    pieces = {}
    labels = []
    for i in range(60):
        label = i % 2
        shift = 1.5 if label else -1.5
        pieces[f"x{i}"] = [rng.normal(shift, 1.0, size=(int(rng.integers(40, 80)), size))]
        labels.append(label)
    items = list(pieces)
    scores = ft.train_and_predict(
        _layers(size),
        lambda item: pieces[item],
        items[:40],
        np.array(labels[:40]),
        items[40:],
        lr=1e-2,
        seed=0,
        steps=60,
        batch_size=8,
        warmup=3,
        crop=30,
    )
    truth = np.array(labels[40:])
    assert scores.shape == (20,)
    assert np.mean((scores >= 0.5) == truth) >= 0.9


def test_training_is_repeatable_with_the_same_seed():
    rng = np.random.default_rng(3)
    pieces = {f"x{i}": [rng.normal(size=(50, 4))] for i in range(16)}
    items = list(pieces)
    labels = np.array([i % 2 for i in range(12)])

    def run():
        import torch

        torch.manual_seed(99)
        layers = _layers(4)
        return ft.train_and_predict(
            layers,
            lambda item: pieces[item],
            items[:12],
            labels,
            items[12:],
            lr=1e-3,
            seed=5,
            steps=10,
            batch_size=4,
            warmup=1,
            crop=20,
        )

    assert np.allclose(run(), run())


def test_reading_follows_the_written_rules():
    assert ft.reading(0.80, 0.80, 0.20, 0.10) == "shrinks"
    assert ft.reading(0.65, 0.66, 0.04, 0.01) == "shrinks"
    assert ft.reading(0.65, 0.66, 0.20, -0.01) == "shrinks"
    assert ft.reading(0.65, 0.69, 0.10, 0.02) == "objection answered for the last two layers"
    assert ft.reading(0.65, 0.71, 0.20, 0.10) == "middle"
    assert ft.reading(0.75, 0.75, 0.20, 0.10) == "middle"


def test_gate_uses_the_frozen_all_score_minus_five_points():
    assert ft.gate(0.785, 0.835, 0.50) == {"all_passes": True, "shuffled_passes": True}
    assert ft.gate(0.784, 0.835, 0.50)["all_passes"] is False
    assert ft.gate(0.80, 0.835, 0.54)["shuffled_passes"] is False


def test_leans_more_on_place():
    assert ft.leans_more_on_place(0.86, 0.80, 0.63, 0.62)
    assert not ft.leans_more_on_place(0.86, 0.80, 0.70, 0.62)
    assert not ft.leans_more_on_place(0.82, 0.80, 0.63, 0.62)


def test_scored_arm_averages_the_repeats():
    y = np.array(["normal", "pathological", "normal", "pathological"])
    scores = np.array([[0.1, 0.9], [0.9, 0.9], [0.2, 0.1], [0.8, 0.2]])
    arm = ft.scored_arm("x", y, np.array(["a", "b", "c", "d"]), scores)
    assert arm.uar == pytest.approx((1.0 + 0.5) / 2)
    assert arm.predictions.shape == (4, 2)


def test_fit_seeds_differ_between_arms_repeats_and_folds():
    plan = {"statistics": {"cv": {"folds": 5, "repeats": 10}, "seeds": {"cv": 1}}}
    table = pd.DataFrame(
        {
            "file_id": [f"avfad__{i:03d}" for i in range(40)],
            "speaker": [f"{i:03d}" for i in range(40)],
            "label": ["normal", "pathological"] * 20,
        }
    )
    y = table["label"].to_numpy()
    jobs = ft.jobs_for("ii", table, y, [0, 1], 1e-5, plan, 0)
    jobs += ft.jobs_for("all", table, y, [0], 1e-5, plan, 1)
    seeds = [job["seed"] for job in jobs]
    assert len(seeds) == len(set(seeds)) == 15
    for job in jobs:
        assert not set(job["train_items"]) & set(job["test_items"])


def test_thresholds_are_the_written_ones():
    assert ft.SHRINK_UAR == 0.80 and ft.ANSWERED_UAR == 0.70
    assert ft.GAP_HOLDS == 0.10 and ft.GAP_GONE == 0.05 and ft.GATE_MARGIN == 0.05
    assert ft.SHUFFLED_RANGE == (0.47, 0.53)
    assert ft.RATES == (1e-5, 1e-4) and ft.TRAINED_LAYERS == 2 and ft.STEPS == 300


def test_each_extra_shuffle_is_its_own_permutation_of_the_same_labels():
    plan = {"statistics": {"seeds": {"negative_control": 20261108}}}
    table = pd.DataFrame(
        {
            "speaker": [f"{i:03d}" for i in range(60)],
            "label": ["pathological"] * 45 + ["normal"] * 15,
        }
    )
    true = ft.labels_of(table, "ii", plan)
    shuffles = [ft.labels_of(table, "shuffled", plan)]
    shuffles += [ft.labels_of(table, f"shuffled_{k}", plan) for k in ft.EXTRA_SHUFFLES]
    for y in shuffles:
        assert sorted(y) == sorted(true)
    assert len({tuple(y) for y in shuffles}) == 5
    assert ft.labels_of(table, "shuffled_2", plan).tolist() == shuffles[2].tolist()


def test_arm_numbers_are_unique():
    assert len(set(ft.ARM_NUMBERS.values())) == len(ft.ARM_NUMBERS)
    assert [ft.ARM_NUMBERS[f"shuffled_{k}"] for k in ft.EXTRA_SHUFFLES] == [13, 14, 15, 16]


def test_post_hoc_gate_reads_the_mean_of_five():
    assert ft.shuffled_mean_passes([0.543, 0.49, 0.50, 0.48, 0.51])
    assert not ft.shuffled_mean_passes([0.543, 0.55, 0.54, 0.53, 0.52])
