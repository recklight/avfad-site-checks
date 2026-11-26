"""Synthetic checks of the sensitivity and exploratory code; no recording is read."""

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

import exploratory_avfad as ex  # noqa: E402
from arms import (  # noqa: E402  # noqa: E402
    HUBERT_C_GRID,
    _use_hubert_grid,
    evaluate_columns,
    evaluate_layers,
    layer_columns,
    merge_features,
)
from extract_hubert import pieces_of, pool_frames  # noqa: E402
from harness import load_plan, make_config  # noqa: E402

PLAN = load_plan()


def quick_config(positive: str = "pathological", repeats: int = 2):
    return make_config(PLAN, positive).override({"evaluation.n_repeats": repeats})


def _labels(n_normal: int, n_sick: int) -> tuple[np.ndarray, np.ndarray]:
    y = np.array(["normal"] * n_normal + ["pathological"] * n_sick)
    return y, np.array([f"s{i:03d}" for i in range(y.size)])


def test_subsample_keeps_every_patient_and_the_asked_normals():
    y, _ = _labels(50, 50)
    train = np.arange(100)
    chosen = ex.subsample_training(train, y, 16, np.random.default_rng(0))
    assert np.sum(y[chosen] == "normal") == 16 and np.sum(y[chosen] == "pathological") == 50
    assert np.array_equal(chosen, np.sort(chosen)) and np.unique(chosen).size == chosen.size


def test_subsample_with_none_or_too_many_keeps_the_fold():
    y, _ = _labels(10, 10)
    train = np.arange(20)
    assert np.array_equal(ex.subsample_training(train, y, None, np.random.default_rng(0)), train)
    assert np.array_equal(ex.subsample_training(train, y, 99, np.random.default_rng(0)), train)


def test_subsample_never_touches_what_is_not_in_the_training_fold():
    y, _ = _labels(20, 20)
    train = np.arange(0, 40, 2)
    chosen = ex.subsample_training(train, y, 4, np.random.default_rng(1))
    assert set(chosen) <= set(train)


def test_curve_recovers_a_known_limit():
    ns = np.array([16.0, 24.0, 32.0, 48.0, 66.0])
    fitted = ex.fit_curve(ns, 0.9 - 0.5 / np.sqrt(ns), 290)
    assert fitted["inverse_root_limit"] == pytest.approx(0.9)
    assert fitted["inverse_root_at_target"] == pytest.approx(0.9 - 0.5 / np.sqrt(290))


def test_flat_curve_stays_flat_in_both_forms():
    ns = np.array([16.0, 24.0, 32.0, 48.0, 66.0])
    fitted = ex.fit_curve(ns, np.full(5, 0.55), 290)
    assert fitted["inverse_root_at_target"] == pytest.approx(0.55)
    assert fitted["log_at_target"] == pytest.approx(0.55)


def test_curve_repeat_uses_the_asked_number_and_scores_every_speaker():
    y, groups = _labels(40, 60)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 6))
    X[:, 0] += 3.0 * (y == "pathological")
    config = quick_config()
    folds = [
        (np.setdiff1d(np.arange(100), np.arange(k, 100, 5)), np.arange(k, 100, 5)) for k in range(5)
    ]
    row = ex._curve_repeat("c", X, y, groups, folds, 16, 0, config, None)
    assert row["normals_in_training"] == 16.0 and row["requested"] == 16
    assert row["uar"] > 0.8
    full = ex._curve_repeat("c", X, y, groups, folds, None, 0, config, None)
    assert full["requested"] == "all" and full["normals_in_training"] == 32.0


def test_bootstrap_mean_brackets_the_mean():
    values = np.random.default_rng(0).normal(0.3, 0.1, 200)
    low, high = ex._bootstrap_mean(values, 0, 2000)
    assert low < values.mean() < high and high - low < 0.05


def test_one_room_recording_per_group_of_identical_ones():
    rooms = pd.DataFrame(
        {
            "path": ["a", "b", "c", "d", "e"],
            "identical_group": [np.nan, "g1", "g1", "g2", np.nan],
        }
    )
    assert ex._one_per_group(rooms)["path"].tolist() == ["a", "b", "d", "e"]


def test_noise_loops_to_the_length_and_starts_where_asked():
    pieces = [np.arange(1.0, 101.0)]
    out = ex.loop_to_length(pieces, 250, 10, 16_000)
    assert out.size == 250 and out[0] == 11.0 and out[90] == 1.0


def test_noise_crossfade_has_no_step_between_spans():
    fade_rate = 16_000
    a, b = np.ones(1000), -np.ones(1000)
    out = ex.loop_to_length([a, b], 1840, 0, fade_rate)
    assert np.max(np.abs(np.diff(out))) < 0.02


def test_hubert_grid_goes_two_decades_below_the_frozen_one():
    from voice_severity_bench import models

    saved = models.SEARCH_SPACES[("classification", "logistic")]
    try:
        _use_hubert_grid()
        assert models.SEARCH_SPACES[("classification", "logistic")]["model__C"] == HUBERT_C_GRID
        assert min(HUBERT_C_GRID) == 0.01 / 100
    finally:
        models.SEARCH_SPACES[("classification", "logistic")] = saved


def test_pieces_cut_long_spans_and_drop_what_is_too_short():
    pieces = pieces_of([np.zeros(1000), np.zeros(399)], limit=400)
    assert [p.size for p in pieces] == [400, 400]


def test_pool_uses_all_frames_of_all_spans():
    first = np.zeros((13, 10, 768))
    second = np.ones((13, 30, 768))
    pooled, n = pool_frames([first, second])
    assert n == 40 and pooled.shape == (13, 1536)
    assert pooled[0, 0] == pytest.approx(0.75)
    assert pooled[0, 768] == pytest.approx(np.std(np.r_[np.zeros(10), np.ones(30)]))


def _small_table(n: int = 50, strength: float = 3.0):
    y, groups = _labels(n, n)
    rng = np.random.default_rng(0)
    meta = pd.DataFrame({"file_id": groups, "speaker": groups, "label": y})
    columns = layer_columns(0) + layer_columns(3)
    values = rng.normal(size=(2 * n, len(columns)))
    # The signal sits only in layer 3, so choosing the layer has to find it.
    values[:, 1536:1546] += strength * (y == "pathological")[:, None]
    features = pd.DataFrame(values, columns=columns)
    features.insert(0, "file_id", groups)
    return merge_features(meta, features, "label")


def test_layer_selection_finds_the_layer_that_holds_the_signal():
    table = _small_table()
    arm = evaluate_layers("t", table, (0, 3), quick_config(repeats=1))
    assert arm.uar > 0.8
    assert max(arm.chosen_c["layer"], key=arm.chosen_c["layer"].get) == "l03"


def test_single_layer_runs_without_block_selection():
    table = _small_table()
    arm = evaluate_layers("t", table, (3,), quick_config(repeats=1))
    assert arm.uar > 0.8 and arm.chosen_c["layer"] == {"None": 5}


def test_evaluate_columns_matches_its_named_arm():
    table = _small_table()
    arm = evaluate_columns("t", table, layer_columns(3)[:20], quick_config(repeats=1))
    assert arm.predictions.shape == (100, 1) and arm.uar > 0.8


def test_merge_refuses_a_speaker_without_features():
    meta = pd.DataFrame({"file_id": ["a", "b"], "speaker": ["a", "b"], "label": ["x", "y"]})
    features = pd.DataFrame({"file_id": ["a"], "f": [1.0]})
    with pytest.raises(ValueError, match="no features"):
        merge_features(meta, features, "label")
