"""Smoke tests of the experiment code on synthetic tables; no recording is read.

The tables keep the real shape of the first batch (speakers, places, age by sex cells, 78
feature columns), with a signal planted where the test expects one.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import freeze
import harness
import run_first_batch as batch
from voice_severity_bench.models import search_space
from harness import (
    FEATURE_COLUMNS,
    SplitCheckError,
    age_sex_cell,
    analysis_table,
    check_folds,
    check_splits,
    evaluate_arm,
    load_plan,
    make_config,
    match_cells,
)

PLAN = load_plan()


def quick_plan(n_repeats: int = 2) -> dict:
    plan = json.loads(json.dumps(PLAN, default=str))
    plan["statistics"]["cv"]["repeats"] = n_repeats
    plan["controls"]["negative"]["shuffles_per_arm"] = 4
    return plan


def speakers_table(counts: dict[tuple[str, str], int], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (place, label), n in counts.items():
        for _ in range(n):
            code = f"S{len(rows):04d}"
            rows.append(
                {
                    "file_id": f"avfad__{code}",
                    "speaker": code,
                    "place": place,
                    "label": label,
                    "age": int(rng.integers(30, 70)),
                    "sex": str(rng.choice(["F", "M"], p=[0.7, 0.3])),
                    "laryngeal_surgery": 0,
                }
            )
    return pd.DataFrame(rows)


def features_for(speakers: pd.DataFrame, signal, strength: float, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(len(speakers), len(FEATURE_COLUMNS)))
    values[:, :4] += strength * np.asarray(signal(speakers), dtype=float)[:, None]
    table = pd.DataFrame(values, columns=FEATURE_COLUMNS)
    table.insert(0, "file_id", speakers["file_id"].to_numpy())
    return table


def is_hsa(speakers: pd.DataFrame) -> np.ndarray:
    return (speakers["place"] == "HSA").to_numpy()


def is_sick(speakers: pd.DataFrame) -> np.ndarray:
    return (speakers["label"] == "pathological").to_numpy()


def test_config_follows_the_frozen_plan():
    config = make_config(PLAN, "HSA")
    assert config.evaluation.n_splits == 5 and config.evaluation.n_repeats == 10
    assert config.evaluation.n_inner_splits == 3
    assert config.model.class_weight == "balanced" and config.model.tune
    assert config.seed == 20261104
    assert search_space(config)["model__C"] == PLAN["detector"]["classifier"]["c_grid"]


def test_permutation_count_is_the_planned_one():
    assert str(batch.C2_PERMUTATIONS) in PLAN["confirmatory_family"]["tests"]["C2"]["test"]


def test_age_band_edges():
    table = pd.DataFrame({"age": [24, 25, 29, 30], "sex": ["F", "F", "M", "M"]})
    assert age_sex_cell(table).tolist() == ["4_F", "5_F", "5_M", "6_M"]


def test_match_takes_the_reference_distribution():
    reference = speakers_table({("HSA", "normal"): 40})
    pool = speakers_table({("ESSUA", "normal"): 2000}, seed=3)
    drawn, cells = match_cells(reference, pool, np.random.default_rng(0))
    assert (cells["found"] == cells["wanted"]).all()
    assert len(drawn) == 40 and drawn["speaker"].is_unique
    assert (
        age_sex_cell(drawn)
        .value_counts()
        .sort_index()
        .equals(age_sex_cell(reference).value_counts().sort_index())
    )
    assert set(drawn["speaker"]) <= set(pool["speaker"])


def test_match_takes_a_short_cell_whole_and_reports_it():
    reference = pd.DataFrame({"speaker": list("abc"), "age": [30, 30, 30], "sex": ["F", "F", "F"]})
    pool = pd.DataFrame({"speaker": ["x"], "age": [31], "sex": ["F"]})
    drawn, cells = match_cells(reference, pool, np.random.default_rng(0))
    assert drawn["speaker"].tolist() == ["x"]
    assert cells[["wanted", "found"]].to_numpy().tolist() == [[3, 1]]


def test_match_is_fixed_by_the_seed_and_changes_with_it():
    reference = speakers_table({("HSA", "normal"): 40})
    pool = speakers_table({("ESSUA", "normal"): 500}, seed=3)
    one, _ = match_cells(reference, pool, np.random.default_rng(5))
    two, _ = match_cells(reference, pool, np.random.default_rng(5))
    three, _ = match_cells(reference, pool, np.random.default_rng(6))
    assert one["speaker"].tolist() == two["speaker"].tolist()
    assert one["speaker"].tolist() != three["speaker"].tolist()


def test_analysis_table_refuses_a_speaker_without_features():
    speakers = speakers_table({("HSA", "normal"): 5})
    features = features_for(speakers, is_hsa, 0.0).iloc[1:]
    with pytest.raises(ValueError, match="no features"):
        analysis_table(speakers, features, "place")


def _labels(n_a: int, n_b: int) -> tuple[np.ndarray, np.ndarray]:
    y = np.array(["a"] * n_a + ["b"] * n_b)
    return y, np.array([f"s{i}" for i in range(y.size)])


def test_split_check_passes_the_real_splitter_at_the_real_sizes():
    config = make_config(PLAN, "b")
    for n_a, n_b in ((82, 82), (76, 82), (329, 82)):
        y, groups = _labels(n_a, n_b)
        assert check_splits(y, groups, config) <= max(0.02, 1.5 / ((n_a + n_b) / 5))


def test_split_check_stops_on_a_shared_speaker():
    y, groups = _labels(10, 10)
    groups[1] = groups[0]
    indices = np.arange(20)
    with pytest.raises(SplitCheckError, match="both sides"):
        check_folds([(indices[1:], indices[:1])], y, groups)


def test_split_check_stops_on_a_lopsided_fold():
    y, groups = _labels(50, 50)
    indices = np.arange(100)
    test_fold = indices[:20]  # all of class a
    with pytest.raises(SplitCheckError, match="share"):
        check_folds([(indices[20:], test_fold)], y, groups)


def test_split_check_stops_when_items_are_not_covered():
    y, groups = _labels(25, 25)
    indices = np.arange(50)
    rng = np.random.default_rng(0)
    test_fold = np.concatenate([rng.choice(25, 5, False), 25 + rng.choice(25, 5, False)])
    train = np.setdiff1d(indices, test_fold)
    with pytest.raises(SplitCheckError, match="cover"):
        check_folds([(train, test_fold)], y, groups)


def test_arm_finds_a_planted_signal():
    speakers = speakers_table({("HSA", "normal"): 40, ("ESSUA", "normal"): 40})
    features = features_for(speakers, is_hsa, 2.0)
    table = analysis_table(speakers, features, "place")
    arm = evaluate_arm("planted", table, make_config(quick_plan(), "HSA"))
    assert arm.uar > 0.9
    assert arm.predictions.shape == (80, 2)
    assert sum(arm.chosen_c.values()) == 10


def test_shuffled_labels_stay_near_chance():
    speakers = speakers_table({("HSA", "normal"): 40, ("ESSUA", "normal"): 40})
    table = analysis_table(speakers, features_for(speakers, is_hsa, 2.0), "place")
    plan = quick_plan()
    row = batch.negative_control("planted", table, make_config(plan, "HSA"), plan, 1)
    assert 0.3 < row["mean_uar"] < 0.7
    assert row["shuffles"] == 4 and row["n_undefined"] == 0


def _arm(name, y, groups, predictions, uar) -> harness.Arm:
    predictions = np.asarray(predictions)
    return harness.Arm(
        name=name,
        y_true=np.asarray(y),
        groups=np.asarray(groups),
        predictions=predictions,
        scores=np.zeros(predictions.shape),
        uar=uar,
        roc_auc=np.nan,
        sensitivity=np.nan,
        specificity=np.nan,
        worst_share_deviation=0.0,
        chosen_c={},
    )


def _two_class(n: int = 40):
    y = np.array(["n"] * n + ["p"] * n)
    return y, np.array([f"s{i}" for i in range(2 * n)])


def test_bootstrap_of_identical_arms_is_exactly_zero():
    y, groups = _two_class()
    rng = np.random.default_rng(0)
    guess = rng.choice(["n", "p"], size=(y.size, 3))
    arm = _arm("a", y, groups, guess, 0.5)
    order = rng.permutation(y.size)
    shuffled = _arm("b", y[order], groups[order], guess[order], 0.5)
    result = batch.c1_bootstrap([arm], shuffled, 500, 0)
    assert np.all(result["samples"] == pytest.approx(0.0, abs=1e-12))
    assert result["p_one_sided"] == 1.0


def test_bootstrap_sees_a_real_gap_and_not_a_reversed_one():
    y, groups = _two_class()
    rng = np.random.default_rng(1)
    good = y[:, None].repeat(3, axis=1)
    poor = rng.choice(["n", "p"], size=(y.size, 3))
    uar = float(np.mean([np.mean(poor[:, c] == y) for c in range(3)]))
    forward = batch.c1_bootstrap(
        [_arm("g", y, groups, good, 1.0)], _arm("p", y, groups, poor, uar), 2000, 0
    )
    assert forward["difference"] == pytest.approx(1.0 - uar)
    assert forward["p_one_sided"] < 0.01
    assert forward["ci_low"] > 0.2
    backward = batch.c1_bootstrap(
        [_arm("p", y, groups, poor, uar)], _arm("g", y, groups, good, 1.0), 2000, 0
    )
    assert backward["p_one_sided"] > 0.99


def test_bootstrap_p_counts_the_observed_sample():
    y, groups = _two_class()
    good = y[:, None]
    result = batch.c1_bootstrap(
        [_arm("g", y, groups, good, 1.0)],
        _arm("w", y, groups, np.where(y == "n", "p", "n")[:, None], 0.0),
        99,
        0,
    )
    assert result["p_one_sided"] == pytest.approx(1 / 100)


def test_bootstrap_averages_over_the_draws():
    y, groups = _two_class()
    good = y[:, None]
    bad = np.where(y == "n", "p", "n")[:, None]
    half = batch.c1_bootstrap(
        [_arm("g", y, groups, good, 1.0), _arm("b", y, groups, bad, 0.0)],
        _arm("b2", y, groups, bad, 0.0),
        200,
        0,
    )
    assert half["difference"] == pytest.approx(0.5)
    assert np.all(half["samples"] == pytest.approx(0.5))


def _first_batch_fixture(gate_strength: float, sick_strength: float):
    speakers = speakers_table(
        {("HSA", "normal"): 40, ("ESSUA", "normal"): 200, ("HSA", "pathological"): 80}, seed=4
    )
    tables = {
        "nonphon_g100": features_for(speakers, is_hsa, gate_strength),
        "full_original": features_for(speakers, is_sick, sick_strength, 2),
        "phonation_norm": features_for(speakers, is_hsa, 0.0, 3),
    }
    return speakers, tables


def test_gate_failure_stops_before_c1(tmp_path, monkeypatch):
    speakers, tables = _first_batch_fixture(gate_strength=0.0, sick_strength=1.0)
    monkeypatch.setattr(batch, "reaches_minimum", lambda _m: set(speakers["file_id"]))
    plan = quick_plan()
    with pytest.raises(batch.StopRun, match="gate"):
        batch.run(speakers, tables, plan, 1, tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert "gate" in summary and "c1" not in summary
    assert summary["gate"]["passed"] is False


def test_negative_control_failure_stops_the_run(tmp_path, monkeypatch):
    speakers, tables = _first_batch_fixture(gate_strength=3.0, sick_strength=1.0)
    monkeypatch.setattr(batch, "reaches_minimum", lambda _m: set(speakers["file_id"]))
    plan = quick_plan()
    plan["controls"]["negative"]["uar_within"] = [0.99, 1.0]
    with pytest.raises(batch.StopRun, match="negative control"):
        batch.run(speakers, tables, plan, 1, tmp_path)


def test_first_batch_runs_end_to_end(tmp_path, monkeypatch):
    speakers, tables = _first_batch_fixture(gate_strength=3.0, sick_strength=2.0)
    monkeypatch.setattr(batch, "reaches_minimum", lambda _m: set(speakers["file_id"]))
    monkeypatch.setattr(batch, "C2_PERMUTATIONS", 9)
    plan = quick_plan()
    plan["confirmatory_family"]["tests"]["C1"]["draws"] = 3
    plan["confirmatory_family"]["tests"]["C1"]["test"]["resamples"] = 200
    plan["statistics"]["seeds"]["c2_other_draws"] = [20261110, 20261111]
    plan["controls"]["negative"]["uar_within"] = [0.2, 0.8]
    summary = batch.run(speakers, tables, plan, 1, tmp_path)
    assert summary["gate"]["uar"] > 0.9
    c1 = summary["c1"]
    assert c1["patients"] == 80 and c1["normals_per_arm"] == 40 and c1["draws"] == 3
    assert set(c1["draw_place_share"]) <= {"ESSUA", "HSA"}
    assert c1["difference"] == pytest.approx(c1["uar_i_mean"] - c1["uar_ii"])
    assert 0.0 < c1["p_one_sided"] <= 1.0
    c2 = summary["c2"]
    assert c2["n_permutations"] == 9
    assert 0.1 <= c2["p_one_sided"] <= 1.0
    assert c2["other_draws_uar_min"] <= c2["other_draws_uar_max"]
    for name in ("c1_arms.csv", "c1_draws.csv", "c2_null.npy", "gate_cells.csv"):
        assert (tmp_path / name).exists()
    arms = pd.read_csv(tmp_path / "c1_arms.csv")
    assert len(arms) == 4 and set(arms["arm"]) == {"ii", "i_00", "i_01", "i_02"}


def test_positive_class_names_exist_in_the_data():
    assert batch.LABEL_POSITIVE == "pathological" and batch.PLACE_POSITIVE == "HSA"


def test_arm_row_reports_class_counts():
    y, groups = _two_class(3)
    row = _arm("a", y, groups, y[:, None], 1.0).row()
    assert row["n"] == 3 and row["p"] == 3 and row["n_speakers"] == 6
    assert replace(_arm("a", y, groups, y[:, None], 1.0), uar=0.5).uar == 0.5


# The freeze record.


def _tree(tmp_path: Path) -> Path:
    for name, text in (("a.txt", "one"), ("sub/b.txt", "two")):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tmp_path


def test_compare_hashes_sees_a_change_a_loss_and_a_new_file(tmp_path):
    root = _tree(tmp_path)
    names = ["a.txt", "sub/b.txt"]
    frozen = freeze.file_hashes(names, root)
    assert freeze.compare_hashes(frozen, names, root) == []
    (root / "a.txt").write_text("changed", encoding="utf-8")
    assert freeze.compare_hashes(frozen, names, root) == ["a.txt changed"]
    (root / "a.txt").unlink()
    assert freeze.compare_hashes(frozen, names, root)[0] == "a.txt is gone"
    assert (
        freeze.compare_hashes(frozen, [*names, "new.txt"], root)[0]
        == "new.txt is not in the freeze"
    )


def test_direct_url_has_to_name_the_commit():
    good = {"url": "file:///x", "vcs_info": {"vcs": "git", "commit_id": freeze.VSB_COMMIT}}
    assert freeze.check_direct_url(json.dumps(good)) == []
    wrong = {"url": "file:///x", "vcs_info": {"vcs": "git", "commit_id": "f" * 40}}
    assert "the freeze says" in freeze.check_direct_url(json.dumps(wrong))[0]
    editable = {"url": "file:///x", "dir_info": {"editable": True}}
    assert "not from a git commit" in freeze.check_direct_url(json.dumps(editable))[0]
    assert "not installed from git" in freeze.check_direct_url(None)[0]


def test_commit_in_the_script_is_the_one_in_the_plan():
    assert freeze.VSB_COMMIT == PLAN["statistics"]["voice_severity_bench_commit"]
    assert freeze.PINNED_VERSIONS["scikit-learn"] == PLAN["statistics"]["cv"]["scikit_learn"]


def test_frozen_plan_files_match_their_recorded_hashes():
    assert freeze.check_plan_hashes() == []


def test_build_refuses_a_changed_plan(tmp_path):
    (tmp_path / "analysis_plan.md").write_text("other", encoding="utf-8")
    (tmp_path / "analysis_plan.yaml").write_text("other", encoding="utf-8")
    with pytest.raises(freeze.FreezeError, match="not the frozen plan"):
        freeze.build_freeze(tmp_path / "freeze.json", tmp_path)


def test_verify_without_a_record_stops(tmp_path):
    with pytest.raises(freeze.FreezeError, match="nothing is frozen"):
        freeze.verify_freeze(tmp_path / "freeze.json", tmp_path)
