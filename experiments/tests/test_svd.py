"""Synthetic checks of the C3 and C4 code; no recording is read."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import freeze_svd
import run_svd as svd
from harness import FEATURE_COLUMNS, load_plan

PLAN = load_plan()


def quick_plan(repeats: int = 2) -> dict:
    plan = json.loads(json.dumps(PLAN, default=str))
    plan["statistics"]["cv"]["repeats"] = repeats
    plan["controls"]["negative"]["shuffles_per_arm"] = 4
    return plan


def speakers(n: int = 120, material: str = "iau", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "file_id": [f"svd__{i}-{material}" for i in range(n)],
            "session": np.arange(n),
            "speaker": [f"p{i}" for i in range(n)],
            "label": np.where(np.arange(n) % 2 == 0, "normal", "pathological"),
            "age": rng.uniform(20, 75, n),
            "sex": rng.choice(["m", "w"], n),
            "year": rng.choice([1997, 1998, 1999, 2000, 2001, 2003], n),
        }
    )


def features(meta: pd.DataFrame, strength: float, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(len(meta), len(FEATURE_COLUMNS)))
    values[:, :4] += strength * (meta["label"] == "pathological").to_numpy()[:, None]
    table = pd.DataFrame(values, columns=FEATURE_COLUMNS)
    table.insert(0, "file_id", meta["file_id"].to_numpy())
    return table


def test_stratum_uses_six_age_bands_and_three_year_bands():
    assert len(svd.AGE_EDGES) - 1 == 6 and len(svd.YEAR_EDGES) - 1 == 3
    table = pd.DataFrame({"age": [18.0, 29.9, 30.0, 94.7], "sex": ["m"] * 4, "year": [1997] * 4})
    cells = svd.stratum(table)
    assert cells[0] == cells[1] and cells[1] != cells[2]


def test_year_bands_are_the_planned_ones():
    table = pd.DataFrame(
        {"age": [40.0] * 6, "sex": ["m"] * 6, "year": [1997, 1998, 1999, 2000, 2001, 2004]}
    )
    cells = svd.stratum(table)
    assert cells[0] == cells[1] and cells[2] == cells[3] and cells[4] == cells[5]
    assert len(set(cells)) == 3


def test_stratum_refuses_what_falls_outside():
    table = pd.DataFrame({"age": [17.0], "sex": ["m"], "year": [1999]})
    with pytest.raises(ValueError, match="outside"):
        svd.stratum(table)


def test_one_sided_p_follows_the_sign():
    assert svd.one_sided_from_t(2.0, 0.04) == pytest.approx(0.02)
    assert svd.one_sided_from_t(-2.0, 0.04) == pytest.approx(0.98)


def test_availability_inside_strata_ignores_what_only_differs_between_them():
    # Same share inside each stratum; the raw shares differ only through the mix of strata.
    table = pd.DataFrame(
        {
            "label": ["normal"] * 10
            + ["pathological"] * 30
            + ["normal"] * 30
            + ["pathological"] * 10,
            "reaches_minimum": np.r_[np.ones(10), np.ones(30), np.zeros(30), np.zeros(10)],
        }
    )
    cells = np.array(["a"] * 40 + ["b"] * 40)
    out = svd.availability_by_label(table, cells, 0)
    assert out["within_strata_difference"] == pytest.approx(0.0)
    assert out["overall_pathological"] == pytest.approx(0.75)
    assert out["overall_normal"] == pytest.approx(0.25)


def test_availability_inside_strata_finds_a_real_difference():
    table = pd.DataFrame(
        {
            "label": ["normal"] * 50 + ["pathological"] * 50,
            "reaches_minimum": np.r_[np.ones(40), np.zeros(10), np.ones(10), np.zeros(40)],
        }
    )
    out = svd.availability_by_label(table, np.array(["a"] * 100), 0)
    assert out["within_strata_difference"] == pytest.approx(-0.6)
    low, high = out["within_strata_ci"]
    assert low < -0.6 < high and high < 0


def test_a_stratum_with_one_label_counts_for_nothing():
    table = pd.DataFrame(
        {
            "label": ["normal"] * 10 + ["pathological"] * 10 + ["normal"] * 5,
            "reaches_minimum": np.r_[np.ones(10), np.ones(10), np.zeros(5)],
        }
    )
    out = svd.availability_by_label(table, np.array(["a"] * 20 + ["b"] * 5), 0)
    assert out["within_strata_difference"] == pytest.approx(0.0)


def _results(p=0.001, gap=True) -> dict:
    return {
        "p_one_sided": p,
        "estimate": 0.2,
        "smallest_gap": 0.1,
        "reaches_smallest_gap": gap,
    }


def test_holm_needs_all_four():
    with pytest.raises(ValueError, match="all four"):
        svd.family_table({"C1": _results(), "C2": _results()})


def test_holm_and_the_three_readings():
    table = svd.family_table(
        {
            "C1": _results(0.0001),
            "C2": _results(0.001),
            "C3": _results(0.01, gap=False),
            "C4": _results(0.3, gap=True),
        }
    )
    assert table["C1"]["p_holm"] == pytest.approx(0.0004)
    assert table["C2"]["p_holm"] == pytest.approx(0.003)
    assert table["C3"]["p_holm"] == pytest.approx(0.02)
    assert table["C4"]["p_holm"] == pytest.approx(0.3)
    assert table["C1"]["reading"] == "yes"
    assert table["C3"]["reading"] == "statistically present, but small"
    assert table["C4"]["reading"].startswith("not significant")


def test_c3_sees_a_signal_only_outside_phonation(tmp_path):
    iau = speakers(120)
    an = speakers(120, "a_n")
    tables = {
        "iau_phonation_norm": features(iau, 0.0, 2),
        "iau_phonation_breath_norm": features(iau, 0.0, 3),
        "iau_full_norm": features(iau, 2.0, 4),
        "a_n_full_norm": features(an, 0.0, 5),
    }
    plan = quick_plan()
    plan["controls"]["negative"]["uar_within"] = [0.2, 0.8]
    result = svd.run_c3(iau, an, tables, plan, 1, tmp_path)
    assert result["uar"]["full"] > 0.85 and result["uar"]["phonation"] < 0.7
    assert result["difference"] == pytest.approx(result["uar"]["full"] - result["uar"]["phonation"])
    assert result["p_one_sided"] < 0.05
    assert result["reaches_smallest_gap"]
    assert result["gap_a_n_to_iau"]["speakers"] == 120
    assert result["gap_a_n_to_iau"]["contrast"]["difference"] > 0.2
    assert (tmp_path / "c3_arms.csv").exists() and (tmp_path / "c3_gap.json").exists()


def test_c4_end_to_end_with_availability(tmp_path, monkeypatch):
    meta = speakers(160)
    quiet = meta["file_id"].to_numpy()
    total = np.where(np.arange(160) % 4 == 0, 0.5, 2.0)
    files = pd.DataFrame(
        {
            "file_id": quiet,
            "nonphonation_g100_s": total,
            "n_nonphonation_g100": np.full(160, 2),
        }
    )
    monkeypatch.setattr(svd, "S1_OUT", tmp_path)
    files.to_csv(tmp_path / "files.csv", index=False)
    monkeypatch.setattr(svd, "C4_PERMUTATIONS", 9)
    plan = quick_plan()
    plan["controls"]["negative"]["uar_within"] = [0.2, 0.8]
    tables = {"iau_nonphon_g100": features(meta, 2.0)}
    result = svd.run_c4(meta, tables, plan, 1, tmp_path)
    assert result["speakers"] == 120 and result["files_left_out"] == 40
    assert result["uar"] > 0.8
    assert result["difference"] == pytest.approx(result["uar"] - result["null_mean"])
    assert result["p_one_sided"] == pytest.approx(0.1)
    assert "availability_alone" in result and "availability_by_label" in result
    assert isinstance(result["holds_only_for_files_with_enough_quiet"], bool)
    assert (tmp_path / "c4_null.npy").exists()
    assert (tmp_path / "c4_availability_by_label.json").exists()


def test_c4_excludes_the_lengths_from_the_features():
    assert "nonphonation_total_s" not in FEATURE_COLUMNS
    assert "nonphonation_segments" not in FEATURE_COLUMNS
    assert "kept_s" not in FEATURE_COLUMNS and "n_spans" not in FEATURE_COLUMNS


def test_second_freeze_sees_a_change(tmp_path):
    for name in freeze_svd.TRACKED:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    record_path = tmp_path / "record.json"
    freeze_svd.build_svd_freeze(record_path, tmp_path)
    assert freeze_svd.verify_svd_freeze(record_path, tmp_path)["files"]
    (tmp_path / "experiments/run_svd.py").write_text("y", encoding="utf-8")
    with pytest.raises(freeze_svd.FreezeError, match=r"run_svd.py changed"):
        freeze_svd.verify_svd_freeze(record_path, tmp_path)


def test_second_freeze_stops_without_a_record(tmp_path):
    with pytest.raises(freeze_svd.FreezeError, match="not frozen"):
        freeze_svd.verify_svd_freeze(tmp_path / "missing.json", tmp_path)
