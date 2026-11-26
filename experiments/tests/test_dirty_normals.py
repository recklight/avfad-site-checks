"""Swapping some A normals for patients, on a synthetic speaker list; no real data is read."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "features"))

import dirty_normals as dn  # noqa: E402
import simulated_place as sim  # noqa: E402
from test_simulated_place import synthetic_meta  # noqa: E402


def people_and_meta(monkeypatch):
    meta = synthetic_meta(n_patients=700, n_normals=600, seed=3)
    monkeypatch.setattr(sim.placebo, "svd_meta", lambda: meta)
    return sim.people(), meta


def test_swaps_are_nested_same_sex_and_near_in_age(monkeypatch):
    who, meta = people_and_meta(monkeypatch)
    swaps = dn.replacements(who, meta)
    assert len(swaps) == max(dn.DIRTY)
    assert swaps["speaker"].is_unique
    assert not set(swaps["speaker"]) & set(who["speaker"])
    assert set(swaps["label"]) == {"pathological"}
    replaced = who.set_index("speaker").loc[swaps["replaced"]]
    assert set(replaced["role"]) == {"A"}
    assert (swaps["sex"].to_numpy() == replaced["sex"].to_numpy()).all()
    assert np.median(np.abs(swaps["age"].to_numpy() - replaced["age"].to_numpy())) <= 1
    assert dn.replacements(who, meta).equals(swaps)


def test_with_dirty_keeps_the_numbers(monkeypatch):
    who, meta = people_and_meta(monkeypatch)
    swaps = dn.replacements(who, meta)
    previous: set[str] = set()
    for n in dn.DIRTY:
        dirty = dn.with_dirty(who, swaps, n)
        assert len(dirty) == len(who)
        assert dirty["role"].value_counts().to_dict() == {
            "patient": sim.N_PATIENTS,
            "B": sim.N_B,
            "A": sim.N_A,
        }
        newcomers = set(dirty.loc[dirty["dirty"], "speaker"])
        assert len(newcomers) == n
        assert previous <= newcomers
        previous = newcomers
        assert set(dirty.loc[dirty["dirty"], "label"]) == {"normal"}
        assert set(dirty.loc[dirty["dirty"], "site"]) == {"A"}
        unchanged = dirty[~dirty["dirty"]].set_index("speaker")
        assert unchanged["label"].equals(who.set_index("speaker").loc[unchanged.index, "label"])
    assert not dn.with_dirty(who, swaps, 0)["dirty"].any()


def test_readings():
    hand = {"difference": 0.18, "ci_low": 0.1, "ci_high": 0.25}
    assert dn.gap_reading(hand, "handcrafted").startswith("this unclean, the normals alone make")
    assert "do not" in dn.gap_reading({**hand, "difference": 0.10}, "handcrafted")
    assert dn.place_reading({"auc": 0.89}, "hubert").endswith("reaches AVFAD's")
    assert "does not" in dn.place_reading({"auc": 0.85}, "hubert")
    assert dn.overall_reading({21: False, 41: True, 62: True}).endswith(
        "at least 41 of HSA's 82 normals"
    )
    assert dn.overall_reading({21: False, 41: False, 62: False}).startswith("three quarters")
    assert dn.overall_reading({21: False}) == "as measured"


def test_place_record_reports_inside_a(monkeypatch):
    who, meta = people_and_meta(monkeypatch)
    dirty = dn.with_dirty(who, dn.replacements(who, meta), 41)
    rng = np.random.default_rng(0)
    scores = pd.DataFrame({"file_id": dirty["file_id"], "score": rng.standard_normal(len(dirty))})
    record = dn.place_record(scores, dirty, np.random.default_rng(1))
    assert 0.0 <= record["inside_a"] <= 1.0
    assert 0.0 <= record["dirty_vs_clean_a"] <= 1.0
    assert record["ci_low"] <= record["auc"] <= record["ci_high"]
