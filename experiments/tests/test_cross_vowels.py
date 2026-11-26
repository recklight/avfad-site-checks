"""Synthetic checks of the /i/ and /u/ cross-database run; no recording is read."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "features"))

import cross_vowels as cv  # noqa: E402


def test_names_follow_the_databases():
    assert cv.avfad_file_id("AAC", "i") == "avfad__AAC-i"
    assert cv.svd_member(1069, "u") == "data/1069/vowels/1069-u_n.nsp"
    assert cv.egg_member(1069, "i") == "data/1069/vowels/1069-i_n-egg.egg"
    assert cv.AVFAD_TASK == {"i": "001", "u": "003"}


def test_files_are_dropped_for_reader_problems_or_a_swapped_session():
    assert cv.drop_reason([], False) == ""
    assert "swap" in cv.drop_reason([], True)
    assert cv.drop_reason(["sample count differs"], False) == "sample count differs"


def test_training_sets_keep_the_speakers_and_change_only_the_file(monkeypatch):
    sets = {
        "ii": pd.DataFrame({"file_id": ["avfad__A", "avfad__B"], "speaker": ["A", "B"]}),
        "all": pd.DataFrame({"file_id": ["avfad__A"], "speaker": ["A"]}),
    }
    monkeypatch.setattr(cv.cd, "training_sets", lambda: sets)
    out = cv.training_sets("u")
    assert out["ii"]["file_id"].tolist() == ["avfad__A-u", "avfad__B-u"]
    assert out["ii"]["speaker"].tolist() == ["A", "B"]
    assert sets["ii"]["file_id"].tolist() == ["avfad__A", "avfad__B"]


@pytest.mark.parametrize(
    ("readings", "expected"),
    [
        (["does not travel: consistent with a place shortcut"] * 2, "repeats"),
        (
            ["does not travel: consistent with a place shortcut", "part of it travels"],
            "does not repeat",
        ),
        (["does not travel: consistent with a place shortcut", "middle"], "no conclusion"),
    ],
)
def test_both_vowels(readings, expected):
    assert cv.both_vowels(dict(zip(cv.VOWELS, readings, strict=True))) == expected


def test_seeds_are_the_written_ones():
    assert cv.SEEDS == {"i": 20261115, "u": 20261116}
