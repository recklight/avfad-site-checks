"""The number check of continue_numbers.py on made-up text; no result file is read."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

import continue_numbers as cn  # noqa: E402


@pytest.fixture
def analysis(tmp_path, monkeypatch):
    monkeypatch.setattr(cn, "values", {})
    path = tmp_path / "analysis.md"
    monkeypatch.setattr(cn, "ANALYSIS", path)
    return path


def test_known_values_match_with_or_without_sign(analysis):
    cn.put("gap", "22.2", "test")
    cn.put("outside", "-4.1", "test")
    cn.put("learned", "+24.0", "test")
    analysis.write_text("差 22.2 點，外部 -4.1，學來的 24.0。", encoding="utf-8")
    matched, unmatched = cn.check_text()
    assert unmatched == []
    assert [m.split(":")[0] for m in matched] == ["22.2", "-4.1", "24.0"]


def test_a_number_from_nowhere_is_reported(analysis):
    cn.put("gap", "22.2", "test")
    analysis.write_text("差 22.2 點，另一個 7.7 點。", encoding="utf-8")
    _, unmatched = cn.check_text()
    assert unmatched == ["7.7"]


def test_dates_labels_code_and_fixed_phrases_are_not_numbers(analysis):
    analysis.write_text(
        "## 3. 標題\n"
        "1. 2026-11-04 15:22Z 寄出，11-10 追信。C1、S2、RT60、SHA-256、`f5174bb`。\n"
        "底噪高 12 dB，模擬裡 0.5 s 迴響，要 1 s 才有。\n",
        encoding="utf-8",
    )
    matched, unmatched = cn.check_text()
    assert unmatched == []
    assert all(m.endswith(": fixed") for m in matched)


def test_formats():
    assert cn.u(0.85312) == "0.853"
    assert cn.pts(0.2217) == "22.2"
    assert cn.signed(-0.0414) == "-4.1"
    assert cn.signed(0.24) == "+24.0"
    assert cn.worker_hours(2076, 12) == "6.9"
