"""paper/figures.py on made-up records; no result file is read."""

from __future__ import annotations

import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))

import figures  # noqa: E402


def test_bars_end_at_the_rounded_interval():
    record = {"difference": 0.22166, "ci_low": 0.17549, "ci_high": 0.27051}
    assert figures.coordinate(1, record) == "(22.2,1) += (4.9,0) -= (4.7,0)"


def test_pre_registered_row_gets_the_filled_marker(monkeypatch):
    gap = {"difference": 0.2, "ci_low": 0.15, "ci_high": 0.25}
    rows = [("C1 row", gap, True), ("other", gap, False), ("third", gap, False)]
    monkeypatch.setattr(figures, "gap_rows", lambda: rows)
    text = figures.fig_c1_gaps()
    filled, hollow = text.split("mark=*", 1)[1].split("mark=o", 1)
    assert "(20.0,1)" in filled and "(20.0,2)" not in filled
    assert "(20.0,2)" in hollow and "(20.0,3)" in hollow
    assert "yticklabels={{C1 row},{other},{third}}" in text


def test_the_same_input_gives_the_same_text(monkeypatch):
    rows = [("a", 0.82, 0.55), ("b", 0.835, 0.621)]
    monkeypatch.setattr(figures, "inside_rows", lambda: rows)
    monkeypatch.setattr(figures, "load", lambda _: {"uar": 0.877})
    assert figures.fig_inside_hsa() == figures.fig_inside_hsa()
    assert "(axis cs:0.877," in figures.fig_inside_hsa()
