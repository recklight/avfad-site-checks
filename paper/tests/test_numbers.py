"""paper/make_numbers.py on made-up values; no result file is read."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))

import make_numbers as pn  # noqa: E402

cn = pn.cn


@pytest.fixture(autouse=True)
def empty_values(monkeypatch):
    monkeypatch.setattr(cn, "values", {})


def test_minus_and_thousands_for_latex():
    assert pn.tex_value("-4.1") == "\\textminus{}4.1"
    assert pn.tex_value("+24.0") == "+24.0"
    assert pn.tex_value("1,462") == "1{,}462"


def test_a_repeated_name_must_agree():
    pn.put("gap", "22.2", "a")
    pn.put("gap", "22.2", "b")
    assert cn.values["gap"] == ("22.2", "a")
    with pytest.raises(ValueError, match="gap"):
        pn.put("gap", "22.3", "c")


def test_tex_file_defines_every_value(tmp_path, monkeypatch):
    monkeypatch.setattr(pn, "TEX", tmp_path / "numbers.tex")
    pn.put("C1.gap", "22.2", "a")
    pn.put("outside.low", "-8.8", "a")
    pn.write_tex()
    text = (tmp_path / "numbers.tex").read_text(encoding="utf-8")
    assert "\\@namedef{pvc@C1.gap}{22.2}" in text
    assert "\\@namedef{pvc@outside.low}{\\textminus{}8.8}" in text
    assert "unknown value" in text


def test_names_latex_cannot_hold_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(pn, "TEX", tmp_path / "numbers.tex")
    pn.put("bad name", "1", "a")
    with pytest.raises(ValueError, match="bad name"):
        pn.write_tex()


def test_paper_phrases_count_only_during_the_check(tmp_path):
    claims = tmp_path / "claims.md"
    claims.write_text("Belo 2015，門檻 10 點，差 22.2 點，另一個 7.7。", encoding="utf-8")
    pn.put("gap", "22.2", "a")
    before = dict(cn.FIXED)
    matched, unmatched = pn.check(claims)
    assert unmatched == ["7.7"]
    assert sum(m.endswith(": fixed") for m in matched) == 2
    assert cn.FIXED == before


def test_tex_check_ignores_macros_and_flags_stray_numbers(tmp_path):
    tex = tmp_path / "main.tex"
    body = [
        r"\documentclass{article}",
        r"\begin{document}",
        r"Gap \val{C1.gap} in C1, see Fig.~\ref{fig:2} and~\cite{ribas2023}. % 99 in a comment",
        r"Frozen on 4 November 2026; another 7.7 points.",
        r"\end{document}",
    ]
    tex.write_text("\n".join(body) + "\n", encoding="utf-8")
    matched, unmatched = pn.check_tex(tex)
    assert unmatched == ["7.7"]
    assert matched == ["4 November 2026: fixed"]
