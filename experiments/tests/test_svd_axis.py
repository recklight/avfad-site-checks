"""Synthetic check of the SVD score table; no real score file is read."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "features"))

import svd_axis as sa  # noqa: E402


def _write(folder: Path, order: list[str]) -> None:
    plain = pd.DataFrame(
        {
            "file_id": ["a", "b", "c"],
            "label": ["normal", "pathological", "normal"],
            "ii": [0.1, 0.9, 0.2],
            "all": [0.3, 0.7, 0.4],
            "i_00": [0.2, 0.8, 0.6],
            "i_01": [0.4, 0.6, 0.0],
        }
    )
    plain.to_csv(folder / "scores_hubert.csv.gz", index=False)
    moved = pd.DataFrame({"file_id": order, "all": [0.5, 0.5, 0.5]})
    moved.to_csv(folder / "aligned.csv.gz", index=False)


def test_draws_are_averaged_and_rows_checked(tmp_path, monkeypatch):
    _write(tmp_path, ["a", "b", "c"])
    monkeypatch.setattr(sa.cd, "RESULTS", tmp_path)
    monkeypatch.setattr(sa, "ALIGNED", tmp_path / "aligned.csv.gz")
    table = sa.svd_scores()
    assert np.allclose(table["i"], [0.3, 0.7, 0.3])
    assert list(table.columns) == ["file_id", "label", "ii", "all", "i", "all_aligned"]
    _write(tmp_path, ["b", "a", "c"])
    with pytest.raises(RuntimeError, match="stored order"):
        sa.svd_scores()
