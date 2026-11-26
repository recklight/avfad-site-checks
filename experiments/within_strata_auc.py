"""AUC inside the strata, from the stored predictions; no model is fitted.

UAR minus the stratified null has little room above a high null, so it can understate what a
detector still does inside a stratum. Inside a stratum the demographic cues are equal and
chance is 0.5 whatever the null. Strata with both labels are pooled with Mantel-Haenszel
weights, per repeat, then averaged; the interval resamples speakers inside each stratum and
label. Reason and timing are in experiments/deviations.md.

    .venv/Scripts/python.exe -I -B experiments/within_strata_auc.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_first_batch as batch  # noqa: E402
import run_svd as svd  # noqa: E402
from harness import age_sex_cell  # noqa: E402

RESULTS = HERE / "results"
OUT = RESULTS / "exploratory" / "within_strata_auc.json"
N_RESAMPLES = 1000
SEED = 20261204


def mann_whitney_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """AUC of the scores, ties counting one half."""
    ordered = np.sort(negative)
    below = np.searchsorted(ordered, positive, side="left")
    through = np.searchsorted(ordered, positive, side="right")
    return float(np.mean((below + through) / 2.0) / negative.size)


def pooled_auc(scores: np.ndarray, sick: np.ndarray, cells: np.ndarray) -> float:
    """Mantel-Haenszel average of the AUC of every stratum with both labels."""
    top, bottom = 0.0, 0.0
    for cell in np.unique(cells):
        inside = cells == cell
        a, b = scores[inside & sick], scores[inside & ~sick]
        if a.size and b.size:
            weight = a.size * b.size / (a.size + b.size)
            top += weight * mann_whitney_auc(a, b)
            bottom += weight
    return top / bottom


def within_strata(matrix: np.ndarray, sick: np.ndarray, cells: np.ndarray, seed: int) -> dict:
    """``matrix`` is (speakers, repeats) of pathology scores."""
    estimate = float(
        np.mean([pooled_auc(matrix[:, r], sick, cells) for r in range(matrix.shape[1])])
    )
    rng = np.random.default_rng(seed)
    groups = [
        np.flatnonzero((cells == cell) & (sick == label))
        for cell in np.unique(cells)
        for label in (True, False)
    ]
    samples = []
    for _ in range(N_RESAMPLES):
        chosen = np.concatenate([rng.choice(g, g.size) for g in groups if g.size])
        values = [
            pooled_auc(matrix[chosen, r], sick[chosen], cells[chosen])
            for r in range(matrix.shape[1])
        ]
        samples.append(np.mean(values))
    low, high = np.quantile(samples, [0.025, 0.975])
    both = sum(
        1
        for cell in np.unique(cells)
        if ((cells == cell) & sick).any() and ((cells == cell) & ~sick).any()
    )
    return {
        "auc": estimate,
        "ci": [float(low), float(high)],
        "strata_with_both_labels": both,
        "strata": int(np.unique(cells).size),
        "speakers": int(sick.size),
    }


def svd_matrix(name: str, predictions: pd.DataFrame) -> np.ndarray:
    part = predictions[predictions["feature_set"] == name]
    return part.pivot(index="item", columns="repeat", values="score_pathological").to_numpy()


def avfad_matrix(path: Path, arm: str) -> pd.DataFrame:
    table = pd.read_csv(path, dtype={"speaker": str})
    part = table[table["arm"] == arm]
    return part.pivot(index="speaker", columns="repeat", values="score")


def main() -> None:
    report = {}
    meta = svd.load_svd_speakers("iau").sort_values("file_id").reset_index(drop=True)
    predictions = pd.read_csv(RESULTS / "svd" / "c3_predictions.csv.gz")
    sick = (meta["label"] == svd.POSITIVE).to_numpy()
    cells = svd.stratum(meta)
    for name in ("full", "phonation", "phonation_breath"):
        report[f"svd_iau_{name}"] = within_strata(svd_matrix(name, predictions), sick, cells, SEED)
        print(f"svd_iau_{name}", report[f"svd_iau_{name}"], flush=True)

    speakers = batch.load_speakers().set_index("speaker")
    sources = (
        ("avfad_hsa_handcrafted", RESULTS / "first_batch" / "c1_predictions.csv.gz"),
        ("avfad_hsa_hubert", RESULTS / "sensitivity" / "hubert" / "c1_predictions.csv.gz"),
    )
    for name, path in sources:
        matrix = avfad_matrix(path, "ii")
        people = speakers.loc[matrix.index]
        report[name] = within_strata(
            matrix.to_numpy(),
            (people["label"] == "pathological").to_numpy(),
            age_sex_cell(people).to_numpy().astype(str),
            SEED,
        )
        print(name, report[name], flush=True)
    OUT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
