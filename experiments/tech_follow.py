"""How much of each detector's ranking of HSA patients the recording measures of D4 explain.

Exploratory. Definitions and readings are in experiments/deviations.md (2026-11-10), written
before any score was set against the recording measures. Nothing is refitted.

    .venv/Scripts/python.exe -I -X utf8 -B experiments/tech_follow.py
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import diagnosis_breakdown as db  # noqa: E402
import outside_view as ov  # noqa: E402
import run_first_batch as batch  # noqa: E402
from arms import write_json  # noqa: E402
from section8_rest import M_TECH, avfad_stats, m_tech_table  # noqa: E402

RESULTS = HERE / "results" / "tech_follow"
SEED = 20261126
N_RESAMPLES = 10_000
# Readings, written before the run (deviations.md, 2026-11-10).
R2_POINTS = 0.10


def rank_r2(y: np.ndarray, design: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted R^2 of y on the design (intercept added), one value per row of ``weights``.

    ``design`` holds the ranks of the covariates; weights are resample counts.
    """
    w = np.atleast_2d(weights)
    x = np.column_stack([np.ones(len(y)), design])
    xtwx = np.einsum("bn,ni,nj->bij", w, x, x)
    xtwy = (w * y) @ x
    beta = np.linalg.solve(xtwx, xtwy[..., None])[..., 0]
    total = w.sum(axis=1)
    mean_term = (w @ y) ** 2 / total
    explained = np.einsum("bi,bi->b", beta, xtwy) - mean_term
    spread = (w * y**2).sum(axis=1) - mean_term
    return explained / spread


def r2_reading(diff: dict) -> str:
    if diff["ci_high"] < 0:
        return "(i) follows them less"
    if diff["ci_low"] <= 0:
        return "the same"
    if diff["difference"] >= R2_POINTS:
        return "(i) follows the recording measures more among patients"
    return "small"


def patient_table() -> pd.DataFrame:
    speakers = batch.load_speakers()
    patients = speakers[(speakers["place"] == "HSA") & (speakers["label"] == "pathological")]
    table = m_tech_table(patients, batch.load_features("phonation_norm"), avfad_stats())
    return table.set_index("speaker")


def main() -> None:
    table = patient_table()
    patients = table.index
    # A column that is constant among the patients (clipping is zero in every file) has no rank.
    columns = [c for c in M_TECH if table[c].nunique() > 1]
    design = np.column_stack([rankdata(table[c]) for c in columns])
    inside, _ = db.speaker_means(ov.INSIDE["a"])
    outside = pd.read_csv(ov.OUTSIDE["a"], dtype={"speaker": str}).set_index("speaker")["score"]
    draws = sorted(a for a in inside if a.startswith("i_"))
    rng = np.random.default_rng(SEED)
    weights = db.resample_counts(len(patients), N_RESAMPLES, rng)
    ones = np.ones(len(patients))

    def measure(scores: np.ndarray) -> tuple[float, np.ndarray, dict]:
        y = rankdata(scores)
        rho = {
            c: float(ov.spearman_weighted(table[c].to_numpy(), scores, ones)[0]) for c in columns
        }
        return float(rank_r2(y, design, ones)[0]), rank_r2(y, design, weights), rho

    ii_point, ii_boot, ii_rho = measure(inside["ii"].loc[patients].to_numpy())
    out_point, out_boot, out_rho = measure(outside.loc[patients].to_numpy())
    i_points, i_boot, i_rho = [], np.zeros(N_RESAMPLES), {c: [] for c in columns}
    for name in draws:
        point, boot, rho = measure(inside[name].loc[patients].to_numpy())
        i_points.append(point)
        i_boot += boot
        for c in columns:
            i_rho[c].append(rho[c])
    i_boot /= len(draws)
    i_point = float(np.mean(i_points))
    diff = ov.interval(i_point - ii_point, i_boot - ii_boot)
    summary = {
        "seed": SEED,
        "n_resamples": N_RESAMPLES,
        "patients": len(patients),
        "columns": columns,
        "constant_left_out": [c for c in M_TECH if c not in columns],
        "r2": {
            "ii": ov.interval(ii_point, ii_boot, "r2"),
            "i": ov.interval(i_point, i_boot, "r2")
            | {"draw_range": [min(i_points), max(i_points)]},
            "outside": ov.interval(out_point, out_boot, "r2"),
        },
        "i_minus_ii": diff,
        "spearman": {
            c: {"ii": ii_rho[c], "i": float(np.mean(i_rho[c])), "outside": out_rho[c]}
            for c in columns
        },
        "reading": r2_reading(diff),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / "summary.json", summary)
    print(summary["reading"], summary["r2"])


if __name__ == "__main__":
    main()
