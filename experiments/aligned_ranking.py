"""How (i) ranks the HSA patients after the ESSUA speakers were aligned, by the outside view.

Exploratory. Definitions and readings are in experiments/deviations.md (2026-11-10), written
before this ran. Nothing is trained: the stored test-fold scores of C1, plain and aligned, against
the scores of the HuBERT detector trained on SVD a_n.

    .venv/Scripts/python.exe -I -X utf8 -B experiments/aligned_ranking.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import diagnosis_breakdown as db  # noqa: E402
import outside_view as ov  # noqa: E402
from arms import write_json  # noqa: E402

RESULTS = HERE / "results" / "aligned_ranking"
SEED = 20261142
N_RESAMPLES = 10_000
HARMONIZE = HERE / "results" / "harmonize_c1"
INSIDE = {
    "hubert": {
        "plain": ov.INSIDE["a"],
        "location": HARMONIZE / "c1_predictions_hubert_location.csv.gz",
        "scale": HARMONIZE / "c1_predictions_hubert_scale.csv.gz",
    },
    "handcrafted": {
        "plain": ov.HANDCRAFTED_INSIDE,
        "location": HARMONIZE / "c1_predictions_handcrafted_location.csv.gz",
        "scale": HARMONIZE / "c1_predictions_handcrafted_scale.csv.gz",
        "coral": HERE / "results" / "coral_c1" / "c1_predictions_logistic_coral.csv.gz",
    },
}
# Readings, written before this ran (deviations.md, 2026-11-10).
RANK_POINTS = 0.10


def draws_rho(
    arms: dict[str, pd.Series], patients, reference: np.ndarray, weights: np.ndarray
) -> tuple[float, np.ndarray]:
    """Mean over the draws of the Spearman correlation with the outside scores, and resamples."""
    draws = sorted(a for a in arms if a.startswith("i_"))
    if not draws:
        raise ValueError("no (i) draws in the table")
    point, boot = [], np.zeros(weights.shape[0])
    for name in draws:
        values = arms[name].loc[patients].to_numpy()
        point.append(float(ov.spearman_weighted(reference, values, np.ones(len(patients)))[0]))
        boot += ov.spearman_weighted(reference, values, weights)
    return float(np.mean(point)), boot / len(draws)


def change_reading(diff: dict) -> str:
    if diff["difference"] >= RANK_POINTS and diff["ci_low"] > 0:
        return "after the alignment (i) ranks patients more like the outside detector"
    if diff["ci_low"] <= 0 <= diff["ci_high"]:
        return "the ranking does not change"
    return "as measured"


def versus_ii_reading(diff: dict) -> str:
    if diff["ci_low"] <= 0 <= diff["ci_high"]:
        return "catches up with (ii)"
    if diff["ci_high"] < 0:
        return "still agrees less than (ii)"
    return "agrees more than (ii)"


def main() -> None:
    speakers = db.load_speakers()
    hsa = speakers["place"] == "HSA"
    patients = speakers.index[hsa & (speakers["label"] == "pathological")]
    outside = pd.read_csv(ov.OUTSIDE["a"], dtype={"speaker": str}).set_index("speaker")["score"]
    reference = outside.loc[patients].to_numpy()
    weights = db.resample_counts(len(patients), N_RESAMPLES, np.random.default_rng(SEED))
    summary: dict = {"patients": len(patients), "seed": SEED, "n_resamples": N_RESAMPLES}
    for detector, paths in INSIDE.items():
        plain, _ = db.speaker_means(paths["plain"])
        ii_values = plain["ii"].loc[patients].to_numpy()
        ii = (
            float(ov.spearman_weighted(reference, ii_values, np.ones(len(patients)))[0]),
            ov.spearman_weighted(reference, ii_values, weights),
        )
        base = draws_rho(plain, patients, reference, weights)
        part: dict = {
            "rho_ii": ov.interval(ii[0], ii[1], "rho"),
            "rho_i_plain": ov.interval(base[0], base[1], "rho"),
        }
        for kind, path in paths.items():
            if kind == "plain" or not path.exists():
                continue
            arms, _ = db.speaker_means(path)
            got = draws_rho(arms, patients, reference, weights)
            change = ov.interval(got[0] - base[0], got[1] - base[1])
            versus = ov.interval(got[0] - ii[0], got[1] - ii[1])
            part[kind] = {
                "rho_i": ov.interval(got[0], got[1], "rho"),
                "minus_plain_i": change,
                "reading": change_reading(change),
                "minus_ii": versus,
                "versus_ii_reading": versus_ii_reading(versus),
            }
        summary[detector] = part
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / "summary.json", summary)
    for detector in INSIDE:
        part = summary[detector]
        print(
            detector,
            "ii",
            round(part["rho_ii"]["rho"], 3),
            "i",
            round(part["rho_i_plain"]["rho"], 3),
        )
        for kind in ("location", "scale", "coral"):
            if kind in part:
                v = part[kind]
                print(
                    " ",
                    kind,
                    round(v["rho_i"]["rho"], 3),
                    v["reading"],
                    "|",
                    v["versus_ii_reading"],
                )


if __name__ == "__main__":
    main()
