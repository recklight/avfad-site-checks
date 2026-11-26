"""HSA patients split by CMVD-I chapter: which diagnoses (ii) separates, and where (i) gains.

Exploratory. Definitions and readings are in experiments/deviations.md (2026-11-10), written
before any per-chapter score was computed. Nothing is refitted: the scores are the stored
test-fold scores of the two C1 detectors, and the SVD-trained HuBERT scored on AVFAD serves as
an outside view of how audible each chapter is.

    .venv/Scripts/python.exe -I -X utf8 -B experiments/diagnosis_breakdown.py
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "4")

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from arms import write_json  # noqa: E402
from harness import ROOT  # noqa: E402

RESULTS = HERE / "results" / "diagnosis_breakdown"
STORED = {
    "hubert": HERE / "results" / "sensitivity" / "hubert" / "c1_predictions.csv.gz",
    "handcrafted": HERE / "results" / "first_batch" / "c1_predictions.csv.gz",
}
OUTSIDE = HERE / "results" / "cross_database" / "reverse_scores_hubert.csv.gz"
SPEAKERS = ROOT / "s0_audit" / "out" / "avfad_speakers.csv"
SEED = 20261121
# /i/ and /u/, added after /a/ (deviations.md, 2026-11-10): HuBERT only.
VOWEL_STORED = {
    v: {"hubert": HERE / "results" / "vowel_fusion" / f"predictions_{v}.csv.gz"} for v in ("i", "u")
}
VOWEL_OUTSIDE = {
    v: HERE / "results" / "outside_view" / f"outside_scores_{v}.csv.gz" for v in ("i", "u")
}
VOWEL_SEEDS = {"i": 20261129, "u": 20261130}
N_RESAMPLES = 10_000
# CMVD-I chapter (first digit of the code) -> group; chapters with fewer than 30 HSA patients
# are pooled and only reported.
CHAPTERS = {1: "structural", 2: "inflammatory", 7: "neurological"}
MAIN = tuple(CHAPTERS.values())
OTHER = "other"
# Readings, written before the run (deviations.md, 2026-11-10).
BEST_AUC, BEST_LOW, WORST_AUC, NONE_ABOVE = 0.75, 0.65, 0.65, 0.70
GAIN_POINTS = 0.10
CONTRAST_POINTS = 0.05


def chapter_group(code: int) -> str:
    return CHAPTERS.get(int(code) // 1000, OTHER)


def load_speakers() -> pd.DataFrame:
    table = pd.read_csv(SPEAKERS, dtype={"speaker": str})
    table["group"] = np.where(
        table["label"] == "pathological",
        [chapter_group(c) for c in table["cmvd_code"]],
        "normal",
    )
    return table.set_index("speaker")


def comparisons(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    """Mann-Whitney kernel: 1 where the patient scores higher, 0.5 for a tie."""
    diff = positive[:, None] - negative[None, :]
    return (diff > 0) + 0.5 * (diff == 0)


def weighted_auc(kernel: np.ndarray, w_pos: np.ndarray, w_neg: np.ndarray) -> np.ndarray:
    """AUC with speaker weights; rows of w_pos and w_neg are resamples (counts)."""
    w_pos = np.atleast_2d(w_pos)
    w_neg = np.atleast_2d(w_neg)
    top = np.einsum("bj,bj->b", w_pos @ kernel, w_neg)
    return top / (w_pos.sum(axis=1) * w_neg.sum(axis=1))


def resample_counts(n: int, size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.multinomial(n, np.full(n, 1.0 / n), size=size).astype(float)


def speaker_means(path: Path) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """Score per speaker, averaged over the repeats, for each arm; and the raw table."""
    table = pd.read_csv(path, dtype={"speaker": str})
    means = table.groupby(["arm", "speaker"])["score"].mean()
    return {arm: means.loc[arm] for arm in table["arm"].unique()}, table


def interval(point: float, samples: np.ndarray) -> dict:
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"auc": float(point), "ci_low": float(low), "ci_high": float(high)}


def difference(point: float, samples: np.ndarray) -> dict:
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"difference": float(point), "ci_low": float(low), "ci_high": float(high)}


def detector_breakdown(
    scores: dict[str, pd.Series],
    speakers: pd.DataFrame,
    weights: dict[str, pd.DataFrame],
    outside: pd.Series,
) -> dict:
    """(ii), (i) and outside AUC per chapter, with resampled values kept for the contrasts."""
    hsa_normals = speakers.index[(speakers["group"] == "normal") & (speakers["place"] == "HSA")]
    draws = sorted(a for a in scores if a.startswith("i_"))
    out: dict = {}
    for group in (*MAIN, OTHER):
        patients = speakers.index[(speakers["group"] == group) & (speakers["place"] == "HSA")]
        w_pos = weights[group].loc[:, patients].to_numpy()
        w_hsa = weights["normal"].loc[:, hsa_normals].to_numpy()

        ii = scores["ii"]
        kernel = comparisons(ii.loc[patients].to_numpy(), ii.loc[hsa_normals].to_numpy())
        ii_point, ii_boot = kernel.mean(), weighted_auc(kernel, w_pos, w_hsa)

        i_points, i_boot = [], np.zeros(w_pos.shape[0])
        for name in draws:
            arm = scores[name]
            normals = arm.index[speakers.loc[arm.index, "group"] == "normal"]
            kernel = comparisons(arm.loc[patients].to_numpy(), arm.loc[normals].to_numpy())
            i_points.append(kernel.mean())
            i_boot += weighted_auc(kernel, w_pos, weights["normal"].loc[:, normals].to_numpy())
        i_point, i_boot = float(np.mean(i_points)), i_boot / len(draws)

        kernel = comparisons(outside.loc[patients].to_numpy(), outside.loc[hsa_normals].to_numpy())
        out_point, out_boot = kernel.mean(), weighted_auc(kernel, w_pos, w_hsa)

        out[group] = {
            "patients": len(patients),
            "ii": interval(ii_point, ii_boot),
            "i": interval(i_point, i_boot) | {"draw_range": [min(i_points), max(i_points)]},
            "outside": interval(out_point, out_boot),
            "gain": difference(i_point - ii_point, i_boot - ii_boot),
            "_gain_samples": i_boot - ii_boot,
        }
    return out


def case_mix_reading(cells: dict[str, dict]) -> str:
    """Applied to the three main chapters, to (ii) or to the outside detector."""
    best = max(cells.values(), key=lambda c: c["auc"])
    worst = min(cells.values(), key=lambda c: c["auc"])
    if best["auc"] >= BEST_AUC and best["ci_low"] > BEST_LOW and worst["auc"] <= WORST_AUC:
        return "case mix in part"
    if all(c["auc"] <= NONE_ABOVE for c in cells.values()):
        return "not case mix"
    return "unclear"


def gains_everywhere(gains: dict[str, dict]) -> bool:
    return all(g["difference"] >= GAIN_POINTS and g["ci_low"] > 0 for g in gains.values())


def contrast_reading(contrast: dict) -> str:
    if contrast["ci_low"] > -CONTRAST_POINTS:
        return "the least audible chapter gains no less"
    if contrast["ci_high"] < 0 and contrast["difference"] <= -CONTRAST_POINTS:
        return "the gain follows how audible the disorder is"
    return "unclear"


def specificity_by_place(table: pd.DataFrame, speakers: pd.DataFrame) -> dict:
    """Share of normals the arm calls normal, over speakers and repeats."""
    normals = table[table["label"] == "normal"].copy()
    normals["hsa"] = speakers.loc[normals["speaker"], "place"].to_numpy() == "HSA"
    normals["right"] = normals["predicted"] == "normal"
    drawn = normals[normals["arm"].str.startswith("i_")]
    return {
        "ii_hsa_normals": float(normals.loc[normals["arm"] == "ii", "right"].mean()),
        "i_hsa_normals": float(drawn.loc[drawn["hsa"], "right"].mean()),
        "i_other_normals": float(drawn.loc[~drawn["hsa"], "right"].mean()),
        "i_hsa_normal_rows": int(drawn["hsa"].sum()),
        "i_other_normal_rows": int((~drawn["hsa"]).sum()),
    }


def run(vowel: str) -> None:
    stored = STORED if vowel == "a" else VOWEL_STORED[vowel]
    speakers = load_speakers()
    outside_path = OUTSIDE if vowel == "a" else VOWEL_OUTSIDE[vowel]
    outside = pd.read_csv(outside_path, dtype={"speaker": str}).set_index("speaker")["score"]
    seed = SEED if vowel == "a" else VOWEL_SEEDS[vowel]
    rng = np.random.default_rng(seed)
    weights: dict[str, pd.DataFrame] = {}
    for group in (*MAIN, OTHER):
        members = speakers.index[(speakers["group"] == group) & (speakers["place"] == "HSA")]
        weights[group] = pd.DataFrame(
            resample_counts(len(members), N_RESAMPLES, rng), columns=members
        )
    # Normals resample within their place, so every draw keeps its HSA share on average.
    parts = []
    for at_hsa in (True, False):
        members = speakers.index[
            (speakers["group"] == "normal") & ((speakers["place"] == "HSA") == at_hsa)
        ]
        parts.append(pd.DataFrame(resample_counts(len(members), N_RESAMPLES, rng), columns=members))
    weights["normal"] = pd.concat(parts, axis=1)

    summary: dict = {"vowel": vowel, "seed": seed, "n_resamples": N_RESAMPLES}
    for detector, path in stored.items():
        scores, table = speaker_means(path)
        cells = detector_breakdown(scores, speakers, weights, outside)
        order = sorted(MAIN, key=lambda g: cells[g]["outside"]["auc"])
        least, most = order[0], order[-1]
        contrast_samples = cells[least]["_gain_samples"] - cells[most]["_gain_samples"]
        contrast = difference(
            cells[least]["gain"]["difference"] - cells[most]["gain"]["difference"],
            contrast_samples,
        )
        for cell in cells.values():
            cell.pop("_gain_samples")
        summary[detector] = {
            "chapters": cells,
            "least_audible": least,
            "most_audible": most,
            "gain_least_minus_most": contrast,
            "specificity": specificity_by_place(table, speakers),
            "readings": {
                "case_mix_ii": case_mix_reading({g: cells[g]["ii"] for g in MAIN}),
                "case_mix_outside": case_mix_reading({g: cells[g]["outside"] for g in MAIN}),
                "gains_in_all_three": gains_everywhere({g: cells[g]["gain"] for g in MAIN}),
                "contrast": contrast_reading(contrast),
            },
        }
        print(detector, summary[detector]["readings"], flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / ("summary.json" if vowel == "a" else f"summary_{vowel}.json"), summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vowel", choices=("a", "i", "u"), default="a")
    run(parser.parse_args().vowel)


if __name__ == "__main__":
    main()
