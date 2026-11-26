"""Inside HSA alone, whether the C1 scores follow the place direction that only the normals define.

Exploratory and descriptive; nothing is trained. Definitions and readings are in
experiments/deviations.md (2026-11-10), written before this ran. The projections are the
cross-fitted ones of patient_position.py (same seed); a larger projection is more ESSUA-like.

    .venv/Scripts/python.exe -I -X utf8 -B experiments/within_place_axis.py a|i|u|aligned

``i``, ``u`` and ``aligned`` were added after the /a/ run: the same view on the other two vowels,
and with the (i) scores of the aligned C1 (harmonize_c1.py, coral_c1.py).
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import diagnosis_breakdown as db  # noqa: E402
import outside_view as ov  # noqa: E402
import patient_position as pp  # noqa: E402
import run_first_batch as batch  # noqa: E402
import site_erasure as se  # noqa: E402
import vowel_fusion as vf  # noqa: E402
from arms import load_hubert, write_json  # noqa: E402
from freeze import verify_freeze  # noqa: E402
from sensitivity_avfad import CHOSEN_LAYERS  # noqa: E402

RESULTS = HERE / "results" / "within_place_axis"
SEED = 20261143
N_RESAMPLES = 10_000
SCORES = {"hubert": ov.INSIDE["a"], "handcrafted": ov.HANDCRAFTED_INSIDE}
VOWEL_SEEDS = {"a": SEED, "i": 20261144, "u": 20261145, "phonation": 20261149}
HARMONIZE = HERE / "results" / "harmonize_c1"
ALIGNED = {
    "hubert": {
        "location": HARMONIZE / "c1_predictions_hubert_location.csv.gz",
        "scale": HARMONIZE / "c1_predictions_hubert_scale.csv.gz",
    },
    "handcrafted": {
        "location": HARMONIZE / "c1_predictions_handcrafted_location.csv.gz",
        "scale": HARMONIZE / "c1_predictions_handcrafted_scale.csv.gz",
        "coral": HERE / "results" / "coral_c1" / "c1_predictions_logistic_coral.csv.gz",
    },
}
# Readings, written before this ran (deviations.md, 2026-11-10).
FOLLOWS = -0.10
LAYERS_FOR_AXIS = 4
LESS = 0.10


def per_speaker(path: Path) -> dict[str, pd.Series]:
    """(ii) and (i) score per speaker; (i) averaged over the draws the speaker is in."""
    arms, _ = db.speaker_means(path)
    draws = pd.concat([s for a, s in arms.items() if a.startswith("i_")])
    return {"ii": arms["ii"], "i": draws.groupby(level=0).mean()}


def projections(
    vowel: str = "a", condition: str = "full"
) -> tuple[dict[int, pd.Series], dict[int, pd.Series]]:
    """Cross-fitted place projections of the HSA normals and the HSA patients, per layer."""
    speakers = batch.load_speakers()
    features = load_hubert(condition, CHOSEN_LAYERS, material=f"avfad_{vowel}")
    normals = vf.with_vowel(se.normal_speakers(), vowel).merge(
        features, on="file_id", how="left", validate="one_to_one"
    )
    patients = speakers[(speakers["place"] == "HSA") & (speakers["label"] == "pathological")]
    patients = vf.with_vowel(patients.sort_values("file_id").reset_index(drop=True), vowel)
    patients = patients.merge(features, on="file_id", how="left", validate="one_to_one")
    for table in (normals, patients):
        if table[features.columns[1]].isna().any():
            raise ValueError(f"/{vowel}/: a speaker has no features")
    essua = (normals["place"] == "ESSUA").to_numpy()
    parts = pp.parts_within_place(
        normals["place"].to_numpy(), pp.N_PARTS, np.random.default_rng(pp.SEED)
    )
    normal_proj, patient_proj = pp.cross_fitted(
        se.layer_blocks(normals), se.layer_blocks(patients), essua, parts
    )
    hsa_ids = normals.loc[~essua, "speaker"].astype(str).to_numpy()
    patient_ids = patients["speaker"].astype(str).to_numpy()
    hsa = {k: pd.Series(v[~essua], index=hsa_ids) for k, v in normal_proj.items()}
    sick = {k: pd.Series(v, index=patient_ids) for k, v in patient_proj.items()}
    return hsa, sick


def follow_reading(diff: dict) -> str:
    if diff["difference"] <= FOLLOWS and diff["ci_high"] < 0:
        return "inside one place, the (i) score follows the place direction too"
    if diff["ci_low"] <= 0 <= diff["ci_high"]:
        return "not visible inside one place"
    return "as measured"


def less_reading(diff: dict) -> str:
    if diff["difference"] >= LESS and diff["ci_low"] > 0:
        return "after the alignment (i) follows the direction less"
    if diff["ci_low"] <= 0 <= diff["ci_high"]:
        return "after the alignment (i) follows it just the same"
    return "as measured"


def majority(readings: list[str]) -> str:
    for text in sorted(set(readings)):
        if readings.count(text) >= LAYERS_FOR_AXIS and text != "as measured":
            return text
    return "as measured"


def correlations(
    projection: pd.Series, scores: dict[str, pd.Series], weights: np.ndarray
) -> dict[str, tuple[float, np.ndarray]]:
    x = projection.to_numpy()
    out = {}
    for name, s in scores.items():
        y = s.loc[projection.index].to_numpy()
        point = float(ov.spearman_weighted(x, y, np.ones(len(x)))[0])
        out[name] = (point, ov.spearman_weighted(x, y, weights))
    return out


def groups_and_weights(vowel: str) -> tuple[dict, dict]:
    if vowel == "phonation":
        hsa, sick = projections("a", "phonation")
    else:
        hsa, sick = projections(vowel)
    rng = np.random.default_rng(VOWEL_SEEDS[vowel])
    groups = {"hsa_patients": sick, "hsa_normals": hsa}
    weights = {
        g: db.resample_counts(len(next(iter(p.values()))), N_RESAMPLES, rng)
        for g, p in groups.items()
    }
    return groups, weights


def main(vowel: str = "a") -> None:
    verify_freeze()
    groups, weights = groups_and_weights(vowel)
    outside_path = ov.OUTSIDE_PHONATION if vowel == "phonation" else ov.OUTSIDE[vowel]
    outside = pd.read_csv(outside_path, dtype={"speaker": str}).set_index("speaker")["score"]
    summary: dict = {"speakers": {g: len(next(iter(p.values()))) for g, p in groups.items()}}
    if vowel == "a":
        sources = SCORES
    elif vowel == "phonation":
        sources = {"hubert": ov.INSIDE_PHONATION}
    else:
        sources = {"hubert": ov.INSIDE[vowel]}
    for detector, path in sources.items():
        scores = per_speaker(path)
        if detector == "hubert":
            scores["outside"] = outside
        part: dict = {}
        for group, proj in groups.items():
            layers = {}
            for layer in CHOSEN_LAYERS:
                got = correlations(proj[layer], scores, weights[group])
                diff = ov.interval(got["i"][0] - got["ii"][0], got["i"][1] - got["ii"][1])
                layers[str(layer)] = {
                    **{f"rho_{n}": ov.interval(v[0], v[1], "rho") for n, v in got.items()},
                    "i_minus_ii": diff,
                    "reading": follow_reading(diff),
                }
            follows = sum(v["reading"].startswith("inside one place") for v in layers.values())
            part[group] = {
                "layers": layers,
                "overall": (
                    "(i) uses the place direction inside one place too"
                    if follows >= LAYERS_FOR_AXIS
                    else "as measured"
                ),
            }
        summary[detector] = part
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / ("summary.json" if vowel == "a" else f"summary_{vowel}.json"), summary)
    for detector in sources:
        for group in groups:
            print(detector, group, summary[detector][group]["overall"])
            for layer, v in summary[detector][group]["layers"].items():
                rhos = {k[4:]: round(x["rho"], 3) for k, x in v.items() if k.startswith("rho_")}
                print("  ", layer, rhos, round(v["i_minus_ii"]["difference"], 3), v["reading"])


def aligned() -> None:
    """The /a/ view with the aligned (i) scores; same projections and resamples as ``main``."""
    verify_freeze()
    groups, weights = groups_and_weights("a")
    summary: dict = {}
    for detector, kinds in ALIGNED.items():
        plain = per_speaker(SCORES[detector])
        for kind, path in kinds.items():
            arms, _ = db.speaker_means(path)
            draws = pd.concat([s for a, s in arms.items() if a.startswith("i_")])
            scores = {"ii": plain["ii"], "i": draws.groupby(level=0).mean(), "i_plain": plain["i"]}
            part: dict = {}
            for group, proj in groups.items():
                layers = {}
                for layer in CHOSEN_LAYERS:
                    got = correlations(proj[layer], scores, weights[group])
                    diff = ov.interval(got["i"][0] - got["ii"][0], got["i"][1] - got["ii"][1])
                    change = ov.interval(
                        got["i"][0] - got["i_plain"][0], got["i"][1] - got["i_plain"][1]
                    )
                    layers[str(layer)] = {
                        **{f"rho_{n}": ov.interval(v[0], v[1], "rho") for n, v in got.items()},
                        "i_minus_ii": diff,
                        "reading": follow_reading(diff),
                        "minus_plain_i": change,
                        "change_reading": less_reading(change),
                    }
                part[group] = {
                    "layers": layers,
                    "overall": majority([v["reading"] for v in layers.values()]),
                    "overall_change": majority([v["change_reading"] for v in layers.values()]),
                }
            summary.setdefault(detector, {})[kind] = part
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / "summary_aligned.json", summary)
    for detector, kinds in summary.items():
        for kind, part in kinds.items():
            for group, v in part.items():
                print(detector, kind, group, v["overall"], "|", v["overall_change"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "step", nargs="?", default="a", choices=("a", "i", "u", "phonation", "aligned")
    )
    step = parser.parse_args().step
    if step == "aligned":
        aligned()
    else:
        main(step)
