"""Where the HSA patients sit on the place direction, which only the normals define.

Exploratory and descriptive; no model is fitted. Definitions and readings are in
experiments/deviations.md (2026-11-10), written before this ran. The direction is cross-fitted:
every normal speaker is projected on a direction estimated without them.

    .venv/Scripts/python.exe -I -X utf8 -B experiments/patient_position.py
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import diagnosis_breakdown as db  # noqa: E402
import run_first_batch as batch  # noqa: E402
import site_erasure as se  # noqa: E402
from arms import load_hubert, write_json  # noqa: E402
from freeze import verify_freeze  # noqa: E402
from sensitivity_avfad import CHOSEN_LAYERS  # noqa: E402

RESULTS = HERE / "results" / "patient_position"
SEED = 20261140
N_PARTS = 5
N_RESAMPLES = 10_000
# Readings, written before this ran (deviations.md, 2026-11-10).
NEAR = 0.25
NEAR_BAND = 0.5
LAYERS_FOR_PLACE = 4


def parts_within_place(place: np.ndarray, n_parts: int, rng: np.random.Generator) -> np.ndarray:
    """Part index of every normal, dealt out at random inside each place."""
    out = np.empty(len(place), dtype=int)
    for value in np.unique(place):
        rows = np.flatnonzero(place == value)
        out[rng.permutation(rows)] = np.arange(len(rows)) % n_parts
    return out


def cross_fitted(
    normal_blocks: dict[int, np.ndarray],
    patient_blocks: dict[int, np.ndarray],
    essua: np.ndarray,
    parts: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Out-of-part projections of the normals and part-averaged projections of the patients."""
    normal_out = {layer: np.full(len(essua), np.nan) for layer in normal_blocks}
    patient_out = {
        layer: np.zeros(len(next(iter(patient_blocks.values())))) for layer in normal_blocks
    }
    for part in np.unique(parts):
        fit = parts != part
        fitted = {layer: block[fit] for layer, block in normal_blocks.items()}
        directions = se.place_directions(fitted, essua[fit], ~essua[fit])
        for layer, direction in directions.items():
            normal_out[layer][~fit] = normal_blocks[layer][~fit] @ direction
            patient_out[layer] += patient_blocks[layer] @ direction / len(np.unique(parts))
    return normal_out, patient_out


def position(patients: np.ndarray, hsa: np.ndarray, essua: np.ndarray) -> float:
    return float((patients.mean() - hsa.mean()) / (essua.mean() - hsa.mean()))


def position_interval(
    patients: np.ndarray, hsa: np.ndarray, essua: np.ndarray, n: int, seed: int
) -> dict:
    rng = np.random.default_rng(seed)
    samples = np.empty(n)
    for b in range(n):
        samples[b] = position(
            patients[rng.integers(0, len(patients), len(patients))],
            hsa[rng.integers(0, len(hsa), len(hsa))],
            essua[rng.integers(0, len(essua), len(essua))],
        )
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "position": position(patients, hsa, essua),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def reading(p: dict) -> str:
    if abs(p["position"]) <= NEAR and -NEAR_BAND <= p["ci_low"] and p["ci_high"] <= NEAR_BAND:
        return "the patients sit with the HSA normals: the direction is place"
    if p["position"] <= -NEAR and p["ci_high"] < 0:
        return "the patients sit further out: the direction runs with pathology"
    if p["position"] >= NEAR and p["ci_low"] > 0:
        return "the patients lean towards ESSUA"
    return "as measured"


def main() -> None:
    verify_freeze()
    speakers = batch.load_speakers()
    features = load_hubert("full", CHOSEN_LAYERS)
    normals = se.normal_speakers().merge(features, on="file_id", how="left", validate="one_to_one")
    patients = speakers[(speakers["place"] == "HSA") & (speakers["label"] == "pathological")]
    patients = patients.sort_values("file_id").reset_index(drop=True)
    patients = patients.merge(features, on="file_id", how="left", validate="one_to_one")
    essua = (normals["place"] == "ESSUA").to_numpy()
    parts = parts_within_place(normals["place"].to_numpy(), N_PARTS, np.random.default_rng(SEED))
    normal_proj, patient_proj = cross_fitted(
        se.layer_blocks(normals), se.layer_blocks(patients), essua, parts
    )
    groups = db.load_speakers()["group"]
    chapter = groups.loc[patients["speaker"].astype(str)].to_numpy()
    layers = {}
    for layer in CHOSEN_LAYERS:
        hsa_n, essua_n = normal_proj[layer][~essua], normal_proj[layer][essua]
        p = position_interval(patient_proj[layer], hsa_n, essua_n, N_RESAMPLES, SEED)
        p["reading"] = reading(p)
        p["by_chapter"] = {
            g: position(patient_proj[layer][chapter == g], hsa_n, essua_n)
            for g in sorted(set(chapter))
        }
        p["normals_auc_out_of_part"] = float(
            np.mean(essua_n[:, None] > hsa_n[None, :])
            + 0.5 * np.mean(essua_n[:, None] == hsa_n[None, :])
        )
        layers[str(layer)] = p
    place_layers = sum(v["reading"].startswith("the patients sit with") for v in layers.values())
    summary = {
        "speakers": {
            "hsa_patients": len(patients),
            "hsa_normals": int((~essua).sum()),
            "essua_normals": int(essua.sum()),
        },
        "chapter_counts": {g: int((chapter == g).sum()) for g in sorted(set(chapter))},
        "layers": layers,
        "overall": "the direction is place" if place_layers >= LAYERS_FOR_PLACE else "as measured",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / "summary.json", summary)
    for layer, v in layers.items():
        print(
            layer,
            round(v["position"], 3),
            round(v["ci_low"], 3),
            round(v["ci_high"], 3),
            v["reading"],
        )
    print(summary["overall"])


if __name__ == "__main__":
    main()
