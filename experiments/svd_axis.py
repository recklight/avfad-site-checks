"""On SVD, whether the AVFAD-trained scores follow the AVFAD place direction, inside each label.

Exploratory and descriptive; nothing is trained. Definitions and readings are in
experiments/deviations.md (2026-11-10), written before this ran. The directions come from all
356 AVFAD normals (site_erasure.fitted_directions); a larger projection is more ESSUA-like.

    .venv/Scripts/python.exe -I -X utf8 -B experiments/svd_axis.py
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cross_database as cd  # noqa: E402
import diagnosis_breakdown as db  # noqa: E402
import outside_view as ov  # noqa: E402
import site_erasure as se  # noqa: E402
import within_place_axis as wa  # noqa: E402
from arms import load_hubert, write_json  # noqa: E402
from freeze import verify_freeze  # noqa: E402
from sensitivity_avfad import CHOSEN_LAYERS  # noqa: E402

RESULTS = HERE / "results" / "svd_axis"
SEED = 20261147
N_RESAMPLES = 10_000
ALIGNED = HERE / "results" / "aligned_transfer" / "aligned_scores_hubert_location.csv.gz"


def svd_scores() -> pd.DataFrame:
    """One column per AVFAD training set, in the SVD row order of the stored scores."""
    plain = pd.read_csv(cd.RESULTS / "scores_hubert.csv.gz")
    draws = [c for c in plain.columns if c.startswith("i_")]
    out = plain[["file_id", "label", "ii", "all"]].assign(i=plain[draws].mean(axis=1))
    moved = pd.read_csv(ALIGNED)
    if not moved["file_id"].equals(plain["file_id"]):
        raise RuntimeError("the aligned SVD rows are not in the stored order")
    return out.assign(all_aligned=moved["all"])


def main() -> None:
    verify_freeze()
    directions = se.fitted_directions(load_hubert("full", CHOSEN_LAYERS))
    scores = svd_scores()
    test = scores.merge(
        load_hubert("full", CHOSEN_LAYERS, material="svd_a_n"),
        on="file_id",
        how="left",
        validate="one_to_one",
    )
    blocks = se.layer_blocks(test)
    rng = np.random.default_rng(SEED)
    groups = {"svd_patients": "pathological", "svd_normals": "normal"}
    summary: dict = {"speakers": {}}
    for group, label in groups.items():
        rows = (test["label"] == label).to_numpy()
        summary["speakers"][group] = int(rows.sum())
        weights = db.resample_counts(int(rows.sum()), N_RESAMPLES, rng)
        index = pd.RangeIndex(int(rows.sum()))
        named = {
            n: pd.Series(test.loc[rows, n].to_numpy(), index=index)
            for n in ("ii", "i", "all", "all_aligned")
        }
        layers = {}
        for layer in CHOSEN_LAYERS:
            projection = pd.Series(blocks[layer][rows] @ directions[layer], index=index)
            got = wa.correlations(projection, named, weights)
            record = {f"rho_{n}": ov.interval(v[0], v[1], "rho") for n, v in got.items()}
            for name in ("i", "all", "all_aligned"):
                diff = ov.interval(got[name][0] - got["ii"][0], got[name][1] - got["ii"][1])
                record[f"{name}_minus_ii"] = diff
                record[f"{name}_reading"] = wa.follow_reading(diff).replace(
                    "inside one place", "on SVD"
                )
            layers[str(layer)] = record
        follows = sum(v["i_reading"].startswith("on SVD") for v in layers.values())
        summary[group] = {
            "layers": layers,
            "overall": (
                "on SVD, (i) follows the AVFAD place direction too"
                if follows >= wa.LAYERS_FOR_AXIS
                else "as measured"
            ),
        }
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / "summary.json", summary)
    for group in groups:
        print(group, summary[group]["overall"])
        for layer, v in summary[group]["layers"].items():
            rhos = {k[4:]: round(x["rho"], 3) for k, x in v.items() if k.startswith("rho_")}
            print("  ", layer, rhos, round(v["i_minus_ii"]["difference"], 3), v["i_reading"])


if __name__ == "__main__":
    main()
