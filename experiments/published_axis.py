"""Inside HSA, whether the published per-file decisions of Ribas 2023 follow the place direction.

Exploratory and descriptive; nothing is trained. Definitions and readings are in
experiments/deviations.md (2026-11-10), written before this ran. The predictions are the local
copy fetched by published_splits.py; only summary numbers are written here. The projections are
the cross-fitted ones of patient_position.py; a larger projection is more ESSUA-like.

    .venv/Scripts/python.exe -I -X utf8 -B experiments/published_axis.py
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

import diagnosis_breakdown as db  # noqa: E402
import published_splits as ps  # noqa: E402
import within_place_axis as wa  # noqa: E402
from arms import write_json  # noqa: E402
from freeze import verify_freeze  # noqa: E402
from sensitivity_avfad import CHOSEN_LAYERS  # noqa: E402

RESULTS = HERE / "results" / "published_axis"
SEED = 20261148
N_RESAMPLES = 10_000
FEWEST = 5
# Reading, written before this ran (deviations.md, 2026-11-10).
FOLLOWS = 0.60


def decisions(material: str) -> pd.Series:
    """Their decision per HSA speaker for one material: True where called normal."""
    lists = pd.read_csv(ps.SOURCE / "ribas_lists.csv", dtype=str).astype({"fold": int})
    predictions = pd.read_csv(ps.SOURCE / "ribas_predictions.csv", dtype=str).astype({"fold": int})
    rows = predictions[predictions["material"] == material]
    listed = lists[(lists["material"] == material) & (lists["side"] == "test")]
    joined = rows.merge(
        listed[["fold", "stem", "speaker"]], on=["fold", "stem"], how="left", validate="one_to_one"
    )
    if joined["speaker"].isna().any():
        raise ValueError(f"{material}: a predicted file is not on the test list of its fold")
    speakers = ps.load_speakers()
    joined = joined.merge(
        speakers[["speaker", "place"]], on="speaker", how="left", validate="many_to_one"
    )
    hsa = joined[joined["place"] == "HSA"]
    if material == "a_n" and hsa["speaker"].duplicated().any():
        raise ValueError(f"{material}: a speaker has more than one test file")
    # Sentences have several files per speaker (found while running): called normal when at
    # least half of that speaker's files are.
    share = (hsa["predicted"] == "normal").groupby(hsa["speaker"]).mean()
    return share >= 0.5


def auc_called_normal(projection: np.ndarray, called: np.ndarray, weights: np.ndarray):
    """P(projection of a speaker called normal > one called pathological), with resamples."""
    kernel = db.comparisons(projection[called], projection[~called])
    point = float(kernel.mean())
    boot = db.weighted_auc(kernel, weights[:, called], weights[:, ~called])
    return point, boot


def reading(record: dict) -> str:
    if record["auc"] >= FOLLOWS and record["ci_low"] > 0.5:
        return "the published system follows the place direction inside one place too"
    if record["ci_low"] <= 0.5 <= record["ci_high"]:
        return "not visible"
    return "as measured"


def main() -> None:
    verify_freeze()
    hsa, sick = wa.projections("a")
    groups = {"hsa_normals": hsa, "hsa_patients": sick}
    rng = np.random.default_rng(SEED)
    summary: dict = {}
    for material in ps.MATERIALS:
        called = decisions(material)
        part: dict = {}
        for group, proj in groups.items():
            ids = proj[CHOSEN_LAYERS[0]].index
            missing = ids.difference(called.index)
            if len(missing):
                raise ValueError(f"{material} {group}: {len(missing)} speakers without a decision")
            flags = called.loc[ids].to_numpy()
            weights = db.resample_counts(len(ids), N_RESAMPLES, rng)
            record: dict = {"speakers": len(ids), "called_normal": int(flags.sum())}
            if flags.sum() >= FEWEST and (~flags).sum() >= FEWEST:
                layers = {}
                for layer in CHOSEN_LAYERS:
                    point, boot = auc_called_normal(proj[layer].to_numpy(), flags, weights)
                    # A resample with nobody called normal has no AUC; it is left out.
                    low, high = np.nanpercentile(boot, [2.5, 97.5])
                    one = {
                        "auc": point,
                        "ci_low": float(low),
                        "ci_high": float(high),
                        "n_undefined": int(np.isnan(boot).sum()),
                    }
                    one["reading"] = reading(one)
                    layers[str(layer)] = one
                record["layers"] = layers
                record["overall"] = wa.majority([v["reading"] for v in layers.values()])
            part[group] = record
        summary[material] = part
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / "summary.json", summary)
    for material, part in summary.items():
        for group, record in part.items():
            print(
                material, group, record["speakers"], record["called_normal"], record.get("overall")
            )
            for layer, v in record.get("layers", {}).items():
                print(
                    "  ", layer, round(v["auc"], 3), round(v["ci_low"], 3), round(v["ci_high"], 3)
                )


if __name__ == "__main__":
    main()
