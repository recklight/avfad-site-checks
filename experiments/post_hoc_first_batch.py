"""Post hoc reading of the first batch; not in the frozen plan and no model is fitted.

It only re-reads the stored out-of-fold predictions, to see where the C1 gap comes from:
which normal speakers arm (i) gets right, and whether the score follows the ESSUA share of
each draw. Writes experiments/results/first_batch/post_hoc.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from voice_severity_bench.bootstrap import bootstrap_metric
from scipy.stats import spearmanr

RESULTS = Path(__file__).resolve().parent / "results" / "first_batch"
BOOTSTRAP_SEED = 20261106
N_RESAMPLES = 10_000


def main() -> None:
    predictions = pd.read_csv(RESULTS / "c1_predictions.csv.gz", dtype={"speaker": str})
    draws = pd.read_csv(RESULTS / "c1_draws.csv", dtype={"speaker": str})
    arms = pd.read_csv(RESULTS / "c1_arms.csv")
    predictions["correct"] = predictions["predicted"] == predictions["label"]
    places = draws.drop_duplicates("speaker").set_index("speaker")["place"]

    normals = predictions[(predictions["label"] == "normal") & (predictions["arm"] != "ii")].copy()
    normals["place"] = normals["speaker"].map(places)
    per_draw = normals.groupby(["arm", "place"])["correct"].mean().unstack()

    ii = predictions[predictions["arm"] == "ii"]
    ii_normals = ii[ii["label"] == "normal"]
    wide = ii.pivot(index="speaker", columns="repeat", values="predicted").sort_index()
    truth = ii.drop_duplicates("speaker").set_index("speaker")["label"].sort_index()
    interval = bootstrap_metric(
        truth.to_numpy(),
        wide.to_numpy(),
        wide.index.to_numpy(),
        metric="uar",
        n_resamples=N_RESAMPLES,
        seed=BOOTSTRAP_SEED,
    )

    share = draws.groupby("draw")["place"].apply(lambda p: float((p == "ESSUA").mean()))
    uar = arms[arms["arm"] != "ii"].assign(draw=lambda t: t["arm"].str[2:].astype(int))
    uar = uar.set_index("draw")["uar"].sort_index()
    rho = spearmanr(share.sort_index().to_numpy(), uar.to_numpy())

    report = {
        "note": "post hoc, not in the frozen plan, no model fitted",
        "arm_i_specificity_by_place": {
            place: {
                "mean_over_draws": float(per_draw[place].mean()),
                "min": float(per_draw[place].min()),
                "max": float(per_draw[place].max()),
                "draws_with_this_place": int(per_draw[place].notna().sum()),
            }
            for place in per_draw.columns
        },
        "arm_ii_specificity": float(ii_normals["correct"].mean()),
        "arm_ii_uar_interval": interval.as_dict(),
        "essua_share_against_uar_over_draws": {
            "spearman_rho": float(rho.statistic),
            "p": float(rho.pvalue),
            "n_draws": int(share.size),
        },
    }
    (RESULTS / "post_hoc.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    np.testing.assert_allclose(interval.estimate, arms.loc[arms["arm"] == "ii", "uar"].iloc[0])


if __name__ == "__main__":
    main()
