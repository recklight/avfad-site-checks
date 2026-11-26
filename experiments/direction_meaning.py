"""Which recording measures the place direction of each HuBERT layer goes with. Description only.

Exploratory. Definitions and the reading are in experiments/deviations.md (2026-11-10), written
before any projection was set against the recording measures. Nothing is trained.

    .venv/Scripts/python.exe -I -X utf8 -B experiments/direction_meaning.py
    .venv/Scripts/python.exe -I -X utf8 -B experiments/direction_meaning.py svd

``svd`` (added later): the same directions projected on SVD a_n, against SVD's labels.
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_first_batch as batch  # noqa: E402
import site_erasure as se  # noqa: E402
from arms import load_hubert, write_json  # noqa: E402
from section8_rest import M_TECH, avfad_stats, m_tech_table  # noqa: E402
from sensitivity_avfad import CHOSEN_LAYERS  # noqa: E402

RESULTS = HERE / "results" / "direction_meaning"
# Reading, written before the run (deviations.md, 2026-11-10).
FOLLOWS = 0.30
FLOOR = "floor_db"
SVD_EDGES = (0.40, 0.60)


def correlations(projection: pd.Series, table: pd.DataFrame, columns: list[str]) -> dict:
    out = {}
    for c in columns:
        rho = spearmanr(projection.to_numpy(), table[c].to_numpy()).statistic
        out[c] = float(rho)
    return out


def reading(within: dict[str, dict[str, dict[str, float]]]) -> dict:
    """``within[group][layer][column]``; groups are the two single-place sets."""
    floor = [abs(v[FLOOR]) for g in within.values() for v in g.values() if FLOOR in v]
    others = sorted(
        {
            c
            for g in within.values()
            for v in g.values()
            for c, r in v.items()
            if c != FLOOR and abs(r) >= FOLLOWS
        }
    )
    if floor and max(floor) >= FOLLOWS:
        text = "the place direction follows the noise floor within one place"
    elif others:
        text = "the place direction follows other measures within one place"
    else:
        text = "little relation to these measures within one place"
    return {"reading": text, "other_columns_at_or_above": others}


def svd_reading(aucs: list[float]) -> str:
    low, high = SVD_EDGES
    if any(a >= high or a <= low for a in aucs):
        return "the place direction separates pathology on SVD too"
    return "the place direction has little to do with pathology on SVD"


def run_svd() -> None:
    import cross_database as cd
    import run_svd as svd
    from within_strata_auc import mann_whitney_auc, pooled_auc

    directions = se.fitted_directions(load_hubert("full", CHOSEN_LAYERS))
    test = cd.svd_test_set().merge(
        load_hubert("full", CHOSEN_LAYERS, material="svd_a_n"),
        on="file_id",
        how="left",
        validate="one_to_one",
    )
    sick = (test["label"] == "pathological").to_numpy()
    cells = svd.stratum(test)
    blocks = se.layer_blocks(test)
    report: dict = {"speakers": len(test), "pathological": int(sick.sum()), "layers": {}}
    for layer in CHOSEN_LAYERS:
        toward_hsa = -(blocks[layer] @ directions[layer])
        report["layers"][str(layer)] = {
            "auc": float(mann_whitney_auc(toward_hsa[sick], toward_hsa[~sick])),
            "within_strata_auc": float(pooled_auc(toward_hsa, sick, cells)),
        }
    report["reading"] = svd_reading([r["auc"] for r in report["layers"].values()])
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / "svd.json", report)
    print(report["reading"], {k: round(v["auc"], 3) for k, v in report["layers"].items()})


def run_avfad() -> None:
    speakers = batch.load_speakers()
    speakers = speakers[speakers["place"].isin(["ESSUA", "HSA"])]
    table = m_tech_table(speakers, batch.load_features("phonation_norm"), avfad_stats())
    features = load_hubert("full", CHOSEN_LAYERS)
    table = table.merge(features, on="file_id", how="left", validate="one_to_one")
    normals = table[table["label"] == "normal"]
    essua = (normals["place"] == "ESSUA").to_numpy()
    directions = se.place_directions(se.layer_blocks(normals), essua, ~essua)
    groups = {
        "all_normals": table["label"] == "normal",
        "essua_normals": (table["label"] == "normal") & (table["place"] == "ESSUA"),
        "hsa_everyone": table["place"] == "HSA",
    }
    summary: dict = {"layers": {}}
    within: dict = {"essua_normals": {}, "hsa_everyone": {}}
    blocks = se.layer_blocks(table)
    for layer in CHOSEN_LAYERS:
        projection = pd.Series(blocks[layer] @ directions[layer], index=table.index)
        record = {}
        for name, rows in groups.items():
            part = table[rows]
            columns = [c for c in M_TECH if part[c].nunique() > 1]
            record[name] = {
                "speakers": int(rows.sum()),
                "spearman": correlations(projection[rows], part, columns),
            }
            if name in within:
                within[name][str(layer)] = record[name]["spearman"]
        summary["layers"][str(layer)] = record
    summary |= reading(within)
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / "summary.json", summary)
    print(summary["reading"], summary["other_columns_at_or_above"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", nargs="?", choices=("avfad", "svd"), default="avfad")
    if parser.parse_args().what == "svd":
        run_svd()
    else:
        run_avfad()


if __name__ == "__main__":
    main()
