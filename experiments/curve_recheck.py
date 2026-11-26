"""The two handcrafted learning curves of 11-05 again, in a process that fits no HuBERT arm.

A check, not a new analysis (deviations.md, 2026-11-10): joblib reuses its workers, and a worker
that fitted a HuBERT arm keeps the HuBERT C grid. The 11-05 run had both kinds in one Parallel.
The original files under results/exploratory/ are left as they are.

    .venv/Scripts/python.exe -I -X utf8 -B experiments/curve_recheck.py --jobs 2
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
from voice_severity_bench.splitting import repeated_split_indices  # noqa: E402
from joblib import Parallel, delayed, parallel_config  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import exploratory_avfad as ex  # noqa: E402
import run_first_batch as batch  # noqa: E402
from arms import merge_features, write_json  # noqa: E402
from extract_handcrafted import FEATURE_COLUMNS  # noqa: E402
from harness import load_plan, make_config, match_cells  # noqa: E402

RESULTS = HERE / "results" / "curve_recheck"
ORIGINAL = HERE / "results" / "exploratory" / "learning_curve.csv"
SAME = 1e-9
# The 11-05 reading: the HSA curve extrapolated to 290 normals below this means more training
# data would not close the gap.
BELOW = 0.70


def run(jobs: int) -> None:
    plan = load_plan()
    seeds = plan["statistics"]["seeds"]
    speakers = batch.load_speakers()
    patients, hsa, normal = ex.batch_groups(speakers)
    essua, _ = match_cells(
        hsa, normal[normal["place"] == "ESSUA"], np.random.default_rng(seeds["c2_essua_draw"])
    )
    handcrafted = batch.load_features("full_original")
    config = make_config(plan, batch.LABEL_POSITIVE)
    tasks = []
    for curve, normals in (("hsa", hsa), ("essua", essua)):
        table = merge_features(pd.concat([patients, normals]), handcrafted, "label")
        X, y, groups = ex.design(table, FEATURE_COLUMNS)
        splits = repeated_split_indices(
            y, groups, task="classification", n_splits=5, n_repeats=10, seed=config.seed
        )
        for n in ex.CURVE_NORMALS:
            for repeat, folds in enumerate(splits):
                tasks.append((f"{curve}_handcrafted", X, y, groups, folds, n, repeat, config, None))
    with parallel_config(backend="loky", inner_max_num_threads=1):
        rows = Parallel(n_jobs=jobs)(delayed(ex._curve_repeat)(*task) for task in tasks)
    table = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / "learning_curve_handcrafted.csv", index=False)

    original = pd.read_csv(ORIGINAL)
    key = ["curve", "requested", "repeat"]
    for frame in (table, original):
        frame["requested"] = frame["requested"].astype(str)
    both = table.merge(original, on=key, suffixes=("_new", "_old"), validate="one_to_one")
    both["uar_change"] = both["uar_new"] - both["uar_old"]
    report: dict = {"rows": len(both)}
    for curve, part in both.groupby("curve"):
        grouped = part.groupby("requested", sort=False).agg(
            normals=("normals_in_training_new", "mean"),
            uar_new=("uar_new", "mean"),
            uar_old=("uar_old", "mean"),
        )
        fitted = ex.fit_curve(
            grouped["normals"].to_numpy(), grouped["uar_new"].to_numpy(), ex.CURVE_TARGET
        )
        report[curve] = {
            "max_abs_uar_change": float(part["uar_change"].abs().max()),
            "same": bool(part["uar_change"].abs().max() <= SAME),
            "points": grouped.reset_index().to_dict("records"),
            **fitted,
        }
    hsa_fit = report["hsa_handcrafted"]
    report["reading_hsa"] = (
        "more training data would not close the gap"
        if max(hsa_fit["inverse_root_at_target"], hsa_fit["log_at_target"]) < BELOW
        else "the 11-05 reading does not hold"
    )
    write_json(RESULTS / "summary.json", report)
    print(
        {
            c: (r["same"], round(r["max_abs_uar_change"], 4))
            for c, r in report.items()
            if isinstance(r, dict)
        }
    )
    print(report["reading_hsa"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=2)
    run(parser.parse_args().jobs)


if __name__ == "__main__":
    main()
