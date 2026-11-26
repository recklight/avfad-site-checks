"""How much of the detector's score age, sex and year cannot explain; see deviations.md.

The confirmatory tests conditioned only the non-phonation arm of SVD. This reads the null of
the detector itself inside the same strata, for the table of unconditioned against
conditioned scores. Exploratory, 200 permutations, only the null mean is read.

    .venv/Scripts/python.exe -I -B experiments/conditioned_detector.py --jobs 16
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from voice_severity_bench.permutation import permutation_test  # noqa: E402
from joblib import parallel_config  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_first_batch as batch  # noqa: E402
import run_svd as svd  # noqa: E402
from arms import write_json  # noqa: E402
from freeze import verify_freeze  # noqa: E402
from freeze_svd import verify_svd_freeze  # noqa: E402
from harness import (  # noqa: E402
    age_sex_cell,
    analysis_table,
    evaluate_arm,
    load_plan,
    make_config,
    to_feature_set,
)

OUT = HERE / "results" / "exploratory" / "conditioned.json"
N_PERMUTATIONS = 200


def row(name, table, cells, positive, plan, jobs) -> dict:
    config = make_config(plan, positive)
    arm = evaluate_arm(name, table, config)
    test = permutation_test(
        to_feature_set(table),
        config,
        n_permutations=N_PERMUTATIONS,
        strata=cells,
        seed=plan["statistics"]["seeds"]["permutation"],
        n_jobs=jobs,
    )
    null_mean = float(np.nanmean(test.null))
    return {
        **arm.row(),
        "null_mean": null_mean,
        "null_sd": float(np.nanstd(test.null, ddof=1)),
        "uar_minus_null": float(arm.uar - null_mean),
        "p_one_sided": float(test.p_value),
        "n_permutations": N_PERMUTATIONS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args()
    verify_freeze()
    verify_svd_freeze()
    plan = load_plan()
    report = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    started = time.perf_counter()
    iau = svd.load_svd_speakers("iau")
    a_n = svd.load_svd_speakers("a_n")
    speakers = batch.load_speakers()
    hsa = speakers[speakers["place"] == "HSA"]
    jobs = [
        ("svd_iau_full", iau, svd.load_table("iau", "full_norm"), svd.stratum, svd.POSITIVE),
        (
            "svd_iau_phonation",
            iau,
            svd.load_table("iau", "phonation_norm"),
            svd.stratum,
            svd.POSITIVE,
        ),
        ("svd_a_n", a_n, svd.load_table("a_n", "full_norm"), svd.stratum, svd.POSITIVE),
        (
            "avfad_hsa_full",
            hsa,
            batch.load_features("full_original"),
            age_sex_cell,
            batch.LABEL_POSITIVE,
        ),
    ]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        for name, meta, features, stratify, positive in jobs:
            if name in report:
                continue
            table = analysis_table(meta, features, "label")
            cells = np.asarray(stratify(table)).astype(str)
            report[name] = row(name, table, cells, positive, plan, args.jobs)
            write_json(OUT, report)
            print(
                name, round(report[name]["uar"], 4), round(report[name]["null_mean"], 4), flush=True
            )
    print(f"done in {time.perf_counter() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
