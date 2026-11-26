"""C1 on the phonation spans only, for the handcrafted and the HuBERT detector.

Exploratory. Definitions and readings are in experiments/deviations.md (2026-11-10), written
before this run. Speakers, draws, folds and detectors are those of C1; only the features change.
The gap is compared with the full-file C1 through the same bootstrap resamples.

    run       102 arms (51 per detector)
    collect   writes results/phonation_c1/summary.json

    .venv/Scripts/python.exe -I -X utf8 -B experiments/phonation_c1.py run --jobs 10
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from voice_severity_bench import models as vsb_models  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_first_batch as batch  # noqa: E402
import vowel_fusion as vf  # noqa: E402
from arms import evaluate_layers, load_hubert, merge_features, write_json  # noqa: E402
from freeze import verify_freeze  # noqa: E402
from harness import analysis_table, evaluate_arm, load_plan, make_config, predictions_table  # noqa: E402
from sensitivity_avfad import CHOSEN_LAYERS  # noqa: E402

RESULTS = HERE / "results" / "phonation_c1"
FULL = {
    "handcrafted": HERE / "results" / "first_batch" / "c1_predictions.csv.gz",
    "hubert": HERE / "results" / "sensitivity" / "hubert" / "c1_predictions.csv.gz",
}
DETECTORS = ("handcrafted", "hubert")
LOGISTIC = ("classification", "logistic")
# The handcrafted grid, taken before any HuBERT fit in this process replaces it. joblib reuses
# its workers, so a worker that fitted a HuBERT arm keeps the HuBERT grid for whatever comes next.
HANDCRAFTED_GRID = {k: list(v) for k, v in vsb_models.SEARCH_SPACES[LOGISTIC].items()}
# Readings, written before the run (deviations.md, 2026-11-10).
GAP_POINTS = 0.10
CHANGE_POINTS = 0.05


def _fit(detector: str, name: str, members: pd.DataFrame, config):
    start = time.perf_counter()
    if detector == "hubert":
        table = merge_features(members, load_hubert("phonation", CHOSEN_LAYERS), "label")
        arm = evaluate_layers(name, table, CHOSEN_LAYERS, config)
    else:
        vsb_models.SEARCH_SPACES[LOGISTIC] = {k: list(v) for k, v in HANDCRAFTED_GRID.items()}
        table = analysis_table(members, batch.load_features("phonation_norm"), "label")
        arm = evaluate_arm(name, table, config)
    return detector, arm, time.perf_counter() - start


def run(jobs: int, detectors: tuple[str, ...] = DETECTORS) -> None:
    verify_freeze()
    plan = load_plan()
    config = make_config(plan, batch.LABEL_POSITIVE)
    sets = [(n, m) for n, m in vf.arm_sets(plan) if not n.startswith("shuffled")]
    # HuBERT first: its arms are the slow ones.
    todo = [(d, n, m) for d in reversed(detectors) for n, m in sets]
    RESULTS.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    done = Parallel(n_jobs=jobs, return_as="generator_unordered")(
        delayed(_fit)(d, n, m, config) for d, n, m in todo
    )
    arms: dict[str, list] = {d: [] for d in detectors}
    for count, (detector, arm, seconds) in enumerate(done, start=1):
        arms[detector].append(arm)
        if count % 10 == 0 or count == len(todo):
            print(
                f"{count} of {len(todo)} after {time.perf_counter() - start:.0f} s "
                f"(last {detector} {seconds:.0f} s)",
                flush=True,
            )
    for detector, part in arms.items():
        part.sort(key=lambda a: a.name)
        predictions_table(part).to_csv(RESULTS / f"c1_predictions_{detector}.csv.gz", index=False)
        pd.DataFrame([a.row() for a in part]).to_csv(
            RESULTS / f"c1_arms_{detector}.csv", index=False
        )


def gap_reading(gap: dict) -> str:
    if gap["difference"] >= GAP_POINTS and gap["ci_low"] > 0:
        return "the gap stays on phonation only: not just length and silence"
    if gap["ci_low"] <= 0:
        return "the gap is gone: length and silence are the shortcut"
    return "the gap shrinks"


def change_reading(change: dict) -> str:
    if change["ci_high"] < 0 and change["difference"] <= -CHANGE_POINTS:
        return "taking out the silence costs the gap"
    if change["ci_low"] <= 0 <= change["ci_high"]:
        return "the gap does not change"
    return "as measured"


def gap_with_samples(table: pd.DataFrame, plan: dict) -> tuple[dict, np.ndarray, dict]:
    names = sorted(set(table["arm"]))
    arms = {name: vf.arm_from_table(table, name) for name in names}
    draws = [arms[f"i_{i:02d}"] for i in range(plan["confirmatory_family"]["tests"]["C1"]["draws"])]
    gap = batch.c1_bootstrap(
        draws, arms["ii"], vf.N_RESAMPLES, plan["statistics"]["seeds"]["bootstrap"]
    )
    samples = gap.pop("samples")
    extra = {
        "ii": {"uar": arms["ii"].uar, "roc_auc": arms["ii"].roc_auc},
        "i_uar_mean": float(np.mean([a.uar for a in draws])),
        "i_uar_range": [min(a.uar for a in draws), max(a.uar for a in draws)],
    }
    return gap, samples, extra


def collect() -> None:
    plan = load_plan()
    summary: dict = {}
    for detector in DETECTORS:
        phon = pd.read_csv(RESULTS / f"c1_predictions_{detector}.csv.gz", dtype={"speaker": str})
        full = pd.read_csv(FULL[detector], dtype={"speaker": str})
        gap, samples, extra = gap_with_samples(phon, plan)
        full_gap, full_samples, _ = gap_with_samples(full, plan)
        diff = samples - full_samples
        finite = diff[np.isfinite(diff)]
        low, high = np.quantile(finite, [0.025, 0.975])
        change = {
            "difference": gap["difference"] - full_gap["difference"],
            "ci_low": float(low),
            "ci_high": float(high),
        }
        summary[detector] = {
            **extra,
            "gap_i_minus_ii": gap,
            "full_gap": full_gap["difference"],
            "phonation_minus_full": change,
            "readings": {"gap": gap_reading(gap), "change": change_reading(change)},
        }
        print(detector, summary[detector]["readings"], round(gap["difference"], 4), flush=True)
    write_json(RESULTS / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("run", "collect"))
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--detector", choices=DETECTORS, help="run one detector only")
    args = parser.parse_args()
    if args.step == "run":
        run(args.jobs, DETECTORS if args.detector is None else (args.detector,))
    else:
        collect()


if __name__ == "__main__":
    main()
