"""C1 as a placebo on SVD a_n, where every speaker comes from one place.

Exploratory. Definitions and readings are in experiments/deviations.md (2026-11-10), written
before this run. Each placebo split takes 329 patients and 82 "HSA" normals at random; (ii) pairs
the patients with those 82, (i) with 82 normals drawn from all SVD normals on the age by sex
cells of the 82, 50 times. Detector, folds and bootstrap are those of C1.

    run       --detector handcrafted|hubert --split 0
    collect   writes results/placebo_c1/summary.json

    .venv/Scripts/python.exe -I -X utf8 -B experiments/placebo_c1.py run --detector handcrafted \
        --split 0 --jobs 12
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "features"))

import run_first_batch as batch  # noqa: E402
import run_svd as svd  # noqa: E402
from arms import evaluate_layers, load_hubert, merge_features, write_json  # noqa: E402
from freeze import verify_freeze  # noqa: E402
from harness import analysis_table, evaluate_arm, load_plan, make_config, match_cells  # noqa: E402
from sensitivity_avfad import CHOSEN_LAYERS  # noqa: E402

RESULTS = HERE / "results" / "placebo_c1"
SEED = 20261146
N_PATIENTS = 329
N_NORMALS = 82
DETECTORS = ("handcrafted", "hubert")
# Readings, written before the run (deviations.md, 2026-11-10).
POINTS = 0.03
# Added after the first three handcrafted splits (deviations.md, 2026-11-10): splits 3-22 give
# the spread of the gap without place.
FIRST_SPLITS = 3
ALL_SPLITS = 23
AVFAD_ALIGNED = {
    "location": ("harmonize_c1", "handcrafted", "location"),
    "scale": ("harmonize_c1", "handcrafted", "scale"),
    "coral": ("coral_c1", "logistic", None),
}


def svd_meta() -> pd.DataFrame:
    meta = svd.load_svd_speakers("a_n")
    return meta.assign(speaker="svd" + meta["speaker"].astype(str)).reset_index(drop=True)


def placebo_sets(meta: pd.DataFrame, split: int, draws: int, draw_seed: int) -> list:
    """(ii) and the draws of (i) for one placebo split, as (name, members) pairs."""
    rng = np.random.default_rng(SEED + split)
    patients = meta[meta["label"] == "pathological"].sort_values("speaker")
    normals = meta[meta["label"] == "normal"].sort_values("speaker")
    patients = patients.iloc[np.sort(rng.choice(len(patients), N_PATIENTS, replace=False))]
    fake = normals.iloc[np.sort(rng.choice(len(normals), N_NORMALS, replace=False))]
    draw_rng = np.random.default_rng(draw_seed)
    sets = [("ii", pd.concat([patients, fake]))]
    for index in range(draws):
        drawn, cells = match_cells(fake, normals, draw_rng)
        if (cells["found"] < cells["wanted"]).any():
            raise RuntimeError("a cell of the SVD normals came up short")
        sets.append((f"i_{index:02d}", pd.concat([patients, drawn])))
    return sets


def _arm(detector: str, name: str, members: pd.DataFrame, config):
    if detector == "hubert":
        features = load_hubert("full", CHOSEN_LAYERS, material="svd_a_n")
        return evaluate_layers(
            name, merge_features(members, features, "label"), CHOSEN_LAYERS, config
        )
    features = pd.read_csv(ROOT / "features" / "out" / "svd_a_n_full_norm.csv")
    return evaluate_arm(name, analysis_table(members, features, "label"), config)


def path_of(detector: str, split: int) -> Path:
    return RESULTS / f"arms_{detector}_split{split}.pkl"


def run(jobs: int, detector: str, split: int) -> None:
    verify_freeze()
    plan = load_plan()
    spec = plan["confirmatory_family"]["tests"]["C1"]
    config = make_config(plan, batch.LABEL_POSITIVE)
    sets = placebo_sets(svd_meta(), split, spec["draws"], plan["statistics"]["seeds"]["c1_draws"])
    start = time.perf_counter()
    arms = Parallel(n_jobs=jobs)(delayed(_arm)(detector, n, m, config) for n, m in sets)
    print(f"{detector} split {split}: {len(arms)} arms in {time.perf_counter() - start:.0f} s")
    RESULTS.mkdir(parents=True, exist_ok=True)
    with path_of(detector, split).open("wb") as handle:
        pickle.dump(arms, handle)


def reading(gap: dict) -> str:
    if gap["ci_low"] <= 0 <= gap["ci_high"]:
        return "no place, no gap"
    if gap["difference"] >= POINTS and gap["ci_low"] > 0:
        return "the design makes a gap by itself"
    if gap["difference"] <= -POINTS and gap["ci_high"] < 0:
        return "the design puts (i) lower by itself"
    return "as measured"


def aligned_gaps() -> dict[str, float]:
    import json

    out = {}
    for kind, (folder, detector, key) in AVFAD_ALIGNED.items():
        record = json.loads((HERE / "results" / folder / "summary.json").read_text("utf-8"))
        part = record[detector][key] if key else record[detector]
        out[kind] = part["gap_i_minus_ii"]["difference"]
    first = json.loads((HERE / "results" / "first_batch" / "summary.json").read_text("utf-8"))
    out["unaligned_c1"] = first["c1"]["difference"]
    return out


def spread_reading(values: list[float], gaps: list[float]) -> str:
    low, high = min(values), max(values)
    if all(low <= g <= high for g in gaps):
        return "what is left after the alignment is within the spread without place"
    if all(g > high for g in gaps):
        return "what is left is larger than the spread without place"
    return "as measured"


def distribution(values: list[float]) -> dict:
    ordered = sorted(values)
    avfad = aligned_gaps()
    aligned = [avfad[k] for k in AVFAD_ALIGNED]
    return {
        "splits": len(values),
        "median": float(np.median(values)),
        "min": ordered[0],
        "max": ordered[-1],
        "third_smallest": ordered[2],
        "third_largest": ordered[-3],
        "avfad": avfad,
        "reading": spread_reading(values, aligned),
        "unaligned_c1_above_max": bool(avfad["unaligned_c1"] > ordered[-1]),
    }


def collect() -> None:
    plan = load_plan()
    spec = plan["confirmatory_family"]["tests"]["C1"]
    summary: dict = {}
    for detector in DETECTORS:
        for split in range(ALL_SPLITS):
            path = path_of(detector, split)
            if not path.exists():
                continue
            with path.open("rb") as handle:
                arms = pickle.load(handle)
            arm_ii, arms_i = arms[0], arms[1:]
            gap = batch.c1_bootstrap(
                arms_i, arm_ii, spec["test"]["resamples"], plan["statistics"]["seeds"]["bootstrap"]
            )
            gap.pop("samples")
            summary.setdefault(detector, {})[str(split)] = {
                "gap_i_minus_ii": gap,
                "reading": reading(gap),
                "uar_ii": arm_ii.uar,
                "uar_i_mean": float(np.mean([a.uar for a in arms_i])),
                "uar_i_range": [min(a.uar for a in arms_i), max(a.uar for a in arms_i)],
                "auc_ii": arm_ii.roc_auc,
                "auc_i_mean": float(np.mean([a.roc_auc for a in arms_i])),
            }
    # The reading written first: handcrafted splits 0-2 and HuBERT split 0.
    first_of = {"handcrafted": FIRST_SPLITS, "hubert": 1}
    first = {
        d: {k: v for k, v in part.items() if int(k) < first_of[d]} for d, part in summary.items()
    }
    readings = [v["reading"] for part in first.values() for v in part.values()]
    complete = len(first.get("handcrafted", {})) == FIRST_SPLITS and "hubert" in summary
    summary["overall"] = (
        "the C1 design makes no gap by itself"
        if complete and all(r == "no place, no gap" for r in readings)
        else "as measured"
    )
    hand = summary.get("handcrafted", {})
    if len(hand) == ALL_SPLITS:
        summary["distribution"] = distribution(
            [v["gap_i_minus_ii"]["difference"] for v in hand.values()]
        )
    hub = summary.get("hubert", {})
    if len(hub) > 1:
        gaps = [v["gap_i_minus_ii"]["difference"] for v in hub.values()]
        summary["hubert_range"] = {"splits": len(gaps), "min": min(gaps), "max": max(gaps)}
    write_json(RESULTS / "summary.json", summary)
    for detector in DETECTORS:
        for split, v in summary.get(detector, {}).items():
            g = v["gap_i_minus_ii"]
            print(
                detector,
                split,
                round(v["uar_ii"], 3),
                round(v["uar_i_mean"], 3),
                round(g["difference"] * 100, 1),
                [round(g["ci_low"] * 100, 1), round(g["ci_high"] * 100, 1)],
                v["reading"],
            )
    print(summary["overall"])
    if "distribution" in summary:
        print(summary["distribution"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("run", "collect"))
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--detector", choices=DETECTORS, default="handcrafted")
    parser.add_argument("--split", type=int, choices=range(ALL_SPLITS), default=0)
    args = parser.parse_args()
    if args.step == "run":
        run(args.jobs, args.detector, args.split)
    else:
        collect()


if __name__ == "__main__":
    main()
