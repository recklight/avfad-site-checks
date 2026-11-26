"""C1 with outer folds grouped by recording day, so no day is in both training and testing.

Exploratory. Definitions and readings are in experiments/deviations.md (2026-11-10), written
before this ran. Speakers, draws, detectors and grids are those of C1; only the outer folds
change. ESSUA's normals were recorded on 22 days, so speaker-grouped folds put people of the same
day on both sides.

    run handcrafted|hubert    51 arms x 10 repeats
    collect                   writes results/c1_by_day/summary.json

    .venv/Scripts/python.exe -I -X utf8 -B experiments/c1_by_day.py run handcrafted --jobs 12
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
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "features"))

import phonation_c1 as pc  # noqa: E402
import published_splits as ps  # noqa: E402
import run_first_batch as batch  # noqa: E402
import vowel_fusion as vf  # noqa: E402
from arms import layer_columns, load_hubert, merge_features, write_json  # noqa: E402
from extract_handcrafted import FEATURE_COLUMNS  # noqa: E402
from freeze import verify_freeze  # noqa: E402
from harness import analysis_table, load_plan, make_config  # noqa: E402
from sensitivity_avfad import CHOSEN_LAYERS  # noqa: E402

RESULTS = HERE / "results" / "c1_by_day"
DAYS = HERE.parent / "s0_audit" / "out" / "avfad_speakers.csv"
SEED = 20261156
STORED = {
    "handcrafted": HERE / "results" / "first_batch" / "c1_predictions.csv.gz",
    "hubert": HERE / "results" / "sensitivity" / "hubert" / "c1_predictions.csv.gz",
}
# Reading, written before the run (deviations.md, 2026-11-10).
GAP_POINTS = 0.10


def day_folds(table: pd.DataFrame, repeat: int, n_splits: int) -> np.ndarray:
    """Outer fold of every row; everyone recorded on one day lands in the same fold."""
    days = pd.read_csv(DAYS, dtype={"speaker": str})[["speaker", "visit_date"]]
    day = table[["speaker"]].astype(str).merge(days, on="speaker", how="left", validate="m:1")
    if day["visit_date"].isna().any():
        raise RuntimeError("a speaker has no visit date")
    folds = np.full(len(table), -1)
    splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=SEED + repeat)
    y = table["label"].astype(str).to_numpy()
    for k, (_, test) in enumerate(splitter.split(np.zeros(len(table)), y, day["visit_date"])):
        folds[test] = k
    return folds


def _fit(detector: str, name: str, members: pd.DataFrame, repeat: int, config):
    start = time.perf_counter()
    if detector == "hubert":
        table = merge_features(members, load_hubert("full", CHOSEN_LAYERS), "label")
        blocks = {f"l{layer:02d}": layer_columns(layer) for layer in CHOSEN_LAYERS}
        columns = [c for block in blocks.values() for c in block]
        tuned = config.override({"model.block_selection": True})
    else:
        vsb_models.SEARCH_SPACES[pc.LOGISTIC] = {k: list(v) for k, v in pc.HANDCRAFTED_GRID.items()}
        table = analysis_table(members, batch.load_features("full_original"), "label")
        blocks, columns, tuned = None, list(FEATURE_COLUMNS), config
    table = table.assign(fold=day_folds(table, repeat, config.evaluation.n_splits))
    score = np.full(len(table), np.nan)
    predicted = np.empty(len(table), dtype=object)
    for k in range(config.evaluation.n_splits):
        output = ps.fit_listed_fold(table, columns, blocks, tuned, k)
        score[output["items"]] = output["scores"]
        predicted[output["items"]] = output["predicted"]
    frame = pd.DataFrame(
        {
            "arm": name,
            "speaker": table["speaker"].to_numpy(),
            "label": table["label"].to_numpy(),
            "repeat": repeat,
            "predicted": predicted,
            "score": score,
        }
    )
    return frame, time.perf_counter() - start


def path_of(detector: str) -> Path:
    return RESULTS / f"c1_predictions_{detector}.csv.gz"


def run(jobs: int, detector: str) -> None:
    verify_freeze()
    plan = load_plan()
    config = make_config(plan, batch.LABEL_POSITIVE)
    sets = [(n, m) for n, m in vf.arm_sets(plan) if not n.startswith("shuffled")]
    todo = [(n, m, r) for n, m in sets for r in range(config.evaluation.n_repeats)]
    RESULTS.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    done = Parallel(n_jobs=jobs, return_as="generator_unordered")(
        delayed(_fit)(detector, n, m, r, config) for n, m, r in todo
    )
    frames = []
    for count, (frame, seconds) in enumerate(done, start=1):
        frames.append(frame)
        if count % 100 == 0 or count == len(todo):
            print(
                f"{count} of {len(todo)} after {time.perf_counter() - start:.0f} s "
                f"(last {seconds:.0f} s)",
                flush=True,
            )
    out = pd.concat(frames).sort_values(["arm", "speaker", "repeat"])
    out.to_csv(path_of(detector), index=False)


def reading(gap: dict) -> str:
    if gap["difference"] >= GAP_POINTS and gap["ci_low"] > 0:
        return "grouped by recording day, C1 still holds"
    if gap["ci_low"] <= 0:
        return "C1 rests on the same-day recordings"
    return "the gap shrinks"


def collect() -> None:
    plan = load_plan()
    summary: dict = {}
    for detector in ("handcrafted", "hubert"):
        if not path_of(detector).exists():
            continue
        table = pd.read_csv(path_of(detector), dtype={"speaker": str})
        gap, samples, extra = pc.gap_with_samples(table, plan)
        stored = pd.read_csv(STORED[detector], dtype={"speaker": str})
        base, base_samples, _ = pc.gap_with_samples(stored, plan)
        diff = samples - base_samples
        low, high = np.quantile(diff[np.isfinite(diff)], [0.025, 0.975])
        summary[detector] = {
            **extra,
            "gap_i_minus_ii": gap,
            "speaker_folds_gap": base["difference"],
            "minus_speaker_folds": {
                "difference": gap["difference"] - base["difference"],
                "ci_low": float(low),
                "ci_high": float(high),
            },
            "reading": reading(gap),
        }
        print(
            detector,
            round(extra["ii"]["uar"], 3),
            round(extra["i_uar_mean"], 3),
            round(gap["difference"], 4),
            round(gap["ci_low"], 4),
            round(gap["ci_high"], 4),
            "| speaker folds",
            round(base["difference"], 4),
            summary[detector]["reading"],
            flush=True,
        )
    write_json(RESULTS / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("run", "collect"))
    parser.add_argument("detector", nargs="?", choices=("handcrafted", "hubert"))
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()
    if args.step == "run":
        if args.detector is None:
            parser.error("run needs a detector")
        run(args.jobs, args.detector)
    else:
        collect()


if __name__ == "__main__":
    main()
