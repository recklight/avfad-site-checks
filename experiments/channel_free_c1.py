"""C1 on the phonation spans with handcrafted features that a fixed channel or gain cannot move.

Exploratory. Definitions and readings are in experiments/deviations.md (2026-11-10), written
before this ran. Speakers, draws, folds and the logistic grid are those of C1; only the columns
change. A fixed linear filter adds one constant to every frame's cepstrum and a gain one constant
to the level in dB, so standard deviations and frame differences see neither.

    check            the fold-wise path with all 78 columns against the stored phonation C1
    run free|bound   51 arms x 10 repeats with one column set
    visibility       ESSUA-vs-HSA AUC among the normals, per column set
    collect          writes results/channel_free_c1/summary.json

    .venv/Scripts/python.exe -I -X utf8 -B experiments/channel_free_c1.py run free --jobs 12
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
import pandas as pd  # noqa: E402
from voice_severity_bench import models as vsb_models  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "features"))

import coral_c1 as cc  # noqa: E402
import patient_position as pp  # noqa: E402
import phonation_c1 as pc  # noqa: E402
import pool_training as pt  # noqa: E402
import published_splits as ps  # noqa: E402
import run_first_batch as batch  # noqa: E402
import site_erasure as se  # noqa: E402
import vowel_fusion as vf  # noqa: E402
from arms import write_json  # noqa: E402
from extract_handcrafted import FEATURE_COLUMNS  # noqa: E402
from freeze import verify_freeze  # noqa: E402
from harness import analysis_table, load_plan, make_config  # noqa: E402

RESULTS = HERE / "results" / "channel_free_c1"
CONDITION = "phonation_norm"
PREFIXES = ("mfcc_std_", "dmfcc_mean_", "dmfcc_std_")
FREE = [c for c in FEATURE_COLUMNS if c.startswith(PREFIXES)] + ["level_std"]
BOUND = [c for c in FEATURE_COLUMNS if c not in FREE]
SETS = {"free": FREE, "bound": BOUND, "all": list(FEATURE_COLUMNS)}
# Readings, written before the run (deviations.md, 2026-11-10).
WEAKEST = 0.60
GAP_POINTS = 0.10


def _fit(set_name: str, name: str, members: pd.DataFrame, repeat: int, config):
    start = time.perf_counter()
    vsb_models.SEARCH_SPACES[pc.LOGISTIC] = {k: list(v) for k, v in pc.HANDCRAFTED_GRID.items()}
    table = analysis_table(members, batch.load_features(CONDITION), "label")
    table = table.assign(fold=pt.fold_ids(table, config)[:, repeat])
    score = np.full(len(table), np.nan)
    predicted = np.empty(len(table), dtype=object)
    for k in range(config.evaluation.n_splits):
        output = ps.fit_listed_fold(table, SETS[set_name], None, config, k)
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


def arm_sets(plan: dict) -> list[tuple[str, pd.DataFrame]]:
    return [(n, m) for n, m in vf.arm_sets(plan) if not n.startswith("shuffled")]


def check(jobs: int) -> None:
    """All 78 columns, fold-wise, against the stored phonation C1 for (ii) and i_00."""
    plan = load_plan()
    config = make_config(plan, batch.LABEL_POSITIVE)
    sets = dict(arm_sets(plan))
    todo = [(n, r) for n in ("ii", "i_00") for r in range(config.evaluation.n_repeats)]
    frames = Parallel(n_jobs=jobs)(delayed(_fit)("all", n, sets[n], r, config) for n, r in todo)
    mine = pd.concat([f for f, _ in frames])
    stored = pd.read_csv(pc.RESULTS / "c1_predictions_handcrafted.csv.gz", dtype={"speaker": str})
    stored = stored[stored["arm"].isin(["ii", "i_00"])]
    both = mine.merge(stored, on=["arm", "speaker", "repeat"], suffixes=("", "_stored"))
    if len(both) != len(stored):
        raise RuntimeError(f"{len(both)} of {len(stored)} rows matched")
    diff = float(np.abs(both["score"] - both["score_stored"]).max())
    same = bool((both["predicted"] == both["predicted_stored"]).all())
    print(f"max score difference {diff:.3g}, same predictions {same}")


def path_of(set_name: str) -> Path:
    return RESULTS / f"c1_predictions_{set_name}.csv.gz"


def run(jobs: int, set_name: str) -> None:
    verify_freeze()
    plan = load_plan()
    config = make_config(plan, batch.LABEL_POSITIVE)
    todo = [(n, m, r) for n, m in arm_sets(plan) for r in range(config.evaluation.n_repeats)]
    RESULTS.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    done = Parallel(n_jobs=jobs, return_as="generator_unordered")(
        delayed(_fit)(set_name, n, m, r, config) for n, m, r in todo
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
    out.to_csv(path_of(set_name), index=False)


def _visibility_one(set_name: str) -> tuple[str, float]:
    columns = SETS[set_name]
    normals = se.normal_speakers().merge(
        batch.load_features(CONDITION)[["file_id", *columns]],
        on="file_id",
        how="left",
        validate="one_to_one",
    )
    values = normals[columns].to_numpy(dtype=float)
    if np.isnan(values).any():
        raise RuntimeError("a normal speaker has no features")
    essua = (normals["place"] == "ESSUA").to_numpy()
    parts = pp.parts_within_place(
        normals["place"].to_numpy(), cc.N_PARTS, np.random.default_rng(cc.SEED)
    )
    groups = [np.arange(len(columns))]
    scores = cc.place_scores(values, essua, parts, groups, "none", "logistic")
    return set_name, float(roc_auc_score(essua, scores))


def visibility(jobs: int) -> None:
    verify_freeze()
    done = Parallel(n_jobs=jobs)(delayed(_visibility_one)(s) for s in SETS)
    out = dict(done)
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / "visibility.json", out)
    print(json.dumps(out, indent=1))


def free_reading(gap: dict, i_uar_mean: float) -> str:
    if i_uar_mean < WEAKEST:
        return "this set is too weak to read"
    if gap["difference"] >= GAP_POINTS and gap["ci_low"] > 0:
        return "a gap without a fixed channel or gain"
    if gap["ci_low"] <= 0:
        return "a fixed channel and gain explain it"
    return "the gap shrinks"


def collect() -> None:
    plan = load_plan()
    phon = pd.read_csv(pc.RESULTS / "c1_predictions_handcrafted.csv.gz", dtype={"speaker": str})
    base, base_samples, base_extra = pc.gap_with_samples(phon, plan)
    summary: dict = {"all": {**base_extra, "columns": len(SETS["all"]), "gap_i_minus_ii": base}}
    for set_name in ("free", "bound"):
        if not path_of(set_name).exists():
            continue
        table = pd.read_csv(path_of(set_name), dtype={"speaker": str})
        gap, samples, extra = pc.gap_with_samples(table, plan)
        diff = samples - base_samples
        low, high = np.quantile(diff[np.isfinite(diff)], [0.025, 0.975])
        record = {
            **extra,
            "columns": len(SETS[set_name]),
            "gap_i_minus_ii": gap,
            "minus_all": {
                "difference": gap["difference"] - base["difference"],
                "ci_low": float(low),
                "ci_high": float(high),
            },
        }
        if set_name == "free":
            record["reading"] = free_reading(gap, extra["i_uar_mean"])
        summary[set_name] = record
    seen = RESULTS / "visibility.json"
    if seen.exists():
        summary["visibility"] = json.loads(seen.read_text(encoding="utf-8"))
    write_json(RESULTS / "summary.json", summary)
    for set_name, record in summary.items():
        if set_name == "visibility":
            print(set_name, record)
            continue
        g = record["gap_i_minus_ii"]
        print(
            set_name,
            record["columns"],
            round(record["ii"]["uar"], 3),
            round(record["i_uar_mean"], 3),
            round(g["difference"], 4),
            round(g["ci_low"], 4),
            round(g["ci_high"], 4),
            record.get("reading", ""),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("check", "run", "visibility", "collect"))
    parser.add_argument("set", nargs="?", choices=("free", "bound"))
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()
    if args.step == "check":
        check(args.jobs)
    elif args.step == "run":
        if args.set is None:
            parser.error("run needs a column set")
        run(args.jobs, args.set)
    elif args.step == "visibility":
        visibility(args.jobs)
    else:
        collect()


if __name__ == "__main__":
    main()
