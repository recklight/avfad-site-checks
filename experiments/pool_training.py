"""Testing inside HSA only, training on the other HSA folds plus data from elsewhere.

Exploratory. Definitions and readings are in experiments/deviations.md (2026-11-10), written
before any arm ran. The outer folds are those of the HuBERT /a/ C1 arm (ii); the added speakers
only ever train. ``ii`` adds nothing and is checked against the stored (ii) scores.

    run       30 jobs (arm x repeat, five folds each)
    collect   writes results/pool_training/summary.json

    .venv/Scripts/python.exe -I -X utf8 -B experiments/pool_training.py run --jobs 8
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
from voice_severity_bench.splitting import repeated_split_indices  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cross_database as cd  # noqa: E402
import published_splits as ps  # noqa: E402
import run_first_batch as batch  # noqa: E402
import vowel_fusion as vf  # noqa: E402
from arms import layer_columns, load_hubert, merge_features, write_json  # noqa: E402
from freeze import verify_freeze  # noqa: E402
from harness import load_plan, make_config  # noqa: E402
from sensitivity_avfad import CHOSEN_LAYERS, hsa_groups  # noqa: E402

RESULTS = HERE / "results" / "pool_training"
STORED_II = HERE / "results" / "sensitivity" / "hubert" / "c1_predictions.csv.gz"
ARMS = ("ii", "ii_essua", "ii_svd")
SEED = 20261122
# Readings, written before the run (deviations.md, 2026-11-10).
HELPS_POINTS = 0.05
REACHES = 0.70
SAME_PATH = 1e-6


def hsa_members() -> pd.DataFrame:
    patients, hsa_normals, _ = hsa_groups(batch.load_speakers())
    return pd.concat([patients, hsa_normals])


def added_members(arm: str) -> pd.DataFrame | None:
    if arm == "ii":
        return None
    if arm == "ii_essua":
        speakers = batch.load_speakers()
        return speakers[(speakers["place"] == "ESSUA") & (speakers["label"] == "normal")]
    if arm == "ii_svd":
        return cd.svd_test_set()
    raise ValueError(arm)


def fold_ids(table: pd.DataFrame, config) -> np.ndarray:
    """Test fold of every HSA row, one column per repeat, in the order the (ii) fit saw them."""
    y = table["label"].to_numpy().astype(str)
    groups = table["speaker"].to_numpy().astype(str)
    splits = repeated_split_indices(
        y,
        groups,
        task="classification",
        n_splits=config.evaluation.n_splits,
        n_repeats=config.evaluation.n_repeats,
        seed=config.seed,
    )
    out = np.full((len(table), len(splits)), -1)
    for r, repeat in enumerate(splits):
        for k, (_, test) in enumerate(repeat):
            out[test, r] = k
    if (out < 0).any():
        raise RuntimeError("a row is in no test fold")
    return out


def stacked(arm: str, repeat: int, config) -> pd.DataFrame:
    """HSA rows first, with their fold for this repeat; added rows get fold -1 (train only)."""
    hsa = merge_features(hsa_members(), load_hubert("full", CHOSEN_LAYERS), "label")
    hsa = hsa.assign(fold=fold_ids(hsa, config)[:, repeat])
    extra = added_members(arm)
    if extra is None:
        return hsa
    material = "svd_a_n" if arm == "ii_svd" else "avfad_a"
    extra = merge_features(extra, load_hubert("full", CHOSEN_LAYERS, material=material), "label")
    return pd.concat([hsa, extra.assign(fold=-1)], ignore_index=True)


def _fit(arm: str, repeat: int, config) -> tuple[str, int, pd.DataFrame, float]:
    start = time.perf_counter()
    table = stacked(arm, repeat, config)
    blocks = {f"l{layer:02d}": layer_columns(layer) for layer in CHOSEN_LAYERS}
    columns = [c for block in blocks.values() for c in block]
    tuned = config.override({"model.block_selection": True})
    n_hsa = int((table["fold"] >= 0).sum())
    score = np.full(n_hsa, np.nan)
    predicted = np.empty(n_hsa, dtype=object)
    for k in range(config.evaluation.n_splits):
        output = ps.fit_listed_fold(table, columns, blocks, tuned, k)
        if (output["items"] >= n_hsa).any():
            raise RuntimeError("an added speaker landed in a test fold")
        score[output["items"]] = output["scores"]
        predicted[output["items"]] = output["predicted"]
    part = table.iloc[:n_hsa]
    frame = pd.DataFrame(
        {
            "arm": arm,
            "speaker": part["speaker"].to_numpy(),
            "label": part["label"].to_numpy(),
            "repeat": repeat,
            "predicted": predicted,
            "score": score,
        }
    )
    return arm, repeat, frame, time.perf_counter() - start


def run(jobs: int) -> None:
    verify_freeze()
    config = make_config(load_plan(), batch.LABEL_POSITIVE)
    todo = [(arm, r) for arm in reversed(ARMS) for r in range(config.evaluation.n_repeats)]
    RESULTS.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    done = Parallel(n_jobs=jobs, return_as="generator_unordered")(
        delayed(_fit)(arm, r, config) for arm, r in todo
    )
    frames = []
    for count, (arm, repeat, frame, seconds) in enumerate(done, start=1):
        frames.append(frame)
        print(
            f"{arm} repeat {repeat} {seconds:.0f} s; {count} of {len(todo)} after "
            f"{time.perf_counter() - start:.0f} s",
            flush=True,
        )
    table = pd.concat(frames).sort_values(["arm", "speaker", "repeat"])
    table.to_csv(RESULTS / "predictions.csv.gz", index=False)


# ------------------------------------------------------------------ collect


def help_reading(diff: dict) -> str:
    if diff["ci_high"] < 0:
        return "worse"
    if diff["ci_low"] <= 0:
        return "does not help"
    if diff["difference"] >= HELPS_POINTS:
        return "helps"
    return "small"


def arm_summary(arm) -> dict:
    return {
        "uar": arm.uar,
        "roc_auc": arm.roc_auc,
        "sensitivity": arm.sensitivity,
        "specificity_hsa_normals": arm.specificity,
    }


def collect() -> None:
    table = pd.read_csv(RESULTS / "predictions.csv.gz", dtype={"speaker": str})
    arms = {name: vf.arm_from_table(table, name) for name in ARMS}
    stored = pd.read_csv(STORED_II, dtype={"speaker": str})
    stored = stored[stored["arm"] == "ii"].pivot(index="speaker", columns="repeat", values="score")
    refit = table[table["arm"] == "ii"].pivot(index="speaker", columns="repeat", values="score")
    gap = float(np.abs(refit.sort_index().to_numpy() - stored.sort_index().to_numpy()).max())
    summary: dict = {
        "same_path_check": {"max_abs_difference": gap, "same_path": gap <= SAME_PATH},
        "arms": {name: arm_summary(arm) for name, arm in arms.items()},
    }
    for name in ARMS[1:]:
        diff = {
            metric: vf.paired_difference(arms[name], arms["ii"], metric, SEED)
            for metric in ("roc_auc", "uar")
        }
        summary[f"{name}_minus_ii"] = diff
        summary.setdefault("readings", {})[name] = {
            "auc": help_reading(diff["roc_auc"]),
            "uar": help_reading(diff["uar"]),
            "reaches_0_70": arms[name].roc_auc >= REACHES,
        }
    write_json(RESULTS / "summary.json", summary)
    print(summary["readings"], summary["same_path_check"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("run", "collect"))
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if args.step == "run":
        run(args.jobs)
    else:
        collect()


if __name__ == "__main__":
    main()
