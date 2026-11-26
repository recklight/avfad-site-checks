"""C1 with the ESSUA speakers moved onto the HSA normals, per dimension, from the normals only.

Exploratory. Definitions and readings are in experiments/deviations.md (2026-11-10), written
before this run. Speakers, draws, outer folds and detectors are those of C1. In every outer fold
the mean (and standard deviation) of each feature comes from the AVFAD normals outside the test
fold, per place, and every ESSUA row is moved onto the HSA values. HSA rows never change, so
(ii) is the stored one.

    check     the fold-wise handcrafted path against the stored first-batch scores
    run       500 jobs per detector and kind (50 draws x 10 repeats, five folds each)
    collect   writes results/harmonize_c1/summary.json

    .venv/Scripts/python.exe -I -X utf8 -B experiments/harmonize_c1.py run --detector hubert \
        --kind location --jobs 12
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
sys.path.insert(0, str(HERE.parent / "features"))

import pool_training as pt  # noqa: E402
import published_splits as ps  # noqa: E402
import run_first_batch as batch  # noqa: E402
import site_erasure as se  # noqa: E402
import vowel_fusion as vf  # noqa: E402
from arms import layer_columns, load_hubert, merge_features, write_json  # noqa: E402
from extract_handcrafted import FEATURE_COLUMNS  # noqa: E402
from freeze import verify_freeze  # noqa: E402
from harness import analysis_table, load_plan, make_config  # noqa: E402
from sensitivity_avfad import CHOSEN_LAYERS  # noqa: E402

RESULTS = HERE / "results" / "harmonize_c1"
PLAIN = {
    "handcrafted": HERE / "results" / "first_batch" / "c1_predictions.csv.gz",
    "hubert": HERE / "results" / "sensitivity" / "hubert" / "c1_predictions.csv.gz",
}
ERASED = HERE / "results" / "erasure_folds" / "summary.json"
DETECTORS = ("hubert", "handcrafted")
KINDS = ("location", "scale")
LOGISTIC = ("classification", "logistic")
# The handcrafted grid, taken before any HuBERT fit in this process replaces it.
HANDCRAFTED_GRID = {k: list(v) for k, v in vsb_models.SEARCH_SPACES[LOGISTIC].items()}
# Readings, written before the run (deviations.md, 2026-11-10).
GAP_POINTS = 0.10
SCALE_POINTS = 0.03


def place_moments(normals: pd.DataFrame, columns: list[str], test_speakers: set[str]) -> dict:
    """Per-dimension mean and SD of the ESSUA and the HSA normals outside the test fold."""
    kept = normals[~normals["speaker"].isin(test_speakers)]
    essua = (kept["place"] == "ESSUA").to_numpy()
    if essua.all() or not essua.any():
        raise RuntimeError("a place has no normal speaker outside the test fold")
    values = kept[columns].to_numpy(dtype=float)
    return {
        "mu_e": values[essua].mean(axis=0),
        "mu_h": values[~essua].mean(axis=0),
        "sd_e": values[essua].std(axis=0, ddof=1),
        "sd_h": values[~essua].std(axis=0, ddof=1),
    }


def aligned(table: pd.DataFrame, columns: list[str], moments: dict, kind: str) -> pd.DataFrame:
    """ESSUA rows moved onto the HSA mean (and SD); HSA rows as they are."""
    rows = (table["place"] == "ESSUA").to_numpy()
    values = table[columns].to_numpy(dtype=float)
    if kind == "location":
        values[rows] = values[rows] - moments["mu_e"] + moments["mu_h"]
    elif kind == "scale":
        sd_e = moments["sd_e"]
        ratio = np.divide(moments["sd_h"], sd_e, out=np.ones_like(sd_e), where=sd_e > 0)
        values[rows] = (values[rows] - moments["mu_e"]) * ratio + moments["mu_h"]
    else:
        raise ValueError(kind)
    out = table.copy()
    out[columns] = values
    return out


def detector_inputs(detector: str, members: pd.DataFrame, config):
    """Arm table, the normals with their features, columns, blocks and the tuned config."""
    if detector == "hubert":
        features = load_hubert("full", CHOSEN_LAYERS, material="avfad_a")
        table = merge_features(vf.with_vowel(members, "a"), features, "label")
        blocks = {f"l{layer:02d}": layer_columns(layer) for layer in CHOSEN_LAYERS}
        columns = [c for block in blocks.values() for c in block]
        tuned = config.override({"model.block_selection": True})
    else:
        vsb_models.SEARCH_SPACES[LOGISTIC] = {k: list(v) for k, v in HANDCRAFTED_GRID.items()}
        features = batch.load_features("full_original")
        table = analysis_table(members, features, "label")
        blocks, columns, tuned = None, list(FEATURE_COLUMNS), config
    normals = se.normal_speakers().merge(
        features[["file_id", *columns]], on="file_id", how="left", validate="one_to_one"
    )
    if normals[columns[0]].isna().any():
        raise RuntimeError("a normal speaker has no features")
    return table, normals, columns, blocks, tuned


def _fit(detector: str, kind: str, name: str, members: pd.DataFrame, repeat: int, config):
    start = time.perf_counter()
    table, normals, columns, blocks, tuned = detector_inputs(detector, members, config)
    table = table.assign(fold=pt.fold_ids(table, config)[:, repeat])
    score = np.full(len(table), np.nan)
    predicted = np.empty(len(table), dtype=object)
    for k in range(config.evaluation.n_splits):
        moved = table
        if kind != "plain":
            test_speakers = set(table.loc[table["fold"] == k, "speaker"])
            moved = aligned(table, columns, place_moments(normals, columns, test_speakers), kind)
        output = ps.fit_listed_fold(moved, columns, blocks, tuned, k)
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


def draw_sets(plan: dict) -> list[tuple[str, pd.DataFrame]]:
    return [(n, m) for n, m in vf.arm_sets(plan) if n.startswith("i_")]


def check(jobs: int) -> None:
    """Handcrafted, no alignment: the fold-wise path against first_batch for (ii) and i_00."""
    plan = load_plan()
    config = make_config(plan, batch.LABEL_POSITIVE)
    sets = dict(vf.arm_sets(plan))
    todo = [(n, r) for n in ("ii", "i_00") for r in range(config.evaluation.n_repeats)]
    frames = Parallel(n_jobs=jobs)(
        delayed(_fit)("handcrafted", "plain", n, sets[n], r, config) for n, r in todo
    )
    mine = pd.concat([f for f, _ in frames])
    stored = pd.read_csv(PLAIN["handcrafted"], dtype={"speaker": str})
    stored = stored[stored["arm"].isin(["ii", "i_00"])]
    both = mine.merge(stored, on=["arm", "speaker", "repeat"], suffixes=("", "_stored"))
    if len(both) != len(stored):
        raise RuntimeError(f"{len(both)} of {len(stored)} rows matched")
    diff = float(np.abs(both["score"] - both["score_stored"]).max())
    same = bool((both["predicted"] == both["predicted_stored"]).all())
    print(f"max score difference {diff:.3g}, same predictions {same}")


def run(jobs: int, detector: str, kind: str) -> None:
    verify_freeze()
    plan = load_plan()
    config = make_config(plan, batch.LABEL_POSITIVE)
    todo = [(n, m, r) for n, m in draw_sets(plan) for r in range(config.evaluation.n_repeats)]
    RESULTS.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    done = Parallel(n_jobs=jobs, return_as="generator_unordered")(
        delayed(_fit)(detector, kind, n, m, r, config) for n, m, r in todo
    )
    frames = []
    for count, (frame, seconds) in enumerate(done, start=1):
        frames.append(frame)
        if count % 50 == 0 or count == len(todo):
            print(
                f"{count} of {len(todo)} after {time.perf_counter() - start:.0f} s "
                f"(last {seconds:.0f} s)",
                flush=True,
            )
    table = pd.concat(frames).sort_values(["arm", "speaker", "repeat"])
    table.to_csv(RESULTS / f"c1_predictions_{detector}_{kind}.csv.gz", index=False)


def gap_reading(gap: dict) -> str:
    if gap["difference"] >= GAP_POINTS and gap["ci_low"] > 0:
        return "the gap stays"
    if gap["ci_low"] <= 0:
        return "the gap is gone"
    return "the gap shrinks"


def scale_reading(change: dict) -> str:
    if change["difference"] <= -SCALE_POINTS and change["ci_high"] < 0:
        return "the spread carries part of the gap"
    if change["ci_low"] <= 0 <= change["ci_high"]:
        return "the spread carries nothing more"
    return "as measured"


def paired(first: tuple[dict, np.ndarray], second: tuple[dict, np.ndarray]) -> dict:
    diff = first[1] - second[1]
    low, high = np.quantile(diff[np.isfinite(diff)], [0.025, 0.975])
    return {
        "difference": first[0]["difference"] - second[0]["difference"],
        "ci_low": float(low),
        "ci_high": float(high),
    }


def gap_of(table: pd.DataFrame, plain: pd.DataFrame, plan: dict) -> tuple[dict, np.ndarray]:
    """C1 gap of the draws in ``table`` against the stored (ii) of ``plain``."""
    ii = vf.arm_from_table(plain, "ii")
    n = plan["confirmatory_family"]["tests"]["C1"]["draws"]
    draws = [vf.arm_from_table(table, f"i_{i:02d}") for i in range(n)]
    gap = batch.c1_bootstrap(draws, ii, vf.N_RESAMPLES, plan["statistics"]["seeds"]["bootstrap"])
    samples = gap.pop("samples")
    gap["i_uar_mean"] = float(np.mean([a.uar for a in draws]))
    gap["ii_uar"] = ii.uar
    return gap, samples


def spread_ratios(detector: str) -> dict[str, float]:
    """Median over dimensions of SD(ESSUA normals) / SD(HSA normals), all normals."""
    members = se.normal_speakers()
    _, normals, _, blocks, _ = detector_inputs(detector, members, make_config(load_plan(), "x"))
    groups = blocks if blocks is not None else {"all": list(FEATURE_COLUMNS)}
    essua = (normals["place"] == "ESSUA").to_numpy()
    out = {}
    for name, columns in groups.items():
        values = normals[columns].to_numpy(dtype=float)
        sd_e, sd_h = values[essua].std(axis=0, ddof=1), values[~essua].std(axis=0, ddof=1)
        keep = sd_h > 0
        out[name] = float(np.median(sd_e[keep] / sd_h[keep]))
    return out


def collect() -> None:
    import json

    plan = load_plan()
    summary: dict = {}
    for detector in DETECTORS:
        plain = pd.read_csv(PLAIN[detector], dtype={"speaker": str})
        base = gap_of(plain, plain, plan)
        part: dict = {"plain_gap": base[0]["difference"], "spread_ratio": spread_ratios(detector)}
        got = {}
        for kind in KINDS:
            path = RESULTS / f"c1_predictions_{detector}_{kind}.csv.gz"
            if not path.exists():
                continue
            got[kind] = gap_of(pd.read_csv(path, dtype={"speaker": str}), plain, plan)
            part[kind] = {
                "gap_i_minus_ii": got[kind][0],
                "gap_reading": gap_reading(got[kind][0]),
                "minus_plain": paired(got[kind], base),
            }
        if len(got) == len(KINDS):
            change = paired(got["scale"], got["location"])
            part["scale_minus_location"] = change
            part["scale_reading"] = scale_reading(change)
        if detector == "hubert":
            erased = json.loads(ERASED.read_text(encoding="utf-8"))
            part["direction_erased_gap"] = erased["gap_i_minus_ii"]["difference"]
        summary[detector] = part
    write_json(RESULTS / "summary.json", summary)
    for detector, part in summary.items():
        for kind in KINDS:
            if kind in part:
                g = part[kind]["gap_i_minus_ii"]
                print(detector, kind, part[kind]["gap_reading"], g["difference"], g["ci_low"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("check", "run", "collect"))
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--detector", choices=DETECTORS, default="hubert")
    parser.add_argument("--kind", choices=KINDS, default="location")
    args = parser.parse_args()
    if args.step == "check":
        check(args.jobs)
    elif args.step == "run":
        run(args.jobs, args.detector, args.kind)
    else:
        collect()


if __name__ == "__main__":
    main()
