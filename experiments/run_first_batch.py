"""The first batch of analysis_plan.md, section 11: positive gate, C1 and C2, on AVFAD.

The gate runs first. If it fails, or a negative control leaves 0.47-0.53, the script writes
what it has and stops with exit code 3. Everything goes to experiments/results/first_batch/.

    .venv/Scripts/python.exe -I -B experiments/run_first_batch.py --jobs 16

One BLAS thread per worker, so the numbers do not depend on the worker count.
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from voice_severity_bench.bootstrap import group_weights, weighted_metric  # noqa: E402
from voice_severity_bench.permutation import permutation_test  # noqa: E402
from joblib import Parallel, delayed, parallel_config  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from freeze import FreezeError, verify_freeze  # noqa: E402
from harness import (  # noqa: E402
    ROOT,
    Arm,
    age_sex_cell,
    analysis_table,
    check_permuted_splits,
    evaluate_arm,
    load_plan,
    make_config,
    match_cells,
    predictions_table,
    to_feature_set,
)

S0_OUT = ROOT / "s0_audit" / "out"
S1_OUT = ROOT / "s1_segments" / "out"
FEATURES = ROOT / "features" / "out"
RESULTS = HERE / "results" / "first_batch"

C2_PERMUTATIONS = 1000
PLACE_POSITIVE = "HSA"
LABEL_POSITIVE = "pathological"

_LOGGER = logging.getLogger("first_batch")


class StopRun(RuntimeError):
    """A stop rule of the plan fired."""


def load_speakers() -> pd.DataFrame:
    table = pd.read_csv(S0_OUT / "avfad_speakers.csv", dtype={"speaker": str})
    table["file_id"] = "avfad__" + table["speaker"]
    return table[["file_id", "speaker", "place", "label", "age", "sex", "laryngeal_surgery"]]


def load_features(condition: str) -> pd.DataFrame:
    return pd.read_csv(FEATURES / f"avfad_a_{condition}.csv")


def reaches_minimum(min_total_s: float) -> set[str]:
    files = pd.read_csv(S1_OUT / "files.csv")
    files = files[(files["database"] == "avfad") & (files["material"] == "a")]
    return set(files.loc[files["nonphonation_g100_s"] >= min_total_s, "file_id"])


def negative_control(
    name: str, table: pd.DataFrame, config, plan: dict, jobs: int
) -> dict[str, float]:
    """Mean UAR of the arm over shuffled speaker labels, full CV each time."""
    controls = plan["controls"]["negative"]
    n = controls["shuffles_per_arm"]
    seed = plan["statistics"]["seeds"]["negative_control"]
    features = to_feature_set(table)
    y = np.asarray(features.y).astype(str)
    groups = np.asarray(features.groups).astype(str)
    worst = check_permuted_splits(y, groups, None, config, n, seed, jobs)
    result = permutation_test(features, config, n_permutations=n, seed=seed, n_jobs=jobs)
    low, high = controls["uar_within"]
    mean = float(np.nanmean(result.null))
    return {
        "arm": name,
        "shuffles": n,
        "mean_uar": mean,
        "sd_uar": float(np.nanstd(result.null, ddof=1)),
        "min_uar": float(np.nanmin(result.null)),
        "max_uar": float(np.nanmax(result.null)),
        "worst_share_deviation": worst,
        "n_undefined": result.n_undefined,
        "passed": bool(low <= mean <= high),
    }


def run_gate(speakers, features, plan, jobs, out) -> dict:
    gate = plan["controls"]["positive_gate"]
    seeds = plan["statistics"]["seeds"]
    minimum = plan["non_phonation_use"]["min_total_s"]
    normal = speakers[speakers["label"] == "normal"]
    available = normal[normal["file_id"].isin(reaches_minimum(minimum))]
    hsa = available[available["place"] == "HSA"]
    essua, cells = match_cells(
        hsa, available[available["place"] == "ESSUA"], np.random.default_rng(seeds["c2_essua_draw"])
    )
    table = analysis_table(pd.concat([hsa, essua]), features, "place")
    config = make_config(plan, PLACE_POSITIVE)
    arm = evaluate_arm("gate", table, config)
    negative = negative_control("gate", table, config, plan, jobs)
    cells.to_csv(out / "gate_cells.csv", index=False)
    table[["speaker", "place", "age", "sex"]].to_csv(out / "gate_speakers.csv", index=False)
    predictions_table([arm]).to_csv(out / "gate_predictions.csv", index=False)
    result = {
        **arm.row(),
        "normal_speakers": {
            place: int((normal["place"] == place).sum()) for place in ("ESSUA", "HSA")
        },
        "reach_minimum": {
            place: int((available["place"] == place).sum()) for place in ("ESSUA", "HSA")
        },
        "cells_short": int((cells["found"] < cells["wanted"]).sum()),
        "min_uar": gate["min_uar"],
        "passed": bool(arm.uar >= gate["min_uar"]),
        "negative": negative,
    }
    _LOGGER.info("gate UAR %.4f, shuffled %.4f", arm.uar, negative["mean_uar"])
    return result


def c1_bootstrap(
    arms_i: list[Arm], arm_ii: Arm, n_resamples: int, seed: int
) -> dict[str, float | np.ndarray]:
    """One-sided speaker bootstrap of the difference, without refitting.

    Every speaker used by either arm gets one weight per resample and keeps it on every
    arm, so the arms stay paired through the patients and the shared normals.
    """
    everyone = sorted({str(g) for arm in [*arms_i, arm_ii] for g in arm.groups})
    position = {speaker: i for i, speaker in enumerate(everyone)}
    weights = group_weights(everyone, len(everyone), n_resamples, seed=seed)

    def over_resamples(arm: Arm) -> np.ndarray:
        columns = [position[str(g)] for g in arm.groups]
        return weighted_metric(arm.y_true, arm.predictions, weights[:, columns], metric="uar")

    mean_i = np.mean([over_resamples(arm) for arm in arms_i], axis=0)
    samples = mean_i - over_resamples(arm_ii)
    estimate = float(np.mean([arm.uar for arm in arms_i]) - arm_ii.uar)
    finite = samples[np.isfinite(samples)]
    low, high = np.quantile(finite, [0.025, 0.975])
    return {
        "difference": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
        "p_one_sided": float((1 + np.count_nonzero(finite <= 0.0)) / (1 + finite.size)),
        "n_resamples": int(n_resamples),
        "n_undefined": int(samples.size - finite.size),
        "n_speakers_resampled": len(everyone),
        "samples": samples,
    }


def _draw_arm(name, table, config) -> Arm:
    return evaluate_arm(name, table, config)


def run_c1(speakers, features, plan, jobs, out) -> dict:
    spec = plan["confirmatory_family"]["tests"]["C1"]
    seeds = plan["statistics"]["seeds"]
    patients = speakers[(speakers["place"] == "HSA") & (speakers["label"] == "pathological")]
    hsa_normals = speakers[(speakers["place"] == "HSA") & (speakers["label"] == "normal")]
    all_normals = speakers[speakers["label"] == "normal"]
    rng = np.random.default_rng(seeds["c1_draws"])
    draws, draw_rows = [], []
    for index in range(spec["draws"]):
        drawn, cells = match_cells(hsa_normals, all_normals, rng)
        if (cells["found"] < cells["wanted"]).any():
            raise StopRun("C1: a cell of the whole normal group came up short")
        draws.append(drawn)
        draw_rows.append(drawn.assign(draw=index)[["draw", "speaker", "place", "age", "sex"]])
    pd.concat(draw_rows).to_csv(out / "c1_draws.csv", index=False)

    config = make_config(plan, LABEL_POSITIVE)
    jobs_list = [("ii", analysis_table(pd.concat([patients, hsa_normals]), features, "label"))]
    jobs_list += [
        (f"i_{index:02d}", analysis_table(pd.concat([patients, drawn]), features, "label"))
        for index, drawn in enumerate(draws)
    ]
    arms = Parallel(n_jobs=jobs)(
        delayed(_draw_arm)(name, table, config) for name, table in jobs_list
    )
    arm_ii, arms_i = arms[0], arms[1:]
    pd.DataFrame([arm.row() for arm in arms]).to_csv(out / "c1_arms.csv", index=False)
    predictions_table(arms).to_csv(out / "c1_predictions.csv.gz", index=False)
    _LOGGER.info("C1 arms done, ii %.4f, i mean %.4f", arm_ii.uar, np.mean([a.uar for a in arms_i]))

    test = c1_bootstrap(arms_i, arm_ii, spec["test"]["resamples"], seeds["bootstrap"])
    np.save(out / "c1_bootstrap_samples.npy", test.pop("samples"))
    negative = [
        negative_control(name, table, config, plan, jobs)
        for name, table in (jobs_list[0], jobs_list[1])
    ]
    places = pd.concat(draw_rows)["place"].value_counts(normalize=True)
    uars = [arm.uar for arm in arms_i]
    return {
        **test,
        "uar_i_mean": float(np.mean(uars)),
        "uar_i_min": float(np.min(uars)),
        "uar_i_max": float(np.max(uars)),
        "uar_ii": arm_ii.uar,
        "sensitivity_i_mean": float(np.mean([a.sensitivity for a in arms_i])),
        "specificity_i_mean": float(np.mean([a.specificity for a in arms_i])),
        "sensitivity_ii": arm_ii.sensitivity,
        "specificity_ii": arm_ii.specificity,
        "roc_auc_i_mean": float(np.mean([a.roc_auc for a in arms_i])),
        "roc_auc_ii": arm_ii.roc_auc,
        "patients": len(patients),
        "normals_per_arm": len(hsa_normals),
        "draws": spec["draws"],
        "draw_place_share": {str(k): float(v) for k, v in places.items()},
        "worst_share_deviation": float(max(a.worst_share_deviation for a in arms)),
        "smallest_gap": spec["smallest_gap_points"] / 100.0,
        "reaches_smallest_gap": bool(test["difference"] >= spec["smallest_gap_points"] / 100.0),
        "negative": negative,
    }


def run_c2(speakers, features, plan, jobs, out) -> dict:
    spec = plan["confirmatory_family"]["tests"]["C2"]
    seeds = plan["statistics"]["seeds"]
    normal = speakers[speakers["label"] == "normal"]
    hsa = normal[normal["place"] == "HSA"]
    pool = normal[normal["place"] == "ESSUA"]
    essua, cells = match_cells(hsa, pool, np.random.default_rng(seeds["c2_essua_draw"]))
    table = analysis_table(pd.concat([hsa, essua]), features, "place")
    config = make_config(plan, PLACE_POSITIVE)
    arm = evaluate_arm("c2", table, config)
    cells.to_csv(out / "c2_cells.csv", index=False)
    table[["speaker", "place", "age", "sex"]].to_csv(out / "c2_speakers.csv", index=False)

    fset = to_feature_set(table)
    y = np.asarray(fset.y).astype(str)
    groups = np.asarray(fset.groups).astype(str)
    strata = age_sex_cell(table).to_numpy().astype(str)
    worst = check_permuted_splits(
        y, groups, strata, config, C2_PERMUTATIONS, seeds["permutation"], jobs
    )
    test = permutation_test(
        fset,
        config,
        n_permutations=C2_PERMUTATIONS,
        strata=strata,
        seed=seeds["permutation"],
        n_jobs=jobs,
    )
    np.save(out / "c2_null.npy", test.null)
    if abs(test.observed - arm.uar) > 1e-9:
        raise StopRun(f"C2: permutation observed {test.observed} but the arm gave {arm.uar}")
    _LOGGER.info("C2 UAR %.4f, p %.4f, null %.4f", arm.uar, test.p_value, np.nanmean(test.null))

    first, last = seeds["c2_other_draws"]
    others = []
    for seed in range(first, last + 1):
        drawn, _ = match_cells(hsa, pool, np.random.default_rng(seed))
        others.append((f"seed_{seed}", analysis_table(pd.concat([hsa, drawn]), features, "place")))
    other_arms = Parallel(n_jobs=jobs)(
        delayed(_draw_arm)(name, other, config) for name, other in others
    )
    pd.DataFrame([a.row() for a in other_arms]).to_csv(out / "c2_other_draws.csv", index=False)
    predictions_table([arm, *other_arms]).to_csv(out / "c2_predictions.csv.gz", index=False)

    low, high = plan["controls"]["negative"]["uar_within"]
    null_mean = float(np.nanmean(test.null))
    uars = [a.uar for a in other_arms]
    gap = spec["smallest_gap_points"] / 100.0
    return {
        **arm.row(),
        "p_one_sided": test.p_value,
        "n_permutations": C2_PERMUTATIONS,
        "n_undefined": test.n_undefined,
        "cells_short": cells[cells["found"] < cells["wanted"]].to_dict("records"),
        "other_draws_uar_min": float(np.min(uars)),
        "other_draws_uar_max": float(np.max(uars)),
        "other_draws_uar_mean": float(np.mean(uars)),
        "worst_share_deviation_permuted": worst,
        "smallest_gap": gap,
        "reaches_smallest_gap": bool(arm.uar >= 0.5 + gap),
        "negative": {
            "arm": "c2",
            "shuffles": C2_PERMUTATIONS,
            "mean_uar": null_mean,
            "sd_uar": float(np.nanstd(test.null, ddof=1)),
            "passed": bool(low <= null_mean <= high),
        },
    }


def negative_failures(section: dict) -> list[str]:
    negative = section["negative"]
    rows = negative if isinstance(negative, list) else [negative]
    return [f"{row['arm']}: {row['mean_uar']:.4f}" for row in rows if not row["passed"]]


def write_summary(summary: dict, out: Path) -> None:
    (out / "summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def run(speakers, tables, plan, jobs, out: Path) -> dict:
    """Gate, C1, C2 in order; each stage is written before the next one starts."""
    summary: dict = {}

    def stage(name: str, function, condition: str) -> dict:
        started = time.perf_counter()
        result = function(speakers, tables[condition], plan, jobs, out)
        result["seconds"] = round(time.perf_counter() - started, 1)
        summary[name] = result
        write_summary(summary, out)
        failures = negative_failures(result)
        if failures:
            raise StopRun(f"{name}: a negative control left the range: {failures}")
        return result

    gate = stage("gate", run_gate, "nonphon_g100")
    if not gate["passed"]:
        raise StopRun(f"gate: UAR {gate['uar']:.4f} is below {gate['min_uar']}")
    stage("c1", run_c1, "full_original")
    stage("c2", run_c2, "phonation_norm")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args()
    try:
        verify_freeze()
    except FreezeError as err:
        sys.exit(str(err))
    sys.path.insert(0, str(ROOT / "s0_audit"))
    from avfad_extract import require_access

    require_access()
    RESULTS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(RESULTS / "run.log", "a", "utf-8")],
    )
    plan = load_plan()
    tables = {c: load_features(c) for c in ("nonphon_g100", "full_original", "phonation_norm")}
    started = time.perf_counter()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        try:
            run(load_speakers(), tables, plan, args.jobs, RESULTS)
        except StopRun as err:
            _LOGGER.error("stopped: %s", err)
            sys.exit(3)
    _LOGGER.info("first batch done in %.0f s", time.perf_counter() - started)


if __name__ == "__main__":
    main()
