"""C3 and C4 of analysis_plan.md on SVD, their sensitivity analyses, and Holm over C1-C4.

C3 first reproduces the a_n to iau gap in the same pipeline. C4 also tests availability by
itself and reports availability by label after unblinding. Holm waits until all four exist.

    .venv/Scripts/python.exe -I -B experiments/run_svd.py c3 --jobs 16
    .venv/Scripts/python.exe -I -B experiments/run_svd.py c4 --jobs 16
    .venv/Scripts/python.exe -I -B experiments/run_svd.py sensitivity --jobs 16
    .venv/Scripts/python.exe -I -B experiments/run_svd.py family
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from voice_severity_bench.association import adjust_pvalues  # noqa: E402
from voice_severity_bench.bootstrap import group_weights  # noqa: E402
from voice_severity_bench.comparison import compare_feature_sets, pairwise_contrasts  # noqa: E402
from voice_severity_bench.dataset import feature_set_from_table  # noqa: E402
from voice_severity_bench.permutation import permutation_test  # noqa: E402
from joblib import parallel_config  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_first_batch as batch  # noqa: E402
from arms import evaluate_columns, write_json  # noqa: E402
from freeze import FreezeError, verify_freeze  # noqa: E402
from freeze_svd import verify_svd_freeze  # noqa: E402
from harness import (  # noqa: E402
    ROOT,
    analysis_table,
    check_permuted_splits,
    check_splits,
    evaluate_arm,
    load_plan,
    make_config,
    to_feature_set,
)

S0_OUT = ROOT / "s0_audit" / "out"
S1_OUT = ROOT / "s1_segments" / "out"
FEATURES = ROOT / "features" / "out"
RESULTS = HERE / "results" / "svd"
FIRST_BATCH = HERE / "results" / "first_batch" / "summary.json"

POSITIVE = "pathological"
AGE_EDGES = (18, 30, 40, 50, 60, 70, 120)
AGE_EDGES_SHIFTED = (18, 28, 38, 48, 58, 68, 78, 120)
YEAR_EDGES = (1997, 1999, 2001, 2005)
C4_PERMUTATIONS = 1000
SENSITIVITY_PERMUTATIONS = 200
AVAILABILITY_COLUMNS = ["reaches_minimum", "nonphonation_total_s", "nonphonation_segments"]
AVAILABILITY_GAP = 0.10
N_RESAMPLES = 10_000


def load_svd_speakers(material: str) -> pd.DataFrame:
    """The analysis set of a material: earliest eligible session, no file problem."""
    status = pd.read_csv(S0_OUT / "svd_file_status.csv", keep_default_na=False, dtype=str)
    chosen = status[
        (status["primary"].str.lower() == "true")
        & (status["file_problem"] == "")
        & (status["material"] == material)
    ].copy()
    chosen["session"] = chosen["session"].astype(int)
    sessions = pd.read_csv(S0_OUT / "svd_sessions.csv")[["session", "sex", "year", "age"]]
    merged = chosen.merge(sessions, on="session", how="left", validate="one_to_one")
    merged["file_id"] = [f"svd__{s}-{material}" for s in merged["session"]]
    return merged[["file_id", "session", "speaker", "label", "age", "sex", "year"]]


def load_table(material: str, condition: str) -> pd.DataFrame:
    return pd.read_csv(FEATURES / f"svd_{material}_{condition}.csv")


def stratum(table: pd.DataFrame, age_edges=AGE_EDGES) -> np.ndarray:
    """Age band by sex by recording-year band."""
    age = pd.cut(table["age"], list(age_edges), right=False).astype(str)
    year = pd.cut(table["year"], list(YEAR_EDGES), right=False).astype(str)
    if (age == "nan").any() or (year == "nan").any():
        raise ValueError("an age or a year falls outside the bands")
    return (age + "|" + table["sex"].astype(str) + "|" + year).to_numpy().astype(str)


def one_sided_from_t(t_statistic: float, p_two_sided: float) -> float:
    """One-sided p for a positive difference, from the two-sided corrected t test."""
    if math.isnan(p_two_sided):
        return math.nan
    return p_two_sided / 2.0 if t_statistic > 0 else 1.0 - p_two_sided / 2.0


def contrast_rows(contrasts: pd.DataFrame) -> list[dict]:
    rows = contrasts.to_dict("records")
    for row in rows:
        row["p_one_sided"] = one_sided_from_t(row["t_statistic"], row["p_corrected_t"])
    return rows


def run_gap(meta_iau, meta_an, tables, config, seed, jobs) -> dict:
    """The a_n to iau gap on the speakers who have both, with folds shared between them."""
    both = set(meta_iau["session"]) & set(meta_an["session"])
    sets = {}
    for name, meta, features in (
        ("a_n", meta_an, tables["a_n_full_norm"]),
        ("iau", meta_iau, tables["iau_full_norm"]),
    ):
        table = analysis_table(meta[meta["session"].isin(both)], features, "label")
        sets[name] = to_feature_set(table.assign(path=table["session"].astype(str)))
    y = np.asarray(sets["a_n"].y).astype(str)
    groups = np.asarray(sets["a_n"].groups).astype(str)
    worst = check_splits(y, groups, config)
    result = compare_feature_sets(sets, config, n_jobs=jobs)
    contrasts = pairwise_contrasts(
        result, [("iau", "a_n")], metric="uar", n_resamples=N_RESAMPLES, seed=seed
    )
    summary = result.summary.set_index("feature_set")
    return {
        "speakers": len(both),
        "uar_a_n": float(summary.loc["a_n", "uar"]),
        "uar_iau": float(summary.loc["iau", "uar"]),
        "roc_auc_a_n": float(summary.loc["a_n", "roc_auc"]),
        "roc_auc_iau": float(summary.loc["iau", "roc_auc"]),
        "contrast": contrast_rows(contrasts)[0],
        "worst_share_deviation": worst,
    }


def run_c3(meta_iau, meta_an, tables, plan, jobs, out) -> dict:
    spec = plan["confirmatory_family"]["tests"]["C3"]
    seeds = plan["statistics"]["seeds"]
    config = make_config(plan, POSITIVE)
    started = time.perf_counter()
    gap = run_gap(meta_iau, meta_an, tables, config, seeds["bootstrap"], jobs)
    write_json(out / "c3_gap.json", gap)
    arms = {
        "phonation": analysis_table(meta_iau, tables["iau_phonation_norm"], "label"),
        "phonation_breath": analysis_table(meta_iau, tables["iau_phonation_breath_norm"], "label"),
        "full": analysis_table(meta_iau, tables["iau_full_norm"], "label"),
    }
    sets = {name: to_feature_set(table) for name, table in arms.items()}
    y = np.asarray(sets["full"].y).astype(str)
    groups = np.asarray(sets["full"].groups).astype(str)
    worst = check_splits(y, groups, config)
    result = compare_feature_sets(sets, config, n_jobs=jobs)
    result.summary.to_csv(out / "c3_arms.csv", index=False)
    result.folds.to_csv(out / "c3_folds.csv", index=False)
    result.predictions.to_csv(out / "c3_predictions.csv.gz", index=False)
    contrasts = contrast_rows(
        pairwise_contrasts(
            result,
            [
                ("full", "phonation"),
                ("phonation_breath", "phonation"),
                ("full", "phonation_breath"),
            ],
            metric="uar",
            n_resamples=N_RESAMPLES,
            seed=seeds["bootstrap"],
        )
    )
    primary = contrasts[0]
    negative = [
        batch.negative_control(name, table, config, plan, jobs) for name, table in arms.items()
    ]
    gap_points = spec["smallest_gap_points"] / 100.0
    summary = result.summary.set_index("feature_set")
    return {
        "speakers": len(meta_iau),
        "labels": meta_iau["label"].value_counts().to_dict(),
        "uar": {name: float(summary.loc[name, "uar"]) for name in arms},
        "roc_auc": {name: float(summary.loc[name, "roc_auc"]) for name in arms},
        "contrasts": contrasts,
        "difference": primary["difference"],
        "ci_low": primary["ci_low"],
        "ci_high": primary["ci_high"],
        "p_one_sided": primary["p_one_sided"],
        "smallest_gap": gap_points,
        "reaches_smallest_gap": bool(primary["difference"] >= gap_points),
        "worst_share_deviation": worst,
        "gap_a_n_to_iau": gap,
        "negative": negative,
        "seconds": round(time.perf_counter() - started, 1),
    }


def with_availability(meta: pd.DataFrame, guard: str, minimum_s: float) -> pd.DataFrame:
    files = pd.read_csv(S1_OUT / "files.csv")
    files = files.set_index("file_id")
    table = meta.copy()
    table["nonphonation_total_s"] = files.loc[
        table["file_id"], f"nonphonation_{guard}_s"
    ].to_numpy()
    table["nonphonation_segments"] = files.loc[
        table["file_id"], f"n_nonphonation_{guard}"
    ].to_numpy()
    table["reaches_minimum"] = (table["nonphonation_total_s"] >= minimum_s).astype(float)
    return table


def availability_by_label(table: pd.DataFrame, cells: np.ndarray, seed: int) -> dict:
    """Share reaching the minimum, pathological minus normal, inside the strata.

    Mantel-Haenszel weights n1 n2 / (n1 + n2), so strata with one label count for nothing,
    with a speaker bootstrap of the weighted difference.
    """
    reach = table["reaches_minimum"].to_numpy(dtype=float)
    sick = (table["label"] == POSITIVE).to_numpy()
    weights = group_weights(None, reach.size, N_RESAMPLES, seed=seed)
    unit = np.ones((1, reach.size))

    def weighted(W: np.ndarray) -> np.ndarray:
        numerator = np.zeros(W.shape[0])
        denominator = np.zeros(W.shape[0])
        for cell in np.unique(cells):
            inside = cells == cell
            a, b = inside & sick, inside & ~sick
            n1, n2 = W[:, a].sum(axis=1), W[:, b].sum(axis=1)
            usable = (n1 > 0) & (n2 > 0)
            share1 = np.divide(W[:, a] @ reach[a], n1, where=n1 > 0, out=np.zeros(W.shape[0]))
            share2 = np.divide(W[:, b] @ reach[b], n2, where=n2 > 0, out=np.zeros(W.shape[0]))
            weight = np.divide(n1 * n2, n1 + n2, where=usable, out=np.zeros(W.shape[0]))
            numerator += weight * (share1 - share2)
            denominator += weight
        return numerator / denominator

    estimate = float(weighted(unit)[0])
    low, high = np.quantile(weighted(weights), [0.025, 0.975])
    rows = []
    for cell in np.unique(cells):
        inside = cells == cell
        rows.append(
            {
                "stratum": str(cell),
                "pathological": int((inside & sick).sum()),
                "normal": int((inside & ~sick).sum()),
                "share_pathological": float(reach[inside & sick].mean())
                if (inside & sick).any()
                else None,
                "share_normal": float(reach[inside & ~sick].mean())
                if (inside & ~sick).any()
                else None,
            }
        )
    return {
        "overall_pathological": float(reach[sick].mean()),
        "overall_normal": float(reach[~sick].mean()),
        "within_strata_difference": estimate,
        "within_strata_ci": [float(low), float(high)],
        "strata": rows,
    }


def _stratified_test(name, features, y, groups, cells, config, n_permutations, seed, jobs) -> dict:
    worst = check_permuted_splits(y, groups, cells, config, n_permutations, seed, jobs)
    test = permutation_test(
        features, config, n_permutations=n_permutations, strata=cells, seed=seed, n_jobs=jobs
    )
    null_mean = float(np.nanmean(test.null))
    return {
        "name": name,
        "speakers": int(y.size),
        "uar": float(test.observed),
        "null_mean": null_mean,
        "null_sd": float(np.nanstd(test.null, ddof=1)),
        "uar_minus_null": float(test.observed - null_mean),
        "p_one_sided": float(test.p_value),
        "n_permutations": int(n_permutations),
        "n_undefined": int(test.n_undefined),
        "worst_share_deviation_permuted": worst,
    }, test.null


def run_nonphonation_arm(
    name, meta, features_table, guard, minimum_s, plan, jobs, age_edges, n_permutations
) -> tuple[dict, np.ndarray, pd.DataFrame]:
    seeds = plan["statistics"]["seeds"]
    table = with_availability(meta, guard, minimum_s)
    kept = table[table["reaches_minimum"] == 1.0]
    analysed = analysis_table(kept, features_table, "label")
    config = make_config(plan, POSITIVE)
    arm = evaluate_arm(name, analysed, config)
    features = to_feature_set(analysed)
    y = np.asarray(features.y).astype(str)
    groups = np.asarray(features.groups).astype(str)
    cells = stratum(analysed, age_edges)
    test, null = _stratified_test(
        name, features, y, groups, cells, config, n_permutations, seeds["permutation"], jobs
    )
    if abs(test["uar"] - arm.uar) > 1e-9:
        raise batch.StopRun(
            f"{name}: permutation observed {test['uar']} but the arm gave {arm.uar}"
        )
    return {**arm.row(), **test, "files_left_out": int(len(table) - len(kept))}, null, analysed


def run_c4(meta, tables, plan, jobs, out) -> dict:
    spec = plan["confirmatory_family"]["tests"]["C4"]
    seeds = plan["statistics"]["seeds"]
    minimum = plan["non_phonation_use"]["min_total_s"]
    started = time.perf_counter()
    result, null, analysed = run_nonphonation_arm(
        "c4",
        meta,
        tables["iau_nonphon_g100"],
        "g100",
        minimum,
        plan,
        jobs,
        AGE_EDGES,
        C4_PERMUTATIONS,
    )
    np.save(out / "c4_null.npy", null)
    analysed[["speaker", "label", "age", "sex", "year"]].to_csv(
        out / "c4_speakers.csv", index=False
    )
    config = make_config(plan, POSITIVE)
    negative = batch.negative_control("c4", analysed, config, plan, jobs)

    everyone = with_availability(meta, "g100", minimum)
    cells = stratum(everyone)
    by_label = availability_by_label(everyone, cells, seeds["bootstrap"])
    write_json(out / "c4_availability_by_label.json", by_label)

    # Availability by itself: can having enough quiet tell the labels apart inside the strata?
    own = everyone.assign(path=everyone["file_id"], group=everyone["speaker"])
    own_features = feature_set_from_table(
        own[["path", "group", "label", *AVAILABILITY_COLUMNS]],
        target="label",
        feature_columns=AVAILABILITY_COLUMNS,
        task="classification",
    )
    own_arm = evaluate_columns("availability", own, AVAILABILITY_COLUMNS, config)
    y = np.asarray(own_features.y).astype(str)
    groups = np.asarray(own_features.groups).astype(str)
    own_test, own_null = _stratified_test(
        "availability",
        own_features,
        y,
        groups,
        cells,
        config,
        C4_PERMUTATIONS,
        seeds["permutation"],
        jobs,
    )
    np.save(out / "c4_availability_null.npy", own_null)

    gap = spec["smallest_gap_points"] / 100.0
    limited = bool(
        own_test["p_one_sided"] < 0.05
        or abs(by_label["within_strata_difference"]) > AVAILABILITY_GAP
    )
    return {
        **result,
        "difference": result["uar_minus_null"],
        "smallest_gap": gap,
        "reaches_smallest_gap": bool(result["uar_minus_null"] >= gap),
        "availability_alone": {**own_arm.row(), **own_test},
        "availability_by_label": {k: v for k, v in by_label.items() if k != "strata"},
        "holds_only_for_files_with_enough_quiet": limited,
        "negative": negative,
        "seconds": round(time.perf_counter() - started, 1),
    }


def run_sensitivity(meta, tables, plan, jobs, out) -> dict:
    n = SENSITIVITY_PERMUTATIONS
    minimum = plan["non_phonation_use"]["min_total_s"]
    cases = [
        ("guard_50_ms", tables["iau_nonphon_g050"], "g050", minimum, AGE_EDGES),
        ("guard_200_ms", tables["iau_nonphon_g200"], "g200", minimum, AGE_EDGES),
        ("minimum_0.5_s", tables["iau_nonphon_g100"], "g100", 0.5, AGE_EDGES),
        ("age_bands_10_years", tables["iau_nonphon_g100"], "g100", minimum, AGE_EDGES_SHIFTED),
    ]
    report = {}
    for name, features, guard, minimum_s, edges in cases:
        result, _, _ = run_nonphonation_arm(
            name, meta, features, guard, minimum_s, plan, jobs, edges, n
        )
        report[name] = result
        write_json(out / "c4_sensitivity.json", report)
    return report


def verdict(significant: bool, reaches_gap: bool) -> str:
    """The reading fixed in section 5 of the plan."""
    if significant and reaches_gap:
        return "yes"
    if significant:
        return "statistically present, but small"
    return "not significant; read the width of the interval"


def family_table(results: dict[str, dict]) -> dict:
    """Holm over the four confirmatory tests, only once all four exist."""
    names = ["C1", "C2", "C3", "C4"]
    missing = [name for name in names if name not in results]
    if missing:
        raise ValueError(f"Holm needs all four tests, missing {missing}")
    raw = np.array([results[name]["p_one_sided"] for name in names])
    adjusted = adjust_pvalues(raw, "holm")
    table = {}
    for name, p, q in zip(names, raw, adjusted, strict=True):
        significant = bool(q < 0.05)
        table[name] = {
            "p_one_sided": float(p),
            "p_holm": float(q),
            "estimate": results[name]["estimate"],
            "smallest_gap": results[name]["smallest_gap"],
            "reaches_smallest_gap": bool(results[name]["reaches_smallest_gap"]),
            "reading": verdict(significant, bool(results[name]["reaches_smallest_gap"])),
        }
    return table


def run_family(out: Path) -> dict:
    first = json.loads(FIRST_BATCH.read_text(encoding="utf-8"))
    c3 = json.loads((out / "c3.json").read_text(encoding="utf-8"))
    c4 = json.loads((out / "c4.json").read_text(encoding="utf-8"))
    results = {
        "C1": {**first["c1"], "estimate": first["c1"]["difference"]},
        # C2 reads against chance: the estimate is the UAR above 0.5.
        "C2": {**first["c2"], "estimate": first["c2"]["uar"] - 0.5},
        "C3": {**c3, "estimate": c3["difference"]},
        "C4": {**c4, "estimate": c4["difference"]},
    }
    table = family_table(results)
    write_json(out.parent / "final_family.json", table)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=["c3", "c4", "sensitivity", "family"])
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args()
    try:
        verify_freeze()
        verify_svd_freeze()
    except FreezeError as err:
        sys.exit(str(err))
    plan = load_plan()
    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.what == "family":
        print(json.dumps(run_family(RESULTS), indent=1))
        return
    meta_iau = load_svd_speakers("iau")
    tables = {
        f"iau_{c}": load_table("iau", c)
        for c in (
            "full_norm",
            "phonation_norm",
            "phonation_breath_norm",
            "nonphon_g100",
            "nonphon_g050",
            "nonphon_g200",
        )
    }
    tables["a_n_full_norm"] = load_table("a_n", "full_norm")
    with parallel_config(backend="loky", inner_max_num_threads=1):
        if args.what == "c3":
            result = run_c3(meta_iau, load_svd_speakers("a_n"), tables, plan, args.jobs, RESULTS)
            write_json(RESULTS / "c3.json", result)
        elif args.what == "c4":
            write_json(RESULTS / "c4.json", run_c4(meta_iau, tables, plan, args.jobs, RESULTS))
        else:
            run_sensitivity(meta_iau, tables, plan, args.jobs, RESULTS)
    print(f"{args.what} done in {time.perf_counter() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
