"""The remaining exploratory items of section 8; details are in experiments/deviations.md.

Results go to experiments/results/exploratory/section8.json, one key per item, written as
each item finishes.

    .venv/Scripts/python.exe -I -B experiments/section8_rest.py features --jobs 16
    .venv/Scripts/python.exe -I -B experiments/section8_rest.py run --jobs 16
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from voice_severity_bench.bootstrap import bootstrap_metric, group_weights  # noqa: E402
from voice_severity_bench.comparison import compare_feature_sets, pairwise_contrasts  # noqa: E402
from voice_severity_bench.dataset import feature_set_from_table  # noqa: E402
from joblib import parallel_config  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_first_batch as batch  # noqa: E402
import run_svd as svd  # noqa: E402
from arms import evaluate_columns, write_json  # noqa: E402
from extract_handcrafted import (  # noqa: E402
    FEATURE_COLUMNS,
    LTAS_COLUMNS,
    extract_file,
    jobs_for,
)
from freeze import verify_freeze  # noqa: E402
from freeze_svd import verify_svd_freeze  # noqa: E402
from harness import ROOT, load_plan, make_config, predictions_table  # noqa: E402

S0_OUT = ROOT / "s0_audit" / "out"
S1_OUT = ROOT / "s1_segments" / "out"
OUT_FEATURES = ROOT / "features" / "out_section8"
RESULTS = HERE / "results" / "exploratory"
SECTION8 = RESULTS / "section8.json"

M_TECH = [
    "duration_s",
    "lead_s",
    "trail_s",
    "phonation_rms_dbfs",
    "floor_db",
    "clip_fraction",
    "dc_offset",
]
M_TECH_NO_LENGTH = [c for c in M_TECH if c != "duration_s"]
LEAK_COLUMNS = ["nonphonation_total_s", "nonphonation_segments"]
ROOM_BANDS = ["ltas_0126", "ltas_0251", "ltas_0501", "ltas_1000", "ltas_1995", "ltas_3981"]
N_PERMUTATIONS = 1000
N_RESAMPLES = 10_000


def head_and_tail(file_ids: pd.Series, durations: pd.Series, spans: pd.DataFrame) -> pd.DataFrame:
    """Quiet before the first and after the last phonation span, by the S1 spans."""
    phonation = spans[spans["kind"] == "phonation"].groupby("file_id")
    first = phonation["start_s"].min()
    last = phonation["end_s"].max()
    lead = file_ids.map(first)
    trail = durations.to_numpy() - file_ids.map(last).to_numpy()
    if lead.isna().any():
        raise ValueError(f"no phonation span in {file_ids[lead.isna()].tolist()[:5]}")
    return pd.DataFrame({"lead_s": lead.to_numpy(), "trail_s": trail})


def m_tech_table(meta: pd.DataFrame, phonation: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """The seven columns of D4 for the speakers of ``meta``; ``stats`` holds clipping and DC."""
    files = pd.read_csv(S1_OUT / "files.csv").set_index("file_id")
    spans = pd.read_csv(S1_OUT / "spans.csv")
    table = meta.reset_index(drop=True).copy()
    table["duration_s"] = files.loc[table["file_id"], "duration_s"].to_numpy()
    table["floor_db"] = files.loc[table["file_id"], "floor_db"].to_numpy()
    edges = head_and_tail(table["file_id"], table["duration_s"], spans)
    table[["lead_s", "trail_s"]] = edges.to_numpy()
    table = table.merge(
        phonation[["file_id", "phonation_rms_dbfs"]],
        on="file_id",
        how="left",
        validate="one_to_one",
    )
    table = table.merge(stats, on="file_id", how="left", validate="one_to_one")
    missing = table[M_TECH].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"M-tech missing for {table.loc[missing, 'file_id'].tolist()[:5]}")
    return table.assign(
        path=table["file_id"],
        group=table["speaker"],
    )


def avfad_stats() -> pd.DataFrame:
    stats = pd.read_csv(S0_OUT / "avfad_stats.csv")[["path", "clip_fraction", "dc_offset"]]
    files = pd.read_csv(S1_OUT / "files.csv")[["file_id", "wav"]]
    merged = files.merge(stats, left_on="wav", right_on="path", validate="one_to_one")
    return merged[["file_id", "clip_fraction", "dc_offset"]]


def svd_stats(material: str) -> pd.DataFrame:
    stats = pd.read_csv(S0_OUT / "svd_stats.csv")
    stats = stats[stats["material"] == material]
    return pd.DataFrame(
        {
            "file_id": [f"svd__{s}-{material}" for s in stats["session"]],
            "clip_fraction": stats["clip_fraction"].to_numpy(),
            "dc_offset": stats["dc_offset"].to_numpy(),
        }
    )


def voiced_stats() -> pd.DataFrame:
    stats = pd.read_csv(S0_OUT / "voiced_files.csv")
    return pd.DataFrame(
        {
            "file_id": "voiced__" + stats["record"].astype(str),
            "clip_fraction": stats["clip_fraction"].to_numpy(),
            "dc_offset": stats["dc_offset"].to_numpy(),
        }
    )


def uar_interval(arm, seed: int) -> list[float]:
    interval = bootstrap_metric(
        arm.y_true, arm.predictions, arm.groups, metric="uar", n_resamples=N_RESAMPLES, seed=seed
    )
    return [float(interval.low), float(interval.high)]


def arm_record(name, table, columns, config, seed) -> dict:
    arm = evaluate_columns(name, table, columns, config)
    return {
        **arm.row(),
        "uar_ci": uar_interval(arm, seed),
    }


def null_record(name, table, columns, config, cells, plan, jobs) -> dict:
    """Speaker-label permutation, inside ``cells`` when given, and the UAR above its mean."""
    features = feature_set_from_table(
        table[["path", "group", "label", *columns]],
        target="label",
        feature_columns=list(columns),
        task="classification",
    )
    y = np.asarray(features.y).astype(str)
    groups = np.asarray(features.groups).astype(str)
    record, _ = svd._stratified_test(
        name,
        features,
        y,
        groups,
        cells,
        config,
        N_PERMUTATIONS,
        plan["statistics"]["seeds"]["permutation"],
        jobs,
    )
    return record


def with_labels(table: pd.DataFrame, target: str) -> pd.DataFrame:
    return table.assign(label=table[target].astype(str))


def nonphonation_table(meta: pd.DataFrame) -> pd.DataFrame:
    files = pd.read_csv(S1_OUT / "files.csv").set_index("file_id")
    table = meta.reset_index(drop=True).copy()
    table["nonphonation_total_s"] = files.loc[table["file_id"], "nonphonation_g100_s"].to_numpy()
    table["nonphonation_segments"] = files.loc[table["file_id"], "n_nonphonation_g100"].to_numpy()
    return table.assign(path=table["file_id"], group=table["speaker"])


def run_avfad(plan, jobs, report) -> None:
    seeds = plan["statistics"]["seeds"]
    speakers = batch.load_speakers()
    label_config = make_config(plan, batch.LABEL_POSITIVE)
    place_config = make_config(plan, batch.PLACE_POSITIVE)
    hsa = speakers[speakers["place"] == "HSA"]
    minimum = plan["non_phonation_use"]["min_total_s"]
    enough = batch.reaches_minimum(minimum)
    quiet = batch.load_features("nonphon_g100")

    section: dict = {"non_phonation": {}, "m_tech": {}, "segmenter_leak": {}}
    for name, people in (("all", speakers), ("hsa", hsa)):
        kept = people[people["file_id"].isin(enough)]
        table = batch.analysis_table(kept, quiet, "label")
        arm = evaluate_columns(f"nonphon_{name}", table, FEATURE_COLUMNS, label_config)
        section["non_phonation"][name] = {
            **arm.row(),
            "uar_ci": uar_interval(arm, seeds["bootstrap"]),
            "left_out": int(len(people) - len(kept)),
            "negative": batch.negative_control(name, table, label_config, plan, jobs),
        }
        predictions_table([arm]).to_csv(RESULTS / f"nonphon_{name}_predictions.csv.gz", index=False)

    phonation = batch.load_features("phonation_norm")
    stats = avfad_stats()
    c2 = pd.read_csv(HERE / "results" / "first_batch" / "c2_speakers.csv", dtype={"speaker": str})
    normals = speakers[speakers["speaker"].isin(c2["speaker"])]
    sets = {
        "all": (m_tech_table(speakers, phonation, stats), "label", label_config),
        "hsa": (m_tech_table(hsa, phonation, stats), "label", label_config),
        "normal_by_place": (m_tech_table(normals, phonation, stats), "place", place_config),
    }
    for name, (table, target, config) in sets.items():
        labelled = with_labels(table, target)
        section["m_tech"][name] = {
            "with_length": {
                **arm_record(f"mtech_{name}", labelled, M_TECH, config, seeds["bootstrap"]),
                **null_record(f"mtech_{name}", labelled, M_TECH, config, None, plan, jobs),
            },
            "without_length": {
                **arm_record(
                    f"mtech_nolen_{name}", labelled, M_TECH_NO_LENGTH, config, seeds["bootstrap"]
                ),
                **null_record(
                    f"mtech_nolen_{name}", labelled, M_TECH_NO_LENGTH, config, None, plan, jobs
                ),
            },
        }
    for name, people in (("all", speakers), ("hsa", hsa)):
        table = with_labels(nonphonation_table(people), "label")
        section["segmenter_leak"][name] = {
            **arm_record(f"leak_{name}", table, LEAK_COLUMNS, label_config, seeds["bootstrap"]),
            **null_record(f"leak_{name}", table, LEAK_COLUMNS, label_config, None, plan, jobs),
        }
    report["avfad"] = section


def demographic_cell(table: pd.DataFrame) -> np.ndarray:
    """Age band by sex, for the question of what year adds beyond them."""
    age = pd.cut(table["age"], list(svd.AGE_EDGES), right=False).astype(str)
    return (age + "|" + table["sex"].astype(str)).to_numpy().astype(str)


def rank_biserial(values: np.ndarray, sick: np.ndarray) -> float:
    """2 x AUC - 1 of one column, positive when patients have the larger values."""
    if sick.all() or (~sick).all():
        return float("nan")
    u = mannwhitneyu(values[sick], values[~sick], alternative="two-sided").statistic
    return float(2.0 * u / (sick.sum() * (~sick).sum()) - 1.0)


def stratified_rank_biserial(values: np.ndarray, sick: np.ndarray, cells: np.ndarray) -> float:
    numerator, denominator = 0.0, 0.0
    for cell in np.unique(cells):
        inside = cells == cell
        n1, n2 = int((inside & sick).sum()), int((inside & ~sick).sum())
        if n1 and n2:
            weight = n1 * n2 / (n1 + n2)
            numerator += weight * rank_biserial(values[inside], sick[inside])
            denominator += weight
    return numerator / denominator


def stratified_difference(
    values: np.ndarray, sick: np.ndarray, cells: np.ndarray, seed: int
) -> dict[str, float | list[float]]:
    """Patients minus normal speakers inside the strata, with a speaker bootstrap."""
    weights = group_weights(None, values.size, N_RESAMPLES, seed=seed)

    def weighted(W: np.ndarray) -> np.ndarray:
        top = np.zeros(W.shape[0])
        bottom = np.zeros(W.shape[0])
        for cell in np.unique(cells):
            inside = cells == cell
            a, b = inside & sick, inside & ~sick
            n1, n2 = W[:, a].sum(axis=1), W[:, b].sum(axis=1)
            usable = (n1 > 0) & (n2 > 0)
            m1 = np.divide(W[:, a] @ values[a], n1, where=n1 > 0, out=np.zeros(W.shape[0]))
            m2 = np.divide(W[:, b] @ values[b], n2, where=n2 > 0, out=np.zeros(W.shape[0]))
            w = np.divide(n1 * n2, n1 + n2, where=usable, out=np.zeros(W.shape[0]))
            top += w * (m1 - m2)
            bottom += w
        return top / bottom

    low, high = np.quantile(weighted(weights), [0.025, 0.975])
    return {
        "estimate": float(weighted(np.ones((1, values.size)))[0]),
        "ci": [float(low), float(high)],
    }


def run_svd_items(plan, jobs, report) -> None:
    seeds = plan["statistics"]["seeds"]
    config = make_config(plan, svd.POSITIVE)
    section: dict = {}
    meta = svd.load_svd_speakers("iau")

    # Age and sex only.
    table = meta.assign(path=meta["file_id"], group=meta["speaker"])
    table = with_labels(table, "label").assign(is_woman=(table["sex"] == "w").astype(float))
    columns = ["age", "is_woman"]
    forest = config.override({"model.name": "random_forest", "model.class_weight": "balanced"})
    section["age_sex"] = {
        "logistic": arm_record("age_sex_logistic", table, columns, config, seeds["bootstrap"]),
        "random_forest": arm_record("age_sex_forest", table, columns, forest, seeds["bootstrap"]),
    }

    # Recording year only.
    years = sorted(table["year"].astype(int).unique().tolist())
    onehot = {f"year_{y}": (table["year"].astype(int) == y).astype(float) for y in years}
    with_year = table.assign(**onehot, year_value=table["year"].astype(float))
    year_columns = ["year_value", *onehot]
    counts = (
        with_year.groupby([with_year["year"].astype(int), "label"]).size().unstack(fill_value=0)
    )
    section["recording_year"] = {
        **arm_record("year", with_year, year_columns, config, seeds["bootstrap"]),
        **null_record(
            "year", with_year, year_columns, config, demographic_cell(with_year), plan, jobs
        ),
        "speakers_by_year": {
            str(year): {str(k): int(v) for k, v in row.items()} for year, row in counts.iterrows()
        },
    }

    # M-tech on iau and a_n.
    section["m_tech"] = {}
    for material in ("iau", "a_n"):
        people = svd.load_svd_speakers(material)
        condition = "phonation_norm" if material == "iau" else "full_norm"
        phonation = svd.load_table(material, condition)
        tech = with_labels(m_tech_table(people, phonation, svd_stats(material)), "label")
        cells = svd.stratum(tech)
        sick = (tech["label"] == svd.POSITIVE).to_numpy()
        effects = {
            column: {
                "overall": rank_biserial(tech[column].to_numpy(dtype=float), sick),
                "within_strata": stratified_rank_biserial(
                    tech[column].to_numpy(dtype=float), sick, cells
                ),
            }
            for column in M_TECH
        }
        section["m_tech"][material] = {
            "with_length": {
                **arm_record(f"mtech_{material}", tech, M_TECH, config, seeds["bootstrap"]),
                **null_record(f"mtech_{material}", tech, M_TECH, config, cells, plan, jobs),
            },
            "without_length": {
                **arm_record(
                    f"mtech_nolen_{material}", tech, M_TECH_NO_LENGTH, config, seeds["bootstrap"]
                ),
                **null_record(
                    f"mtech_nolen_{material}", tech, M_TECH_NO_LENGTH, config, cells, plan, jobs
                ),
            },
            "effects": effects,
        }

    # Phrase length.
    status = pd.read_csv(S0_OUT / "svd_file_status.csv", keep_default_na=False, dtype=str)
    phrase = status[
        (status["primary"].str.lower() == "true")
        & (status["file_problem"] == "")
        & (status["material"] == "phrase")
    ].copy()
    phrase["session"] = phrase["session"].astype(int)
    phrase["duration_s"] = phrase["duration_s"].astype(float)
    sessions = pd.read_csv(S0_OUT / "svd_sessions.csv")[["session", "sex", "year", "age"]]
    phrase = phrase.merge(sessions, on="session", validate="one_to_one")
    table = phrase.assign(
        path=[f"svd__{s}-phrase" for s in phrase["session"]],
        group=phrase["speaker"],
    )
    table = with_labels(table, "label")
    cells = svd.stratum(table)
    sick = (table["label"] == svd.POSITIVE).to_numpy()
    length = table["duration_s"].to_numpy(dtype=float)
    section["phrase_length"] = {
        **arm_record("phrase_length", table, ["duration_s"], config, seeds["bootstrap"]),
        **null_record("phrase_length", table, ["duration_s"], config, cells, plan, jobs),
        "median_s": {
            svd.POSITIVE: float(np.median(length[sick])),
            "normal": float(np.median(length[~sick])),
        },
        "within_strata_difference_s": stratified_difference(
            length, sick, cells, seeds["bootstrap"]
        ),
        "within_strata_rank_biserial": stratified_rank_biserial(length, sick, cells),
    }
    report["svd"] = section


def voiced_labels() -> pd.DataFrame:
    meta = pd.read_csv(S0_OUT / "voiced_meta.csv")
    label = meta["label_header"].where(meta["label_header"].notna(), meta["label_info"])
    return pd.DataFrame(
        {
            "file_id": "voiced__" + meta["record"].astype(str),
            "speaker": meta["record"].astype(str),
            "label": label.astype(str).str.strip().str.lower(),
        }
    )


def run_voiced(plan, jobs, report) -> None:
    seeds = plan["statistics"]["seeds"]
    meta = voiced_labels()
    full = pd.read_csv(OUT_FEATURES / "voiced_a_full_norm.csv")
    phonation = pd.read_csv(OUT_FEATURES / "voiced_a_phonation_norm.csv")
    usable = [c for c in FEATURE_COLUMNS if not (full[c].isna().all() or phonation[c].isna().all())]
    positive = "pathological"
    config = make_config(plan, positive)

    def feature_set(frame: pd.DataFrame):
        table = meta.merge(frame[["file_id", *usable]], on="file_id", validate="one_to_one")
        table = table.sort_values("file_id").assign(
            path=lambda t: t["file_id"], group=lambda t: t["speaker"]
        )
        return feature_set_from_table(
            table[["path", "group", "label", *usable]],
            target="label",
            feature_columns=usable,
            task="classification",
        )

    result = compare_feature_sets(
        {"full": feature_set(full), "phonation": feature_set(phonation)}, config, n_jobs=jobs
    )
    contrast = svd.contrast_rows(
        pairwise_contrasts(
            result,
            [("full", "phonation")],
            metric="uar",
            n_resamples=N_RESAMPLES,
            seed=seeds["bootstrap"],
        )
    )[0]
    summary = result.summary.set_index("feature_set")
    tech = with_labels(m_tech_table(meta, phonation, voiced_stats()), "label")
    report["voiced"] = {
        "speakers": meta["label"].value_counts().to_dict(),
        "columns_used": len(usable),
        "uar": {n: float(summary.loc[n, "uar"]) for n in ("full", "phonation")},
        "roc_auc": {n: float(summary.loc[n, "roc_auc"]) for n in ("full", "phonation")},
        "full_minus_phonation": contrast,
        "m_tech": {
            "with_length": {
                **arm_record("mtech_voiced", tech, M_TECH, config, seeds["bootstrap"]),
                **null_record("mtech_voiced", tech, M_TECH, config, None, plan, jobs),
            },
            "without_length": {
                **arm_record(
                    "mtech_nolen_voiced", tech, M_TECH_NO_LENGTH, config, seeds["bootstrap"]
                ),
                **null_record(
                    "mtech_nolen_voiced", tech, M_TECH_NO_LENGTH, config, None, plan, jobs
                ),
            },
        },
    }


def run_rooms(report) -> None:
    rooms = pd.read_csv(S0_OUT / "avfad_silence.csv", dtype={"identical_group": str})
    files = pd.read_csv(S1_OUT / "files.csv")
    files = files[files["material"] == "silence"]
    merged = rooms.merge(files, left_on="path", right_on="wav", validate="one_to_one")
    bands = pd.read_csv(OUT_FEATURES / "avfad_silence_nonphon_g100.csv")
    merged = merged.merge(bands[["file_id", *LTAS_COLUMNS]], on="file_id", how="left")
    merged["share_quiet"] = merged["nonphonation_raw_s"] / merged["duration_s_y"]
    rows = {}
    for place, part in merged.groupby("place"):
        groups = part["identical_group"].dropna().unique()
        rows[str(place)] = {
            "files": len(part),
            "groups_of_identical_files": len(groups),
            "files_in_those_groups": int(part["identical_group"].notna().sum()),
            "median_duration_s": float(part["duration_s_y"].median()),
            "median_rms_dbfs": float(part["rms_dbfs"].median()),
            "median_floor_db": float(part["floor_db"].median()),
            "median_share_quiet": float(part["share_quiet"].median()),
            "median_voiced_fraction": float(part["voiced_fraction"].median()),
            "median_band_db": {band: float(part[band].median()) for band in ROOM_BANDS},
        }
    merged[
        ["name", "place", "identical_group", "identical_files", "rms_dbfs", "floor_db", *ROOM_BANDS]
    ].to_csv(RESULTS / "room_silence.csv", index=False)
    report["rooms"] = rows


def run_features(jobs: int) -> None:
    OUT_FEATURES.mkdir(parents=True, exist_ok=True)
    sets = (
        ("voiced", "a", ["full_norm", "phonation_norm"]),
        ("avfad", "silence", ["nonphon_g100"]),
    )
    for database, material, conditions in sets:
        todo = jobs_for(database, material, conditions)
        with ProcessPoolExecutor(jobs) as pool:
            outputs = list(pool.map(extract_file, todo, chunksize=4))
        for condition in conditions:
            rows = [
                {
                    "file_id": job[0],
                    **output[condition][1],
                    **dict(zip(FEATURE_COLUMNS, output[condition][0], strict=True)),
                }
                for job, output in zip(todo, outputs, strict=True)
                if output[condition][0] is not None
            ]
            path = OUT_FEATURES / f"{database}_{material}_{condition}.csv"
            pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")
            print(path.name, len(rows), "of", len(todo), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=["features", "run"])
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args()
    verify_freeze()
    verify_svd_freeze()
    sys.path.insert(0, str(ROOT / "s0_audit"))
    from avfad_extract import require_access

    require_access()
    started = time.perf_counter()
    if args.what == "features":
        run_features(args.jobs)
    else:
        RESULTS.mkdir(parents=True, exist_ok=True)
        plan = load_plan()
        report = json.loads(SECTION8.read_text(encoding="utf-8")) if SECTION8.exists() else {}
        with parallel_config(backend="loky", inner_max_num_threads=1):
            for name, step in (
                ("rooms", lambda: run_rooms(report)),
                ("voiced", lambda: run_voiced(plan, args.jobs, report)),
                ("svd", lambda: run_svd_items(plan, args.jobs, report)),
                ("avfad", lambda: run_avfad(plan, args.jobs, report)),
            ):
                if name in report:
                    continue
                step()
                write_json(SECTION8, report)
                print(name, "done", round(time.perf_counter() - started), "s", flush=True)
    print(f"{args.what} done in {time.perf_counter() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
