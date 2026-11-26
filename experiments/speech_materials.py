"""C1 and the non-phonation arms on AVFAD's sentences and reading; exploratory.

Definitions and readings are in experiments/deviations.md, written before this ran. The
speakers, draws, matching and detector are those of the /a/ runs; only the audio changes.

    index        file list after the exclusion rules, for the segmenter
    full         handcrafted features of whole files
    hubert       HuBERT per file, then pooled into each set of materials
    c1-hand      C1 on every set, handcrafted
    c1-hubert    C1 on cape_v and reading in full, arm (ii) on the other sets
    quiet        handcrafted features of the non-phonation spans (after the segmenter)
    nonphon      non-phonation and pause counts, everyone against HSA
    figures      blind acceptance figures for the segmenter on speech
    collect      fusion, AUC inside the strata, the verdicts; writes speech/summary.json
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
from voice_severity_bench.bootstrap import bootstrap_metric  # noqa: E402
from voice_severity_bench.evaluation import classification_metrics  # noqa: E402
from joblib import Parallel, delayed, parallel_config  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_first_batch as batch  # noqa: E402
import section8_rest as s8  # noqa: E402
import within_strata_auc as strata_auc  # noqa: E402
from arms import evaluate_columns, evaluate_layers, load_hubert, write_json  # noqa: E402
from arms import merge_features as merge  # noqa: E402
from common import cut, load_signal  # noqa: E402
from extract_handcrafted import FEATURE_COLUMNS, features_of  # noqa: E402
from extract_hubert import extract_one  # noqa: E402
from freeze import verify_freeze  # noqa: E402
from freeze_svd import verify_svd_freeze  # noqa: E402
from harness import (  # noqa: E402
    ROOT,
    age_sex_cell,
    analysis_table,
    evaluate_arm,
    load_plan,
    make_config,
    match_cells,
    predictions_table,
)
from sensitivity_avfad import CHOSEN_LAYERS, hsa_groups, shuffled_uar  # noqa: E402

S0_OUT = ROOT / "s0_audit" / "out"
S1_SPEECH = ROOT / "s1_segments" / "out_speech"
FEATURES = ROOT / "features" / "out_speech"
HUBERT_OUT = ROOT / "features" / "out_hubert"
RESULTS = HERE / "results" / "speech"

SENTENCE_TASKS = ("004", "005", "006", "007", "008", "009")
READING_TASK = "010"
SETS: dict[str, tuple[str, ...]] = {
    **{f"s{task}": (task,) for task in SENTENCE_TASKS},
    "cape_v": SENTENCE_TASKS,
    "reading": (READING_TASK,),
    "connected": (*SENTENCE_TASKS, READING_TASK),
}
FULL_C1_HUBERT = ("cape_v", "reading")
MIN_SENTENCE_S = 0.5
MAX_SENTENCE_S = 60.0
MIN_READING_S = 20.0
HUBERT_CONDITION = "speech_full"
HUBERT_MATERIAL = "avfad_speech"
N_SHUFFLES = 5
N_RESAMPLES = 10_000
FIGURES_PER_STRATUM = 5
FIGURE_SEED = 20261205

# Fixed before any result; see deviations.md.
SHRINK_UAR = 0.80
MIDDLE_UAR = 0.70
GAP_HOLDS = 0.10
GAP_GONE = 0.05
QUIET_ALL_HOLDS = 0.75
QUIET_HSA_CHANCE = 0.60


def excluded_reason(task: str, duration_s: float) -> str:
    """Why a file stays out, by its task and length alone; empty when it is used."""
    if task == READING_TASK:
        return "reading_shorter_than_20s" if duration_s < MIN_READING_S else ""
    if duration_s < MIN_SENTENCE_S:
        return "sentence_shorter_than_0.5s"
    if duration_s > MAX_SENTENCE_S:
        return "sentence_longer_than_60s"
    return ""


def file_table() -> pd.DataFrame:
    """Tasks 004-010 with a file id and the reason a file is left out, if any."""
    status = pd.read_csv(
        S0_OUT / "avfad_file_status.csv", dtype={"task": str}, keep_default_na=False
    )
    wanted = status[status["task"].isin((*SENTENCE_TASKS, READING_TASK))].copy()
    wanted["duration_s"] = pd.to_numeric(wanted["duration_s"], errors="coerce").fillna(0.0)
    wanted["excluded"] = [
        excluded_reason(task, duration)
        for task, duration in zip(wanted["task"], wanted["duration_s"], strict=True)
    ]
    wanted["file_id"] = "avfad__" + wanted["speaker"] + "-" + wanted["task"]
    return wanted[["file_id", "speaker", "task", "path", "duration_s", "file_problem", "excluded"]]


def run_index() -> None:
    table = file_table()
    S1_SPEECH.mkdir(parents=True, exist_ok=True)
    table[table["excluded"] != ""].to_csv(S1_SPEECH / "excluded.csv", index=False)
    kept = table[table["excluded"] == ""]
    pd.DataFrame(
        {
            "file_id": kept["file_id"],
            "database": "avfad",
            "material": kept["task"].map(lambda t: "reading" if t == READING_TASK else f"s{t}"),
            "wav": kept["path"],
        }
    ).to_csv(S1_SPEECH / "index.csv", index=False, lineterminator="\n")
    print(len(kept), "files for the segmenter,", int((table["excluded"] != "").sum()), "left out")


def _speaker_full(job: tuple[str, dict[str, str]]) -> dict[str, list]:
    """Features of whole files for every set this speaker has; frames never cross a file."""
    speaker, wavs = job
    signals = {task: load_signal(path) for task, path in wavs.items()}
    rows: dict[str, list] = {}
    for name, tasks in SETS.items():
        pieces = [signals[t][0] for t in tasks if t in signals]
        if pieces:
            rate = signals[next(t for t in tasks if t in signals)][1]
            rows[name] = [
                f"avfad__{speaker}",
                float(sum(p.size for p in pieces) / rate),
                len(pieces),
                *features_of(pieces, rate),
            ]
    return rows


def _speaker_quiet(job: tuple[str, dict[str, str], dict[str, list]]) -> dict[str, list]:
    """Features of the non-phonation spans of every set, pooled across its files."""
    speaker, wavs, spans = job
    signals = {task: load_signal(path) for task, path in wavs.items()}
    rows: dict[str, list] = {}
    for name, tasks in SETS.items():
        pieces, kept, n_spans = [], 0.0, 0
        rate = 16_000
        for task in tasks:
            if task in signals and spans.get(task):
                x, rate = signals[task]
                pieces.extend(cut(x, rate, spans[task]))
                kept += sum(end - start for start, end in spans[task])
                n_spans += len(spans[task])
        if pieces:
            rows[name] = [f"avfad__{speaker}", kept, n_spans, *features_of(pieces, rate)]
    return rows


def _write_rows(outputs: list[dict[str, list]], suffix: str) -> None:
    FEATURES.mkdir(parents=True, exist_ok=True)
    columns = ["file_id", "kept_s", "n_spans", *FEATURE_COLUMNS]
    for name in SETS:
        rows = [output[name] for output in outputs if name in output]
        table = pd.DataFrame(rows, columns=columns).sort_values("file_id")
        table.to_csv(FEATURES / f"{name}_{suffix}.csv", index=False, lineterminator="\n")
        print(name, suffix, len(table), flush=True)


def wavs_by_speaker() -> dict[str, dict[str, str]]:
    table = file_table()
    table = table[table["excluded"] == ""]
    return {
        speaker: dict(zip(part["task"], part["path"], strict=True))
        for speaker, part in table.groupby("speaker")
    }


def run_full(jobs: int, limit: int | None) -> None:
    todo = list(wavs_by_speaker().items())
    todo = todo[:limit] if limit else todo
    started = time.perf_counter()
    with ProcessPoolExecutor(jobs) as pool:
        outputs = list(pool.map(_speaker_full, todo, chunksize=2))
    print(len(todo), "speakers in", round(time.perf_counter() - started, 1), "s", flush=True)
    if not limit:
        _write_rows(outputs, "full_original")


def run_quiet(jobs: int) -> None:
    spans = pd.read_csv(S1_SPEECH / "spans.csv")
    spans = spans[spans["kind"] == "nonphonation_g100"]
    by_file = {
        file_id: list(zip(part["start_s"], part["end_s"], strict=True))
        for file_id, part in spans.groupby("file_id")
    }
    todo = []
    for speaker, wavs in wavs_by_speaker().items():
        here = {t: by_file.get(f"avfad__{speaker}-{t}", []) for t in wavs}
        todo.append((speaker, wavs, here))
    with ProcessPoolExecutor(jobs) as pool:
        outputs = list(pool.map(_speaker_quiet, todo, chunksize=2))
    _write_rows(outputs, "nonphon_g100")


def combine_pooled(pooled: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Mean and std of each layer over several files, from each file's mean, std and frames.

    ``pooled`` is (files, layers, 2 x 768); this gives what pooling all frames together would.
    """
    half = pooled.shape[2] // 2
    means, stds = pooled[:, :, :half], pooled[:, :, half:]
    weights = frames[:, None, None].astype(np.float64)
    total = weights.sum()
    mean = (weights * means).sum(axis=0) / total
    second = (weights * (stds.astype(np.float64) ** 2 + means.astype(np.float64) ** 2)).sum(
        axis=0
    ) / total
    return np.concatenate([mean, np.sqrt(np.maximum(second - mean**2, 0.0))], axis=1)


def run_hubert(jobs: int, limit: int | None) -> None:
    table = file_table()
    table = table[table["excluded"] == ""]
    cache = HUBERT_OUT / "cache" / HUBERT_CONDITION
    cache.mkdir(parents=True, exist_ok=True)
    todo = [
        (HUBERT_CONDITION, row.file_id, row.path, row.duration_s, [(0.0, row.duration_s)])
        for row in table.itertuples()
    ]
    if limit:
        todo = todo[:limit]
    started = time.perf_counter()
    with ProcessPoolExecutor(jobs) as pool:
        done = list(pool.map(extract_one, todo, chunksize=2))
    print(
        len(done),
        "files",
        sum(1 for _, state in done if state == "done"),
        "new, in",
        round(time.perf_counter() - started, 1),
        "s; audio",
        round(sum(job[3] for job in todo)),
        "s",
        flush=True,
    )
    if limit:
        return
    for name, tasks in SETS.items():
        ids, pooled = [], []
        for speaker, part in table[table["task"].isin(tasks)].groupby("speaker"):
            loaded = [np.load(cache / f"{file_id}.npz") for file_id in part["file_id"]]
            ids.append(f"avfad__{speaker}")
            pooled.append(
                combine_pooled(
                    np.stack([item["pooled"] for item in loaded]),
                    np.array([int(item["n_frames"]) for item in loaded]),
                ).astype(np.float32)
            )
        np.savez(
            HUBERT_OUT / f"{HUBERT_MATERIAL}_{name}.npz",
            file_id=np.array(ids),
            pooled=np.stack(pooled),
        )
        print(name, len(ids), flush=True)


def speakers_with(features: pd.DataFrame) -> pd.DataFrame:
    speakers = batch.load_speakers()
    return speakers[speakers["file_id"].isin(features["file_id"])]


def uar_ci(arm, seed: int) -> list[float]:
    interval = bootstrap_metric(
        arm.y_true, arm.predictions, arm.groups, metric="uar", n_resamples=N_RESAMPLES, seed=seed
    )
    return [float(interval.low), float(interval.high)]


def _everyone_arm(name, table, plan):
    return evaluate_columns(name, table, FEATURE_COLUMNS, make_config(plan, batch.LABEL_POSITIVE))


def run_c1_hand(plan, jobs) -> None:
    report_path = RESULTS / "c1_handcrafted.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    for name in SETS:
        if name in report:
            continue
        features = pd.read_csv(FEATURES / f"{name}_full_original.csv")
        speakers = speakers_with(features)
        out = RESULTS / "handcrafted" / name
        out.mkdir(parents=True, exist_ok=True)
        result = batch.run_c1(speakers, features, plan, jobs, out)
        table = analysis_table(speakers, features, "label")
        everyone = _everyone_arm(f"all_{name}", table, plan)
        result["all"] = {
            **everyone.row(),
            "uar_ci": uar_ci(everyone, plan["statistics"]["seeds"]["bootstrap"]),
        }
        result["speakers_left_out"] = int(709 - len(speakers))
        report[name] = result
        write_json(report_path, report)
        print(name, round(result["uar_ii"], 3), round(result["uar_i_mean"], 3), flush=True)


def run_c1_hubert(plan, jobs) -> None:
    seeds = plan["statistics"]["seeds"]
    spec = plan["confirmatory_family"]["tests"]["C1"]
    config = make_config(plan, batch.LABEL_POSITIVE)
    report_path = RESULTS / "c1_hubert.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    for name in sorted(SETS, key=lambda n: n not in FULL_C1_HUBERT):
        if name in report:
            continue
        features = load_hubert(name, CHOSEN_LAYERS, HUBERT_MATERIAL)
        speakers = speakers_with(features)
        patients, hsa, normals = hsa_groups(speakers)
        out = RESULTS / "hubert" / name
        out.mkdir(parents=True, exist_ok=True)
        jobs_list = [("ii", merge(pd.concat([patients, hsa]), features, "label"))]
        if name in FULL_C1_HUBERT:
            rng = np.random.default_rng(seeds["c1_draws"])
            for i in range(spec["draws"]):
                drawn, _ = match_cells(hsa, normals, rng)
                jobs_list.append(
                    (f"i_{i:02d}", merge(pd.concat([patients, drawn]), features, "label"))
                )
            jobs_list.append(("all", merge(speakers, features, "label")))
        arms = Parallel(n_jobs=jobs)(
            delayed(evaluate_layers)(arm, table, CHOSEN_LAYERS, config) for arm, table in jobs_list
        )
        by_name = {arm.name: arm for arm in arms}
        arm_ii = by_name["ii"]
        predictions_table(arms).to_csv(out / "c1_predictions.csv.gz", index=False)
        record = {
            "uar_ii": arm_ii.uar,
            "roc_auc_ii": arm_ii.roc_auc,
            "uar_ii_ci": uar_ci(arm_ii, seeds["bootstrap"]),
            "chosen_ii": arm_ii.chosen_c,
            "patients": len(patients),
            "normals_per_arm": len(hsa),
        }
        if name in FULL_C1_HUBERT:
            arms_i = [arm for key, arm in by_name.items() if key.startswith("i_")]
            test = batch.c1_bootstrap(arms_i, arm_ii, spec["test"]["resamples"], seeds["bootstrap"])
            test.pop("samples")
            uars = [arm.uar for arm in arms_i]
            shuffles = Parallel(n_jobs=jobs)(
                delayed(shuffled_uar)(
                    jobs_list[0][1], CHOSEN_LAYERS, config, seeds["negative_control"] + k
                )
                for k in range(N_SHUFFLES)
            )
            record |= {
                **test,
                "draws": len(arms_i),
                "uar_i_mean": float(np.mean(uars)),
                "uar_i_min": float(np.min(uars)),
                "uar_i_max": float(np.max(uars)),
                "roc_auc_i_mean": float(np.mean([arm.roc_auc for arm in arms_i])),
                "shuffled_ii_mean_uar": float(np.mean(shuffles)),
                "all": {
                    **by_name["all"].row(),
                    "uar_ci": uar_ci(by_name["all"], seeds["bootstrap"]),
                },
            }
        report[name] = record
        write_json(report_path, report)
        print(name, round(arm_ii.uar, 3), flush=True)


def run_nonphon(plan, jobs) -> None:
    seeds = plan["statistics"]["seeds"]
    config = make_config(plan, batch.LABEL_POSITIVE)
    minimum = plan["non_phonation_use"]["min_total_s"]
    speakers = batch.load_speakers()
    report: dict = {}
    for name in ("cape_v", "reading", "connected"):
        quiet = pd.read_csv(FEATURES / f"{name}_nonphon_g100.csv")
        enough = quiet.loc[quiet["kept_s"] >= minimum, "file_id"]
        record: dict = {"non_phonation": {}, "pauses_alone": {}}
        for group, people in (("all", speakers), ("hsa", speakers[speakers["place"] == "HSA"])):
            kept = people[people["file_id"].isin(enough)]
            table = analysis_table(kept, quiet, "label")
            arm = evaluate_arm(f"quiet_{name}_{group}", table, config)
            record["non_phonation"][group] = {
                **arm.row(),
                "uar_ci": uar_ci(arm, seeds["bootstrap"]),
                "left_out": int(len(people) - len(kept)),
                "negative": batch.negative_control(group, table, config, plan, jobs),
            }
            counts = people.merge(
                quiet[["file_id", "kept_s", "n_spans"]], on="file_id", how="left"
            ).fillna({"kept_s": 0.0, "n_spans": 0.0})
            counts = counts.assign(
                path=counts["file_id"],
                group=counts["speaker"],
                label=counts["label"].astype(str),
                nonphonation_total_s=counts["kept_s"],
                nonphonation_segments=counts["n_spans"],
            )
            record["pauses_alone"][group] = s8.arm_record(
                f"pauses_{name}_{group}", counts, s8.LEAK_COLUMNS, config, seeds["bootstrap"]
            )
        report[name] = record
        write_json(RESULTS / "nonphonation.json", report)
        print(name, {g: round(v["uar"], 3) for g, v in record["non_phonation"].items()}, flush=True)


def choose_figures() -> pd.DataFrame:
    """Five files per place and label, for sentences and for reading, shuffled and numbered."""
    index = pd.read_csv(S1_SPEECH / "index.csv", dtype=str)
    index["speaker"] = index["file_id"].str.extract(r"avfad__(.+)-\d+$")[0]
    speakers = batch.load_speakers().set_index("speaker")
    index["place"] = index["speaker"].map(speakers["place"])
    index["label"] = index["speaker"].map(speakers["label"])
    rng = np.random.default_rng(FIGURE_SEED)
    picked = []
    cells = {
        "essua": index["place"] == "ESSUA",
        "hsa_normal": (index["place"] == "HSA") & (index["label"] == "normal"),
        "hsa_pathological": (index["place"] == "HSA") & (index["label"] == "pathological"),
    }
    for kind, is_kind in (
        ("sentence", index["material"] != "reading"),
        ("reading", index["material"] == "reading"),
    ):
        for cell, inside in cells.items():
            members = index[is_kind & inside]
            chosen = rng.choice(members.index.to_numpy(), FIGURES_PER_STRATUM, replace=False)
            picked.append(members.loc[chosen].assign(stratum=f"{kind}_{cell}"))
    sample = pd.concat(picked, ignore_index=True)
    sample = sample.iloc[rng.permutation(len(sample))].reset_index(drop=True)
    sample["figure"] = [f"check_{i + 1:03d}" for i in range(len(sample))]
    return sample


def run_figures() -> None:
    sys.path.insert(0, str(ROOT / "s1_segments"))
    from check_figures import draw_step

    folder = S1_SPEECH / "check"
    folder.mkdir(parents=True, exist_ok=True)
    sample = choose_figures()
    sample[["figure", "file_id", "stratum"]].to_csv(folder / "check_key.csv", index=False)
    sample[["figure"]].assign(correct="", note="").sort_values("figure").to_csv(
        folder / "check_sheet.csv", index=False
    )
    print(draw_step(S1_SPEECH, folder), "figures", flush=True)


def logit(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def fuse(matrices: list[pd.DataFrame]) -> pd.DataFrame:
    """Average the logits of the out-of-fold scores, per repeat, on the speakers all of them have.

    Each score is out-of-fold, so no model is fitted and nothing leaks.
    """
    common = matrices[0].index
    for matrix in matrices[1:]:
        common = common.intersection(matrix.index)
    stacked = np.stack([logit(m.loc[common].to_numpy()) for m in matrices])
    mean = stacked.mean(axis=0)
    return pd.DataFrame(1.0 / (1.0 + np.exp(-mean)), index=common, columns=matrices[0].columns)


def fusion_record(matrix: pd.DataFrame, speakers: pd.DataFrame, seed: int) -> dict:
    """UAR and AUC of the fused scores, per repeat, with the 0.5 cut the classifiers use."""
    people = speakers.loc[matrix.index]
    truth = people["label"].astype(str).to_numpy()
    classes = ("normal", "pathological")
    scores = matrix.to_numpy()
    rows = []
    for repeat in range(scores.shape[1]):
        positive = scores[:, repeat]
        predicted = np.where(positive >= 0.5, "pathological", "normal")
        rows.append(
            classification_metrics(
                truth,
                predicted,
                np.column_stack([1.0 - positive, positive]),
                classes=classes,
                positive_class="pathological",
            )
        )
    return {
        "speakers": int(truth.size),
        "uar": float(np.mean([row["uar"] for row in rows])),
        "roc_auc": float(np.mean([row["roc_auc"] for row in rows])),
        "within_strata": strata_auc.within_strata(
            scores,
            truth == "pathological",
            age_sex_cell(people).to_numpy().astype(str),
            seed,
        ),
    }


def verdict(uar_ii: float, gap: float | None, ci_low: float | None) -> str:
    """The reading fixed before the run, for one material and one detector."""
    if uar_ii >= SHRINK_UAR or (
        gap is not None and (gap < GAP_GONE or (ci_low is not None and ci_low <= 0))
    ):
        return "shrinks to /a/"
    if uar_ii < MIDDLE_UAR and (gap is None or gap >= GAP_HOLDS):
        return "not only /a/"
    return "middle"


def quiet_verdict(uar_all: float, uar_hsa: float) -> str:
    if uar_hsa > QUIET_HSA_CHANCE:
        return "pauses carry the label inside HSA"
    if uar_all >= QUIET_ALL_HOLDS:
        return "not only /a/"
    return "place cue weaker here"


def overall(cells: dict[str, str]) -> str:
    """Across the detectors and the two main materials."""
    values = set(cells.values())
    if "shrinks to /a/" in values:
        return "shrinks to /a/"
    return "not only /a/" if values == {"not only /a/"} else "middle"


def run_collect(plan) -> None:
    seed = strata_auc.SEED
    hand = json.loads((RESULTS / "c1_handcrafted.json").read_text(encoding="utf-8"))
    hub = json.loads((RESULTS / "c1_hubert.json").read_text(encoding="utf-8"))
    speakers = batch.load_speakers().set_index("speaker")
    report: dict = {"inside_strata": {}, "fusion": {}, "verdicts": {}}
    first = HERE / "results"
    sources = {
        "handcrafted": (
            first / "first_batch" / "c1_predictions.csv.gz",
            lambda name: RESULTS / "handcrafted" / name / "c1_predictions.csv.gz",
        ),
        "hubert": (
            first / "sensitivity" / "hubert" / "c1_predictions.csv.gz",
            lambda name: RESULTS / "hubert" / name / "c1_predictions.csv.gz",
        ),
    }
    for kind, (vowel, where) in sources.items():
        report["inside_strata"][kind] = {}
        for name in SETS:
            matrix = strata_auc.avfad_matrix(where(name), "ii")
            people = speakers.loc[matrix.index]
            report["inside_strata"][kind][name] = strata_auc.within_strata(
                matrix.to_numpy(),
                (people["label"] == "pathological").to_numpy(),
                age_sex_cell(people).to_numpy().astype(str),
                seed,
            )
        report["fusion"][kind] = fusion_record(
            fuse(
                [
                    strata_auc.avfad_matrix(vowel, "ii"),
                    strata_auc.avfad_matrix(where("cape_v"), "ii"),
                    strata_auc.avfad_matrix(where("reading"), "ii"),
                ]
            ),
            speakers,
            seed,
        )
    for material in FULL_C1_HUBERT:
        report["verdicts"][material] = {
            "handcrafted": verdict(
                hand[material]["uar_ii"], hand[material]["difference"], hand[material]["ci_low"]
            ),
            "hubert": verdict(
                hub[material]["uar_ii"], hub[material]["difference"], hub[material]["ci_low"]
            ),
        }
    cells = {f"{m}_{k}": v for m, row in report["verdicts"].items() for k, v in row.items()}
    report["overall"] = overall(cells)
    report["fusion_reaches_shrink"] = {
        kind: bool(record["uar"] >= SHRINK_UAR) for kind, record in report["fusion"].items()
    }
    quiet_path = RESULTS / "nonphonation.json"
    if quiet_path.exists():
        quiet = json.loads(quiet_path.read_text(encoding="utf-8"))
        report["quiet_verdicts"] = {
            name: quiet_verdict(
                row["non_phonation"]["all"]["uar"], row["non_phonation"]["hsa"]["uar"]
            )
            for name, row in quiet.items()
        }
    write_json(RESULTS / "summary.json", report)
    print(json.dumps({k: report[k] for k in report if k != "inside_strata"}, indent=1)[:1500])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "what",
        choices=[
            "index",
            "full",
            "hubert",
            "c1-hand",
            "c1-hubert",
            "quiet",
            "nonphon",
            "figures",
            "collect",
        ],
    )
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--limit", type=int, help="first N, to measure speed; writes no table")
    args = parser.parse_args()
    verify_freeze()
    verify_svd_freeze()
    sys.path.insert(0, str(ROOT / "s0_audit"))
    from avfad_extract import require_access

    require_access()
    RESULTS.mkdir(parents=True, exist_ok=True)
    plan = load_plan()
    started = time.perf_counter()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        if args.what == "index":
            run_index()
        elif args.what == "full":
            run_full(args.jobs, args.limit)
        elif args.what == "hubert":
            run_hubert(args.jobs, args.limit)
        elif args.what == "c1-hand":
            run_c1_hand(plan, args.jobs)
        elif args.what == "c1-hubert":
            run_c1_hubert(plan, args.jobs)
        elif args.what == "quiet":
            run_quiet(args.jobs)
        elif args.what == "nonphon":
            run_nonphon(plan, args.jobs)
        elif args.what == "figures":
            run_figures()
        else:
            run_collect(plan)
    print(f"{args.what} done in {time.perf_counter() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
