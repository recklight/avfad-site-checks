"""Published AVFAD splits against the recording places; exploratory.

Definitions and readings are in experiments/deviations.md, written before anything here was
split by place. Only text lists are read; no audio is fetched and nothing is installed.

    fetch        Ribas 2023 fold lists and test predictions, at a fixed commit, into local tables
    composition  normal speakers and patients by place, per fold, per side; speaker overlap
    published    their own predictions split by place
    mine         C1 on their folds with this study's detectors
    collect      the readings; writes published_splits/summary.json
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from voice_severity_bench import models as vsb_models  # noqa: E402
from voice_severity_bench.bootstrap import group_weights, weighted_metric  # noqa: E402
from voice_severity_bench.comparison import _fit_fold  # noqa: E402
from voice_severity_bench.dataset import feature_set_from_table  # noqa: E402
from voice_severity_bench.evaluation import classification_metrics  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import hypergeom  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_first_batch as batch  # noqa: E402
from arms import HUBERT_C_GRID, layer_columns, load_hubert, write_json  # noqa: E402
from extract_handcrafted import FEATURE_COLUMNS  # noqa: E402
from freeze import verify_freeze  # noqa: E402
from freeze_svd import verify_svd_freeze  # noqa: E402
from harness import ROOT, Arm, load_plan, make_config  # noqa: E402
from sensitivity_avfad import CHOSEN_LAYERS  # noqa: E402

GH = r"C:\Program Files\GitHub CLI\gh.exe"
RIBAS_REPO = "dayanavivolab/voicedisorder"
RIBAS_COMMIT = "e89522bbe9610726ca60778e045be1b042dad619"
WBR_COMMIT = "118f8bb4bd3dbf41e433f65daf39a781eaca2bc6"
MATERIALS = ("a_n", "phrase")
N_FOLDS = 5
SENTENCE_TASKS = ("004", "005", "006", "007", "008", "009")
LABELS = {"HEALTH": "normal", "PATH": "pathological"}

# Table 3 of Ribas et al. 2023, Indep. row, UAR in percent.
PAPER_UAR = {"a_n": 86.70, "phrase": 87.03}
PAPER_TOLERANCE = 0.05

N_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261106

# Fixed before any result; see deviations.md.
DROP_HOLDS = 0.10
DROP_GONE = 0.05
HSA_SEPARATES = 0.80
II_BELOW = 0.70
LEAK_MATTERS = 0.05

RESULTS = HERE / "results" / "published_splits"
SOURCE = RESULTS / "source"
S0_OUT = ROOT / "s0_audit" / "out"
SPEECH_OUT = ROOT / "s1_segments" / "out_speech"
SPEECH_FEATURES = ROOT / "features" / "out_speech"


def gh_raw(repo: str, path: str, ref: str) -> str:
    done = subprocess.run(
        [
            GH,
            "api",
            f"repos/{repo}/contents/{path}?ref={ref}",
            "-H",
            "Accept: application/vnd.github.raw",
        ],
        capture_output=True,
        check=True,
    )
    return done.stdout.decode("utf-8")


def parse_list(text: str) -> list[dict]:
    """Rows of one fold list; the file stem carries the speaker and the task."""
    rows = []
    for item in json.loads(text)["meta_data"]:
        stem = item["path"].rsplit("/", 1)[-1].rsplit(".", 1)[0]
        rows.append({"stem": stem, "speaker": item["speaker"], "label": LABELS[item["label"]]})
    return rows


def parse_predictions(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        if line.strip():
            stem, truth, predicted = line.split()
            rows.append({"stem": stem, "label": LABELS[truth], "predicted": LABELS[predicted]})
    return rows


def run_fetch() -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)
    lists, predictions = [], []
    for material in MATERIALS:
        for fold in range(1, N_FOLDS + 1):
            for side in ("train", "test"):
                path = f"data/lst/AVFAD/{side}_{material}_both_meta_data_fold{fold}.json"
                for row in parse_list(gh_raw(RIBAS_REPO, path, RIBAS_COMMIT)):
                    lists.append({"material": material, "fold": fold, "side": side, **row})
            path = f"data/result/AVFAD/output_{material}_both_fold{fold}_polyd1c1.log"
            for row in parse_predictions(gh_raw(RIBAS_REPO, path, RIBAS_COMMIT)):
                predictions.append({"material": material, "fold": fold, **row})
    pd.DataFrame(lists).to_csv(SOURCE / "ribas_lists.csv", index=False, lineterminator="\n")
    pd.DataFrame(predictions).to_csv(
        SOURCE / "ribas_predictions.csv", index=False, lineterminator="\n"
    )
    write_json(
        SOURCE / "provenance.json",
        {"repo": RIBAS_REPO, "commit": RIBAS_COMMIT, "read": time.strftime("%Y-%m-%d")},
    )
    print(len(lists), "list rows,", len(predictions), "predictions")


def load_speakers() -> pd.DataFrame:
    return pd.read_csv(S0_OUT / "avfad_speakers.csv", dtype={"speaker": str})[
        ["speaker", "place", "label", "age", "sex"]
    ]


def place_group(place: pd.Series) -> pd.Series:
    return place.where(place.isin(["ESSUA", "HSA"]), "other")


def composition_of(table: pd.DataFrame) -> dict:
    """Speakers of one side of one fold, by label and place; each speaker counts once."""
    people = table.drop_duplicates("speaker")
    normal = people[people["label"] == "normal"]
    patients = people[people["label"] == "pathological"]

    def counts(group: pd.DataFrame) -> dict[str, int]:
        found = place_group(group["place"]).value_counts()
        return {p: int(found.get(p, 0)) for p in ("ESSUA", "HSA", "other")}

    share = counts(normal)["ESSUA"] / max(len(normal), 1)
    return {
        "files": len(table),
        "speakers": len(people),
        "normal": counts(normal),
        "patients": counts(patients),
        "essua_share_of_normals": share,
    }


def fold_table_composition(lists: pd.DataFrame) -> list[dict]:
    rows = []
    for fold, in_fold in lists.groupby("fold"):
        train = in_fold[in_fold["side"] == "train"]
        test = in_fold[in_fold["side"] == "test"]
        shared = set(train["speaker"]) & set(test["speaker"])
        rows.append(
            {
                "fold": int(fold),
                "train": composition_of(train),
                "test": composition_of(test),
                "speakers_on_both_sides": len(shared),
            }
        )
    return rows


def is_mixed(folds: list[dict]) -> bool:
    """Every side of every fold holds normal speakers of both places."""
    return all(
        fold[side]["normal"]["ESSUA"] > 0 and fold[side]["normal"]["HSA"] > 0
        for fold in folds
        for side in ("train", "test")
    )


def describe_folds(folds: list[dict]) -> dict:
    train = [f["train"]["essua_share_of_normals"] for f in folds]
    test = [f["test"]["essua_share_of_normals"] for f in folds]
    return {
        "folds": folds,
        "mixed": is_mixed(folds),
        "essua_share_train": [min(train), float(np.mean(train)), max(train)],
        "essua_share_test": [min(test), float(np.mean(test)), max(test)],
        "speakers_on_both_sides": [f["speakers_on_both_sides"] for f in folds],
    }


def wbr_folds(speakers: pd.DataFrame) -> pd.DataFrame:
    """Their StratifiedKFold rebuilt on 708 speakers; PLS has an empty sentence file."""
    kept = speakers[speakers["speaker"] != "PLS"].copy()
    pairs = sorted(zip(kept["speaker"], kept["label"], strict=True))
    names = np.array([s for s, _ in pairs])
    y = np.array([0 if label == "normal" else 1 for _, label in pairs])
    rows = []
    splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for k, (train, test) in enumerate(splitter.split(names, y), start=1):
        rows += [{"fold": k, "side": "train", "speaker": s} for s in names[train]]
        rows += [{"fold": k, "side": "test", "speaker": s} for s in names[test]]
    return pd.DataFrame(rows).merge(speakers, on="speaker", validate="many_to_one")


def expected_share(normals_in_set: int, normals: int, essua: int) -> list[float]:
    """Where the ESSUA share of a random set of normal speakers falls, 95%."""
    low, high = hypergeom.ppf([0.025, 0.975], normals, essua, normals_in_set)
    return [float(low / normals_in_set), float(high / normals_in_set)]


def run_composition() -> None:
    speakers = load_speakers()
    lists = pd.read_csv(SOURCE / "ribas_lists.csv", dtype=str).astype({"fold": int})
    merged = lists.merge(speakers, on="speaker", how="left", suffixes=("", "_meta"))
    unknown = sorted(merged.loc[merged["place"].isna(), "speaker"].unique())
    disagree = merged[merged["place"].notna() & (merged["label"] != merged["label_meta"])]
    report: dict = {
        "corpus": {
            "normal": {
                p: int(
                    ((speakers["label"] == "normal") & (place_group(speakers["place"]) == p)).sum()
                )
                for p in ("ESSUA", "HSA", "other")
            },
            "patients": {
                p: int(
                    (
                        (speakers["label"] == "pathological")
                        & (place_group(speakers["place"]) == p)
                    ).sum()
                )
                for p in ("ESSUA", "HSA", "other")
            },
        },
        "ribas": {},
        "unknown_speakers": unknown,
        "label_disagreements": sorted(disagree["speaker"].unique().tolist()),
    }
    normals = int((speakers["label"] == "normal").sum())
    essua = report["corpus"]["normal"]["ESSUA"]
    for material in MATERIALS:
        table = merged[merged["material"] == material]
        known = table[table["place"].notna()]
        test = table[table["side"] == "test"]
        on_list = set(table["speaker"])
        record = describe_folds(fold_table_composition(known))
        record["speakers_on_list"] = len(on_list)
        record["left_out"] = sorted(set(speakers["speaker"]) - on_list)
        record["test_files_per_speaker"] = float(test.groupby("speaker").size().mean())
        record["folds_per_speaker"] = float(test.groupby("speaker")["fold"].nunique().mean())
        report["ribas"][material] = record
        # A fold's train list has to be the other folds' test lists, or the CV below is wrong.
        for fold in range(1, N_FOLDS + 1):
            train = set(table[(table["fold"] == fold) & (table["side"] == "train")]["stem"])
            rest = set(test[test["fold"] != fold]["stem"])
            if train != rest:
                raise ValueError(f"{material} fold {fold}: train is not the other test folds")
    rebuilt = wbr_folds(speakers)
    rebuilt_folds = []
    for fold, in_fold in rebuilt.groupby("fold"):
        train = in_fold[in_fold["side"] == "train"]
        held = in_fold[in_fold["side"] == "test"]
        rebuilt_folds.append(
            {
                "fold": int(fold),
                "train": composition_of(train),
                "test": composition_of(held),
                "speakers_on_both_sides": len(set(train["speaker"]) & set(held["speaker"])),
            }
        )
    report["wbr_rebuilt"] = {
        "commit": WBR_COMMIT,
        "note": "rebuilt from their script on the 708 speakers; no list of theirs to check against",
        **describe_folds(rebuilt_folds),
    }
    report["expected_without_place_control"] = {
        "share_of_normals_overall": essua / normals,
        # Normal speakers in a test set of these sizes, at the corpus's share of normal speakers.
        "speechdx_test_142_subjects": expected_share(round(142 * normals / 709), normals, essua),
        "mvp_test_fold_71_subjects": expected_share(round(71 * normals / 709), normals, essua),
    }
    write_json(RESULTS / "composition.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "ribas"}, indent=1)[:900])
    for material, record in report["ribas"].items():
        print(
            material,
            {k: v for k, v in record.items() if k != "folds"},
        )


def uar_of(y_true: np.ndarray, predicted: np.ndarray) -> float:
    recalls = [float(np.mean(predicted[y_true == c] == c)) for c in ("normal", "pathological")]
    return float(np.mean(recalls))


def weighted_uar(table: pd.DataFrame, weights: np.ndarray, position: dict[str, int]) -> np.ndarray:
    columns = [position[s] for s in table["speaker"]]
    return weighted_metric(
        table["label"].to_numpy().astype(str),
        table["predicted"].to_numpy().astype(str)[:, None],
        weights[:, columns],
        metric="uar",
    )


def interval(samples: np.ndarray) -> list[float]:
    finite = samples[np.isfinite(samples)]
    return [float(v) for v in np.quantile(finite, [0.025, 0.975])]


def split_by_place(table: pd.DataFrame, seed: int = BOOTSTRAP_SEED, n: int = N_RESAMPLES) -> dict:
    """UAR of one prediction table, overall and by place, with a speaker bootstrap."""
    y = table["label"].to_numpy().astype(str)
    predicted = table["predicted"].to_numpy().astype(str)
    is_normal = y == "normal"
    place = place_group(table["place"]).to_numpy().astype(str)
    hsa = table[place == "HSA"]
    against_essua = table[(place == "ESSUA") | ((place == "HSA") & ~is_normal)]
    everyone = sorted(table["speaker"].unique())
    position = {speaker: i for i, speaker in enumerate(everyone)}
    weights = group_weights(everyone, len(everyone), n, seed=seed)
    over_all = weighted_uar(table, weights, position)
    over_hsa = weighted_uar(hsa, weights, position)
    over_essua = weighted_uar(against_essua, weights, position)

    def rate(mask: np.ndarray, wanted: str) -> dict:
        people = table.loc[mask, "speaker"].nunique()
        return {
            "rate": float(np.mean(predicted[mask] == wanted)),
            "files": int(mask.sum()),
            "speakers": int(people),
        }

    all_uar = uar_of(y, predicted)
    hsa_uar = uar_of(hsa["label"].to_numpy().astype(str), hsa["predicted"].to_numpy().astype(str))
    return {
        "uar_all": all_uar,
        "uar_all_ci": interval(over_all),
        "uar_hsa": hsa_uar,
        "uar_hsa_ci": interval(over_hsa),
        "drop": all_uar - hsa_uar,
        "drop_ci": interval(over_all - over_hsa),
        "uar_essua_normals_vs_hsa_patients": uar_of(
            against_essua["label"].to_numpy().astype(str),
            against_essua["predicted"].to_numpy().astype(str),
        ),
        "uar_essua_normals_vs_hsa_patients_ci": interval(over_essua),
        "specificity_essua": rate(is_normal & (place == "ESSUA"), "normal"),
        "specificity_hsa": rate(is_normal & (place == "HSA"), "normal"),
        "sensitivity_hsa": rate(~is_normal & (place == "HSA"), "pathological"),
    }


def run_published() -> None:
    speakers = load_speakers()
    lists = pd.read_csv(SOURCE / "ribas_lists.csv", dtype=str).astype({"fold": int})
    predictions = pd.read_csv(SOURCE / "ribas_predictions.csv", dtype=str).astype({"fold": int})
    report = {}
    for material in MATERIALS:
        rows = predictions[predictions["material"] == material]
        listed = lists[(lists["material"] == material) & (lists["side"] == "test")]
        joined = rows.merge(
            listed[["fold", "stem", "speaker"]],
            on=["fold", "stem"],
            how="left",
            validate="one_to_one",
        )
        if joined["speaker"].isna().any():
            raise ValueError(f"{material}: a predicted file is not on the test list of its fold")
        joined = joined.merge(
            speakers[["speaker", "place"]], on="speaker", how="left", validate="many_to_one"
        )
        fold_uars = [
            uar_of(group["label"].to_numpy().astype(str), group["predicted"].to_numpy().astype(str))
            for _, group in joined.groupby("fold")
        ]
        mean_over_folds = float(np.mean(fold_uars)) * 100
        report[material] = {
            "uar_per_fold_percent": [u * 100 for u in fold_uars],
            "uar_mean_over_folds_percent": mean_over_folds,
            "paper_uar_percent": PAPER_UAR[material],
            "reproduces_paper": bool(abs(mean_over_folds - PAPER_UAR[material]) <= PAPER_TOLERANCE),
            **split_by_place(joined),
        }
        print(material, {k: v for k, v in report[material].items() if k != "uar_per_fold_percent"})
    write_json(RESULTS / "published_predictions.json", report)


def _use_hubert_grid() -> None:
    vsb_models.SEARCH_SPACES[("classification", "logistic")] = {"model__C": list(HUBERT_C_GRID)}


def fit_listed_fold(table, columns, blocks, config, fold) -> dict:
    """One outer fold of the list: rows of this fold are tested, every other row trains."""
    if blocks is not None:
        _use_hubert_grid()
    features = feature_set_from_table(
        table[["path", "group", "label", *columns]],
        target="label",
        feature_columns=None if blocks is not None else list(columns),
        blocks=blocks,
        task="classification",
    )
    y = np.asarray(features.y).astype(str)
    groups = np.asarray(features.groups).astype(str)
    test = np.flatnonzero(table["fold"].to_numpy() == fold)
    train = np.flatnonzero(table["fold"].to_numpy() != fold)
    classes = tuple(sorted(set(y)))
    predicted, scores, chosen, _ = _fit_fold(
        features.X, y, groups, train, test, config, features.blocks, None, classes
    )
    return {
        "items": test,
        "predicted": predicted,
        "scores": scores[:, classes.index(batch.LABEL_POSITIVE)],
        "chosen": chosen,
    }


def arm_on_folds(name, table, columns, blocks, config) -> Arm:
    """This study's detector with the outer folds taken from ``table['fold']``."""
    table = table.reset_index(drop=True)
    tuned = config.override({"model.block_selection": True}) if blocks is not None else config
    outputs = [
        fit_listed_fold(table, columns, blocks, tuned, fold)
        for fold in sorted(table["fold"].unique())
    ]
    n = len(table)
    predicted = np.empty(n, dtype=object)
    scores = np.full(n, np.nan)
    for output in outputs:
        predicted[output["items"]] = output["predicted"]
        scores[output["items"]] = output["scores"]
    y = table["label"].to_numpy().astype(str)
    classes = ("normal", "pathological")
    metrics = classification_metrics(
        y,
        predicted.astype(str),
        np.column_stack([1 - scores, scores]),
        classes=classes,
        positive_class=batch.LABEL_POSITIVE,
    )
    chosen = Counter(json.dumps(o["chosen"], sort_keys=True) for o in outputs)
    return Arm(
        name=name,
        y_true=y,
        groups=table["speaker"].to_numpy().astype(str),
        predictions=predicted.astype(str)[:, None],
        scores=scores[:, None],
        uar=float(metrics["uar"]),
        roc_auc=float(metrics["roc_auc"]),
        sensitivity=float(metrics["sensitivity"]),
        specificity=float(metrics["specificity"]),
        worst_share_deviation=float("nan"),
        chosen_c=dict(sorted(chosen.items())),
    )


def subset_arm(name: str, arm: Arm, keep: np.ndarray) -> Arm:
    """The same predictions on some speakers; nothing is fitted again."""
    y = arm.y_true[keep]
    predicted = arm.predictions[keep]
    scores = arm.scores[keep]
    classes = ("normal", "pathological")
    metrics = classification_metrics(
        y,
        predicted[:, 0],
        np.column_stack([1 - scores[:, 0], scores[:, 0]]),
        classes=classes,
        positive_class=batch.LABEL_POSITIVE,
    )
    return Arm(
        name=name,
        y_true=y,
        groups=arm.groups[keep],
        predictions=predicted,
        scores=scores,
        uar=float(metrics["uar"]),
        roc_auc=float(metrics["roc_auc"]),
        sensitivity=float(metrics["sensitivity"]),
        specificity=float(metrics["specificity"]),
        worst_share_deviation=float("nan"),
        chosen_c={},
    )


def arm_summary(arm: Arm) -> dict:
    return {
        "items": int(arm.y_true.size),
        "speakers": int(np.unique(arm.groups).size),
        "uar": arm.uar,
        "roc_auc": arm.roc_auc,
        "sensitivity": arm.sensitivity,
        "specificity": arm.specificity,
    }


def contrast(first: list[Arm], second: Arm) -> dict:
    test = batch.c1_bootstrap(first, second, N_RESAMPLES, BOOTSTRAP_SEED)
    test.pop("samples")
    return test


def listed_table(material: str) -> pd.DataFrame:
    """One row per listed file, with its test fold, the speaker's place and the feature key."""
    lists = pd.read_csv(SOURCE / "ribas_lists.csv", dtype=str).astype({"fold": int})
    test = lists[(lists["material"] == material) & (lists["side"] == "test")]
    speakers = load_speakers()
    table = test.merge(
        speakers, on="speaker", how="left", suffixes=("", "_meta"), validate="many_to_one"
    )
    table = table.assign(
        task=table["stem"].str[-3:],
        file_id="avfad__" + table["speaker"],
        path=table["stem"],
        group=table["speaker"],
    )
    return table.drop(columns=["label_meta", "side", "material"]).reset_index(drop=True)


def sentence_features(kind: str, table: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Rows of ``table`` joined to the feature file of each row's sentence."""
    parts = []
    columns = FEATURE_COLUMNS if kind == "handcrafted" else None
    for task in SENTENCE_TASKS:
        rows = table[table["task"] == task]
        if kind == "handcrafted":
            found = pd.read_csv(SPEECH_FEATURES / f"s{task}_full_original.csv")[
                ["file_id", *FEATURE_COLUMNS]
            ]
        else:
            found = load_hubert(f"s{task}", CHOSEN_LAYERS, material="avfad_speech")
        parts.append(rows.merge(found, on="file_id", how="inner", validate="one_to_one"))
    merged = pd.concat(parts, ignore_index=True)
    if columns is None:
        columns = [c for layer in CHOSEN_LAYERS for c in layer_columns(layer)]
    return merged, list(columns)


def fix_speakers(table: pd.DataFrame) -> pd.DataFrame:
    """Every sentence of a speaker goes to the fold of that speaker's lowest task."""
    first = table.sort_values("task").groupby("speaker")["fold"].first()
    return table.assign(fold=table["speaker"].map(first))


def blocks_of(kind: str) -> dict[str, list[str]] | None:
    if kind == "handcrafted":
        return None
    return {f"l{layer:02d}": layer_columns(layer) for layer in CHOSEN_LAYERS}


def run_a_n(kind: str, plan: dict, jobs: int) -> dict:
    config = make_config(plan, batch.LABEL_POSITIVE)
    table = listed_table("a_n")
    if kind == "handcrafted":
        features = batch.load_features("full_original")[["file_id", *FEATURE_COLUMNS]]
        columns = list(FEATURE_COLUMNS)
    else:
        features = load_hubert("full", CHOSEN_LAYERS)
        columns = [c for layer in CHOSEN_LAYERS for c in layer_columns(layer)]
    table = table.merge(features, on="file_id", how="left", validate="one_to_one")
    if table[columns[0]].isna().any():
        raise ValueError("a listed speaker has no /a/ features")
    blocks = blocks_of(kind)
    fold_of = table.set_index("speaker")["fold"]
    is_hsa = table["place"] == "HSA"
    patients = table[is_hsa & (table["label"] == "pathological")]
    hsa_normals = table[is_hsa & (table["label"] == "normal")]
    draws = pd.read_csv(HERE / "results" / "first_batch" / "c1_draws.csv", dtype={"speaker": str})
    by_speaker = table.set_index("speaker", drop=False)
    jobs_list = [("listed", table), ("ii", pd.concat([patients, hsa_normals]))]
    for index, drawn in draws.groupby("draw"):
        on_list = [s for s in drawn["speaker"] if s in fold_of.index]
        jobs_list.append((f"i_{index:02d}", pd.concat([patients, by_speaker.loc[on_list]])))
    started = time.perf_counter()
    arms = Parallel(n_jobs=jobs)(
        delayed(arm_on_folds)(name, part, columns, blocks, config) for name, part in jobs_list
    )
    print(kind, "a_n arms in", round(time.perf_counter() - started, 1), "s", flush=True)
    listed, arm_ii, arms_i = arms[0], arms[1], arms[2:]
    in_hsa = np.array([p == "HSA" for p in table.sort_index()["place"]])
    listed_hsa = subset_arm("listed_hsa", listed, in_hsa)
    uars = [arm.uar for arm in arms_i]
    return {
        "listed": arm_summary(listed),
        "listed_hsa": arm_summary(listed_hsa),
        "ii": arm_summary(arm_ii),
        "i": {
            "draws": len(arms_i),
            "uar_mean": float(np.mean(uars)),
            "uar_min": float(np.min(uars)),
            "uar_max": float(np.max(uars)),
            "roc_auc_mean": float(np.mean([arm.roc_auc for arm in arms_i])),
        },
        "gap_i_minus_ii": contrast(arms_i, arm_ii),
        "listed_minus_ii": contrast([listed], arm_ii),
        "listed_minus_listed_hsa": contrast([listed], listed_hsa),
        "chosen_ii": arm_ii.chosen_c,
    }


def _phrase_arm(name, table, columns, blocks, config) -> Arm:
    return arm_on_folds(name, table, columns, blocks, config)


def run_phrase(kind: str, plan: dict, jobs: int) -> dict:
    config = make_config(plan, batch.LABEL_POSITIVE)
    listed_rows = listed_table("phrase")
    excluded = pd.read_csv(SPEECH_OUT / "excluded.csv", dtype=str)
    table, columns = sentence_features(kind, listed_rows)
    blocks = blocks_of(kind)
    fixed = fix_speakers(table)
    in_hsa = table["place"] == "HSA"
    jobs_list = [
        ("listed", table),
        ("ii", table[in_hsa]),
        ("speaker_fixed", fixed),
        ("speaker_fixed_ii", fixed[in_hsa]),
    ]
    started = time.perf_counter()
    arms = Parallel(n_jobs=jobs)(
        delayed(_phrase_arm)(name, part, columns, blocks, config) for name, part in jobs_list
    )
    print(kind, "phrase arms in", round(time.perf_counter() - started, 1), "s", flush=True)
    listed, arm_ii, fixed_arm, fixed_ii = arms
    mask = in_hsa.to_numpy()
    listed_hsa = subset_arm("listed_hsa", listed, mask)
    fixed_hsa = subset_arm("speaker_fixed_hsa", fixed_arm, mask)
    return {
        "excluded_listed_rows_missing_features": len(listed_rows) - len(table),
        "excluded_by_rule": len(excluded),
        "listed": arm_summary(listed),
        "listed_hsa": arm_summary(listed_hsa),
        "ii": arm_summary(arm_ii),
        "speaker_fixed": arm_summary(fixed_arm),
        "speaker_fixed_hsa": arm_summary(fixed_hsa),
        "speaker_fixed_ii": arm_summary(fixed_ii),
        "listed_minus_ii": contrast([listed], arm_ii),
        "listed_minus_listed_hsa": contrast([listed], listed_hsa),
        "fixed_minus_fixed_ii": contrast([fixed_arm], fixed_ii),
        "chosen_ii": arm_ii.chosen_c,
    }


def run_mine(material: str, kind: str, jobs: int) -> None:
    plan = load_plan()
    path = RESULTS / "mine.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    runner = run_a_n if material == "a_n" else run_phrase
    report[f"{material}_{kind}"] = runner(kind, plan, jobs)
    write_json(path, report)
    print(json.dumps(report[f"{material}_{kind}"], indent=1)[:1500])


def published_verdict(drop: float, ci_low: float) -> str:
    if drop >= DROP_HOLDS and ci_low > 0:
        return "falls past the threshold inside one place"
    if drop < DROP_GONE or ci_low <= 0:
        return "no clear reliance on place"
    return "in between"


def my_verdict(uar_ii: float, gap: float) -> str:
    return "same as C1" if uar_ii < II_BELOW and gap >= DROP_HOLDS else "differs from C1"


def leak_verdict(listed: float, fixed: float) -> str:
    return (
        "overlap raises the score" if listed - fixed >= LEAK_MATTERS else "overlap matters little"
    )


def run_collect() -> None:
    composition = json.loads((RESULTS / "composition.json").read_text(encoding="utf-8"))
    published = json.loads((RESULTS / "published_predictions.json").read_text(encoding="utf-8"))
    path = RESULTS / "mine.json"
    mine = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    report: dict = {"published": {}, "mine": {}, "lists_mixed": {}}
    for material, record in composition["ribas"].items():
        report["lists_mixed"][material] = record["mixed"]
    report["lists_mixed"]["wbr_rebuilt"] = composition["wbr_rebuilt"]["mixed"]
    for material, record in published.items():
        report["published"][material] = {
            "verdict": published_verdict(record["drop"], record["drop_ci"][0]),
            "hsa_separates": bool(record["uar_hsa"] >= HSA_SEPARATES),
            "reproduces_paper": record["reproduces_paper"],
        }
    for key, record in mine.items():
        if key.startswith("a_n"):
            report["mine"][key] = my_verdict(
                record["ii"]["uar"], record["gap_i_minus_ii"]["difference"]
            )
        else:
            report["mine"][key] = {
                "listed": leak_verdict(record["listed"]["uar"], record["speaker_fixed"]["uar"]),
                "inside_hsa": leak_verdict(record["ii"]["uar"], record["speaker_fixed_ii"]["uar"]),
            }
    write_json(RESULTS / "summary.json", report)
    print(json.dumps(report, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("fetch", "composition", "published", "mine", "collect"))
    parser.add_argument("--material", choices=MATERIALS, default="a_n")
    parser.add_argument("--kind", choices=("handcrafted", "hubert"), default="handcrafted")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    verify_freeze()
    verify_svd_freeze()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.step == "fetch":
        run_fetch()
    elif args.step == "composition":
        run_composition()
    elif args.step == "published":
        run_published()
    elif args.step == "mine":
        run_mine(args.material, args.kind, args.jobs)
    else:
        run_collect()


if __name__ == "__main__":
    main()
