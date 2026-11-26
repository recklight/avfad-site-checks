"""The detector of Ribas et al. 2023 trained inside HSA only; exploratory.

Definitions and readings are in experiments/deviations.md, written before any ComParE feature
was extracted. Their setting, from their repository: ComParE 2016 functionals, z-scored with
the training mean and population std (+1e-20), ``SVC(C=1, kernel="poly", degree=1)``, no
tuning. The features come from features/extract_compare.py, run in opensmile_env.

    jobs        file lists for the extractor (features/out_compare/jobs_*.csv)
    reproduce   their lists, everyone trained, against their own per-file predictions
    hsa         their folds, HSA only; sentences in a split without repeated speakers
    c1          their detector inside C1 (5x10, 50 draws, same bootstrap)
    collect     the readings; writes results/ribas_method/summary.json
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
from voice_severity_bench.evaluation import classification_metrics  # noqa: E402
from voice_severity_bench.splitting import repeated_split_indices  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import published_splits as ps  # noqa: E402
import run_first_batch as batch  # noqa: E402
from arms import write_json  # noqa: E402
from freeze import verify_freeze  # noqa: E402
from freeze_svd import verify_svd_freeze  # noqa: E402
from harness import ROOT, Arm, load_plan  # noqa: E402

COMPARE = ROOT / "features" / "out_compare"
RESULTS = HERE / "results" / "ribas_method"
RAW = ROOT / "data" / "avfad" / "raw"
SENTENCE_TASKS = ("004", "005", "006", "007", "008", "009")
CLASSES = ("normal", "pathological")
MATERIAL_OF = {"a_n": "a", "phrase": "sentences"}

# Fixed before any result; see deviations.md.
REPRODUCE_POINTS = 2.0
REPRODUCE_AGREEMENT = 0.90
SEPARATES = 0.80
FAILS = 0.70
DROP_HOLDS = 0.10


# ------------------------------------------------------------------ jobs and features


def sentence_path(folder: str, speaker: str, task: str) -> str | None:
    """The sentence file as it is on disk; some extensions are upper case."""
    found = [
        p for p in (RAW / folder / speaker).glob(f"{speaker}{task}.*") if p.suffix.lower() == ".wav"
    ]
    return found[0].relative_to(ROOT).as_posix() if found else None


def run_jobs() -> None:
    speakers = pd.read_csv(ROOT / "s0_audit" / "out" / "avfad_speakers.csv", dtype={"speaker": str})
    COMPARE.mkdir(parents=True, exist_ok=True)
    a = pd.DataFrame({"stem": speakers["speaker"] + "002", "path": speakers["a_path"]})
    a.to_csv(COMPARE / "jobs_a.csv", index=False, lineterminator="\n")
    rows = []
    for speaker, folder in zip(speakers["speaker"], speakers["folder"], strict=True):
        for task in SENTENCE_TASKS:
            path = sentence_path(folder, speaker, task)
            if path is not None:
                rows.append({"stem": f"{speaker}{task}", "path": path})
    pd.DataFrame(rows).to_csv(COMPARE / "jobs_sentences.csv", index=False, lineterminator="\n")
    print(len(a), "/a/ files,", len(rows), "sentence files")


def load_compare(material: str) -> pd.DataFrame:
    table = pd.read_csv(COMPARE / f"compare_{material}.csv", dtype={"stem": str})
    return table.set_index("stem")


SUFFIX = ""
BALANCED = False  # set by --balanced: the same SVM with class_weight="balanced", not theirs


def fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, balanced=None):
    """Their z-score and SVM; returns 0/1 predictions and a monotone score in (0, 1)."""
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0) + 1e-20
    model = SVC(
        C=1.0,
        kernel="poly",
        degree=1,
        class_weight="balanced" if (BALANCED if balanced is None else balanced) else None,
    )
    model.fit((x_train - mean) / std, y_train)
    test = (x_test - mean) / std
    return model.predict(test), 1.0 / (1.0 + np.exp(-model.decision_function(test)))


def arm_from(name: str, table: pd.DataFrame, predicted: np.ndarray, scores: np.ndarray) -> Arm:
    """An Arm from (items, repeats) matrices of 0/1 predictions and scores."""
    y = table["label"].to_numpy().astype(str)
    labels = np.where(predicted == 1, "pathological", "normal")
    per_repeat = [
        classification_metrics(
            y,
            labels[:, r],
            np.column_stack([1 - scores[:, r], scores[:, r]]),
            classes=CLASSES,
            positive_class="pathological",
        )
        for r in range(predicted.shape[1])
    ]
    mean = {k: float(np.mean([m[k] for m in per_repeat])) for k in per_repeat[0]}
    return Arm(
        name=name,
        y_true=y,
        groups=table["speaker"].to_numpy().astype(str),
        predictions=labels,
        scores=scores,
        uar=mean["uar"],
        roc_auc=mean["roc_auc"],
        sensitivity=mean["sensitivity"],
        specificity=mean["specificity"],
        worst_share_deviation=float("nan"),
        chosen_c={},
    )


def on_folds(name: str, table: pd.DataFrame, features: pd.DataFrame) -> Arm:
    """Every fold of ``table['fold']`` tested once, the other folds training it."""
    table = table.reset_index(drop=True)
    x = features.loc[table["stem"]].to_numpy(dtype=np.float64)
    y = (table["label"] == "pathological").to_numpy().astype(int)
    predicted = np.zeros(len(table), dtype=int)
    scores = np.zeros(len(table))
    for fold in sorted(table["fold"].unique()):
        test = table["fold"].to_numpy() == fold
        predicted[test], scores[test] = fit_predict(x[~test], y[~test], x[test])
    return arm_from(name, table, predicted[:, None], scores[:, None])


def listed_with_features(material: str, features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    table = ps.listed_table(material)
    missing = sorted(set(table["stem"]) - set(features.index))
    return table[table["stem"].isin(features.index)].reset_index(drop=True), missing


# ------------------------------------------------------------------ their lists


def run_reproduce() -> None:
    predictions = pd.read_csv(ps.SOURCE / "ribas_predictions.csv", dtype=str).astype({"fold": int})
    report = {}
    for material, source in MATERIAL_OF.items():
        features = load_compare(source)
        table, missing = listed_with_features(material, features)
        arm = on_folds("listed", table, features)
        theirs = predictions[predictions["material"] == material].merge(
            table[["fold", "stem"]].assign(ours=arm.predictions[:, 0]),
            on=["fold", "stem"],
            how="inner",
            validate="one_to_one",
        )
        fold_uars = [
            ps.uar_of(g["label"].to_numpy(), g["ours"].to_numpy()) * 100
            for _, g in theirs.groupby("fold")
        ]
        mean_over_folds = float(np.mean(fold_uars))
        agreement = float(np.mean(theirs["ours"] == theirs["predicted"]))
        report[material] = {
            "files": len(table),
            "missing_features": missing,
            "compared_files": len(theirs),
            "uar_mean_over_folds_percent": mean_over_folds,
            "paper_uar_percent": ps.PAPER_UAR[material],
            "agreement_with_their_predictions": agreement,
            "reproduces": bool(
                abs(mean_over_folds - ps.PAPER_UAR[material]) <= REPRODUCE_POINTS
                and agreement >= REPRODUCE_AGREEMENT
            ),
            "uar_pooled": arm.uar,
        }
        print(material, report[material], flush=True)
    write_json(RESULTS / "reproduce.json", report)


def run_hsa() -> None:
    report = {}
    for material, source in MATERIAL_OF.items():
        features = load_compare(source)
        table, missing = listed_with_features(material, features)
        if material == "phrase":
            table = ps.fix_speakers(table)
        overlap = sum(
            len(
                set(table.loc[table["fold"] == k, "speaker"])
                & set(table.loc[table["fold"] != k, "speaker"])
            )
            for k in table["fold"].unique()
        )
        if overlap:
            raise ValueError(f"{material}: speakers on both sides of a fold")
        in_hsa = (table["place"] == "HSA").to_numpy()
        everyone = on_folds("everyone", table, features)
        inside = on_folds("ii", table[in_hsa], features)
        scored = ps.subset_arm("everyone_hsa", everyone, in_hsa)
        report[material] = {
            "split": "their folds"
            if material == "a_n"
            else "speaker_fixed (each speaker in the fold of 004)",
            "missing_features": missing,
            "everyone": ps.arm_summary(everyone),
            "everyone_scored_in_hsa": ps.arm_summary(scored),
            "ii": ps.arm_summary(inside),
            "everyone_minus_ii": ps.contrast([everyone], inside),
            "everyone_minus_everyone_hsa": ps.contrast([everyone], scored),
        }
        print(material, {k: report[material][k] for k in ("everyone", "ii")}, flush=True)
    write_json(RESULTS / f"hsa{SUFFIX}.json", report)


# ------------------------------------------------------------------ inside C1


def c1_items(material: str, features: pd.DataFrame) -> pd.DataFrame:
    """One row per item: a speaker's /a/, or each of a speaker's usable sentences."""
    speakers = batch.load_speakers()
    if material == "a":
        rows = speakers.assign(stem=speakers["speaker"] + "002")
    else:
        excluded = pd.read_csv(ROOT / "s1_segments" / "out_speech" / "excluded.csv", dtype=str)
        dropped = set(excluded["speaker"] + excluded["task"])
        rows = pd.concat(
            [speakers.assign(stem=speakers["speaker"] + task) for task in SENTENCE_TASKS]
        )
        rows = rows[~rows["stem"].isin(dropped)]
    return rows[rows["stem"].isin(features.index)].reset_index(drop=True)


def c1_tables(items: pd.DataFrame) -> dict[str, pd.DataFrame]:
    patients = items[(items["place"] == "HSA") & (items["label"] == "pathological")]
    hsa_normals = items[(items["place"] == "HSA") & (items["label"] == "normal")]
    draws = pd.read_csv(HERE / "results" / "first_batch" / "c1_draws.csv", dtype={"speaker": str})
    tables = {"ii": pd.concat([patients, hsa_normals]), "all": items}
    for index, drawn in draws.groupby("draw"):
        chosen = items[items["speaker"].isin(drawn["speaker"]) & (items["label"] == "normal")]
        tables[f"i_{index:02d}"] = pd.concat([patients, chosen])
    return {k: t.sort_values("stem").reset_index(drop=True) for k, t in tables.items()}


def _one_repeat(x, y, folds, balanced) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.zeros(y.size, dtype=int)
    scores = np.zeros(y.size)
    for train, test in folds:
        predicted[test], scores[test] = fit_predict(x[train], y[train], x[test], balanced)
    return predicted, scores


def run_c1(plan: dict, jobs: int) -> None:
    stats = plan["statistics"]
    n = plan["confirmatory_family"]["tests"]["C1"]["test"]["resamples"]
    path = RESULTS / f"c1{SUFFIX}.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for material in ("a", "sentences"):
        if material in report:
            continue
        features = load_compare(material)
        tables = c1_tables(c1_items(material, features))
        work = []
        for name, table in tables.items():
            x = features.loc[table["stem"]].to_numpy(dtype=np.float64)
            y_text = table["label"].to_numpy().astype(str)
            splits = repeated_split_indices(
                y_text,
                table["speaker"].to_numpy().astype(str),
                task="classification",
                n_splits=stats["cv"]["folds"],
                n_repeats=stats["cv"]["repeats"],
                seed=stats["seeds"]["cv"],
            )
            y = (y_text == "pathological").astype(int)
            work += [(name, r, x, y, folds) for r, folds in enumerate(splits)]
        started = time.perf_counter()
        outputs = Parallel(n_jobs=jobs)(
            delayed(_one_repeat)(x, y, f, BALANCED) for _, _, x, y, f in work
        )
        print(
            material, len(work), "repeats in", round(time.perf_counter() - started), "s", flush=True
        )
        arms = {}
        for name, table in tables.items():
            parts = [
                out for (arm, _, _, _, _), out in zip(work, outputs, strict=True) if arm == name
            ]
            predicted = np.column_stack([p for p, _ in parts])
            scores = np.column_stack([s for _, s in parts])
            arms[name] = arm_from(name, table, predicted, scores)
        long = []
        for name in ("ii", "all"):
            arm, table = arms[name], tables[name]
            repeats = arm.predictions.shape[1]
            long.append(
                pd.DataFrame(
                    {
                        "arm": name,
                        "stem": np.repeat(table["stem"].to_numpy(), repeats),
                        "speaker": np.repeat(arm.groups, repeats),
                        "label": np.repeat(arm.y_true, repeats),
                        "repeat": np.tile(np.arange(repeats), len(table)),
                        "predicted": arm.predictions.ravel(),
                        "score": arm.scores.ravel(),
                    }
                )
            )
        pd.concat(long).to_csv(RESULTS / f"c1_predictions_{material}{SUFFIX}.csv.gz", index=False)
        arms_i = [arm for name, arm in arms.items() if name.startswith("i_")]
        test = batch.c1_bootstrap(arms_i, arms["ii"], n, stats["seeds"]["bootstrap"])
        test.pop("samples")
        uars = [arm.uar for arm in arms_i]
        report[material] = {
            "items_ii": int(arms["ii"].y_true.size),
            "speakers_ii": int(np.unique(arms["ii"].groups).size),
            "ii": ps.arm_summary(arms["ii"]),
            "all": ps.arm_summary(arms["all"]),
            "i_uar_mean": float(np.mean(uars)),
            "i_uar_min": float(np.min(uars)),
            "i_uar_max": float(np.max(uars)),
            "i_roc_auc_mean": float(np.mean([arm.roc_auc for arm in arms_i])),
            "gap_i_minus_ii": test,
        }
        write_json(path, report)
        print(material, {k: report[material][k] for k in ("ii", "i_uar_mean")}, flush=True)


# ------------------------------------------------------------------ readings


def primary_reading(uar_a: float, uar_sentences: float) -> str:
    if max(uar_a, uar_sentences) >= SEPARATES:
        return "their method separates inside one place"
    if uar_a < FAILS and uar_sentences < FAILS:
        return "their method does not separate inside one place either"
    return "middle"


def c1_reading(uar_ii: float, gap: float) -> str:
    return "same as C1" if uar_ii < FAILS and gap >= DROP_HOLDS else "differs from C1"


def run_collect() -> None:
    reproduce = json.loads((RESULTS / "reproduce.json").read_text(encoding="utf-8"))
    hsa = json.loads((RESULTS / "hsa.json").read_text(encoding="utf-8"))
    path = RESULTS / "c1.json"
    c1 = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    report = {
        "reproduces": {m: r["reproduces"] for m, r in reproduce.items()},
        "name": (
            "their method"
            if all(r["reproduces"] for r in reproduce.values())
            else "ComParE plus SVM as reproduced here"
        ),
        "primary": primary_reading(hsa["a_n"]["ii"]["uar"], hsa["phrase"]["ii"]["uar"]),
        "mixed_training_higher": {
            m: bool(
                r["everyone_minus_ii"]["difference"] >= DROP_HOLDS
                and r["everyone_minus_ii"]["ci_low"] > 0
            )
            for m, r in hsa.items()
        },
        "c1": {
            m: c1_reading(r["ii"]["uar"], r["gap_i_minus_ii"]["difference"]) for m, r in c1.items()
        },
    }
    write_json(RESULTS / "summary.json", report)
    print(json.dumps(report, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("jobs", "reproduce", "hsa", "c1", "collect"))
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--balanced", action="store_true", help="class_weight='balanced'")
    args = parser.parse_args()
    global BALANCED, SUFFIX
    BALANCED = args.balanced
    SUFFIX = "_balanced" if args.balanced else ""
    verify_freeze()
    verify_svd_freeze()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.step == "jobs":
        run_jobs()
    elif args.step == "reproduce":
        run_reproduce()
    elif args.step == "hsa":
        run_hsa()
    elif args.step == "c1":
        run_c1(load_plan(), args.jobs)
    else:
        run_collect()


if __name__ == "__main__":
    main()
