"""A detector that only learns ESSUA versus HSA among the normals, scored as a pathology detector.

Exploratory. Definitions and readings are in experiments/deviations.md (2026-11-10), written
before this ran. It never sees a patient: every fit uses the ESSUA and HSA normals of the other
parts, and its "HSA-like" score is then read as a pathology score on all of AVFAD /a/.

    .venv/Scripts/python.exe -I -X utf8 -B experiments/place_only.py [step]

``step`` is one of materials, variants, compare, outside, outside_phonation, strata, c1,
c1_phonation, c1_materials, uar, room, sex, agreement, dose, few, published, oracle,
cross_material, paired, paired_a, inside_hsa, by_day; without it the /a/ run.

``materials`` was added after the /a/ run: the same detector on /i/, /u/, the sentences, the
reading and the /a/ phonation spans. ``variants`` came after that: C = 0.01, C = 100 and a plain
mean-difference projection on /a/. ``compare`` uses the ComParE features of the published
system instead, and ``outside`` fits once on all the normals and scores SVD and VOICED;
``outside_phonation`` does the same for SVD on the phonation spans. ``strata`` asks whether
the /a/ score is age and sex, ``c1`` scores the C1 arms with it, and ``uar`` reads its own
boundary as a pathology call. ``room`` uses only the non-phonation spans, and ``sex``
splits the /a/ results by sex.
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "features"))

import diagnosis_breakdown as db  # noqa: E402
import run_first_batch as batch  # noqa: E402
import vowel_fusion as vf  # noqa: E402
from arms import layer_columns, load_hubert, write_json  # noqa: E402
from extract_handcrafted import FEATURE_COLUMNS  # noqa: E402
from freeze import verify_freeze  # noqa: E402
from sensitivity_avfad import CHOSEN_LAYERS  # noqa: E402

RESULTS = HERE / "results" / "place_only"
SEED = 20261151
N_PARTS = 5
N_REPEATS = 10
N_RESAMPLES = 10_000
# The real detectors on all of AVFAD /a/ (section 2 of the summary), for the share only.
FULL_AUC = {"handcrafted": 0.872, "hubert": 0.908}
SPEECH = ("cape_v", "reading")
COMPARE = HERE.parent / "features" / "out_compare"
S1_OUT = HERE.parent / "s1_segments" / "out"
ROOM_SECONDS = 1.0
SPEECH_FEATURES = HERE.parent / "features" / "out_speech"
SPEECH_RESULTS = HERE / "results" / "speech"
MATERIALS = {
    "hubert": ("a_phonation", "i", "u", "cape_v", "reading"),
    "handcrafted": ("a_phonation", "cape_v", "reading"),
}
# Reading, written before the run (deviations.md, 2026-11-10).
MOST = 0.80
MOST_LOW = 0.75
LITTLE = 0.65


def features_of(detector: str, material: str) -> pd.DataFrame:
    """Feature table of one material; speech and /a/ files are keyed by the /a/ file id."""
    if detector == "hubert":
        if material in SPEECH:
            return load_hubert(material, CHOSEN_LAYERS, "avfad_speech")
        if material == "a_phonation":
            return load_hubert("phonation", CHOSEN_LAYERS)
        return load_hubert("full", CHOSEN_LAYERS, material=f"avfad_{material}")
    if material in SPEECH:
        return pd.read_csv(SPEECH_FEATURES / f"{material}_full_original.csv")
    return batch.load_features("phonation_norm" if material == "a_phonation" else "full_original")


def compare_table() -> tuple[pd.DataFrame, list[str]]:
    """ComParE features of /a/, one row per speaker; the file stem is the speaker code plus 002."""
    features = pd.read_csv(COMPARE / "compare_a.csv", dtype={"stem": str})
    columns = [c for c in features.columns if c != "stem"]
    speakers = batch.load_speakers()
    speakers = speakers.assign(stem=speakers["speaker"].astype(str) + "002")
    table = speakers.merge(features, on="stem", how="left", validate="one_to_one")
    if table[columns].isna().any().any():
        raise RuntimeError("a speaker has no ComParE features")
    return table.sort_values("file_id").reset_index(drop=True), columns


def room_table() -> tuple[pd.DataFrame, list[str]]:
    """Non-phonation handcrafted features of the speakers with at least 1.0 s of them."""
    files = pd.read_csv(S1_OUT / "files.csv", dtype={"file_id": str})
    files = files[(files["database"] == "avfad") & (files["material"] == "a")]
    kept = files.loc[files["nonphonation_g100_s"] >= ROOM_SECONDS, "file_id"]
    columns = list(FEATURE_COLUMNS)
    speakers = batch.load_speakers()
    table = speakers[speakers["file_id"].isin(kept)].merge(
        batch.load_features("nonphon_g100")[["file_id", *columns]],
        on="file_id",
        how="left",
        validate="one_to_one",
    )
    if table[columns[0]].isna().any():
        raise RuntimeError("a kept speaker has no non-phonation features")
    return table.sort_values("file_id").reset_index(drop=True), columns


def table_of(detector: str, material: str = "a") -> tuple[pd.DataFrame, list[str]]:
    if detector == "compare":
        return compare_table()
    if detector == "room":
        return room_table()
    speakers = batch.load_speakers()
    if material in ("i", "u"):
        speakers = vf.with_vowel(speakers, material)
    if detector == "hubert":
        columns = [c for layer in CHOSEN_LAYERS for c in layer_columns(layer)]
    else:
        columns = list(FEATURE_COLUMNS)
    features = features_of(detector, material)
    if material == "a":
        table = speakers.merge(
            features[["file_id", *columns]], on="file_id", how="left", validate="one_to_one"
        )
        if table[columns[0]].isna().any():
            raise RuntimeError("a speaker has no features")
    else:
        # Other materials: a speaker without that recording is left out.
        table = speakers.merge(
            features[["file_id", *columns]], on="file_id", how="inner", validate="one_to_one"
        )
        table = table[table[columns].notna().all(axis=1)]
    return table.sort_values("file_id").reset_index(drop=True), columns


def place_scores(
    values: np.ndarray,
    place: np.ndarray,
    normal: np.ndarray,
    test: np.ndarray,
    variant: str = "logistic",
) -> np.ndarray:
    """HSA-vs-ESSUA decision values of the ``test`` rows, fitted on the other rows' normals.

    ``variant`` is ``logistic`` (C = 1.0), ``c0.01``, ``c100`` or ``mean`` (projection on the
    standardised HSA-minus-ESSUA mean difference, no classifier).
    """
    train = ~test & normal & np.isin(place, ["ESSUA", "HSA"])
    hsa = place[train] == "HSA"
    if variant == "mean":
        scaler = StandardScaler().fit(values[train])
        z = scaler.transform(values[train])
        direction = z[hsa].mean(axis=0) - z[~hsa].mean(axis=0)
        return scaler.transform(values[test]) @ direction
    c = {"logistic": 1.0, "c0.01": 0.01, "c100": 100.0}[variant]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))
    clf.fit(values[train], hsa)
    return clf.decision_function(values[test])


def _repeat(
    detector: str, repeat: int, material: str = "a", variant: str = "logistic"
) -> tuple[str, int, np.ndarray]:
    table, columns = table_of(detector, material)
    values = table[columns].to_numpy(dtype=float)
    place = table["place"].to_numpy()
    normal = (table["label"] == "normal").to_numpy()
    strata = table["place"].astype(str) + "_" + table["label"].astype(str)
    folds = StratifiedKFold(N_PARTS, shuffle=True, random_state=SEED + repeat)
    score = np.full(len(table), np.nan)
    for _, test_index in folds.split(values, strata):
        test = np.zeros(len(table), dtype=bool)
        test[test_index] = True
        score[test] = place_scores(values, place, normal, test, variant)
    return detector, repeat, score


def auc_with_ci(score: np.ndarray, sick: np.ndarray, rng: np.random.Generator) -> dict:
    kernel = db.comparisons(score[sick], score[~sick])
    w_sick = db.resample_counts(int(sick.sum()), N_RESAMPLES, rng)
    w_well = db.resample_counts(int((~sick).sum()), N_RESAMPLES, rng)
    boot = db.weighted_auc(kernel, w_sick, w_well)
    low, high = np.percentile(boot, [2.5, 97.5])
    return {"auc": float(kernel.mean()), "ci_low": float(low), "ci_high": float(high)}


def reading(record: dict) -> str:
    if record["auc"] >= MOST and record["ci_low"] > MOST_LOW:
        return "a detector that never saw a patient gets most of the full score"
    if record["auc"] < LITTLE:
        return "place alone does not get the full score"
    return "as measured"


def main(jobs: int = 12) -> None:
    verify_freeze()
    start = time.perf_counter()
    todo = [(d, r) for d in ("handcrafted", "hubert") for r in range(N_REPEATS)]
    done = Parallel(n_jobs=jobs)(delayed(_repeat)(d, r) for d, r in todo)
    rng = np.random.default_rng(SEED)
    summary: dict = {}
    for detector in ("handcrafted", "hubert"):
        table, _ = table_of(detector)
        score = np.mean([s for d, _, s in done if d == detector], axis=0)
        sick = (table["label"] == "pathological").to_numpy()
        hsa = (table["place"] == "HSA").to_numpy()
        everyone = auc_with_ci(score, sick, rng)
        everyone["reading"] = reading(everyone)
        everyone["share_of_full"] = (everyone["auc"] - 0.5) / (FULL_AUC[detector] - 0.5)
        summary[detector] = {
            "speakers": {"all": len(table), "pathological": int(sick.sum())},
            "all_avfad": everyone,
            "hsa_only": auc_with_ci(score[hsa], sick[hsa], rng),
            "median_score": {
                f"{p}_{lab}": float(
                    np.median(score[(table["place"] == p) & (table["label"] == lab)])
                )
                for p, lab in (("ESSUA", "normal"), ("HSA", "normal"), ("HSA", "pathological"))
            },
        }
    summary["seconds"] = round(time.perf_counter() - start, 1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / "summary.json", summary)
    for detector in ("handcrafted", "hubert"):
        s = summary[detector]
        a, h = s["all_avfad"], s["hsa_only"]
        print(
            detector,
            f"all {a['auc']:.3f} [{a['ci_low']:.3f}, {a['ci_high']:.3f}] {a['reading']}",
            f"share {a['share_of_full']:.2f}",
            f"| HSA {h['auc']:.3f} [{h['ci_low']:.3f}, {h['ci_high']:.3f}]",
        )
    print(summary["seconds"], "s")


def full_auc(detector: str, material: str) -> float | None:
    """The real detector on all of AVFAD for this material, where it was stored."""
    if material not in SPEECH:
        return None
    report = json.loads((SPEECH_RESULTS / f"c1_{detector}.json").read_text(encoding="utf-8"))
    return float(report[material]["all"]["roc_auc"])


def materials(jobs: int = 12) -> None:
    verify_freeze()
    start = time.perf_counter()
    todo = [(d, m) for d, ms in MATERIALS.items() for m in ms]
    done = Parallel(n_jobs=jobs)(
        delayed(_repeat)(d, r, m) for d, m in todo for r in range(N_REPEATS)
    )
    rng = np.random.default_rng(SEED)
    summary: dict = {}
    for index, (detector, material) in enumerate(todo):
        table, _ = table_of(detector, material)
        runs = done[index * N_REPEATS : (index + 1) * N_REPEATS]
        score = np.mean([s for _, _, s in runs], axis=0)
        sick = (table["label"] == "pathological").to_numpy()
        hsa = (table["place"] == "HSA").to_numpy()
        everyone = auc_with_ci(score, sick, rng)
        everyone["reading"] = reading(everyone)
        real = full_auc(detector, material)
        if real is not None:
            everyone["real_detector_auc"] = real
            everyone["share_of_full"] = (everyone["auc"] - 0.5) / (real - 0.5)
        summary.setdefault(detector, {})[material] = {
            "speakers": {"all": len(table), "pathological": int(sick.sum())},
            "all_avfad": everyone,
            "hsa_only": auc_with_ci(score[hsa], sick[hsa], rng),
        }
    most = "a detector that never saw a patient gets most of the full score"
    every = all(r["all_avfad"]["reading"] == most for d in summary.values() for r in d.values())
    phonation = all(summary[d]["a_phonation"]["all_avfad"]["reading"] == most for d in MATERIALS)
    if every:
        summary["overall"] = "the same on every material and on phonation only"
    elif phonation:
        summary["overall"] = "the same on phonation only"
    else:
        summary["overall"] = "as measured"
    summary["seconds"] = round(time.perf_counter() - start, 1)
    write_json(RESULTS / "summary_materials.json", summary)
    for detector, part in summary.items():
        if not isinstance(part, dict):
            continue
        for material, r in part.items():
            a, h = r["all_avfad"], r["hsa_only"]
            print(
                detector,
                material,
                r["speakers"]["all"],
                f"all {a['auc']:.3f} [{a['ci_low']:.3f}, {a['ci_high']:.3f}]",
                f"real {a.get('real_detector_auc', float('nan')):.3f}",
                f"| HSA {h['auc']:.3f} [{h['ci_low']:.3f}, {h['ci_high']:.3f}]",
            )
    print(summary["overall"], summary["seconds"], "s")


VARIANTS = ("c0.01", "c100", "mean")


def variants(jobs: int = 12) -> None:
    verify_freeze()
    start = time.perf_counter()
    todo = [(d, v) for d in ("handcrafted", "hubert") for v in VARIANTS]
    done = Parallel(n_jobs=jobs)(
        delayed(_repeat)(d, r, "a", v) for d, v in todo for r in range(N_REPEATS)
    )
    rng = np.random.default_rng(SEED)
    summary: dict = {}
    for index, (detector, variant) in enumerate(todo):
        table, _ = table_of(detector)
        runs = done[index * N_REPEATS : (index + 1) * N_REPEATS]
        score = np.mean([s for _, _, s in runs], axis=0)
        sick = (table["label"] == "pathological").to_numpy()
        hsa = (table["place"] == "HSA").to_numpy()
        summary.setdefault(detector, {})[variant] = {
            "all_avfad": auc_with_ci(score, sick, rng),
            "hsa_only": auc_with_ci(score[hsa], sick[hsa], rng),
        }
    for detector in ("handcrafted", "hubert"):
        part = summary[detector]
        steady = all(part[v]["all_avfad"]["auc"] >= MOST for v in VARIANTS)
        part["reading"] = "does not depend on the classifier" if steady else "as measured"
    summary["seconds"] = round(time.perf_counter() - start, 1)
    write_json(RESULTS / "summary_variants.json", summary)
    for detector in ("handcrafted", "hubert"):
        for variant in VARIANTS:
            a, h = summary[detector][variant]["all_avfad"], summary[detector][variant]["hsa_only"]
            print(
                detector,
                variant,
                f"all {a['auc']:.3f} [{a['ci_low']:.3f}, {a['ci_high']:.3f}]",
                f"| HSA {h['auc']:.3f} [{h['ci_low']:.3f}, {h['ci_high']:.3f}]",
            )
        print(detector, summary[detector]["reading"])
    print(summary["seconds"], "s")


def compare(jobs: int = 12) -> None:
    verify_freeze()
    start = time.perf_counter()
    done = Parallel(n_jobs=jobs)(delayed(_repeat)("compare", r) for r in range(N_REPEATS))
    rng = np.random.default_rng(SEED)
    table, columns = table_of("compare")
    score = np.mean([s for _, _, s in done], axis=0)
    sick = (table["label"] == "pathological").to_numpy()
    hsa = (table["place"] == "HSA").to_numpy()
    everyone = auc_with_ci(score, sick, rng)
    everyone["reading"] = reading(everyone)
    summary = {
        "columns": len(columns),
        "speakers": {"all": len(table), "pathological": int(sick.sum())},
        "all_avfad": everyone,
        "hsa_only": auc_with_ci(score[hsa], sick[hsa], rng),
        "seconds": round(time.perf_counter() - start, 1),
    }
    write_json(RESULTS / "summary_compare.json", summary)
    a, h = summary["all_avfad"], summary["hsa_only"]
    print(
        f"compare all {a['auc']:.3f} [{a['ci_low']:.3f}, {a['ci_high']:.3f}] {a['reading']}",
        f"| HSA {h['auc']:.3f} [{h['ci_low']:.3f}, {h['ci_high']:.3f}]",
        summary["seconds"],
        "s",
    )


def outside_sets() -> dict[str, tuple[pd.DataFrame, pd.DataFrame, list[str]]]:
    """(AVFAD normals of the two places, outside test set, columns) per detector and database."""
    import cross_database as cd
    import run_svd as svd
    import section8_rest as s8

    normals = batch.load_speakers()
    normals = normals[(normals["label"] == "normal") & normals["place"].isin(["ESSUA", "HSA"])]
    svd_meta = svd.load_svd_speakers("a_n")[["file_id", "label"]]
    columns = [c for layer in CHOSEN_LAYERS for c in layer_columns(layer)]
    hubert_avfad = normals.merge(load_hubert("full", CHOSEN_LAYERS), on="file_id", how="left")
    hubert_svd = svd_meta.merge(
        load_hubert("full", CHOSEN_LAYERS, material="svd_a_n"), on="file_id", how="inner"
    )
    data = np.load(cd.VOICED_HUBERT)
    values = np.concatenate([data["pooled"][:, layer, :] for layer in CHOSEN_LAYERS], axis=1)
    voiced = pd.DataFrame(values, columns=columns)
    voiced.insert(0, "file_id", data["file_id"].astype(str))
    labels = s8.voiced_labels()
    labels = labels.assign(label=labels["label"].replace({"healthy": "normal"}))
    voiced = labels[["file_id", "label"]].merge(voiced, on="file_id", how="inner")
    hand = list(FEATURE_COLUMNS)
    hand_avfad = normals.merge(
        pd.read_csv(cd.EXPLORATORY / "avfad_a_full_norm.csv")[["file_id", *hand]],
        on="file_id",
        how="left",
    )
    hand_svd = svd_meta.merge(
        pd.read_csv(HERE.parent / "features" / "out" / "svd_a_n_full_norm.csv")[["file_id", *hand]],
        on="file_id",
        how="inner",
    )
    return {
        "hubert_svd": (hubert_avfad, hubert_svd, columns),
        "hubert_voiced": (hubert_avfad, voiced, columns),
        "handcrafted_svd": (hand_avfad, hand_svd, hand),
    }


def outside_reading(record: dict) -> str:
    if record["ci_low"] <= 0.5 <= record["ci_high"]:
        return "the place direction carries no pathology outside"
    if record["auc"] >= 0.60 and record["ci_low"] > 0.5:
        return "the place direction carries some pathology outside"
    return "as measured"


def outside() -> None:
    verify_freeze()
    start = time.perf_counter()
    rng = np.random.default_rng(SEED)
    summary: dict = {}
    for name, (train, test, columns) in outside_sets().items():
        x_train = train[columns].to_numpy(dtype=float)
        x_test = test[columns].to_numpy(dtype=float)
        if np.isnan(x_train).any() or np.isnan(x_test).any():
            raise RuntimeError(f"{name}: missing features")
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        clf.fit(x_train, (train["place"] == "HSA").to_numpy())
        score = clf.decision_function(x_test)
        sick = (test["label"] == "pathological").to_numpy()
        record = auc_with_ci(score, sick, rng)
        record["reading"] = outside_reading(record)
        record["speakers"] = {"all": len(test), "pathological": int(sick.sum())}
        record["trained_on"] = {p: int((train["place"] == p).sum()) for p in ("ESSUA", "HSA")}
        summary[name] = record
        print(
            name,
            record["speakers"],
            f"{record['auc']:.3f} [{record['ci_low']:.3f}, {record['ci_high']:.3f}]",
            record["reading"],
        )
    summary["seconds"] = round(time.perf_counter() - start, 1)
    write_json(RESULTS / "summary_outside.json", summary)


def outside_phonation() -> None:
    verify_freeze()
    import run_svd as svd

    normals = batch.load_speakers()
    normals = normals[(normals["label"] == "normal") & normals["place"].isin(["ESSUA", "HSA"])]
    columns = [c for layer in CHOSEN_LAYERS for c in layer_columns(layer)]
    train = normals.merge(load_hubert("phonation", CHOSEN_LAYERS), on="file_id", how="left")
    test = svd.load_svd_speakers("a_n")[["file_id", "label"]].merge(
        load_hubert("phonation", CHOSEN_LAYERS, material="svd_a_n"), on="file_id", how="inner"
    )
    x_train = train[columns].to_numpy(dtype=float)
    x_test = test[columns].to_numpy(dtype=float)
    if np.isnan(x_train).any() or np.isnan(x_test).any():
        raise RuntimeError("missing phonation features")
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    clf.fit(x_train, (train["place"] == "HSA").to_numpy())
    score = clf.decision_function(x_test)
    sick = (test["label"] == "pathological").to_numpy()
    record = auc_with_ci(score, sick, np.random.default_rng(SEED))
    record["reading"] = outside_reading(record)
    if record["reading"] == "the place direction carries some pathology outside":
        record["source"] = "what goes out is in the voiced part"
    elif record["ci_low"] <= 0.5 <= record["ci_high"]:
        record["source"] = "what goes out is silence and length"
    else:
        record["source"] = "as measured"
    record["speakers"] = {"all": len(test), "pathological": int(sick.sum())}
    write_json(RESULTS / "summary_outside_phonation.json", record)
    print(
        f"hubert_svd_phonation {record['auc']:.3f} [{record['ci_low']:.3f}, "
        f"{record['ci_high']:.3f}] {record['reading']} | {record['source']}"
    )


def strata_reading(record: dict) -> str:
    if record["auc"] >= MOST and record["ci"][0] > MOST_LOW:
        return "not age and sex"
    if record["auc"] < LITTLE:
        return "mostly age and sex"
    return "as measured"


def _age_sex_repeat(repeat: int) -> np.ndarray:
    """Out-of-part pathology scores from age and sex alone; unlike the rest, it sees patients."""
    table, columns = table_of("handcrafted")
    male = (table["sex"].astype(str).str.upper().str[0] == "M").to_numpy(dtype=float)
    age = table["age"].to_numpy(dtype=float)
    x = np.column_stack([male, age, male * age])
    sick = (table["label"] == "pathological").to_numpy()
    strata = table["place"].astype(str) + "_" + table["label"].astype(str)
    folds = StratifiedKFold(N_PARTS, shuffle=True, random_state=SEED + repeat)
    score = np.full(len(table), np.nan)
    for train, test in folds.split(table[columns].to_numpy(), strata):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        clf.fit(x[train], sick[train])
        score[test] = clf.decision_function(x[test])
    return score


def strata(jobs: int = 12) -> None:
    verify_freeze()
    import within_strata_auc as wsa
    from harness import age_sex_cell

    start = time.perf_counter()
    todo = [(d, r) for d in ("handcrafted", "hubert") for r in range(N_REPEATS)]
    done = Parallel(n_jobs=jobs)(delayed(_repeat)(d, r) for d, r in todo)
    summary: dict = {}
    for detector in ("handcrafted", "hubert"):
        table, _ = table_of(detector)
        matrix = np.column_stack([s for d, _, s in done if d == detector])
        sick = (table["label"] == "pathological").to_numpy()
        cells = age_sex_cell(table).to_numpy().astype(str)
        record = wsa.within_strata(matrix, sick, cells, SEED)
        record["reading"] = strata_reading(record)
        summary[detector] = record
        print(
            detector,
            round(record["auc"], 3),
            [round(c, 3) for c in record["ci"]],
            record["reading"],
        )
    table, _ = table_of("handcrafted")
    sick = (table["label"] == "pathological").to_numpy()
    plain = np.mean([_age_sex_repeat(r) for r in range(N_REPEATS)], axis=0)
    summary["age_sex_only"] = auc_with_ci(plain, sick, np.random.default_rng(SEED))
    print("age_sex_only", {k: round(v, 3) for k, v in summary["age_sex_only"].items()})
    summary["seconds"] = round(time.perf_counter() - start, 1)
    write_json(RESULTS / "summary_strata.json", summary)


REAL_ARMS = {
    "a": {
        "handcrafted": HERE / "results" / "first_batch" / "c1_arms.csv",
        "hubert": HERE / "results" / "sensitivity" / "hubert" / "c1_arms.csv",
    },
    "a_phonation": {
        "handcrafted": HERE / "results" / "phonation_c1" / "c1_arms_handcrafted.csv",
        "hubert": HERE / "results" / "phonation_c1" / "c1_arms_hubert.csv",
    },
}


def c1_reading(place_gap: float, real_gap: float) -> str:
    if place_gap >= 0.8 * real_gap:
        return "a detector that never saw a patient reproduces most of the C1 gap"
    if place_gap <= 0.2 * real_gap:
        return "it does not reproduce the gap"
    return "as measured"


def c1(jobs: int = 12, material: str = "a") -> None:
    verify_freeze()
    from harness import load_plan

    start = time.perf_counter()
    todo = [(d, r) for d in ("handcrafted", "hubert") for r in range(N_REPEATS)]
    done = Parallel(n_jobs=jobs)(delayed(_repeat)(d, r, material) for d, r in todo)
    sets = [(n, m) for n, m in vf.arm_sets(load_plan()) if not n.startswith("shuffled")]
    summary: dict = {}
    for detector in ("handcrafted", "hubert"):
        table, _ = table_of(detector, material)
        score = pd.Series(
            np.mean([s for d, _, s in done if d == detector], axis=0),
            index=table["speaker"].astype(str),
        )
        place = {}
        for name, members in sets:
            ids = members["speaker"].astype(str).to_numpy()
            sick = (members["label"] == "pathological").to_numpy()
            values = score.loc[ids].to_numpy()
            place[name] = float(db.comparisons(values[sick], values[~sick]).mean())
        real = pd.read_csv(REAL_ARMS[material][detector]).set_index("arm")["roc_auc"]
        draws = [n for n in place if n.startswith("i_")]
        place_i = [place[n] for n in draws]
        real_i = [float(real[n]) for n in draws]
        record = {
            "place_ii": place["ii"],
            "place_i_mean": float(np.mean(place_i)),
            "place_i_range": [min(place_i), max(place_i)],
            "real_ii": float(real["ii"]),
            "real_i_mean": float(np.mean(real_i)),
            "place_gap": float(np.mean(place_i)) - place["ii"],
            "real_gap": float(np.mean(real_i)) - float(real["ii"]),
        }
        record["reading"] = c1_reading(record["place_gap"], record["real_gap"])
        summary[detector] = record
        print(
            detector, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in record.items()}
        )
    summary["seconds"] = round(time.perf_counter() - start, 1)
    name = "summary_c1.json" if material == "a" else f"summary_c1_{material.split('_')[1]}.json"
    write_json(RESULTS / name, summary)


REAL_UAR = {"handcrafted": 0.820, "hubert": 0.835, "compare": 0.858}


def uar_at_boundary(score: np.ndarray, sick: np.ndarray) -> float:
    """UAR when "HSA" (decision > 0) is read as pathological."""
    called = score > 0
    return float((called[sick].mean() + (~called[~sick]).mean()) / 2.0)


def uar(jobs: int = 12) -> None:
    verify_freeze()
    start = time.perf_counter()
    todo = [(d, r) for d in REAL_UAR for r in range(N_REPEATS)]
    done = Parallel(n_jobs=jobs)(delayed(_repeat)(d, r) for d, r in todo)
    summary: dict = {}
    for detector, real in REAL_UAR.items():
        table, _ = table_of(detector)
        sick = (table["label"] == "pathological").to_numpy()
        runs = [s for d, _, s in done if d == detector]
        values = [uar_at_boundary(s, sick) for s in runs]
        record = {
            "uar": float(np.mean(values)),
            "uar_range": [min(values), max(values)],
            "sensitivity": float(np.mean([(s > 0)[sick].mean() for s in runs])),
            "specificity": float(np.mean([(s <= 0)[~sick].mean() for s in runs])),
            "real_uar": real,
        }
        record["reading"] = (
            "about the same in UAR" if record["uar"] >= real - 0.05 else "as measured"
        )
        summary[detector] = record
        print(
            detector, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in record.items()}
        )
    summary["seconds"] = round(time.perf_counter() - start, 1)
    write_json(RESULTS / "summary_uar.json", summary)


def room(jobs: int = 12) -> None:
    verify_freeze()
    start = time.perf_counter()
    done = Parallel(n_jobs=jobs)(delayed(_repeat)("room", r) for r in range(N_REPEATS))
    table, _ = table_of("room")
    runs = [s for _, _, s in done]
    score = np.mean(runs, axis=0)
    sick = (table["label"] == "pathological").to_numpy()
    hsa = (table["place"] == "HSA").to_numpy()
    rng = np.random.default_rng(SEED)
    everyone = auc_with_ci(score, sick, rng)
    everyone["reading"] = reading(everyone)
    summary = {
        "speakers": {
            "all": len(table),
            "pathological": int(sick.sum()),
            "hsa": int(hsa.sum()),
        },
        "all_avfad": everyone,
        "hsa_only": auc_with_ci(score[hsa], sick[hsa], rng),
        "uar": float(np.mean([uar_at_boundary(s, sick) for s in runs])),
        "seconds": round(time.perf_counter() - start, 1),
    }
    write_json(RESULTS / "summary_room.json", summary)
    a, h = summary["all_avfad"], summary["hsa_only"]
    print(
        summary["speakers"],
        f"all {a['auc']:.3f} [{a['ci_low']:.3f}, {a['ci_high']:.3f}] {a['reading']}",
        f"| HSA {h['auc']:.3f} [{h['ci_low']:.3f}, {h['ci_high']:.3f}]",
        f"| UAR {summary['uar']:.3f}",
    )


REAL_PREDICTIONS = {
    "handcrafted": HERE / "results" / "first_batch" / "c1_predictions.csv.gz",
    "hubert": HERE / "results" / "sensitivity" / "hubert" / "c1_predictions.csv.gz",
}


def c1_auc_by_sex(path: Path, sex_of: pd.Series) -> dict[str, dict]:
    """Per sex: (ii) AUC, mean (i) AUC and the gap, each AUC averaged over the repeats."""
    table = pd.read_csv(path, dtype={"speaker": str})
    table = table[~table["arm"].str.startswith("shuffled")]
    table = table.assign(sex=table["speaker"].map(sex_of))
    if table["sex"].isna().any():
        raise RuntimeError("a C1 speaker has no sex")
    out = {}
    for sex, part in table.groupby("sex"):
        aucs = {}
        for (arm, _), rows in part.groupby(["arm", "repeat"]):
            sick = (rows["label"] == "pathological").to_numpy()
            values = rows["score"].to_numpy()
            aucs.setdefault(arm, []).append(
                float(db.comparisons(values[sick], values[~sick]).mean())
            )
        mean = {arm: float(np.mean(v)) for arm, v in aucs.items()}
        draws = [v for arm, v in mean.items() if arm.startswith("i_")]
        out[str(sex)] = {
            "ii": mean["ii"],
            "i_mean": float(np.mean(draws)),
            "gap": float(np.mean(draws)) - mean["ii"],
            "normals_in_ii": int(
                ((part["arm"] == "ii") & (part["label"] == "normal")).sum() // N_REPEATS
            ),
        }
    return out


def by_sex(jobs: int = 12) -> None:
    verify_freeze()
    start = time.perf_counter()
    todo = [(d, r) for d in ("handcrafted", "hubert") for r in range(N_REPEATS)]
    done = Parallel(n_jobs=jobs)(delayed(_repeat)(d, r) for d, r in todo)
    rng = np.random.default_rng(SEED)
    summary: dict = {}
    for detector in ("handcrafted", "hubert"):
        table, _ = table_of(detector)
        score = np.mean([s for d, _, s in done if d == detector], axis=0)
        sick = (table["label"] == "pathological").to_numpy()
        part: dict = {"place_only": {}}
        for sex in sorted(table["sex"].astype(str).unique()):
            rows = (table["sex"].astype(str) == sex).to_numpy()
            record = auc_with_ci(score[rows], sick[rows], rng)
            record["speakers"] = int(rows.sum())
            record["pathological"] = int((rows & sick).sum())
            part["place_only"][sex] = record
        sex_of = table.set_index(table["speaker"].astype(str))["sex"].astype(str)
        part["c1"] = c1_auc_by_sex(REAL_PREDICTIONS[detector], sex_of)
        steady = all(r["auc"] >= MOST for r in part["place_only"].values())
        part["reading"] = "the same for women and men" if steady else "as measured"
        summary[detector] = part
        for sex, r in part["place_only"].items():
            c = part["c1"][sex]
            print(
                detector,
                sex,
                r["speakers"],
                f"place {r['auc']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]",
                f"| C1 ii {c['ii']:.3f} i {c['i_mean']:.3f} gap {c['gap']:+.3f}",
            )
        print(detector, part["reading"])
    summary["seconds"] = round(time.perf_counter() - start, 1)
    write_json(RESULTS / "summary_sex.json", summary)


C1_PREDICTIONS = {
    ("hubert", "a"): HERE / "results" / "sensitivity" / "hubert" / "c1_predictions.csv.gz",
    ("handcrafted", "a"): HERE / "results" / "first_batch" / "c1_predictions.csv.gz",
    ("hubert", "i"): HERE / "results" / "vowel_fusion" / "predictions_i.csv.gz",
    ("hubert", "u"): HERE / "results" / "vowel_fusion" / "predictions_u.csv.gz",
    ("hubert", "cape_v"): SPEECH_RESULTS / "hubert" / "cape_v" / "c1_predictions.csv.gz",
    ("hubert", "reading"): SPEECH_RESULTS / "hubert" / "reading" / "c1_predictions.csv.gz",
    ("handcrafted", "cape_v"): SPEECH_RESULTS / "handcrafted" / "cape_v" / "c1_predictions.csv.gz",
    ("handcrafted", "reading"): SPEECH_RESULTS
    / "handcrafted"
    / "reading"
    / "c1_predictions.csv.gz",
}


def c1_arms_from(path: Path) -> tuple[dict[str, float], dict[str, pd.DataFrame]]:
    """Mean AUC over the repeats of (ii) and every (i) draw, and each arm's speakers."""
    table = pd.read_csv(path, dtype={"speaker": str})
    table = table[(table["arm"] == "ii") | table["arm"].str.startswith("i_")]
    aucs: dict[str, list[float]] = {}
    for (arm, _), rows in table.groupby(["arm", "repeat"]):
        sick = (rows["label"] == "pathological").to_numpy()
        values = rows["score"].to_numpy()
        aucs.setdefault(arm, []).append(float(db.comparisons(values[sick], values[~sick]).mean()))
    first = table[table["repeat"] == table["repeat"].min()]
    members = {arm: rows[["speaker", "label"]] for arm, rows in first.groupby("arm")}
    return {arm: float(np.mean(v)) for arm, v in aucs.items()}, members


def c1_materials(jobs: int = 12) -> None:
    verify_freeze()
    start = time.perf_counter()
    pairs = list(C1_PREDICTIONS)
    done = Parallel(n_jobs=jobs)(
        delayed(_repeat)(d, r, m) for d, m in pairs for r in range(N_REPEATS)
    )
    summary: dict = {}
    for index, (detector, material) in enumerate(pairs):
        table, _ = table_of(detector, material)
        runs = done[index * N_REPEATS : (index + 1) * N_REPEATS]
        score = pd.Series(
            np.mean([s for _, _, s in runs], axis=0), index=table["speaker"].astype(str)
        )
        real, members = c1_arms_from(C1_PREDICTIONS[(detector, material)])
        place = {}
        for arm, rows in members.items():
            sick = (rows["label"] == "pathological").to_numpy()
            values = score.loc[rows["speaker"].astype(str)].to_numpy()
            place[arm] = float(db.comparisons(values[sick], values[~sick]).mean())
        draws = sorted(a for a in place if a.startswith("i_"))
        record = {
            "draws": len(draws),
            "place_ii": place["ii"],
            "place_i_mean": float(np.mean([place[a] for a in draws])),
            "real_ii": real["ii"],
            "real_i_mean": float(np.mean([real[a] for a in draws])),
        }
        record["place_gap"] = record["place_i_mean"] - record["place_ii"]
        record["real_gap"] = record["real_i_mean"] - record["real_ii"]
        record["reading"] = c1_reading(record["place_gap"], record["real_gap"])
        summary.setdefault(detector, {})[material] = record
        print(
            detector,
            material,
            {k: (round(v, 3) if isinstance(v, float) else v) for k, v in record.items()},
            flush=True,
        )
    reproduced = "a detector that never saw a patient reproduces most of the C1 gap"
    every = all(r["reading"] == reproduced for part in summary.values() for r in part.values())
    summary["overall"] = "the same on every material" if every else "as measured"
    summary["seconds"] = round(time.perf_counter() - start, 1)
    write_json(RESULTS / "summary_c1_materials.json", summary)
    print(summary["overall"], summary["seconds"], "s")


def agreement_reading(rhos: list[float]) -> str:
    if all(r >= 0.70 for r in rhos):
        return "the full detector's ranking is mostly place"
    if all(r < 0.40 for r in rhos):
        return "the rankings differ"
    return "as measured"


def agreement(jobs: int = 12) -> None:
    verify_freeze()
    import outside_view as ov

    start = time.perf_counter()
    done = Parallel(n_jobs=jobs)(
        delayed(_repeat)("hubert", r, m) for m in SPEECH for r in range(N_REPEATS)
    )
    rng = np.random.default_rng(SEED)
    summary: dict = {}
    for index, material in enumerate(SPEECH):
        table, _ = table_of("hubert", material)
        runs = done[index * N_REPEATS : (index + 1) * N_REPEATS]
        place = pd.Series(
            np.mean([s for _, _, s in runs], axis=0), index=table["speaker"].astype(str)
        )
        stored = pd.read_csv(C1_PREDICTIONS[("hubert", material)], dtype={"speaker": str})
        full = stored[stored["arm"] == "all"].groupby("speaker")["score"].mean()
        people = table.set_index(table["speaker"].astype(str)).loc[full.index]
        groups = {
            "everyone": np.ones(len(people), dtype=bool),
            "essua_normals": (
                (people["place"] == "ESSUA") & (people["label"] == "normal")
            ).to_numpy(),
            "hsa_normals": ((people["place"] == "HSA") & (people["label"] == "normal")).to_numpy(),
            "hsa_patients": (
                (people["place"] == "HSA") & (people["label"] == "pathological")
            ).to_numpy(),
        }
        part = {"speakers": len(people)}
        for name, rows in groups.items():
            x = full.to_numpy()[rows]
            y = place.loc[full.index].to_numpy()[rows]
            point = float(ov.spearman_weighted(x, y, np.ones(len(x)))[0])
            weights = db.resample_counts(len(x), N_RESAMPLES, rng)
            part[name] = ov.interval(point, ov.spearman_weighted(x, y, weights), "rho")
            part[name]["n"] = int(rows.sum())
        summary[material] = part
        print(
            material,
            {k: (round(v["rho"], 3), v["n"]) for k, v in part.items() if isinstance(v, dict)},
            flush=True,
        )
    # Not "reading": that is one of the materials.
    summary["verdict"] = agreement_reading([summary[m]["everyone"]["rho"] for m in SPEECH])
    summary["seconds"] = round(time.perf_counter() - start, 1)
    write_json(RESULTS / "summary_agreement.json", summary)
    print(summary["verdict"], summary["seconds"], "s")


DOSE = HERE / "results" / "dose_response"


def dose_reading(place_slope: float, real_slope: float) -> str:
    if place_slope >= 0.8 * real_slope:
        return "the place detector draws the dose curve too"
    if place_slope <= 0.2 * real_slope:
        return "it does not draw the dose curve"
    return "as measured"


def dose(jobs: int = 12) -> None:
    verify_freeze()
    import dose_response as dr

    start = time.perf_counter()
    todo = [(d, r) for d in ("handcrafted", "hubert") for r in range(N_REPEATS)]
    done = Parallel(n_jobs=jobs)(delayed(_repeat)(d, r) for d, r in todo)
    members = pd.read_csv(DOSE / "arms_members.csv.gz", dtype={"speaker": str})
    normals = members[members["label"] == "normal"]
    share = normals.groupby("arm")["place"].apply(lambda p: float((p == "ESSUA").mean()))
    summary: dict = {}
    for detector in ("handcrafted", "hubert"):
        table, _ = table_of(detector)
        score = pd.Series(
            np.mean([s for d, _, s in done if d == detector], axis=0),
            index=table["speaker"].astype(str),
        )
        real, arms = c1_arms_from(REAL_PREDICTIONS[detector])
        dose_real, dose_arms = {}, {}
        stored = pd.read_csv(DOSE / f"predictions_{detector}.csv.gz", dtype={"speaker": str})
        for (arm, _), rows in stored.groupby(["arm", "repeat"]):
            sick = (rows["label"] == "pathological").to_numpy()
            values = rows["score"].to_numpy()
            dose_real.setdefault(arm, []).append(
                float(db.comparisons(values[sick], values[~sick]).mean())
            )
        first = stored[stored["repeat"] == stored["repeat"].min()]
        dose_arms = {arm: rows[["speaker", "label"]] for arm, rows in first.groupby("arm")}
        shares, place_auc, real_auc = [0.0], [], [real["ii"]]
        rows = arms["ii"]
        sick = (rows["label"] == "pathological").to_numpy()
        values = score.loc[rows["speaker"]].to_numpy()
        place_auc.append(float(db.comparisons(values[sick], values[~sick]).mean()))
        for arm in sorted(dose_arms):
            rows = dose_arms[arm]
            sick = (rows["label"] == "pathological").to_numpy()
            values = score.loc[rows["speaker"]].to_numpy()
            shares.append(share[arm])
            place_auc.append(float(db.comparisons(values[sick], values[~sick]).mean()))
            real_auc.append(float(np.mean(dose_real[arm])))
        x = np.array(shares)
        record = {
            "arms": len(shares),
            "place_slope": dr.slope(x, np.array(place_auc)),
            "real_slope": dr.slope(x, np.array(real_auc)),
            "place_at_0": place_auc[0],
            "place_at_top": float(
                np.mean([a for a, sh in zip(place_auc, shares) if sh == max(shares)])
            ),
            "real_at_0": real_auc[0],
            "real_at_top": float(
                np.mean([a for a, sh in zip(real_auc, shares) if sh == max(shares)])
            ),
            "top_share": float(max(shares)),
        }
        record["reading"] = dose_reading(record["place_slope"], record["real_slope"])
        summary[detector] = record
        print(
            detector, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in record.items()}
        )
    summary["seconds"] = round(time.perf_counter() - start, 1)
    write_json(RESULTS / "summary_dose.json", summary)


FEW = (5, 10, 20, 40)
FEW_SEED = 20261153


def _few_repeat(detector: str, n: int, repeat: int) -> tuple[str, int, np.ndarray]:
    """Like ``_repeat``, but each fit sees only ``n`` normals of each place."""
    table, columns = table_of(detector)
    values = table[columns].to_numpy(dtype=float)
    place = table["place"].to_numpy()
    normal = (table["label"] == "normal").to_numpy()
    strata = table["place"].astype(str) + "_" + table["label"].astype(str)
    folds = StratifiedKFold(N_PARTS, shuffle=True, random_state=SEED + repeat)
    score = np.full(len(table), np.nan)
    for part, (_, test_index) in enumerate(folds.split(values, strata)):
        test = np.zeros(len(table), dtype=bool)
        test[test_index] = True
        rng = np.random.default_rng([FEW_SEED, n, repeat, part])
        keep = np.zeros(len(table), dtype=bool)
        for site in ("ESSUA", "HSA"):
            pool = np.flatnonzero(~test & normal & (place == site))
            keep[rng.choice(pool, size=n, replace=False)] = True
        # Rows outside ``keep`` and outside the test part are marked as patients so they are
        # not used for fitting; their scores are not read.
        masked = normal & (keep | test)
        score[test] = place_scores(values, place, masked, test)
    return detector, n, score


def few_reading(at_10: list[float], at_40: list[float]) -> str:
    if all(a >= MOST for a in at_10):
        return "ten normals per place are enough"
    if any(a < MOST for a in at_40):
        return "it needs many normals"
    return "as measured"


def few(jobs: int = 12) -> None:
    verify_freeze()
    start = time.perf_counter()
    todo = [(d, n, r) for d in ("handcrafted", "hubert") for n in FEW for r in range(N_REPEATS)]
    done = Parallel(n_jobs=jobs)(delayed(_few_repeat)(d, n, r) for d, n, r in todo)
    rng = np.random.default_rng(FEW_SEED)
    summary: dict = {}
    for detector in ("handcrafted", "hubert"):
        table, _ = table_of(detector)
        sick = (table["label"] == "pathological").to_numpy()
        part = {}
        for n in FEW:
            score = np.mean([s for d, m, s in done if d == detector and m == n], axis=0)
            part[str(n)] = auc_with_ci(score, sick, rng)
        summary[detector] = part
        print(detector, {n: round(r["auc"], 3) for n, r in part.items()}, flush=True)
    summary["reading"] = few_reading(
        [summary[d]["10"]["auc"] for d in ("handcrafted", "hubert")],
        [summary[d]["40"]["auc"] for d in ("handcrafted", "hubert")],
    )
    summary["seconds"] = round(time.perf_counter() - start, 1)
    write_json(RESULTS / "summary_few.json", summary)
    print(summary["reading"], summary["seconds"], "s")


def published_calls() -> pd.Series:
    """Their /a/ call per speaker, True where called pathological (one test file per speaker)."""
    import published_splits as pub

    lists = pd.read_csv(pub.SOURCE / "ribas_lists.csv", dtype=str)
    calls = pd.read_csv(pub.SOURCE / "ribas_predictions.csv", dtype=str)
    calls = calls[calls["material"] == "a_n"]
    listed = lists[(lists["material"] == "a_n") & (lists["side"] == "test")]
    joined = calls.merge(
        listed[["fold", "stem", "speaker"]], on=["fold", "stem"], how="left", validate="one_to_one"
    )
    if joined["speaker"].isna().any() or joined["speaker"].duplicated().any():
        raise ValueError("a_n: a call without a speaker, or a speaker called twice")
    return (joined.set_index("speaker")["predicted"] == "pathological").rename("called")


def published_reading(record: dict) -> str:
    if record["auc"] >= MOST and record["ci_low"] > 0.70:
        return "which normals it calls pathological follows place"
    if record["ci_low"] <= 0.5 <= record["ci_high"]:
        return "not visible"
    return "as measured"


def published(jobs: int = 12) -> None:
    verify_freeze()
    start = time.perf_counter()
    todo = [(d, r) for d in ("handcrafted", "hubert") for r in range(N_REPEATS)]
    done = Parallel(n_jobs=jobs)(delayed(_repeat)(d, r) for d, r in todo)
    called = published_calls()
    rng = np.random.default_rng(SEED)
    summary: dict = {}
    for detector in ("handcrafted", "hubert"):
        table, _ = table_of(detector)
        score = pd.Series(
            np.mean([s for d, _, s in done if d == detector], axis=0),
            index=table["speaker"].astype(str),
        )
        people = table.set_index(table["speaker"].astype(str))
        ids = people.index.intersection(called.index)
        flags = called.loc[ids].to_numpy()
        values = score.loc[ids].to_numpy()
        truth = (people.loc[ids, "label"] == "pathological").to_numpy()
        part: dict = {"speakers": len(ids), "called_pathological": int(flags.sum())}
        for name, rows in (
            ("everyone", np.ones(len(ids), dtype=bool)),
            ("true_normals", ~truth),
            ("true_patients", truth),
        ):
            record = auc_with_ci(values[rows], flags[rows], rng)
            record["called_pathological"] = int(flags[rows].sum())
            record["n"] = int(rows.sum())
            part[name] = record
        part["reading"] = published_reading(part["true_normals"])
        sites = people.loc[ids, "place"].to_numpy()
        part["called_normals_by_place"] = {
            site: [
                int((flags & ~truth & (sites == site)).sum()),
                int((~truth & (sites == site)).sum()),
            ]
            for site in ("ESSUA", "HSA")
        }
        summary[detector] = part
        print(
            detector,
            {
                k: (round(v["auc"], 3), v["called_pathological"], v["n"])
                for k, v in part.items()
                if k in ("everyone", "true_normals", "true_patients")
            },
            part["reading"],
            flush=True,
        )
    summary["seconds"] = round(time.perf_counter() - start, 1)
    write_json(RESULTS / "summary_published.json", summary)


REAL_FULL_UAR = {"handcrafted": 0.820, "hubert": 0.835, "hubert_finetuned": 0.828, "compare": 0.858}


def oracle() -> None:
    """The rule "not recorded at ESSUA means pathological", on all of AVFAD /a/; no audio."""
    verify_freeze()
    speakers = batch.load_speakers()
    called = (speakers["place"] != "ESSUA").to_numpy()
    sick = (speakers["label"] == "pathological").to_numpy()
    sensitivity = float(called[sick].mean())
    specificity = float((~called[~sick]).mean())
    record = {
        "speakers": len(speakers),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "uar": (sensitivity + specificity) / 2.0,
        "patients_at_essua": int((sick & ~called).sum()),
        "real_uar": REAL_FULL_UAR,
    }
    best = max(REAL_FULL_UAR.values())
    record["reading"] = (
        "knowing the place alone beats every detector" if record["uar"] >= best else "as measured"
    )
    write_json(RESULTS / "summary_oracle.json", record)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in record.items()})


def _cross_repeat(detector: str, material: str, repeat: int) -> tuple[str, str, np.ndarray]:
    """Fitted on /a/ of the other parts' normals, scored on ``material`` of this part."""
    source, columns = table_of(detector)
    target, _ = table_of(detector, material)
    both = source[["speaker"]].merge(target[["speaker"]], on="speaker")["speaker"]
    source = source[source["speaker"].isin(both)].sort_values("speaker").reset_index(drop=True)
    target = target[target["speaker"].isin(both)].sort_values("speaker").reset_index(drop=True)
    x_source = source[columns].to_numpy(dtype=float)
    x_target = target[columns].to_numpy(dtype=float)
    place = source["place"].to_numpy()
    normal = (source["label"] == "normal").to_numpy()
    strata = source["place"].astype(str) + "_" + source["label"].astype(str)
    folds = StratifiedKFold(N_PARTS, shuffle=True, random_state=SEED + repeat)
    score = np.full(len(source), np.nan)
    for _, test_index in folds.split(x_source, strata):
        test = np.zeros(len(source), dtype=bool)
        test[test_index] = True
        train = ~test & normal & np.isin(place, ["ESSUA", "HSA"])
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        clf.fit(x_source[train], place[train] == "HSA")
        score[test] = clf.decision_function(x_target[test])
    return detector, material, score


def cross_reading(aucs: list[float]) -> str:
    if all(a >= MOST for a in aucs):
        return "the place cue follows the recording chain, no relearning per material"
    if all(a < LITTLE for a in aucs):
        return "it has to be learned on the same material"
    return "as measured"


def cross_material(jobs: int = 12) -> None:
    verify_freeze()
    start = time.perf_counter()
    pairs = [(d, m) for d in ("handcrafted", "hubert") for m in SPEECH]
    done = Parallel(n_jobs=jobs)(
        delayed(_cross_repeat)(d, m, r) for d, m in pairs for r in range(N_REPEATS)
    )
    rng = np.random.default_rng(SEED)
    summary: dict = {}
    for detector, material in pairs:
        source, _ = table_of(detector)
        target, _ = table_of(detector, material)
        both = source[["speaker"]].merge(target[["speaker"]], on="speaker")["speaker"]
        labels = source[source["speaker"].isin(both)].sort_values("speaker")["label"].to_numpy()
        runs = [s for d, m, s in done if d == detector and m == material]
        score = np.mean(runs, axis=0)
        record = auc_with_ci(score, labels == "pathological", rng)
        record["speakers"] = len(labels)
        summary.setdefault(detector, {})[material] = record
        print(
            detector,
            material,
            len(labels),
            f"{record['auc']:.3f} [{record['ci_low']:.3f}, {record['ci_high']:.3f}]",
        )
    summary["verdict"] = cross_reading([summary[d][m]["auc"] for d, m in pairs])
    summary["seconds"] = round(time.perf_counter() - start, 1)
    write_json(RESULTS / "summary_cross_material.json", summary)
    print(summary["verdict"], summary["seconds"], "s")


def paired_reading(record: dict) -> str:
    if record["ci_low"] <= 0 <= record["ci_high"]:
        return "no better or worse than the full detector"
    if record["ci_high"] < 0:
        return "below the full detector"
    return "above the full detector"


def paired(jobs: int = 12) -> None:
    verify_freeze()
    start = time.perf_counter()
    done = Parallel(n_jobs=jobs)(
        delayed(_repeat)("hubert", r, m) for m in SPEECH for r in range(N_REPEATS)
    )
    rng = np.random.default_rng(SEED)
    summary: dict = {}
    for index, material in enumerate(SPEECH):
        table, _ = table_of("hubert", material)
        runs = done[index * N_REPEATS : (index + 1) * N_REPEATS]
        place = pd.Series(
            np.mean([s for _, _, s in runs], axis=0), index=table["speaker"].astype(str)
        )
        stored = pd.read_csv(C1_PREDICTIONS[("hubert", material)], dtype={"speaker": str})
        full = stored[stored["arm"] == "all"].groupby("speaker")["score"].mean()
        people = table.set_index(table["speaker"].astype(str)).loc[full.index]
        sick = (people["label"] == "pathological").to_numpy()
        x_place = place.loc[full.index].to_numpy()
        x_full = full.to_numpy()
        k_place = db.comparisons(x_place[sick], x_place[~sick])
        k_full = db.comparisons(x_full[sick], x_full[~sick])
        w_sick = db.resample_counts(int(sick.sum()), N_RESAMPLES, rng)
        w_well = db.resample_counts(int((~sick).sum()), N_RESAMPLES, rng)
        diff = db.weighted_auc(k_place, w_sick, w_well) - db.weighted_auc(k_full, w_sick, w_well)
        low, high = np.percentile(diff, [2.5, 97.5])
        record = {
            "speakers": len(people),
            "place_auc": float(k_place.mean()),
            "full_auc": float(k_full.mean()),
            "difference": float(k_place.mean() - k_full.mean()),
            "ci_low": float(low),
            "ci_high": float(high),
        }
        record["reading"] = paired_reading(record)
        summary[material] = record
        print(
            material, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in record.items()}
        )
    summary["seconds"] = round(time.perf_counter() - start, 1)
    write_json(RESULTS / "summary_paired.json", summary)


ALL_REFERENCE = HERE / "results" / "exploratory" / "avfad_all.json"


def _full_fold(detector: str, repeat: int, fold: int):
    """One outer fold of the all-AVFAD detector of ``exploratory_avfad.py``."""
    import phonation_c1 as pc
    import pool_training as pt
    import published_splits as pub
    from voice_severity_bench import models as vsb_models
    from arms import merge_features
    from harness import load_plan, make_config

    config = make_config(load_plan(), batch.LABEL_POSITIVE)
    speakers = batch.load_speakers()
    if detector == "hubert":
        table = merge_features(speakers, load_hubert("full", CHOSEN_LAYERS), "label")
        blocks = {f"l{layer:02d}": layer_columns(layer) for layer in CHOSEN_LAYERS}
        columns = [c for block in blocks.values() for c in block]
        config = config.override({"model.block_selection": True})
    else:
        vsb_models.SEARCH_SPACES[pc.LOGISTIC] = {k: list(v) for k, v in pc.HANDCRAFTED_GRID.items()}
        table = merge_features(speakers, batch.load_features("full_original"), "label")
        blocks, columns = None, list(FEATURE_COLUMNS)
    table = table.assign(fold=pt.fold_ids(table, config)[:, repeat])
    output = pub.fit_listed_fold(table, columns, blocks, config, fold)
    return detector, repeat, output["items"], output["scores"]


def paired_a(jobs: int = 12) -> None:
    verify_freeze()
    start = time.perf_counter()
    reference = json.loads(ALL_REFERENCE.read_text(encoding="utf-8"))
    todo = [
        (d, r, k)
        for d in ("handcrafted", "hubert")
        for r in range(N_REPEATS)
        for k in range(N_PARTS)
    ]
    done = Parallel(n_jobs=jobs)(delayed(_full_fold)(d, r, k) for d, r, k in todo)
    places = Parallel(n_jobs=jobs)(
        delayed(_repeat)(d, r) for d in ("handcrafted", "hubert") for r in range(N_REPEATS)
    )
    rng = np.random.default_rng(SEED)
    summary: dict = {}
    for detector in ("handcrafted", "hubert"):
        table, _ = table_of(detector)
        matrix = np.full((len(table), N_REPEATS), np.nan)
        for d, repeat, items, scores in done:
            if d == detector:
                matrix[items, repeat] = scores
        if np.isnan(matrix).any():
            raise RuntimeError("a speaker was never tested")
        sick = (table["label"] == "pathological").to_numpy()
        rerun = float(
            np.mean(
                [db.comparisons(matrix[sick, r], matrix[~sick, r]).mean() for r in range(N_REPEATS)]
            )
        )
        stored = reference[f"all_{detector}"]["roc_auc"]
        record: dict = {"rerun_auc": rerun, "stored_auc": stored}
        pd.DataFrame(matrix, columns=[f"r{r}" for r in range(N_REPEATS)]).assign(
            speaker=table["speaker"].to_numpy()
        ).to_csv(RESULTS / f"all_scores_{detector}.csv.gz", index=False)
        if abs(rerun - stored) > 0.002:
            record["reading"] = "the rerun does not match the stored detector; not compared"
            summary[detector] = record
            print(detector, record)
            continue
        x_full = matrix.mean(axis=1)
        x_place = np.mean([s for d, _, s in places if d == detector], axis=0)
        k_place = db.comparisons(x_place[sick], x_place[~sick])
        k_full = db.comparisons(x_full[sick], x_full[~sick])
        w_sick = db.resample_counts(int(sick.sum()), N_RESAMPLES, rng)
        w_well = db.resample_counts(int((~sick).sum()), N_RESAMPLES, rng)
        diff = db.weighted_auc(k_place, w_sick, w_well) - db.weighted_auc(k_full, w_sick, w_well)
        low, high = np.percentile(diff, [2.5, 97.5])
        record.update(
            {
                "place_auc": float(k_place.mean()),
                "full_auc": float(k_full.mean()),
                "difference": float(k_place.mean() - k_full.mean()),
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )
        record["reading"] = paired_reading(record)
        summary[detector] = record
        print(
            detector,
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in record.items()},
            flush=True,
        )
    summary["seconds"] = round(time.perf_counter() - start, 1)
    write_json(RESULTS / "summary_paired_a.json", summary)


def inside_reading(record: dict) -> str:
    if record["ci_low"] <= 0 <= record["ci_high"]:
        return "the ESSUA normals do not help the ranking inside HSA"
    if record["ci_high"] < 0:
        return "the ESSUA normals make the ranking inside HSA worse"
    return "the ESSUA normals help"


def inside_hsa() -> None:
    verify_freeze()
    rng = np.random.default_rng(SEED)
    speakers = batch.load_speakers().set_index("speaker")
    summary: dict = {}
    for detector in ("handcrafted", "hubert"):
        full = pd.read_csv(RESULTS / f"all_scores_{detector}.csv.gz", dtype={"speaker": str})
        full = full.set_index("speaker")
        stored = pd.read_csv(REAL_PREDICTIONS[detector], dtype={"speaker": str})
        ii = stored[stored["arm"] == "ii"].pivot(index="speaker", columns="repeat", values="score")
        ids = ii.index
        if not ids.isin(full.index).all():
            raise RuntimeError("an HSA speaker has no all-AVFAD score")
        sick = (speakers.loc[ids, "label"] == "pathological").to_numpy()
        m_full = full.loc[ids].to_numpy()
        m_ii = ii.to_numpy()

        def mean_auc(matrix: np.ndarray) -> float:
            return float(
                np.mean(
                    [
                        db.comparisons(matrix[sick, r], matrix[~sick, r]).mean()
                        for r in range(matrix.shape[1])
                    ]
                )
            )

        x_full, x_ii = m_full.mean(axis=1), m_ii.mean(axis=1)
        k_full = db.comparisons(x_full[sick], x_full[~sick])
        k_ii = db.comparisons(x_ii[sick], x_ii[~sick])
        w_sick = db.resample_counts(int(sick.sum()), N_RESAMPLES, rng)
        w_well = db.resample_counts(int((~sick).sum()), N_RESAMPLES, rng)
        diff = db.weighted_auc(k_full, w_sick, w_well) - db.weighted_auc(k_ii, w_sick, w_well)
        low, high = np.percentile(diff, [2.5, 97.5])
        record = {
            "speakers": len(ids),
            "full_auc": mean_auc(m_full),
            "ii_auc": mean_auc(m_ii),
            "difference": float(k_full.mean() - k_ii.mean()),
            "ci_low": float(low),
            "ci_high": float(high),
            "full_specificity_hsa_normals": float((m_full[~sick] <= 0.5).mean()),
            "ii_specificity_hsa_normals": float((m_ii[~sick] <= 0.5).mean()),
        }
        record["reading"] = inside_reading(record)
        summary[detector] = record
        print(
            detector, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in record.items()}
        )
    write_json(RESULTS / "summary_inside_hsa.json", summary)


DAYS = HERE.parent / "s0_audit" / "out" / "avfad_speakers.csv"
DAY_SEED = 20261155


def _day_repeat(detector: str, repeat: int) -> tuple[str, int, np.ndarray]:
    """Like ``_repeat``, but everyone recorded on the same day is in the same part."""
    from sklearn.model_selection import StratifiedGroupKFold

    table, columns = table_of(detector)
    days = pd.read_csv(DAYS, dtype={"speaker": str})[["speaker", "visit_date"]]
    day = table[["speaker"]].merge(days, on="speaker", how="left", validate="one_to_one")
    if day["visit_date"].isna().any():
        raise RuntimeError("a speaker has no visit date")
    values = table[columns].to_numpy(dtype=float)
    place = table["place"].to_numpy()
    normal = (table["label"] == "normal").to_numpy()
    strata = table["place"].astype(str) + "_" + table["label"].astype(str)
    folds = StratifiedGroupKFold(N_PARTS, shuffle=True, random_state=DAY_SEED + repeat)
    score = np.full(len(table), np.nan)
    for _, test_index in folds.split(values, strata, day["visit_date"].astype(str)):
        test = np.zeros(len(table), dtype=bool)
        test[test_index] = True
        score[test] = place_scores(values, place, normal, test)
    return detector, repeat, score


def day_reading(aucs: list[float]) -> str:
    if all(a >= MOST for a in aucs):
        return "the place cue holds across recording days"
    if all(a < LITTLE for a in aucs):
        return "mostly the conditions of the same day"
    return "as measured"


def by_day(jobs: int = 12) -> None:
    verify_freeze()
    start = time.perf_counter()
    todo = [(d, r) for d in ("handcrafted", "hubert") for r in range(N_REPEATS)]
    done = Parallel(n_jobs=jobs)(delayed(_day_repeat)(d, r) for d, r in todo)
    plain = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(SEED)
    summary: dict = {}
    for detector in ("handcrafted", "hubert"):
        table, _ = table_of(detector)
        score = np.mean([s for d, _, s in done if d == detector], axis=0)
        sick = (table["label"] == "pathological").to_numpy()
        hsa = (table["place"] == "HSA").to_numpy()
        summary[detector] = {
            "all_avfad": auc_with_ci(score, sick, rng),
            "hsa_only": auc_with_ci(score[hsa], sick[hsa], rng),
            "random_parts_auc": plain[detector]["all_avfad"]["auc"],
        }
        a = summary[detector]["all_avfad"]
        print(
            detector,
            f"by day {a['auc']:.3f} [{a['ci_low']:.3f}, {a['ci_high']:.3f}]",
            f"(random parts {summary[detector]['random_parts_auc']:.3f})",
            f"| HSA {summary[detector]['hsa_only']['auc']:.3f}",
            flush=True,
        )
    summary["verdict"] = day_reading(
        [summary[d]["all_avfad"]["auc"] for d in ("handcrafted", "hubert")]
    )
    summary["seconds"] = round(time.perf_counter() - start, 1)
    write_json(RESULTS / "summary_by_day.json", summary)
    print(summary["verdict"], summary["seconds"], "s")


if __name__ == "__main__":
    step = sys.argv[1:]
    if step == ["materials"]:
        materials()
    elif step == ["variants"]:
        variants()
    elif step == ["compare"]:
        compare()
    elif step == ["outside"]:
        outside()
    elif step == ["outside_phonation"]:
        outside_phonation()
    elif step == ["strata"]:
        strata()
    elif step == ["c1"]:
        c1()
    elif step == ["c1_phonation"]:
        c1(material="a_phonation")
    elif step == ["c1_materials"]:
        c1_materials()
    elif step == ["uar"]:
        uar()
    elif step == ["room"]:
        room()
    elif step == ["sex"]:
        by_sex()
    elif step == ["agreement"]:
        agreement()
    elif step == ["dose"]:
        dose()
    elif step == ["few"]:
        few()
    elif step == ["published"]:
        published()
    elif step == ["oracle"]:
        oracle()
    elif step == ["cross_material"]:
        cross_material()
    elif step == ["paired"]:
        paired()
    elif step == ["paired_a"]:
        paired_a()
    elif step == ["inside_hsa"]:
        inside_hsa()
    elif step == ["by_day"]:
        by_day()
    else:
        main()
