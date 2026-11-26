"""Exploratory runs on AVFAD; definitions and readings are in experiments/deviations.md.

    curve         how much of the low score inside HSA is too few normal speakers to train on
    all           the same detector on all 709 speakers, against the published scores
    transfer      train inside HSA, test the specificity on ESSUA's normal speakers
    noise-tables  features of the recordings with room noise added
    intervention  how often room noise turns a normal speaker's prediction into pathological

None of these enter the family of confirmatory tests.
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
from voice_severity_bench.bootstrap import bootstrap_metric, group_weights  # noqa: E402
from voice_severity_bench.evaluation import class_scores, classification_metrics  # noqa: E402
from voice_severity_bench.models import fit_model  # noqa: E402
from voice_severity_bench.splitting import repeated_split_indices  # noqa: E402
from joblib import Parallel, delayed, parallel_config  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_first_batch as batch  # noqa: E402
from arms import (  # noqa: E402
    _use_hubert_grid,
    evaluate_columns,
    evaluate_layers,
    layer_columns,
    load_hubert,
    merge_features,
    write_json,
)
from common import cut, load_signal  # noqa: E402
from extract_handcrafted import FEATURE_COLUMNS, features_of, level_statistics  # noqa: E402
from freeze import FreezeError, verify_freeze  # noqa: E402
from harness import ROOT, load_plan, make_config, match_cells  # noqa: E402
from sensitivity_avfad import CHOSEN_LAYERS  # noqa: E402

RESULTS = HERE / "results" / "exploratory"
NOISE_TABLES = ROOT / "features" / "out_exploratory"
S0_OUT = ROOT / "s0_audit" / "out"
S1_OUT = ROOT / "s1_segments" / "out"

CURVE_NORMALS = (16, 24, 32, 48, None)  # None: every normal speaker of the training fold
CURVE_TARGET = 290  # 363 x 0.8: if all of AVFAD's normal speakers could train
DOSES_DB = (-6, 0, 6)
MIN_NOISE_S = 5.0
CURVE_SEED = 20261201
NOISE_SEED = 20261202


def design(table: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        table[columns].to_numpy(dtype=np.float64),
        table["label"].to_numpy().astype(str),
        table["group"].to_numpy().astype(str),
    )


def layer_design(table, layers):
    """HuBERT layers side by side, with the blocks the inner loop chooses from."""
    columns = [c for layer in layers for c in layer_columns(layer)]
    blocks = {f"l{layer:02d}": (i * 1536, (i + 1) * 1536) for i, layer in enumerate(layers)}
    return (*design(table, columns), blocks)


def subsample_training(
    train: np.ndarray, y: np.ndarray, n_normals: int | None, rng: np.random.Generator
) -> np.ndarray:
    """All patients of the fold, and ``n_normals`` of its normal speakers."""
    normals = train[y[train] == "normal"]
    if n_normals is None or n_normals >= normals.size:
        return train
    picked = rng.choice(normals, size=n_normals, replace=False)
    return np.sort(np.concatenate([train[y[train] != "normal"], picked]))


def _curve_repeat(curve, X, y, groups, folds, n_normals, repeat, config, blocks) -> dict:
    if blocks:
        _use_hubert_grid()
    classes = tuple(sorted(np.unique(y).tolist()))
    predicted = np.empty(y.size, dtype=object)
    scores = np.zeros((y.size, len(classes)))
    used = []
    for fold, (train, test) in enumerate(folds):
        rng = np.random.default_rng([CURVE_SEED, n_normals or 0, repeat, fold])
        chosen = subsample_training(train, y, n_normals, rng)
        used.append(int(np.sum(y[chosen] == "normal")))
        model = fit_model(X[chosen], y[chosen], config, groups=groups[chosen], blocks=blocks)
        predicted[test] = np.asarray(model.predict(X[test])).astype(str)
        scores[test] = class_scores(model, X[test], classes)
    metrics = classification_metrics(
        y, predicted.astype(str), scores, classes=classes, positive_class="pathological"
    )
    return {
        "curve": curve,
        "requested": n_normals if n_normals is not None else "all",
        "repeat": repeat,
        "normals_in_training": float(np.mean(used)),
        "uar": float(metrics["uar"]),
        "roc_auc": float(metrics["roc_auc"]),
    }


def fit_curve(ns: np.ndarray, uars: np.ndarray, target: float) -> dict[str, float]:
    """Two rough forms; the gap between their answers is part of the answer."""
    root = np.linalg.lstsq(np.column_stack([np.ones(ns.size), 1 / np.sqrt(ns)]), uars, rcond=None)[
        0
    ]
    log = np.linalg.lstsq(np.column_stack([np.ones(ns.size), np.log(ns)]), uars, rcond=None)[0]
    return {
        "inverse_root_at_target": float(root[0] + root[1] / np.sqrt(target)),
        "inverse_root_limit": float(root[0]),
        "log_at_target": float(log[0] + log[1] * np.log(target)),
    }


def run_curve(plan, jobs) -> None:
    seeds = plan["statistics"]["seeds"]
    speakers = batch.load_speakers()
    patients, hsa, normal = batch_groups(speakers)
    essua, _ = match_cells(
        hsa, normal[normal["place"] == "ESSUA"], np.random.default_rng(seeds["c2_essua_draw"])
    )
    handcrafted = batch.load_features("full_original")
    hubert = load_hubert("full", CHOSEN_LAYERS)
    config = make_config(plan, batch.LABEL_POSITIVE)
    tuned = config.override({"model.block_selection": True})
    sets = []
    for curve, normals in (("hsa", hsa), ("essua", essua)):
        meta = pd.concat([patients, normals])
        table = merge_features(meta, handcrafted, "label")
        X, y, groups = design(table, FEATURE_COLUMNS)
        sets.append((f"{curve}_handcrafted", X, y, groups, config, None))
        table = merge_features(meta, hubert, "label")
        X, y, groups, blocks = layer_design(table, CHOSEN_LAYERS)
        sets.append((f"{curve}_hubert", X, y, groups, tuned, blocks))
    tasks = []
    for name, X, y, groups, cfg, blocks in sets:
        splits = repeated_split_indices(
            y, groups, task="classification", n_splits=5, n_repeats=10, seed=cfg.seed
        )
        for n in CURVE_NORMALS:
            for repeat, folds in enumerate(splits):
                tasks.append((name, X, y, groups, folds, n, repeat, cfg, blocks))
    rows = Parallel(n_jobs=jobs)(delayed(_curve_repeat)(*task) for task in tasks)
    table = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / "learning_curve.csv", index=False)
    report = {}
    for curve, part in table.groupby("curve"):
        grouped = part.groupby(part["requested"].astype(str), sort=False).agg(
            normals=("normals_in_training", "mean"),
            uar=("uar", "mean"),
            uar_sd=("uar", "std"),
            roc_auc=("roc_auc", "mean"),
        )
        ns = grouped["normals"].to_numpy()
        fitted = fit_curve(ns, grouped["uar"].to_numpy(), CURVE_TARGET)
        per_repeat = [
            fit_curve(
                r.sort_values("normals_in_training")["normals_in_training"].to_numpy(),
                r.sort_values("normals_in_training")["uar"].to_numpy(),
                CURVE_TARGET,
            )
            for _, r in part.groupby("repeat")
        ]
        report[curve] = {
            "points": grouped.reset_index().to_dict("records"),
            "extrapolated_to_normals": CURVE_TARGET,
            **fitted,
            "inverse_root_at_target_over_repeats": [
                float(min(p["inverse_root_at_target"] for p in per_repeat)),
                float(max(p["inverse_root_at_target"] for p in per_repeat)),
            ],
        }
    write_json(RESULTS / "learning_curve.json", report)


def batch_groups(speakers):
    patients = speakers[(speakers["place"] == "HSA") & (speakers["label"] == "pathological")]
    hsa = speakers[(speakers["place"] == "HSA") & (speakers["label"] == "normal")]
    normal = speakers[speakers["label"] == "normal"]
    return patients, hsa, normal


def run_all(plan, jobs) -> None:
    """All 709 speakers, with the same detector: handcrafted, then HuBERT."""
    seeds = plan["statistics"]["seeds"]
    speakers = batch.load_speakers()
    config = make_config(plan, batch.LABEL_POSITIVE)
    handcrafted = merge_features(speakers, batch.load_features("full_original"), "label")
    hubert = merge_features(speakers, load_hubert("full", CHOSEN_LAYERS), "label")
    results = Parallel(n_jobs=min(jobs, 2))(
        [
            delayed(evaluate_columns)("all_handcrafted", handcrafted, FEATURE_COLUMNS, config),
            delayed(evaluate_layers)("all_hubert", hubert, CHOSEN_LAYERS, config),
        ]
    )
    report = {}
    for arm in results:
        interval = bootstrap_metric(
            arm.y_true,
            arm.scores,
            arm.groups,
            metric="roc_auc",
            n_resamples=10_000,
            seed=seeds["bootstrap"],
            positive_class=batch.LABEL_POSITIVE,
        )
        report[arm.name] = {**arm.row(), "roc_auc_interval": interval.as_dict()}
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / "avfad_all.json", report)


def _transfer_fold(X, y, groups, train, test, X_other, config, blocks) -> dict:
    if blocks:
        _use_hubert_grid()
    model = fit_model(X[train], y[train], config, groups=groups[train], blocks=blocks)
    normals = test[y[test] == "normal"]
    return {
        "hsa_normals": groups[normals],
        "hsa_correct": np.asarray(model.predict(X[normals])).astype(str) == "normal",
        "other_correct": np.asarray(model.predict(X_other)).astype(str) == "normal",
    }


def _bootstrap_mean(values: np.ndarray, seed: int, n: int = 10_000) -> tuple[float, float]:
    weights = group_weights(None, values.size, n, seed=seed)
    samples = (weights @ values) / weights.sum(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def run_transfer(plan, jobs) -> None:
    seeds = plan["statistics"]["seeds"]
    speakers = batch.load_speakers()
    patients, hsa, normal = batch_groups(speakers)
    essua = normal[normal["place"] == "ESSUA"].copy()
    c2 = pd.read_csv(HERE / "results" / "first_batch" / "c2_speakers.csv", dtype={"speaker": str})
    matched = c2.loc[c2["place"] == "ESSUA", "speaker"].to_numpy().astype(str)
    config = make_config(plan, batch.LABEL_POSITIVE)
    tuned = config.override({"model.block_selection": True})
    report = {}
    cases = [
        ("handcrafted_full", batch.load_features("full_original"), None),
        ("handcrafted_phonation", batch.load_features("phonation_norm"), None),
        ("hubert_full", load_hubert("full", CHOSEN_LAYERS), CHOSEN_LAYERS),
        ("hubert_phonation", load_hubert("phonation", CHOSEN_LAYERS), CHOSEN_LAYERS),
    ]
    for name, features, layers in cases:
        inside = merge_features(pd.concat([patients, hsa]), features, "label")
        outside = merge_features(essua, features, "label")
        if layers is None:
            X, y, groups = design(inside, FEATURE_COLUMNS)
            X_other = design(outside, FEATURE_COLUMNS)[0]
            blocks, cfg = None, config
        else:
            X, y, groups, blocks = layer_design(inside, layers)
            X_other = layer_design(outside, layers)[0]
            cfg = tuned
        splits = repeated_split_indices(
            y, groups, task="classification", n_splits=5, n_repeats=10, seed=cfg.seed
        )
        fits = Parallel(n_jobs=jobs)(
            delayed(_transfer_fold)(X, y, groups, train, test, X_other, cfg, blocks)
            for folds in splits
            for train, test in folds
        )
        other = np.stack([f["other_correct"] for f in fits]).mean(axis=0)
        per_model_other = np.array([f["other_correct"].mean() for f in fits])
        per_model_hsa = np.array([f["hsa_correct"].mean() for f in fits])
        hsa_rows = pd.DataFrame(
            {
                "speaker": np.concatenate([f["hsa_normals"] for f in fits]),
                "correct": np.concatenate([f["hsa_correct"] for f in fits]),
            }
        )
        hsa_means = hsa_rows.groupby("speaker")["correct"].mean().to_numpy(dtype=float)
        # The 274 are not matched on age and sex; the people C2 matched are read on their own.
        speakers_out = outside["group"].to_numpy().astype(str)
        is_matched = np.isin(speakers_out, matched)
        pd.DataFrame(
            {"speaker": speakers_out, "matched": is_matched, "correct_rate": other}
        ).to_csv(RESULTS / f"hsa_to_essua_{name}_speakers.csv", index=False)
        report[name] = {
            "essua_normals": int(other.size),
            "models": len(fits),
            "specificity_essua": float(other.mean()),
            "specificity_essua_ci": _bootstrap_mean(other, seeds["bootstrap"]),
            "specificity_essua_per_model": [
                float(per_model_other.min()),
                float(per_model_other.max()),
            ],
            "specificity_essua_matched": float(other[is_matched].mean()),
            "specificity_essua_matched_ci": _bootstrap_mean(other[is_matched], seeds["bootstrap"]),
            "essua_matched_speakers": int(is_matched.sum()),
            "specificity_hsa_out_of_fold": float(hsa_means.mean()),
            "specificity_hsa_ci": _bootstrap_mean(hsa_means, seeds["bootstrap"]),
            "specificity_hsa_per_model_mean": float(per_model_hsa.mean()),
            "difference_essua_minus_hsa": float(other.mean() - hsa_means.mean()),
        }
        RESULTS.mkdir(parents=True, exist_ok=True)
        write_json(RESULTS / "hsa_to_essua.json", report)


def noise_files(place: str) -> pd.DataFrame:
    """Room recordings of a place, one per group of identical files, with enough quiet."""
    rooms = pd.read_csv(S0_OUT / "avfad_silence.csv", dtype={"identical_group": str})
    rooms = _one_per_group(rooms[rooms["place"] == place])
    files = pd.read_csv(S1_OUT / "files.csv")
    files = files[files["material"] == "silence"][["file_id", "wav", "nonphonation_g100_s"]]
    merged = rooms.merge(files, left_on="path", right_on="wav", validate="one_to_one")
    return merged[merged["nonphonation_g100_s"] >= MIN_NOISE_S].sort_values("file_id")


def _one_per_group(rooms: pd.DataFrame) -> pd.DataFrame:
    rooms = rooms.sort_values("path")
    grouped = rooms["identical_group"].notna()
    return pd.concat(
        [rooms[~grouped], rooms[grouped].drop_duplicates("identical_group", keep="first")]
    ).sort_values("path")


def loop_to_length(pieces: list[np.ndarray], n: int, start: int, rate: int) -> np.ndarray:
    """Join quiet spans with 10 ms cross-fades, then repeat from ``start`` up to ``n`` samples."""
    fade = round(0.010 * rate)
    joined = pieces[0]
    ramp = np.linspace(0.0, 1.0, fade)
    for piece in pieces[1:]:
        if joined.size < fade or piece.size < fade:
            joined = np.concatenate([joined, piece])
            continue
        mix = joined[-fade:] * (1.0 - ramp) + piece[:fade] * ramp
        joined = np.concatenate([joined[:-fade], mix, piece[fade:]])
    indices = (start + np.arange(n)) % joined.size
    return joined[indices]


_ROOM_CACHE: dict[str, tuple[list[np.ndarray], float]] = {}


def room_quiet(file_id: str, wav: str, spans: list[tuple[float, float]]):
    """Quiet spans of one room recording and their level, the measure the target uses."""
    if file_id not in _ROOM_CACHE:
        x, rate = load_signal(wav)
        pieces = cut(x, rate, spans)
        _ROOM_CACHE.clear()
        _ROOM_CACHE[file_id] = (pieces, float(level_statistics(pieces, rate)[0]))
    return _ROOM_CACHE[file_id]


def _noisy_row(file_id, wav, room, target_db, dose, seed_key) -> dict:
    x, rate = load_signal(wav)
    pieces, level = room_quiet(*room)
    rng = np.random.default_rng([NOISE_SEED, *seed_key])
    total = int(sum(p.size for p in pieces))
    noise = loop_to_length(pieces, x.size, int(rng.integers(total)), rate)
    noise = noise * 10.0 ** ((target_db + dose - level) / 20.0)
    values = features_of([x + noise], rate)
    return {
        "file_id": file_id,
        "room_file": room[0],
        "room_level_db": level,
        **dict(zip(FEATURE_COLUMNS, values, strict=True)),
    }


def hsa_target_db(speakers: pd.DataFrame) -> float:
    """Median quiet level of HSA's normal speakers who reach 1.0 s, the 0 dB of the doses."""
    table = batch.load_features("nonphon_g100")
    normal = speakers[(speakers["label"] == "normal") & (speakers["place"] == "HSA")]
    table = table[table["file_id"].isin(normal["file_id"]) & (table["kept_s"] >= 1.0)]
    return float(table["level_mean"].median())


def run_noise_tables(jobs) -> None:
    speakers = batch.load_speakers()
    files = pd.read_csv(S1_OUT / "files.csv").set_index("file_id")
    spans = pd.read_csv(S1_OUT / "spans.csv")
    quiet = spans[spans["kind"] == "nonphonation_g100"]
    target = hsa_target_db(speakers)
    NOISE_TABLES.mkdir(parents=True, exist_ok=True)
    cells = [("ESSUA", "HSA"), ("HSA", "HSA"), ("ESSUA", "ESSUA")]
    record = {"target_db": target, "doses_db": list(DOSES_DB), "cells": []}
    for people, room_place in cells:
        rooms = noise_files(room_place)
        room_jobs = []
        for row in rooms.itertuples():
            part = quiet[quiet["file_id"] == row.file_id]
            room_jobs.append(
                (row.file_id, row.wav, list(zip(part["start_s"], part["end_s"], strict=True)))
            )
        chosen = speakers[(speakers["label"] == "normal") & (speakers["place"] == people)]
        chosen = chosen.sort_values("file_id")
        rng = np.random.default_rng([NOISE_SEED, len(people), len(room_place)])
        picks = rng.integers(len(room_jobs), size=len(chosen))
        for dose in DOSES_DB:
            rows = Parallel(n_jobs=jobs)(
                delayed(_noisy_row)(
                    file_id,
                    files.loc[file_id, "wav"],
                    room_jobs[pick],
                    target,
                    dose,
                    (i, dose + 100, len(people), len(room_place)),
                )
                for i, (file_id, pick) in enumerate(zip(chosen["file_id"], picks, strict=True))
            )
            name = f"{people}_with_{room_place}_noise_{dose:+d}.csv"
            pd.DataFrame(rows).to_csv(NOISE_TABLES / name, index=False, lineterminator="\n")
        record["cells"].append(
            {
                "people": people,
                "room": room_place,
                "speakers": len(chosen),
                "room_files_available": len(room_jobs),
                "room_files_used": int(np.unique(picks).size),
            }
        )
    write_json(NOISE_TABLES / "noise_tables.json", record)


def _intervention_fold(X, y, groups, train, test, cases, config, classes) -> list[dict]:
    model = fit_model(X[train], y[train], config, groups=groups[train])
    test_set = set(groups[test].tolist())
    rows = []
    for name, ids, Z in cases:
        keep = np.array([g in test_set for g in ids])
        if keep.any():
            scores = class_scores(model, Z[keep], classes)[:, classes.index("pathological")]
            predicted = np.asarray(model.predict(Z[keep])).astype(str) == "pathological"
            rows.append({"case": name, "speaker": ids[keep], "flag": predicted, "score": scores})
    return rows


def run_intervention(plan, jobs) -> None:
    seeds = plan["statistics"]["seeds"]
    speakers = batch.load_speakers()
    table = merge_features(speakers, batch.load_features("full_original"), "label")
    X, y, groups = design(table, FEATURE_COLUMNS)
    classes = tuple(sorted(np.unique(y).tolist()))
    config = make_config(plan, batch.LABEL_POSITIVE)
    cases: list[tuple[str, np.ndarray, np.ndarray]] = []
    names = {"clean_ESSUA": "ESSUA", "clean_HSA": "HSA"}
    for name, place in names.items():
        part = table[table["place"] == place].query("label == 'normal'")
        cases.append((name, part["group"].to_numpy().astype(str), part[FEATURE_COLUMNS].to_numpy()))
    cells = [("ESSUA", "HSA"), ("HSA", "HSA"), ("ESSUA", "ESSUA")]
    lookup = table.set_index("file_id")["group"]
    for people, room in cells:
        for dose in DOSES_DB:
            noisy = pd.read_csv(NOISE_TABLES / f"{people}_with_{room}_noise_{dose:+d}.csv")
            cases.append(
                (
                    f"{people}_with_{room}_{dose:+d}",
                    noisy["file_id"].map(lookup).to_numpy().astype(str),
                    noisy[FEATURE_COLUMNS].to_numpy(),
                )
            )
    splits = repeated_split_indices(
        y, groups, task="classification", n_splits=5, n_repeats=10, seed=config.seed
    )
    outputs = Parallel(n_jobs=jobs)(
        delayed(_intervention_fold)(X, y, groups, train, test, cases, config, classes)
        for folds in splits
        for train, test in folds
    )
    parts = [
        pd.DataFrame(
            {
                "case": row["case"],
                "speaker": row["speaker"],
                "flag": row["flag"],
                "score": row["score"],
            }
        )
        for output in outputs
        for row in output
    ]
    long = pd.concat(parts, ignore_index=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    long.to_csv(RESULTS / "intervention_predictions.csv.gz", index=False)
    means = long.groupby(["case", "speaker"])["flag"].mean().unstack("case")
    report: dict = {"flagged_clean": {}, "cells": {}}
    for name, place in names.items():
        report["flagged_clean"][place] = float(means[name].dropna().mean())
    for people, room in cells:
        for dose in DOSES_DB:
            case = f"{people}_with_{room}_{dose:+d}"
            flips = (means[case] - means[f"clean_{people}"]).dropna().to_numpy(dtype=float)
            low, high = _bootstrap_mean(flips, seeds["bootstrap"])
            report["cells"][case] = {
                "speakers": int(flips.size),
                "flagged": float(means[case].dropna().mean()),
                "added_flag_rate": float(flips.mean()),
                "ci": [low, high],
            }
    # The flag rate has a ceiling for speakers who are mostly flagged already; the logit has none.
    clipped = long["score"].clip(1e-6, 1 - 1e-6)
    long["logit"] = np.log(clipped / (1 - clipped))
    logits = long.groupby(["case", "speaker"])["logit"].mean().unstack("case")
    for people, room in cells:
        for dose in DOSES_DB:
            case = f"{people}_with_{room}_{dose:+d}"
            moved = (logits[case] - logits[f"clean_{people}"]).dropna().to_numpy(dtype=float)
            low, high = _bootstrap_mean(moved, seeds["bootstrap"])
            report["cells"][case]["logit_shift"] = float(moved.mean())
            report["cells"][case]["logit_shift_ci"] = [low, high]
    write_json(RESULTS / "intervention.json", report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "what", choices=["curve", "all", "transfer", "noise-tables", "intervention"]
    )
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args()
    try:
        verify_freeze()
    except FreezeError as err:
        sys.exit(str(err))
    plan = load_plan()
    started = time.perf_counter()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        if args.what == "curve":
            run_curve(plan, args.jobs)
        elif args.what == "all":
            run_all(plan, args.jobs)
        elif args.what == "transfer":
            run_transfer(plan, args.jobs)
        elif args.what == "noise-tables":
            run_noise_tables(args.jobs)
        else:
            run_intervention(plan, args.jobs)
    print(f"{args.what} done in {time.perf_counter() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
