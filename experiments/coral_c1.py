"""C1 with the ESSUA speakers moved onto the HSA normals by CORAL, and how well place still shows.

Exploratory. Definitions and readings are in experiments/deviations.md (2026-11-10), written
before this run. CORAL here also matches the mean: z-scores on the HSA normals, then
(z - m_E) C_E^(-1/2) C_H^(1/2) with Ledoit-Wolf covariances (Sun, Feng and Saenko 2016). The
fold-wise path, the speakers and the readings of the gap are those of harmonize_c1.py.

    run         --model logistic|svc|random_forest   50 draws x 10 repeats, handcrafted
    visibility  place AUC among the 356 normals after each alignment
    collect     writes results/coral_c1/summary.json

    .venv/Scripts/python.exe -I -X utf8 -B experiments/coral_c1.py run --model logistic --jobs 12
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
from joblib import Parallel, delayed  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "features"))

import harmonize_c1 as hc  # noqa: E402
import nonlinear_c1 as nl  # noqa: E402
import patient_position as pp  # noqa: E402
import pool_training as pt  # noqa: E402
import published_splits as ps  # noqa: E402
import run_first_batch as batch  # noqa: E402
import site_erasure as se  # noqa: E402
from arms import layer_columns, load_hubert, write_json  # noqa: E402
from extract_handcrafted import FEATURE_COLUMNS  # noqa: E402
from freeze import verify_freeze  # noqa: E402
from harness import load_plan, make_config  # noqa: E402
from sensitivity_avfad import CHOSEN_LAYERS  # noqa: E402

RESULTS = HERE / "results" / "coral_c1"
MODELS = ("logistic", "svc", "random_forest")
KINDS = ("none", "location", "scale", "coral")
SEED = 20261141
N_PARTS = 5
N_TREES = 500
# Reading, written before the run (deviations.md, 2026-11-10).
CORAL_POINTS = 0.03


def matrix_power(matrix: np.ndarray, power: float) -> np.ndarray:
    """Power of a symmetric positive definite matrix through its eigenvalues."""
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, values.max() * 1e-12, None)
    return (vectors * values**power) @ vectors.T


def coral_from(essua: np.ndarray, hsa: np.ndarray) -> dict:
    """CORAL map of the ESSUA rows onto the HSA rows, estimated on z-scores of the HSA rows."""
    mu_h = hsa.mean(axis=0)
    sd_h = hsa.std(axis=0, ddof=1)
    sd_h = np.where(sd_h > 0, sd_h, 1.0)
    z_e = (essua - mu_h) / sd_h
    z_h = (hsa - mu_h) / sd_h
    c_e = LedoitWolf().fit(z_e).covariance_
    c_h = LedoitWolf().fit(z_h).covariance_
    return {
        "mu_h": mu_h,
        "sd_h": sd_h,
        "m_e": z_e.mean(axis=0),
        "transform": matrix_power(c_e, -0.5) @ matrix_power(c_h, 0.5),
    }


def coral_apply(values: np.ndarray, moments: dict) -> np.ndarray:
    z = (values - moments["mu_h"]) / moments["sd_h"]
    z = (z - moments["m_e"]) @ moments["transform"]
    return z * moments["sd_h"] + moments["mu_h"]


def moved_by(essua: np.ndarray, hsa: np.ndarray, kind: str, rows: np.ndarray) -> np.ndarray:
    """``rows`` moved from the place of ``essua`` onto that of ``hsa``, both normals only."""
    mu_e, mu_h = essua.mean(axis=0), hsa.mean(axis=0)
    if kind == "location":
        return rows - mu_e + mu_h
    if kind == "scale":
        sd_e, sd_h = essua.std(axis=0, ddof=1), hsa.std(axis=0, ddof=1)
        ratio = np.divide(sd_h, sd_e, out=np.ones_like(sd_e), where=sd_e > 0)
        return (rows - mu_e) * ratio + mu_h
    if kind == "coral":
        return coral_apply(rows, coral_from(essua, hsa))
    raise ValueError(kind)


def align_values(
    values: np.ndarray, essua: np.ndarray, fit: np.ndarray, groups: list[np.ndarray], kind: str
) -> np.ndarray:
    """ESSUA rows moved onto the HSA normals of ``fit``; ``groups`` are column indices aligned
    on their own (one HuBERT layer each). ``essua`` and ``fit`` mark normals only here."""
    out = values.copy()
    if kind == "none":
        return out
    for idx in groups:
        block = values[:, idx]
        moved = moved_by(block[fit & essua], block[fit & ~essua], kind, block[essua])
        out[np.ix_(essua, idx)] = moved
    return out


def coral_table(
    table: pd.DataFrame, columns: list[str], normals: pd.DataFrame, test_speakers: set[str]
) -> pd.DataFrame:
    """ESSUA rows of ``table`` moved by CORAL, estimated on the normals outside the test fold."""
    kept = normals[~normals["speaker"].isin(test_speakers)]
    essua = (kept["place"] == "ESSUA").to_numpy()
    if essua.all() or not essua.any():
        raise RuntimeError("a place has no normal speaker outside the test fold")
    values = kept[columns].to_numpy(dtype=float)
    moments = coral_from(values[essua], values[~essua])
    rows = (table["place"] == "ESSUA").to_numpy()
    out = table.copy()
    moved = table[columns].to_numpy(dtype=float)
    moved[rows] = coral_apply(moved[rows], moments)
    out[columns] = moved
    return out


def config_for(model: str):
    config = make_config(load_plan(), batch.LABEL_POSITIVE)
    return config if model == "logistic" else config.override({"model.name": model})


def _fit(name: str, members: pd.DataFrame, repeat: int, config):
    start = time.perf_counter()
    table, normals, columns, blocks, tuned = hc.detector_inputs("handcrafted", members, config)
    table = table.assign(fold=pt.fold_ids(table, config)[:, repeat])
    score = np.full(len(table), np.nan)
    predicted = np.empty(len(table), dtype=object)
    for k in range(config.evaluation.n_splits):
        test_speakers = set(table.loc[table["fold"] == k, "speaker"])
        moved = coral_table(table, columns, normals, test_speakers)
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


def path_of(model: str) -> Path:
    return RESULTS / f"c1_predictions_{model}_coral.csv.gz"


def run(jobs: int, model: str) -> None:
    verify_freeze()
    plan = load_plan()
    config = config_for(model)
    todo = [(n, m, r) for n, m in hc.draw_sets(plan) for r in range(config.evaluation.n_repeats)]
    RESULTS.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    done = Parallel(n_jobs=jobs, return_as="generator_unordered")(
        delayed(_fit)(n, m, r, config) for n, m, r in todo
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
    pd.concat(frames).sort_values(["arm", "speaker", "repeat"]).to_csv(path_of(model), index=False)


def place_scores(
    values: np.ndarray,
    essua: np.ndarray,
    parts: np.ndarray,
    groups: list[np.ndarray],
    kind: str,
    model: str,
) -> np.ndarray:
    """Out-of-part ESSUA-vs-HSA scores among the normals, alignment estimated on the other parts."""
    scores = np.full(len(essua), np.nan)
    for part in np.unique(parts):
        fit = parts != part
        moved = align_values(values, essua, fit, groups, kind)
        if model == "logistic":
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        else:
            clf = RandomForestClassifier(n_estimators=N_TREES, random_state=SEED, n_jobs=1)
        clf.fit(moved[fit], essua[fit])
        scores[~fit] = clf.predict_proba(moved[~fit])[:, 1]
    return scores


def _visibility_one(features: str, kind: str, model: str) -> tuple[str, str, str, float]:
    normals = se.normal_speakers()
    if features == "hubert":
        hubert = load_hubert("full", CHOSEN_LAYERS, material="avfad_a")
        columns = [c for layer in CHOSEN_LAYERS for c in layer_columns(layer)]
        normals = normals.merge(
            hubert[["file_id", *columns]], on="file_id", how="left", validate="one_to_one"
        )
        width = len(layer_columns(CHOSEN_LAYERS[0]))
        groups = [np.arange(i * width, (i + 1) * width) for i in range(len(CHOSEN_LAYERS))]
    else:
        handcrafted = batch.load_features("full_original")
        columns = list(FEATURE_COLUMNS)
        normals = normals.merge(
            handcrafted[["file_id", *columns]], on="file_id", how="left", validate="one_to_one"
        )
        groups = [np.arange(len(columns))]
    values = normals[columns].to_numpy(dtype=float)
    if np.isnan(values).any():
        raise RuntimeError("a normal speaker has no features")
    essua = (normals["place"] == "ESSUA").to_numpy()
    parts = pp.parts_within_place(normals["place"].to_numpy(), N_PARTS, np.random.default_rng(SEED))
    scores = place_scores(values, essua, parts, groups, kind, model)
    return features, kind, model, float(roc_auc_score(essua, scores))


def visibility(jobs: int) -> None:
    verify_freeze()
    todo = [("handcrafted", k, m) for k in KINDS for m in ("logistic", "forest")]
    todo += [("hubert", k, "logistic") for k in KINDS]
    start = time.perf_counter()
    done = Parallel(n_jobs=jobs)(delayed(_visibility_one)(*t) for t in todo)
    out: dict = {}
    for features, kind, model, auc in done:
        out.setdefault(features, {}).setdefault(model, {})[kind] = auc
    out["seconds"] = round(time.perf_counter() - start, 1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / "visibility.json", out)
    print(json.dumps(out, indent=1))


def coral_reading(change: dict) -> str:
    if change["difference"] <= -CORAL_POINTS and change["ci_high"] < 0:
        return "the correlations carry part of the gap"
    if change["ci_low"] <= 0 <= change["ci_high"]:
        return "the correlations carry nothing more"
    return "as measured"


def baselines(model: str) -> tuple[Path, Path]:
    """Unaligned and mean-plus-SD predictions of the same detector."""
    if model == "logistic":
        return hc.PLAIN["handcrafted"], hc.RESULTS / "c1_predictions_handcrafted_scale.csv.gz"
    return nl.path_of(model, "plain"), nl.path_of(model, "scale")


def collect() -> None:
    plan = load_plan()
    summary: dict = {}
    for model in MODELS:
        plain_path, scale_path = baselines(model)
        if not (path_of(model).exists() and plain_path.exists() and scale_path.exists()):
            continue
        plain = nl.read(plain_path)
        base = hc.gap_of(plain, plain, plan)
        scale = hc.gap_of(nl.read(scale_path), plain, plan)
        got = hc.gap_of(nl.read(path_of(model)), plain, plan)
        change = hc.paired(got, scale)
        summary[model] = {
            "plain_gap": base[0]["difference"],
            "scale_gap": scale[0]["difference"],
            "gap_i_minus_ii": got[0],
            "gap_reading": hc.gap_reading(got[0]),
            "minus_plain": hc.paired(got, base),
            "minus_scale": change,
            "coral_reading": coral_reading(change),
        }
    seen = RESULTS / "visibility.json"
    if seen.exists():
        summary["visibility"] = json.loads(seen.read_text(encoding="utf-8"))
    write_json(RESULTS / "summary.json", summary)
    for model in MODELS:
        if model in summary:
            g = summary[model]["gap_i_minus_ii"]
            print(model, summary[model]["gap_reading"], round(g["difference"], 4))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("run", "visibility", "collect"))
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--model", choices=MODELS, default="logistic")
    args = parser.parse_args()
    if args.step == "run":
        run(args.jobs, args.model)
    elif args.step == "visibility":
        visibility(args.jobs)
    else:
        collect()


if __name__ == "__main__":
    main()
