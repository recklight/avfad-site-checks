"""Pitch statistics alone: do they tell the two places' normals apart, and do they carry the C1 gap.

Exploratory. Definitions and readings are in experiments/deviations.md (2026-11-10), written
before this ran. A linear channel cannot move the fundamental frequency, and noise or
reverberation only disturb its estimate at a low signal-to-noise ratio, so the pitch is used as a
probe of the voices themselves. F0 is YIN (de Cheveigne and Kawahara 2002) on the phonation spans.

    extract       per-file pitch features of every AVFAD /a/ file (results folder only)
    visibility    ESSUA-vs-HSA AUC among the normals, after age and sex
    run           C1, 51 arms x 10 repeats
    collect       writes results/pitch_c1/summary.json

    .venv/Scripts/python.exe -I -X utf8 -B experiments/pitch_c1.py extract --jobs 12
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
from voice_severity_bench import models as vsb_models  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "features"))

import coral_c1 as cc  # noqa: E402
import diagnosis_breakdown as db  # noqa: E402
import patient_position as pp  # noqa: E402
import phonation_c1 as pc  # noqa: E402
import pool_training as pt  # noqa: E402
import published_splits as ps  # noqa: E402
import run_first_batch as batch  # noqa: E402
import site_erasure as se  # noqa: E402
import vowel_fusion as vf  # noqa: E402
from arms import write_json  # noqa: E402
from common import cut, load_signal  # noqa: E402
from freeze import verify_freeze  # noqa: E402
from harness import load_plan, make_config  # noqa: E402

RESULTS = HERE / "results" / "pitch_c1"
S1_OUT = HERE.parent / "s1_segments" / "out"
FEATURES = RESULTS / "pitch_features.csv.gz"
PITCH = ["f0_median_st", "f0_spread_st", "f0_iqr_st", "f0_step_st"]
SEED = 20261150
N_RESAMPLES = 10_000
FRAME_S = 0.040
HOP_S = 0.010
F0_LOW_HZ = 60.0
F0_HIGH_HZ = 500.0
THRESHOLD = 0.15
FEWEST_VOICED = 20
# Readings, written before the run (deviations.md, 2026-11-10).
SEPARATES = 0.65
WEAKEST = 0.60
GAP_POINTS = 0.10


def _frames(x: np.ndarray, length: int, hop: int) -> np.ndarray:
    n = (x.size - length) // hop + 1
    if n <= 0:
        return np.empty((0, length))
    return np.lib.stride_tricks.sliding_window_view(x, length)[::hop][:n]


def yin(x: np.ndarray, rate: int) -> np.ndarray:
    """F0 in Hz per 10 ms frame, NaN where the frame is not voiced."""
    window = round(FRAME_S * rate)
    hop = round(HOP_S * rate)
    tau_min = int(np.floor(rate / F0_HIGH_HZ))
    tau_max = int(np.ceil(rate / F0_LOW_HZ))
    frames = _frames(np.asarray(x, dtype=float), window + tau_max, hop)
    if not frames.shape[0]:
        return np.empty(0)
    # Difference function d(tau) = sum_j (x_j - x_{j+tau})^2 over the window, through the
    # cross-correlation of the first `window` samples with the whole frame.
    n_fft = 1 << int(np.ceil(np.log2(2 * (window + tau_max))))
    head = np.fft.rfft(frames[:, :window], n_fft)
    whole = np.fft.rfft(frames, n_fft)
    corr = np.fft.irfft(np.conj(head) * whole, n_fft)[:, : tau_max + 1]
    energy = np.cumsum(np.pad(frames**2, ((0, 0), (1, 0))), axis=1)
    taus = np.arange(tau_max + 1)
    shifted = energy[:, taus + window] - energy[:, taus]
    diff = np.maximum(shifted[:, :1] + shifted - 2.0 * corr, 0.0)
    # Cumulative mean normalised difference; d'(0) = 1 by definition.
    running = np.cumsum(diff[:, 1:], axis=1)
    norm = np.ones_like(diff)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm[:, 1:] = np.where(running > 0, diff[:, 1:] * taus[1:] / running, 1.0)
    f0 = np.full(frames.shape[0], np.nan)
    for i, row in enumerate(norm):
        below = np.flatnonzero(row[tau_min : tau_max + 1] < THRESHOLD)
        if not below.size:
            continue
        tau = tau_min + int(below[0])
        while tau + 1 <= tau_max and row[tau + 1] < row[tau]:
            tau += 1
        shift = 0.0
        if tau_min < tau < tau_max:
            a, b, c = row[tau - 1], row[tau], row[tau + 1]
            bend = a - 2.0 * b + c
            if bend > 0:
                shift = 0.5 * (a - c) / bend
        f0[i] = rate / (tau + shift)
    return f0


def pitch_features(pieces: list[np.ndarray], rate: int) -> dict[str, float]:
    """The four semitone statistics (re 100 Hz) and the voiced share, over all spans."""
    tones, steps, total = [], [], 0
    for piece in pieces:
        f0 = yin(piece, rate)
        total += f0.size
        st = 12.0 * np.log2(f0 / 100.0)
        tones.append(st[np.isfinite(st)])
        # Steps only between neighbouring voiced frames of the same span.
        step = np.abs(np.diff(st))
        steps.append(step[np.isfinite(step)])
    voiced = np.concatenate(tones) if tones else np.empty(0)
    step = np.concatenate(steps) if steps else np.empty(0)
    out = {
        "voiced_frames": float(voiced.size),
        "voiced_share": voiced.size / total if total else 0.0,
    }
    if voiced.size < FEWEST_VOICED:
        return {**out, **dict.fromkeys(PITCH, np.nan)}
    median = float(np.median(voiced))
    p25, p75 = np.percentile(voiced, [25, 75])
    return {
        **out,
        "f0_median_st": median,
        "f0_spread_st": float(1.4826 * np.median(np.abs(voiced - median))),
        "f0_iqr_st": float(p75 - p25),
        "f0_step_st": float(np.median(step)) if step.size else np.nan,
    }


def _extract_one(file_id: str, wav: str, spans: list[tuple[float, float]]) -> dict:
    x, rate = load_signal(wav)
    return {"file_id": file_id, **pitch_features(cut(x, rate, spans), rate)}


def extract(jobs: int) -> None:
    verify_freeze()
    files = pd.read_csv(S1_OUT / "files.csv", dtype={"file_id": str})
    chosen = files[
        (files["database"] == "avfad") & (files["material"] == "a") & (files["status"] == "ok")
    ].sort_values("file_id")
    spans = pd.read_csv(S1_OUT / "spans.csv", dtype={"file_id": str})
    spans = spans[(spans["kind"] == "phonation") & spans["file_id"].isin(chosen["file_id"])]
    grouped = {
        f: list(zip(p["start_s"].astype(float), p["end_s"].astype(float), strict=True))
        for f, p in spans.groupby("file_id")
    }
    start = time.perf_counter()
    rows = Parallel(n_jobs=jobs)(
        delayed(_extract_one)(r.file_id, r.wav, grouped.get(r.file_id, []))
        for r in chosen.itertuples()
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows).sort_values("file_id")
    table.to_csv(FEATURES, index=False)
    print(f"{len(table)} files in {time.perf_counter() - start:.0f} s, ", end="")
    print(f"{int(table[PITCH[0]].isna().sum())} without enough voiced frames")


def load_pitch() -> pd.DataFrame:
    return pd.read_csv(FEATURES, dtype={"file_id": str})


def pitch_table(members: pd.DataFrame) -> pd.DataFrame:
    """C1 arm table with the pitch columns, in the row order of ``harness.analysis_table``."""
    merged = members.merge(
        load_pitch()[["file_id", *PITCH]], on="file_id", how="left", validate="one_to_one"
    )
    merged = merged.assign(
        path=merged["file_id"], group=merged["speaker"], label=merged["label"].astype(str)
    )
    return merged.sort_values("file_id").reset_index(drop=True)


def _fit(name: str, members: pd.DataFrame, repeat: int, config):
    start = time.perf_counter()
    vsb_models.SEARCH_SPACES[pc.LOGISTIC] = {k: list(v) for k, v in pc.HANDCRAFTED_GRID.items()}
    table = pitch_table(members)
    table = table.assign(fold=pt.fold_ids(table, config)[:, repeat])
    score = np.full(len(table), np.nan)
    predicted = np.empty(len(table), dtype=object)
    for k in range(config.evaluation.n_splits):
        output = ps.fit_listed_fold(table, PITCH, None, config, k)
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


def run(jobs: int) -> None:
    verify_freeze()
    plan = load_plan()
    config = make_config(plan, batch.LABEL_POSITIVE).override({"model.impute": "median"})
    sets = [(n, m) for n, m in vf.arm_sets(plan) if not n.startswith("shuffled")]
    todo = [(n, m, r) for n, m in sets for r in range(config.evaluation.n_repeats)]
    start = time.perf_counter()
    done = Parallel(n_jobs=jobs, return_as="generator_unordered")(
        delayed(_fit)(n, m, r, config) for n, m, r in todo
    )
    frames = []
    for count, (frame, seconds) in enumerate(done, start=1):
        frames.append(frame)
        if count % 100 == 0 or count == len(todo):
            print(
                f"{count} of {len(todo)} after {time.perf_counter() - start:.0f} s "
                f"(last {seconds:.0f} s)",
                flush=True,
            )
    out = pd.concat(frames).sort_values(["arm", "speaker", "repeat"])
    out.to_csv(RESULTS / "c1_predictions_pitch.csv.gz", index=False)


def residual_scores(
    values: np.ndarray, covariates: np.ndarray, essua: np.ndarray, parts: np.ndarray
) -> np.ndarray:
    """Out-of-part ESSUA-vs-HSA scores of ``values`` after the covariates, both fitted per part."""
    scores = np.full(len(essua), np.nan)
    for part in np.unique(parts):
        fit = parts != part
        filled = np.where(np.isnan(values), np.nanmedian(values[fit], axis=0), values)
        trend = LinearRegression().fit(covariates[fit], filled[fit])
        left = filled - trend.predict(covariates)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        clf.fit(left[fit], essua[fit])
        scores[~fit] = clf.decision_function(left[~fit])
    return scores


def covariates_of(normals: pd.DataFrame) -> np.ndarray:
    male = (normals["sex"].astype(str).str.upper().str[0] == "M").to_numpy(dtype=float)
    age = normals["age"].to_numpy(dtype=float)
    return np.column_stack([male, age, male * age])


def visibility_reading(record: dict) -> str:
    if record["auc"] >= SEPARATES and record["ci_low"] > 0.5:
        return "the two places' normals differ in pitch"
    if record["ci_low"] <= 0.5 <= record["ci_high"]:
        return "pitch does not show the place"
    return "as measured"


def visibility() -> None:
    verify_freeze()
    normals = se.normal_speakers().merge(
        load_pitch(), on="file_id", how="left", validate="one_to_one"
    )
    normals = normals[normals["place"].isin(["ESSUA", "HSA"])].reset_index(drop=True)
    if normals["age"].isna().any() or normals["sex"].isna().any():
        raise RuntimeError("a normal speaker has no age or sex")
    essua = (normals["place"] == "ESSUA").to_numpy()
    parts = pp.parts_within_place(
        normals["place"].to_numpy(), cc.N_PARTS, np.random.default_rng(cc.SEED)
    )
    scores = residual_scores(
        normals[PITCH].to_numpy(dtype=float), covariates_of(normals), essua, parts
    )
    kernel = db.comparisons(scores[essua], scores[~essua])
    rng = np.random.default_rng(SEED)
    w_e = db.resample_counts(int(essua.sum()), N_RESAMPLES, rng)
    w_h = db.resample_counts(int((~essua).sum()), N_RESAMPLES, rng)
    boot = db.weighted_auc(kernel, w_e, w_h)
    low, high = np.percentile(boot, [2.5, 97.5])
    record = {"auc": float(kernel.mean()), "ci_low": float(low), "ci_high": float(high)}
    record["reading"] = visibility_reading(record)
    missing = {
        p: int(normals.loc[normals["place"] == p, PITCH[0]].isna().sum()) for p in ("ESSUA", "HSA")
    }
    out = {
        "speakers": {"ESSUA": int(essua.sum()), "HSA": int((~essua).sum())},
        "missing": missing,
        "voiced_share_median": {
            p: float(normals.loc[normals["place"] == p, "voiced_share"].median())
            for p in ("ESSUA", "HSA")
        },
        **record,
    }
    write_json(RESULTS / "visibility.json", out)
    print(json.dumps(out, indent=1))


def gap_reading(gap: dict, i_uar_mean: float) -> str:
    if i_uar_mean < WEAKEST:
        return "pitch is too weak to read"
    if gap["difference"] >= GAP_POINTS and gap["ci_low"] > 0:
        return "pitch alone carries a gap"
    if gap["ci_low"] <= 0:
        return "pitch carries no gap"
    return "the gap shrinks"


def joint_reading(seen: str, gap: str) -> str:
    if seen == "the two places' normals differ in pitch" and gap == "pitch alone carries a gap":
        return "more like the two places recruited different voices"
    if seen == "pitch does not show the place" and gap == "pitch carries no gap":
        return "the cue is not in the pitch"
    return "as measured"


def collect() -> None:
    plan = load_plan()
    table = pd.read_csv(RESULTS / "c1_predictions_pitch.csv.gz", dtype={"speaker": str})
    gap, _, extra = pc.gap_with_samples(table, plan)
    seen = json.loads((RESULTS / "visibility.json").read_text(encoding="utf-8"))
    speakers = batch.load_speakers()
    pitch = load_pitch().merge(
        speakers[["file_id", "place", "label"]], on="file_id", how="left", validate="one_to_one"
    )
    missing = {
        f"{p}_{lab}": int(part[PITCH[0]].isna().sum())
        for (p, lab), part in pitch.groupby(["place", "label"])
    }
    summary = {
        **extra,
        "gap_i_minus_ii": gap,
        "gap_reading": gap_reading(gap, extra["i_uar_mean"]),
        "visibility": seen,
        "missing_by_place_and_label": missing,
    }
    summary["joint_reading"] = joint_reading(seen["reading"], summary["gap_reading"])
    write_json(RESULTS / "summary.json", summary)
    print(
        round(extra["ii"]["uar"], 3),
        round(extra["i_uar_mean"], 3),
        round(gap["difference"], 4),
        round(gap["ci_low"], 4),
        round(gap["ci_high"], 4),
        summary["gap_reading"],
        "|",
        summary["joint_reading"],
    )
    print(missing)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("extract", "visibility", "run", "collect"))
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()
    if args.step == "extract":
        extract(args.jobs)
    elif args.step == "visibility":
        visibility()
    elif args.step == "run":
        run(args.jobs)
    else:
        collect()


if __name__ == "__main__":
    main()
