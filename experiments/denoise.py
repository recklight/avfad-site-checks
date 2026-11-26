"""Log-MMSE denoising before the detector: does the shortcut shrink? Exploratory.

Definitions and readings are in experiments/deviations.md, written before any file was
denoised. The estimator is speechdsp.log_mmse. Each file's own quiet spans (at most 0.5 s)
are put in front of it so the estimator learns the noise from them, and are cut off again.

    features      handcrafted features of the denoised /a/ files, and of the intervention's
                  main cell (ESSUA normal speakers with HSA room noise at 0 dB), denoised
    run           C1, everyone and the intervention on the denoised features
    collect       comparisons with the plain detector; writes denoise/summary.json
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
import speechdsp  # noqa: E402
from voice_severity_bench.splitting import repeated_split_indices  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "features"))

import exploratory_avfad as ex  # noqa: E402
import noise_augmentation as na  # noqa: E402
import run_first_batch as batch  # noqa: E402
from arms import write_json  # noqa: E402
from common import cut, load_signal  # noqa: E402
from extract_handcrafted import FEATURE_COLUMNS, features_of, level_statistics  # noqa: E402
from freeze import verify_freeze  # noqa: E402
from freeze_svd import verify_svd_freeze  # noqa: E402
from harness import ROOT, evaluate_arm, load_plan, make_config  # noqa: E402

PREFIX_MAX_S = 0.5
FALLBACK_NOISE_MS = 100.0
SMOOTHING = 0.98
EXPLORATORY = ROOT / "features" / "out_exploratory"
CLEAN = EXPLORATORY / "avfad_a_full_original_logmmse.csv"
NOISY = EXPLORATORY / "ESSUA_with_HSA_noise_+0_logmmse.csv"
RESULTS = HERE / "results" / "denoise"
S1_OUT = ROOT / "s1_segments" / "out"

# Copied from voice-clinic-desktop, src/voice_clinic_desktop/noise_reduction.py (HEAD 38196e4,
# the file last changed in 57d432e). The speechdsp estimators use an FFT near 32 ms with a
# quarter hop; this counts the frames that lie wholly inside the first noise_ms.
_SPEECHDSP_WINDOW_S = 0.032


def leading_noise_frames(noise_ms: float, sample_rate: int) -> int:
    n_fft = int(2 ** max(6, round(np.log2(_SPEECHDSP_WINDOW_S * sample_rate))))
    hop = n_fft // 4
    noise_samples = noise_ms * sample_rate / 1000.0
    return max(1, int(np.floor((noise_samples - n_fft / 2) / hop)) + 1)


def denoise(x: np.ndarray, rate: int, quiet: list[np.ndarray]) -> np.ndarray:
    """Log-MMSE with the noise learned from the file's own quiet spans, put in front."""
    if quiet:
        prefix = np.concatenate(quiet)[: round(PREFIX_MAX_S * rate)]
        joined = np.concatenate([prefix, x])
        frames = leading_noise_frames(1000.0 * prefix.size / rate, rate)
        out = speechdsp.log_mmse(joined, rate, noise_frames=frames, alpha=SMOOTHING)
        return np.asarray(out, dtype=np.float64)[prefix.size : prefix.size + x.size]
    frames = leading_noise_frames(FALLBACK_NOISE_MS, rate)
    return np.asarray(speechdsp.log_mmse(x, rate, noise_frames=frames, alpha=SMOOTHING))


def quiet_spans() -> dict[str, list[tuple[float, float]]]:
    spans = pd.read_csv(S1_OUT / "spans.csv")
    spans = spans[spans["kind"] == "nonphonation_g100"]
    return {
        file_id: list(zip(part["start_s"], part["end_s"], strict=True))
        for file_id, part in spans.groupby("file_id")
    }


def _clean_row(file_id: str, wav: str, spans) -> dict:
    x, rate = load_signal(wav)
    quiet = cut(x, rate, spans)
    y = denoise(x, rate, quiet)
    before = float(level_statistics(quiet, rate)[0]) if quiet else np.nan
    after = float(level_statistics(cut(y, rate, spans), rate)[0]) if quiet else np.nan
    return {
        "file_id": file_id,
        "quiet_level_before_db": before,
        "quiet_level_after_db": after,
        **dict(zip(FEATURE_COLUMNS, features_of([y], rate), strict=True)),
    }


def _noisy_row(file_id, wav, room, target_db, seed_key, spans) -> dict:
    """The intervention's noisy file, built exactly as exploratory_avfad does, then denoised."""
    x, rate = load_signal(wav)
    pieces, level = ex.room_quiet(*room)
    rng = np.random.default_rng([ex.NOISE_SEED, *seed_key])
    total = int(sum(p.size for p in pieces))
    noise = ex.loop_to_length(pieces, x.size, int(rng.integers(total)), rate)
    noisy = x + noise * 10.0 ** ((target_db + 0 - level) / 20.0)
    y = denoise(noisy, rate, cut(noisy, rate, spans))
    return {"file_id": file_id, **dict(zip(FEATURE_COLUMNS, features_of([y], rate), strict=True))}


def run_features(jobs: int) -> None:
    speakers = batch.load_speakers().sort_values("file_id")
    files = pd.read_csv(S1_OUT / "files.csv").set_index("file_id")
    spans = quiet_spans()
    started = time.perf_counter()
    rows = Parallel(n_jobs=jobs)(
        delayed(_clean_row)(f, files.loc[f, "wav"], spans.get(f, [])) for f in speakers["file_id"]
    )
    EXPLORATORY.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(CLEAN, index=False, lineterminator="\n")
    print(len(rows), "clean files in", round(time.perf_counter() - started), "s", flush=True)
    # The intervention's main cell, with the same room, start and level as exploratory_avfad.
    target = ex.hsa_target_db(speakers)
    rooms = ex.noise_files("HSA")
    room_jobs = [(row.file_id, row.wav, spans.get(row.file_id, [])) for row in rooms.itertuples()]
    chosen = speakers[(speakers["label"] == "normal") & (speakers["place"] == "ESSUA")]
    chosen = chosen.sort_values("file_id")
    rng = np.random.default_rng([ex.NOISE_SEED, len("ESSUA"), len("HSA")])
    picks = rng.integers(len(room_jobs), size=len(chosen))
    noisy = Parallel(n_jobs=jobs)(
        delayed(_noisy_row)(
            file_id,
            files.loc[file_id, "wav"],
            room_jobs[pick],
            target,
            (i, 0 + 100, len("ESSUA"), len("HSA")),
            spans.get(file_id, []),
        )
        for i, (file_id, pick) in enumerate(zip(chosen["file_id"], picks, strict=True))
    )
    pd.DataFrame(noisy).to_csv(NOISY, index=False, lineterminator="\n")
    print(len(noisy), "noisy files in", round(time.perf_counter() - started), "s", flush=True)


def run_arms(jobs: int) -> None:
    plan = load_plan()
    config = make_config(plan, batch.LABEL_POSITIVE)
    speakers = batch.load_speakers()
    features = pd.read_csv(CLEAN)
    RESULTS.mkdir(parents=True, exist_ok=True)
    c1 = batch.run_c1(speakers, features[["file_id", *FEATURE_COLUMNS]], plan, jobs, RESULTS)
    write_json(RESULTS / "c1_summary.json", c1)
    table = speakers.merge(features[["file_id", *FEATURE_COLUMNS]], on="file_id", how="left")
    table = table.sort_values("file_id").reset_index(drop=True)
    X = table[FEATURE_COLUMNS].to_numpy()
    y = table["label"].to_numpy().astype(str)
    groups = table["speaker"].to_numpy().astype(str)
    classes = tuple(sorted(np.unique(y).tolist()))
    lookup = table.set_index("file_id")["speaker"]
    noisy = pd.read_csv(NOISY)
    essua = table[(table["place"] == "ESSUA") & (table["label"] == "normal")]
    cases = [
        ("clean_ESSUA", essua["speaker"].to_numpy().astype(str), essua[FEATURE_COLUMNS].to_numpy()),
        (
            na.MAIN_CASE,
            noisy["file_id"].map(lookup).to_numpy().astype(str),
            noisy[FEATURE_COLUMNS].to_numpy(),
        ),
    ]
    splits = repeated_split_indices(
        y, groups, task="classification", n_splits=5, n_repeats=10, seed=config.seed
    )
    outputs = Parallel(n_jobs=jobs)(
        delayed(ex._intervention_fold)(X, y, groups, train, test, cases, config, classes)
        for folds in splits
        for train, test in folds
    )
    long = pd.concat(
        [
            pd.DataFrame({"case": r["case"], "speaker": r["speaker"], "flag": r["flag"]})
            for output in outputs
            for r in output
        ],
        ignore_index=True,
    )
    long.to_csv(RESULTS / "intervention_predictions.csv.gz", index=False)
    every = evaluate_arm("all", table.assign(path=table["file_id"], group=table["speaker"]), config)
    write_json(RESULTS / "all.json", {"uar": every.uar, "roc_auc": every.roc_auc})


def run_collect() -> None:
    plan = load_plan()
    seed = plan["statistics"]["seeds"]["bootstrap"]
    n = plan["confirmatory_family"]["tests"]["C1"]["test"]["resamples"]
    den = na.arms_from(RESULTS / "c1_predictions.csv.gz")
    base = na.arms_from(HERE / "results" / "first_batch" / "c1_predictions.csv.gz")
    den_i = [a for k, a in den.items() if k.startswith("i_")]
    base_i = [a for k, a in base.items() if k.startswith("i_")]
    gap_den = batch.c1_bootstrap(den_i, den["ii"], n, seed)
    gap_base = batch.c1_bootstrap(base_i, base["ii"], n, seed)
    change = gap_den.pop("samples") - gap_base.pop("samples")
    finite = change[np.isfinite(change)]
    ii_change = batch.c1_bootstrap([den["ii"]], base["ii"], n, seed)
    ii_change.pop("samples")
    flip_den = na.flips(RESULTS / "intervention_predictions.csv.gz", "ESSUA", na.MAIN_CASE)
    flip_base = na.flips(
        HERE / "results" / "exploratory" / "intervention_predictions.csv.gz", "ESSUA", na.MAIN_CASE
    )
    common = flip_den.index.intersection(flip_base.index)
    paired = (flip_den.loc[common] - flip_base.loc[common]).to_numpy(dtype=float)
    flip_ci = na.paired_mean_ci(paired, seed)
    levels = pd.read_csv(CLEAN)[["quiet_level_before_db", "quiet_level_after_db"]].dropna()
    plain_all = json.loads(
        (HERE / "results" / "exploratory" / "avfad_all.json").read_text(encoding="utf-8")
    )["all_handcrafted"]
    report = {
        "all": {
            "denoised": json.loads((RESULTS / "all.json").read_text(encoding="utf-8")),
            "plain": {"uar": plain_all["uar"], "roc_auc": plain_all["roc_auc"]},
        },
        "ii": {"denoised": den["ii"].uar, "plain": base["ii"].uar, "change": ii_change},
        "i": {
            "denoised_mean": float(np.mean([a.uar for a in den_i])),
            "plain_mean": float(np.mean([a.uar for a in base_i])),
        },
        "gap": {
            "denoised": gap_den,
            "plain": gap_base,
            "change": float(gap_den["difference"] - gap_base["difference"]),
            "change_ci": [float(v) for v in np.quantile(finite, [0.025, 0.975])],
        },
        "intervention": {
            "case": na.MAIN_CASE,
            "denoised": float(flip_den.mean()),
            "plain": float(flip_base.mean()),
            "speakers": int(common.size),
            "change": float(paired.mean()),
            "change_ci": flip_ci,
        },
        "quiet_level_db": {
            "files": len(levels),
            "median_before": float(levels["quiet_level_before_db"].median()),
            "median_after": float(levels["quiet_level_after_db"].median()),
        },
    }
    report["reading"] = na.reading(
        report["intervention"]["denoised"],
        report["intervention"]["plain"],
        flip_ci,
        report["gap"]["change"],
        report["gap"]["change_ci"],
        report["ii"]["denoised"] - report["ii"]["plain"],
    )
    write_json(RESULTS / "summary.json", report)
    print(json.dumps(report, indent=1)[:3000])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("features", "run", "collect"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    verify_freeze()
    verify_svd_freeze()
    if args.step == "features":
        run_features(args.jobs)
    elif args.step == "run":
        run_arms(args.jobs)
    else:
        run_collect()


if __name__ == "__main__":
    main()
