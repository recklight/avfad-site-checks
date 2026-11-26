"""Which non-phonation rule keeps its accuracy from one room to the other? Synthetic only.

Usage, with the Kudio venv:
  python -I -B s1_segments/threshold_eval.py [--workers 6] [--seeds 3]

Every rule marks frames as quiet. Two numbers come out per room: the share of
the true pause time marked quiet, and the share of the true phonation time
marked quiet. A rule fit for H1 has both numbers the same in both rooms; a gap
between rooms would pass through to H1 as a difference in the segmenter.
Frames within 50 ms of a true boundary count for neither.

Two entries are not rules for quiet frames but pYIN's own decision, on the full
band and on 150-3500 Hz: there "marked" means called voiced.

Writes s1_segments/out/threshold_eval.csv and threshold_eval_summary.csv.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import librosa
import numpy as np
import pandas as pd
import segment as sg
import synth
from kudio.core.vad import vad

HERE = Path(__file__).resolve().parent
RATE = 16000
BOUNDARY_S = 0.05
FLOOR_RULES = [(p, m) for p in (2.0, 5.0, 15.0) for m in (6.0, 8.0, 10.0, 12.0)]


def level_full_db(y: np.ndarray, rate: int) -> np.ndarray:
    cum = np.concatenate(([0.0], np.cumsum(y**2)))
    centres = np.round(sg.frame_times(y.size, rate, 0.010) * rate).astype(int)
    half = round(0.030 * rate / 2)
    lo, hi = np.clip(centres - half, 0, y.size), np.clip(centres + half, 0, y.size)
    return 10 * np.log10((cum[hi] - cum[lo]) / np.maximum(hi - lo, 1) + 1e-14)


def rules_for(rec: synth.Recording) -> dict[str, np.ndarray]:
    y, rate = rec.x, rec.rate
    times = sg.frame_times(y.size, rate, 0.010)
    full = level_full_db(y, rate)
    _, band = sg.band_level_db(y, rate)
    rules = {
        "absolute_-50dB": full < -50.0,
        "peak_minus_20dB": full < np.percentile(full, 95) - 20.0,
        "peak_minus_30dB": full < np.percentile(full, 95) - 30.0,
        "full_band_p15_plus_8dB": full < np.percentile(full, 15) + 8.0,
    }
    for pct, margin in FLOOR_RULES:
        rules[f"band_p{pct:g}_plus_{margin:g}dB"] = band < np.percentile(band, pct) + margin
    in_vad = sg.spans_to_mask(vad(y, rate), times)
    _, flag, _ = librosa.pyin(y, fmin=60, fmax=800, sr=rate, frame_length=1024, hop_length=160)
    voiced = np.resize(flag, times.size).astype(bool)
    default = sg.PARAMS
    rules["pipeline_full_band_pyin"] = (
        (band < np.percentile(band, default.floor_percentile) + default.margin_db)
        & ~in_vad
        & ~voiced
    )
    rules["pyin_voiced_full_band"] = voiced
    contains_voice = bool(rec.phonation)
    seg = sg.segment_signal(y, rate, contains_voice=contains_voice)
    rules["pyin_voiced_band"] = seg.tracks["voiced"]
    for kind, rule in (("nonphonation_raw", "pipeline"), ("nonphonation_g100", "pipeline_g100")):
        spans = [(s["start_s"], s["end_s"]) for s in seg.spans if s["kind"] == kind]
        rules[rule] = sg.spans_to_mask(spans, times)
    return rules


def one(job: tuple[str, str, str, int]) -> list[dict]:
    case, room, voice, seed = job
    rec = synth.recording(case, room, RATE, seed, voice=voice)
    times = sg.frame_times(rec.x.size, RATE, 0.010)
    phon = sg.spans_to_mask([(a + BOUNDARY_S, b - BOUNDARY_S) for a, b in rec.phonation], times)
    pause = sg.spans_to_mask([(a + BOUNDARY_S, b - BOUNDARY_S) for a, b in rec.pauses], times)
    rows = []
    for rule, quiet in rules_for(rec).items():
        rows.append(
            {
                "rule": rule,
                "room": room,
                "case": case,
                "voice": voice,
                "seed": seed,
                "pause_marked": float(quiet[pause].mean()) if pause.any() else np.nan,
                "phonation_marked": float(quiet[phon].mean()) if phon.any() else np.nan,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    jobs = []
    for seed in range(args.seeds):
        for room in synth.ROOMS:
            jobs.append(("silence_only", room, "modal", seed))
            for case in ("three_reps", "short_pauses", "one_short_pause", "no_pause"):
                for voice in synth.VOICES:
                    jobs.append((case, room, voice, seed))
    summary_rooms = ["quiet_broadband", "noisy_lowfreq", "noisy_tonal"]
    sg.warm_jit_cache()
    with ProcessPoolExecutor(args.workers) as pool:
        rows = [r for batch in pool.map(one, jobs) for r in batch]
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "threshold_eval.csv", index=False)
    summary = (
        table.groupby(["rule", "room"])[["pause_marked", "phonation_marked"]].mean().unstack("room")
    )
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    pause = summary[[f"pause_marked_{r}" for r in summary_rooms]]
    phonation = summary[[f"phonation_marked_{r}" for r in summary_rooms]]
    summary["pause_gap"] = pause.max(axis=1) - pause.min(axis=1)
    summary["phonation_gap"] = phonation.max(axis=1) - phonation.min(axis=1)
    summary.sort_values("pause_gap").round(3).to_csv(out / "threshold_eval_summary.csv")
    with pd.option_context("display.width", 200, "display.max_rows", 50):
        print(summary.sort_values("pause_gap").round(3).to_string())


if __name__ == "__main__":
    main()
