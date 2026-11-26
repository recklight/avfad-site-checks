"""Frozen HuBERT base: mean and std of every layer over the frames of the kept spans.

Each span goes through the model by itself, so no frame sees across a gap, and spans longer
than 30 s are cut in 30 s pieces to bound the attention memory. The preprocessor's
zero-mean, unit-variance step runs on each piece, which is why the recording level cannot
reach the model and no -26 dBFS scaling is needed here.

Files go to features/out_hubert/, outside features/out/, because the tables there are under
the first freeze. Each file is cached by itself, so a stopped run resumes.

    .venv/Scripts/python.exe -I -B features/extract_hubert.py --jobs 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import ROOT, Spans, cut, load_signal  # noqa: E402

MODEL_DIR = ROOT / "models" / "facebook__hubert-base-ls960"
REVISION = (MODEL_DIR / "REVISION").read_text(encoding="utf-8").strip()
OUT = HERE / "out_hubert"
S1_OUT = ROOT / "s1_segments" / "out"
PIECE_S = 30.0
RATE = 16_000
N_LAYERS = 13  # projection output plus 12 transformer layers
HIDDEN = 768

_MODEL = None
_EXTRACTOR = None


def pieces_of(
    signal_pieces: list[np.ndarray], limit: int = round(PIECE_S * RATE)
) -> list[np.ndarray]:
    """Long spans in pieces of ``limit`` samples; what is too short for the model is dropped."""
    out = []
    for piece in signal_pieces:
        out.extend(piece[start : start + limit] for start in range(0, piece.size, limit))
    return [p for p in out if p.size >= 400]  # 25 ms, the first convolution's window


def pool_frames(layers_frames: list[np.ndarray]) -> tuple[np.ndarray, int]:
    """Mean, then std, of every layer over all frames of arrays (layers, frames, dim)."""
    joined = np.concatenate(layers_frames, axis=1)
    pooled = np.concatenate([joined.mean(axis=1), joined.std(axis=1)], axis=1)
    return pooled, joined.shape[1]


def _load_model() -> None:
    global _MODEL, _EXTRACTOR
    if _MODEL is not None:
        return
    import torch
    from transformers import HubertModel, Wav2Vec2FeatureExtractor

    torch.set_num_threads(1)
    _EXTRACTOR = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_DIR, local_files_only=True)
    _MODEL = HubertModel.from_pretrained(MODEL_DIR, local_files_only=True).eval()


def embed(signal_pieces: list[np.ndarray]) -> tuple[np.ndarray, int]:
    import torch

    _load_model()
    frames = []
    with torch.no_grad():
        for piece in pieces_of(signal_pieces):
            inputs = _EXTRACTOR(
                piece.astype(np.float32), sampling_rate=RATE, return_tensors="pt"
            ).input_values
            hidden = _MODEL(inputs, output_hidden_states=True).hidden_states
            frames.append(np.stack([h[0].numpy() for h in hidden]))
    if not frames:
        return np.full((N_LAYERS, 2 * HIDDEN), np.nan, dtype=np.float32), 0
    pooled, n_frames = pool_frames(frames)
    return pooled.astype(np.float32), n_frames


def extract_one(job: tuple[str, str, str, float, Spans]) -> tuple[str, str]:
    condition, file_id, wav, duration_s, spans = job
    target = OUT / "cache" / condition / f"{file_id}.npz"
    if target.exists():
        return file_id, "cached"
    x, rate = load_signal(wav)
    if rate != RATE:
        raise ValueError(f"{file_id}: HuBERT needs 16 kHz, got {rate}")
    kept = [(0.0, duration_s)] if condition == "full" else spans
    pooled, n_frames = embed(cut(x, rate, kept))
    temporary = target.with_suffix(".tmp.npz")
    np.savez(temporary, pooled=pooled, n_frames=n_frames)
    temporary.replace(target)
    return file_id, "done"


def jobs_for(database: str, material: str, condition: str) -> list[tuple]:
    files = pd.read_csv(S1_OUT / "files.csv")
    chosen = files[
        (files["database"] == database)
        & (files["material"] == material)
        & (files["status"] == "ok")
    ].sort_values("file_id")
    spans = pd.read_csv(S1_OUT / "spans.csv")
    spans = spans[spans["file_id"].isin(chosen["file_id"]) & (spans["kind"] == "phonation")]
    grouped = {
        file_id: list(zip(part["start_s"].astype(float), part["end_s"].astype(float), strict=True))
        for file_id, part in spans.groupby("file_id", sort=False)
    }
    return [
        (condition, row.file_id, row.wav, float(row.duration_s), grouped.get(row.file_id, []))
        for row in chosen.itertuples()
    ]


def assemble(database: str, material: str, condition: str, jobs: list[tuple]) -> Path:
    ids, pooled, frames = [], [], []
    for job in jobs:
        data = np.load(OUT / "cache" / condition / f"{job[1]}.npz")
        ids.append(job[1])
        pooled.append(data["pooled"])
        frames.append(int(data["n_frames"]))
    path = OUT / f"{database}_{material}_{condition}.npz"
    np.savez(
        path,
        file_id=np.array(ids),
        pooled=np.stack(pooled),
        n_frames=np.array(frames),
        revision=np.array(REVISION),
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", default="avfad")
    parser.add_argument("--material", default="a")
    parser.add_argument("--conditions", nargs="*", default=["full", "phonation"])
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--limit", type=int, help="first N files, to measure speed")
    args = parser.parse_args()
    if args.database == "avfad":
        sys.path.insert(0, str(ROOT / "s0_audit"))
        from avfad_extract import require_access

        require_access()
    for condition in args.conditions:
        (OUT / "cache" / condition).mkdir(parents=True, exist_ok=True)
        jobs = jobs_for(args.database, args.material, condition)
        if args.limit:
            jobs = jobs[: args.limit]
        started = time.perf_counter()
        if args.jobs > 1:
            with ProcessPoolExecutor(args.jobs) as pool:
                done = list(pool.map(extract_one, jobs, chunksize=2))
        else:
            done = [extract_one(job) for job in jobs]
        elapsed = time.perf_counter() - started
        seconds = sum(job[3] for job in jobs) if condition == "full" else None
        print(
            json.dumps(
                {
                    "condition": condition,
                    "files": len(done),
                    "new": sum(1 for _, state in done if state == "done"),
                    "seconds": round(elapsed, 1),
                    "audio_s": seconds,
                }
            ),
            flush=True,
        )
        if not args.limit:
            print(assemble(args.database, args.material, condition, jobs), flush=True)


if __name__ == "__main__":
    main()
