"""How long fine-tuning HuBERT's last layers would take on this CPU; timing only.

Nothing is evaluated: no prediction is scored and no result is kept from the labels. A few
training steps on a handful of /a/ files give seconds per step and peak memory. Each setting
runs in its own process, so its peak memory is its own and its thread count holds. The
weights are the ones already on disk; nothing is downloaded or installed.

    .venv/Scripts/python.exe -I -B experiments/finetune_timing.py
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_DIR = ROOT / "models" / "facebook__hubert-base-ls960"
OUT = HERE / "results" / "finetune_timing"
RATE = 16_000
FRAMES_PER_S = 50
HIDDEN = 768
BYTES_PER_FRAME = HIDDEN * 4
N_SPEAKERS = 16
SAMPLE_SEED = 20261107
STEPS_MEASURED = 6
STEPS_DROPPED = 1  # the first step allocates


class _Counters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def peak_memory_gb() -> float:
    counters = _Counters()
    counters.cb = ctypes.sizeof(_Counters)
    # The handle is pointer-sized; without these types ctypes cuts it to 32 bits and the call fails.
    kernel32 = ctypes.WinDLL("kernel32")
    psapi = ctypes.WinDLL("psapi")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_Counters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise OSError("GetProcessMemoryInfo failed")
    return counters.PeakWorkingSetSize / 2**30


def sample_paths() -> list[str]:
    import numpy as np
    import pandas as pd

    speakers = pd.read_csv(ROOT / "s0_audit" / "out" / "avfad_speakers.csv", dtype={"speaker": str})
    hsa = speakers[speakers["place"] == "HSA"]
    rng = np.random.default_rng(SAMPLE_SEED)
    chosen = []
    for label in ("normal", "pathological"):
        group = hsa[hsa["label"] == label].sort_values("speaker")
        chosen += rng.choice(group["a_path"].to_numpy(), N_SPEAKERS // 2, replace=False).tolist()
    return chosen


def load_wave(path: str):
    sys.path.insert(0, str(ROOT / "features"))
    from common import load_signal

    signal, rate = load_signal(path)
    if rate != RATE:
        raise ValueError(f"{path}: expected 16 kHz, got {rate}")
    return signal.astype("float32")


def crop(wave, seconds: float, rng):
    length = round(seconds * RATE)
    if wave.size <= length:
        return wave
    start = int(rng.integers(0, wave.size - length))
    return wave[start : start + length]


def build_model(trainable_layers: int):
    import torch
    from transformers import HubertModel

    model = HubertModel.from_pretrained(MODEL_DIR, local_files_only=True)
    for parameter in model.parameters():
        parameter.requires_grad = False
    layers = model.encoder.layers
    for layer in layers[len(layers) - trainable_layers :]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    head = torch.nn.Linear(HIDDEN, 2)
    return model, head


def measure_end_to_end(trainable: int, batch: int, seconds: float) -> dict:
    """The whole model runs every step; gradients stop at the frozen layers."""
    import numpy as np
    import torch
    from transformers import Wav2Vec2FeatureExtractor

    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_DIR, local_files_only=True)
    model, head = build_model(trainable)
    model.train()
    wanted = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    optimizer = torch.optim.AdamW(wanted, lr=1e-5)
    waves = [load_wave(path) for path in sample_paths()]
    rng = np.random.default_rng(SAMPLE_SEED)
    times = []
    for step in range(STEPS_MEASURED + STEPS_DROPPED):
        picks = rng.choice(len(waves), batch, replace=False)
        chunks = [crop(waves[i], seconds, rng) for i in picks]
        inputs = extractor(chunks, sampling_rate=RATE, return_tensors="pt", padding=True)
        target = torch.as_tensor(rng.integers(0, 2, batch))
        started = time.perf_counter()
        hidden = model(inputs.input_values).last_hidden_state
        loss = torch.nn.functional.cross_entropy(head(hidden.mean(dim=1)), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step >= STEPS_DROPPED:
            times.append(time.perf_counter() - started)
    return {"seconds_per_step": float(np.median(times)), "peak_gb": peak_memory_gb()}


def through(layer, hidden):
    """One encoder layer; some transformers versions return a tuple, some the tensor."""
    output = layer(hidden)
    return output[0] if isinstance(output, tuple) else output


def measure_cached(trainable: int, batch: int, seconds: float) -> dict:
    """The frozen part ran once; each step only runs the trainable layers on its output."""
    import numpy as np
    import torch
    from transformers import Wav2Vec2FeatureExtractor

    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_DIR, local_files_only=True)
    model, head = build_model(trainable)
    model.eval()
    layers = model.encoder.layers
    split = len(layers) - trainable
    waves = [load_wave(path)[: 30 * RATE] for path in sample_paths()]
    started = time.perf_counter()
    cached = []
    with torch.no_grad():
        for wave in waves:
            inputs = extractor(wave, sampling_rate=RATE, return_tensors="pt").input_values
            hidden = model(inputs, output_hidden_states=True).hidden_states[split]
            cached.append(hidden[0].numpy())
    extract_seconds = time.perf_counter() - started
    audio_seconds = sum(wave.size for wave in waves) / RATE
    trainable_layers = torch.nn.ModuleList(layers[split:])
    trainable_layers.train()
    wanted = list(trainable_layers.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(wanted, lr=1e-5)
    rng = np.random.default_rng(SAMPLE_SEED)
    frames = round(seconds * FRAMES_PER_S)
    times = []
    for step in range(STEPS_MEASURED + STEPS_DROPPED):
        picks = rng.choice(len(cached), batch, replace=False)
        chunks = []
        for i in picks:
            item = cached[i]
            start = int(rng.integers(0, max(item.shape[0] - frames, 0) + 1))
            part = item[start : start + frames]
            if part.shape[0] < frames:
                part = np.pad(part, ((0, frames - part.shape[0]), (0, 0)))
            chunks.append(part)
        states = torch.as_tensor(np.stack(chunks))
        target = torch.as_tensor(rng.integers(0, 2, batch))
        began = time.perf_counter()
        hidden = states
        for layer in trainable_layers:
            hidden = through(layer, hidden)
        loss = torch.nn.functional.cross_entropy(head(hidden.mean(dim=1)), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step >= STEPS_DROPPED:
            times.append(time.perf_counter() - began)
    trainable_layers.eval()
    began = time.perf_counter()
    with torch.no_grad():
        for item in cached:
            hidden = torch.as_tensor(item[None])
            for layer in trainable_layers:
                hidden = through(layer, hidden)
    infer_seconds = time.perf_counter() - began
    return {
        "seconds_per_step": float(np.median(times)),
        "peak_gb": peak_memory_gb(),
        "extract_seconds_per_audio_second": extract_seconds / audio_seconds,
        "infer_seconds_per_audio_second": infer_seconds / audio_seconds,
    }


SETTINGS = [
    # mode, trainable layers, batch, seconds of audio per item, threads
    ("end_to_end", 2, 8, 4.0, 4),
    ("end_to_end", 2, 8, 10.0, 4),
    ("cached", 1, 8, 4.0, 4),
    ("cached", 2, 8, 4.0, 4),
    ("cached", 4, 8, 4.0, 4),
    ("cached", 1, 8, 10.0, 4),
    ("cached", 2, 8, 10.0, 4),
    ("cached", 4, 8, 10.0, 4),
    ("cached", 2, 8, 10.0, 1),
    ("cached", 2, 8, 10.0, 2),
    ("cached", 2, 8, 10.0, 8),
    ("cached", 2, 4, 10.0, 4),
]


# Second pass: the memory of the end-to-end settings, and every transformer layer trained.
EXTRA_SETTINGS = [
    ("end_to_end", 2, 8, 4.0, 4),
    ("end_to_end", 2, 8, 10.0, 4),
    ("end_to_end", 12, 8, 4.0, 4),
    ("end_to_end", 12, 8, 10.0, 4),
    ("cached", 1, 8, 4.0, 4),
]


def run_one(mode: str, trainable: int, batch: int, seconds: float, threads: int) -> dict:
    import torch

    torch.set_num_threads(threads)
    torch.manual_seed(SAMPLE_SEED)
    measure = measure_end_to_end if mode == "end_to_end" else measure_cached
    return {
        "mode": mode,
        "trainable_layers": trainable,
        "batch": batch,
        "audio_seconds_per_item": seconds,
        "threads": threads,
        **measure(trainable, batch, seconds),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one", help="JSON of one setting; used by the parent process")
    parser.add_argument("--extra", action="store_true", help="run EXTRA_SETTINGS")
    args = parser.parse_args()
    if args.one:
        print("RESULT " + json.dumps(run_one(*json.loads(args.one))), flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for setting in EXTRA_SETTINGS if args.extra else SETTINGS:
        threads = str(setting[-1])
        env = {**os.environ, "OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads}
        done = subprocess.run(
            [sys.executable, "-I", "-X", "utf8", "-B", __file__, "--one", json.dumps(setting)],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        lines = [line for line in done.stdout.splitlines() if line.startswith("RESULT ")]
        if not lines:
            print(setting, "failed:", done.stderr[-800:], flush=True)
            rows.append({"setting": list(setting), "failed": done.stderr[-800:]})
        else:
            row = json.loads(lines[0][len("RESULT ") :])
            rows.append(row)
            print(row, flush=True)
        target = OUT / ("timing_extra.json" if args.extra else "timing.json")
        target.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
