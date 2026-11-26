"""Fine-tuning HuBERT's last two layers on /a/, inside HSA and against the matched draws.

Exploratory. The definitions, the gate and the readings are in experiments/deviations.md,
written before any fine-tuned score existed. Layers 0-9 run once into a cache; each training
step only runs layers 10 and 11 and the head. Nothing is downloaded: the weights are the
copy under models/.

    cache     layer-10 input of every /a/ file, in pieces of at most 30 s
    screen    arm (ii), repeat 0, both learning rates
    main      the other fits at the chosen learning rate
    shuffles  four more label shuffles and one repeated fit (added after the first shuffle
              gave 0.543; deviations.md, 2026-11-09)
    collect   gate, gaps and readings; writes results/finetune/summary.json

    .venv/Scripts/python.exe -I -X utf8 -B experiments/finetune_hubert.py cache
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable, Sequence  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from voice_severity_bench.evaluation import classification_metrics  # noqa: E402
from voice_severity_bench.permutation import permute_group_labels  # noqa: E402
from voice_severity_bench.splitting import repeated_split_indices  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "features"))

import run_first_batch as batch  # noqa: E402
from arms import write_json  # noqa: E402
from common import load_signal  # noqa: E402
from extract_hubert import HIDDEN, MODEL_DIR, RATE, REVISION, pieces_of  # noqa: E402
from freeze import verify_freeze  # noqa: E402
from freeze_svd import verify_svd_freeze  # noqa: E402
from harness import ROOT, Arm, load_plan  # noqa: E402

SPLIT_LAYER = 10  # hidden_states[10] is the input of layer 10
TRAINED_LAYERS = 2
STEPS = 300
BATCH = 8
WARMUP = 3  # 1% of 300, as in the Koudounas et al. 2024 detection runs
CROP_FRAMES = 500  # 10 s at 50 frames per second
RATES = (1e-5, 1e-4)
THREADS = 4
WORKERS = 2  # 8 threads in all
FIT_SEED = 20261110
DRAWS_USED = 5
II_REPEATS = (0, 1, 2)
EXTRA_SHUFFLES = (1, 2, 3, 4)  # seed offsets; added after the first shuffle, post hoc

# Fixed before any result; see deviations.md.
GATE_MARGIN = 0.05
SHUFFLED_RANGE = (0.47, 0.53)
SHRINK_UAR = 0.80
ANSWERED_UAR = 0.70
GAP_HOLDS = 0.10
GAP_GONE = 0.05
LEANS_MORE = 0.05

CACHE = ROOT / "features" / "out_finetune"
RESULTS = HERE / "results" / "finetune"
FITS = RESULTS / "fits"
FITS_RECHECK = RESULTS / "fits_recheck"
RECHECKED = "ii_r0_f0_lr1e-05"
CLASSES = ("normal", "pathological")

_STATE: dict = {}


# ------------------------------------------------------------------ the cache


def _load_model():
    import torch
    from transformers import HubertModel, Wav2Vec2FeatureExtractor

    torch.set_num_threads(THREADS)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_DIR, local_files_only=True)
    model = HubertModel.from_pretrained(MODEL_DIR, local_files_only=True).eval()
    return extractor, model


def _cache_one(job: tuple[str, str]) -> tuple[str, list[np.ndarray]]:
    import torch

    if "model" not in _STATE:
        _STATE["extractor"], _STATE["model"] = _load_model()
    file_id, wav = job
    x, rate = load_signal(wav)
    if rate != RATE:
        raise ValueError(f"{file_id}: HuBERT needs 16 kHz, got {rate}")
    pieces = []
    with torch.no_grad():
        for piece in pieces_of([x]):
            inputs = _STATE["extractor"](
                piece.astype(np.float32), sampling_rate=RATE, return_tensors="pt"
            ).input_values
            hidden = _STATE["model"](inputs, output_hidden_states=True).hidden_states
            pieces.append(hidden[SPLIT_LAYER][0].numpy().astype(np.float32))
    return file_id, pieces


def run_cache() -> None:
    speakers = pd.read_csv(ROOT / "s0_audit" / "out" / "avfad_speakers.csv", dtype={"speaker": str})
    jobs = [(f"avfad__{s}", path) for s, path in zip(speakers["speaker"], speakers["a_path"])]
    CACHE.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with ProcessPoolExecutor(WORKERS) as pool:
        done = list(pool.map(_cache_one, jobs, chunksize=4))
    rows, blocks, start = [], [], 0
    for file_id, pieces in sorted(done):
        for number, piece in enumerate(pieces):
            rows.append({"file_id": file_id, "piece": number, "start": start, "frames": len(piece)})
            blocks.append(piece)
            start += len(piece)
    frames = np.memmap(CACHE / "a_full_l10.f32", dtype=np.float32, mode="w+", shape=(start, HIDDEN))
    position = 0
    for block in blocks:
        frames[position : position + len(block)] = block
        position += len(block)
    frames.flush()
    pd.DataFrame(rows).to_csv(CACHE / "a_full_l10_index.csv", index=False, lineterminator="\n")
    write_json(
        CACHE / "a_full_l10.json",
        {"revision": REVISION, "split_layer": SPLIT_LAYER, "frames": start, "files": len(done)},
    )
    print(len(done), "files,", start, "frames in", round(time.perf_counter() - started), "s")


def cached_pieces() -> Callable[[str], list[np.ndarray]]:
    meta = json.loads((CACHE / "a_full_l10.json").read_text(encoding="utf-8"))
    frames = np.memmap(
        CACHE / "a_full_l10.f32", dtype=np.float32, mode="r", shape=(meta["frames"], HIDDEN)
    )
    index = pd.read_csv(CACHE / "a_full_l10_index.csv")
    spans = {
        file_id: list(zip(group["start"], group["frames"], strict=True))
        for file_id, group in index.groupby("file_id")
    }

    def pieces(file_id: str) -> list[np.ndarray]:
        return [frames[start : start + n] for start, n in spans[file_id]]

    return pieces


# ------------------------------------------------------------------ training


def through(layer, hidden):
    """One encoder layer; some transformers versions return a tuple, some the tensor."""
    output = layer(hidden)
    return output[0] if isinstance(output, tuple) else output


def linear_schedule(step: int, steps: int = STEPS, warmup: int = WARMUP) -> float:
    """HF's linear schedule with warmup: 0 at step 0, 1 after warmup, 0 at the end."""
    if step < warmup:
        return step / max(1, warmup)
    return max(0.0, (steps - step) / max(1, steps - warmup))


def class_weights(labels: np.ndarray) -> np.ndarray:
    """n / (2 n_c), the 'balanced' weights of the frozen detector."""
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    return labels.size / (2.0 * counts)


def crop_batch(
    chosen: Sequence[list[np.ndarray]], rng: np.random.Generator, crop: int = CROP_FRAMES
) -> np.ndarray:
    """One piece per item, picked by length, then the same number of frames from each.

    A piece shorter than ``crop`` sets the length for the whole batch, so nothing is padded.
    """
    picked = []
    for pieces in chosen:
        lengths = np.array([len(p) for p in pieces], dtype=np.float64)
        picked.append(pieces[int(rng.choice(len(pieces), p=lengths / lengths.sum()))])
    length = min(crop, *(len(p) for p in picked))
    out = []
    for piece in picked:
        start = int(rng.integers(0, len(piece) - length + 1))
        out.append(np.asarray(piece[start : start + length], dtype=np.float32))
    return np.stack(out)


def train_and_predict(
    layers,
    pieces_of: Callable[[str], list[np.ndarray]],
    train_items: Sequence[str],
    train_labels: np.ndarray,
    test_items: Sequence[str],
    lr: float,
    seed: int,
    steps: int = STEPS,
    batch_size: int = BATCH,
    warmup: int = WARMUP,
    crop: int = CROP_FRAMES,
) -> np.ndarray:
    """Train ``layers`` plus a linear head; returns P(pathological) of each test item."""
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    hidden_size = int(np.asarray(pieces_of(train_items[0])[0]).shape[1])
    head = torch.nn.Linear(hidden_size, 2)
    parameters = list(layers.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: linear_schedule(step, steps, warmup)
    )
    weights = torch.as_tensor(class_weights(train_labels), dtype=torch.float32)
    labels = torch.as_tensor(train_labels, dtype=torch.long)
    layers.train()
    head.train()
    order, position = rng.permutation(len(train_items)), 0
    for _ in range(steps):
        if position + batch_size > len(order):
            order, position = rng.permutation(len(train_items)), 0
        chosen = order[position : position + batch_size]
        position += batch_size
        hidden = torch.from_numpy(
            crop_batch([pieces_of(train_items[i]) for i in chosen], rng, crop)
        )
        for layer in layers:
            hidden = through(layer, hidden)
        loss = torch.nn.functional.cross_entropy(
            head(hidden.mean(dim=1)), labels[chosen], weight=weights
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    layers.eval()
    head.eval()
    scores = []
    with torch.no_grad():
        for item in test_items:
            outputs = []
            for piece in pieces_of(item):
                hidden = torch.from_numpy(np.asarray(piece, dtype=np.float32)[None])
                for layer in layers:
                    hidden = through(layer, hidden)
                outputs.append(hidden[0])
            pooled = torch.cat(outputs).mean(dim=0)
            scores.append(float(torch.softmax(head(pooled), dim=-1)[1]))
    return np.array(scores)


def _init_worker() -> None:
    import torch
    from transformers import HubertModel

    torch.set_num_threads(THREADS)
    model = HubertModel.from_pretrained(MODEL_DIR, local_files_only=True)
    kept = model.encoder.layers[SPLIT_LAYER : SPLIT_LAYER + TRAINED_LAYERS]
    _STATE["template"] = copy.deepcopy(torch.nn.ModuleList(kept))
    del model
    _STATE["pieces"] = cached_pieces()


def fit_name(arm: str, repeat: int, fold: int, lr: float) -> str:
    return f"{arm}_r{repeat}_f{fold}_lr{lr:.0e}"


def _fit(job: dict) -> dict:
    target = Path(job.get("folder", str(FITS))) / f"{job['name']}.json"
    if target.exists():
        return json.loads(target.read_text(encoding="utf-8"))
    started = time.perf_counter()
    layers = copy.deepcopy(_STATE["template"])
    scores = train_and_predict(
        layers,
        _STATE["pieces"],
        job["train_items"],
        np.asarray(job["train_labels"]),
        job["test_items"],
        job["lr"],
        job["seed"],
    )
    record = {
        "name": job["name"],
        "arm": job["arm"],
        "repeat": job["repeat"],
        "fold": job["fold"],
        "lr": job["lr"],
        "seed": job["seed"],
        "test_items": job["test_items"],
        "scores": scores.tolist(),
        "seconds": time.perf_counter() - started,
    }
    write_json(target, record)
    return record


# ------------------------------------------------------------------ arms and folds


def arm_tables() -> dict[str, pd.DataFrame]:
    """Each arm's speakers, sorted by file_id like the frozen runs, so the folds match."""
    speakers = batch.load_speakers()
    patients = speakers[(speakers["place"] == "HSA") & (speakers["label"] == "pathological")]
    hsa_normals = speakers[(speakers["place"] == "HSA") & (speakers["label"] == "normal")]
    draws = pd.read_csv(HERE / "results" / "first_batch" / "c1_draws.csv", dtype={"speaker": str})
    tables = {"ii": pd.concat([patients, hsa_normals]), "all": speakers}
    for index in range(DRAWS_USED):
        drawn = speakers[speakers["speaker"].isin(draws.loc[draws["draw"] == index, "speaker"])]
        tables[f"i_{index:02d}"] = pd.concat([patients, drawn])
    return {name: t.sort_values("file_id").reset_index(drop=True) for name, t in tables.items()}


def labels_of(table: pd.DataFrame, name: str, plan: dict) -> np.ndarray:
    """True labels, or the speaker-level shuffle of arm ``shuffled`` / ``shuffled_<k>``."""
    y = table["label"].to_numpy().astype(str)
    if not name.startswith("shuffled"):
        return y
    offset = 0 if name == "shuffled" else int(name.removeprefix("shuffled_"))
    rng = np.random.default_rng(plan["statistics"]["seeds"]["negative_control"] + offset)
    return np.asarray(permute_group_labels(y, table["speaker"].to_numpy(), None, rng)).astype(str)


def folds_of(y: np.ndarray, groups: np.ndarray, plan: dict) -> list:
    stats = plan["statistics"]
    return repeated_split_indices(
        y,
        groups,
        task="classification",
        n_splits=stats["cv"]["folds"],
        n_repeats=stats["cv"]["repeats"],
        seed=stats["seeds"]["cv"],
    )


def jobs_for(arm: str, table: pd.DataFrame, y: np.ndarray, repeats, lr, plan, arm_number) -> list:
    groups = table["speaker"].to_numpy().astype(str)
    splits = folds_of(y, groups, plan)
    items = table["file_id"].tolist()
    jobs = []
    for repeat in repeats:
        for fold, (train, test) in enumerate(splits[repeat]):
            jobs.append(
                {
                    "name": fit_name(arm, repeat, fold, lr),
                    "arm": arm,
                    "repeat": repeat,
                    "fold": fold,
                    "lr": lr,
                    "seed": FIT_SEED + 1000 * arm_number + 10 * repeat + fold,
                    "train_items": [items[i] for i in train],
                    "train_labels": [int(y[i] == "pathological") for i in train],
                    "test_items": [items[i] for i in test],
                }
            )
    return jobs


ARM_NUMBERS = {
    "ii": 0,
    "all": 1,
    "shuffled": 2,
    **{f"i_{k:02d}": 3 + k for k in range(10)},
    **{f"shuffled_{k}": 12 + k for k in EXTRA_SHUFFLES},
}


def run_jobs(jobs: list) -> list:
    FITS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records = []
    with ProcessPoolExecutor(WORKERS, initializer=_init_worker) as pool:
        for record in pool.map(_fit, jobs):
            records.append(record)
            print(
                record["name"],
                round(record["seconds"]),
                "s;",
                len(records),
                "of",
                len(jobs),
                "after",
                round(time.perf_counter() - started),
                "s",
                flush=True,
            )
    return records


def run_screen(plan: dict) -> None:
    table = arm_tables()["ii"]
    y = labels_of(table, "ii", plan)
    jobs = []
    for lr in RATES:
        jobs += jobs_for("ii", table, y, [0], lr, plan, ARM_NUMBERS["ii"])
    run_jobs(jobs)


def chosen_rate(plan: dict) -> tuple[float, dict]:
    table = arm_tables()["ii"]
    y = labels_of(table, "ii", plan)
    found = {}
    for lr in RATES:
        arm = arm_from_fits("ii", table, y, [0], lr)
        found[f"{lr:.0e}"] = arm.uar
    best = max(RATES, key=lambda lr: (round(found[f"{lr:.0e}"], 12), -lr))
    return best, found


def run_main(plan: dict) -> None:
    lr, _ = chosen_rate(plan)
    tables = arm_tables()
    jobs = []
    for name, table in tables.items():
        y = labels_of(table, name, plan)
        repeats = [r for r in II_REPEATS if r != 0] if name == "ii" else [0]
        jobs += jobs_for(name, table, y, repeats, lr, plan, ARM_NUMBERS[name])
    shuffled = tables["ii"]
    jobs += jobs_for(
        "shuffled",
        shuffled,
        labels_of(shuffled, "shuffled", plan),
        [0],
        lr,
        plan,
        ARM_NUMBERS["shuffled"],
    )
    print("learning rate", lr, "for", len(jobs), "fits", flush=True)
    run_jobs(jobs)


def run_shuffles(plan: dict) -> None:
    lr, _ = chosen_rate(plan)
    table = arm_tables()["ii"]
    FITS_RECHECK.mkdir(parents=True, exist_ok=True)
    again = [
        dict(job, folder=str(FITS_RECHECK))
        for job in jobs_for("ii", table, labels_of(table, "ii", plan), [0], 1e-5, plan, 0)
        if job["name"] == RECHECKED
    ]
    jobs = list(again)
    for k in EXTRA_SHUFFLES:
        name = f"shuffled_{k}"
        jobs += jobs_for(
            name, table, labels_of(table, name, plan), [0], lr, plan, ARM_NUMBERS[name]
        )
    print("learning rate", lr, "for", len(jobs), "fits", flush=True)
    run_jobs(jobs)


def recheck() -> dict | None:
    """The repeated fit against the stored one; None until it exists."""
    again = FITS_RECHECK / f"{RECHECKED}.json"
    if not again.exists():
        return None
    first = json.loads((FITS / f"{RECHECKED}.json").read_text(encoding="utf-8"))
    second = json.loads(again.read_text(encoding="utf-8"))
    a, b = np.asarray(first["scores"]), np.asarray(second["scores"])
    return {
        "fit": RECHECKED,
        "same_items": first["test_items"] == second["test_items"],
        "max_abs_difference": float(np.max(np.abs(a - b))),
        "same_decisions": bool(np.array_equal(a >= 0.5, b >= 0.5)),
    }


def shuffled_mean_passes(uars: Sequence[float]) -> bool:
    return bool(SHUFFLED_RANGE[0] <= float(np.mean(uars)) <= SHUFFLED_RANGE[1])


# ------------------------------------------------------------------ reading


def scored_arm(name: str, y: np.ndarray, groups: np.ndarray, scores: np.ndarray) -> Arm:
    """An Arm from a (speakers, repeats) matrix of P(pathological); UAR is the repeat mean."""
    predictions = np.where(scores >= 0.5, "pathological", "normal")
    per_repeat = [
        classification_metrics(
            y,
            predictions[:, r],
            np.column_stack([1 - scores[:, r], scores[:, r]]),
            classes=CLASSES,
            positive_class="pathological",
        )
        for r in range(scores.shape[1])
    ]
    mean = {k: float(np.mean([m[k] for m in per_repeat])) for k in per_repeat[0]}
    return Arm(
        name=name,
        y_true=y,
        groups=groups,
        predictions=predictions,
        scores=scores,
        uar=mean["uar"],
        roc_auc=mean["roc_auc"],
        sensitivity=mean["sensitivity"],
        specificity=mean["specificity"],
        worst_share_deviation=float("nan"),
        chosen_c={},
    )


def arm_from_fits(name: str, table: pd.DataFrame, y: np.ndarray, repeats, lr) -> Arm:
    position = {item: i for i, item in enumerate(table["file_id"])}
    scores = np.full((len(table), len(repeats)), np.nan)
    for column, repeat in enumerate(repeats):
        for fold in range(5):
            record = json.loads(
                (FITS / f"{fit_name(name, repeat, fold, lr)}.json").read_text(encoding="utf-8")
            )
            for item, score in zip(record["test_items"], record["scores"], strict=True):
                scores[position[item], column] = score
    if np.isnan(scores).any():
        raise ValueError(f"{name}: some speakers have no fine-tuned score")
    return scored_arm(name, y, table["speaker"].to_numpy().astype(str), scores)


def frozen_arm(name: str, table: pd.DataFrame, repeats) -> Arm:
    """The frozen HuBERT arm of the same speakers and repeats, from its saved predictions."""
    saved = pd.read_csv(HERE / "results" / "sensitivity" / "hubert" / "c1_predictions.csv.gz")
    saved = saved[(saved["arm"] == name) & (saved["repeat"].isin(repeats))]
    wide = saved.pivot(index="speaker", columns="repeat", values="score").loc[table["speaker"]]
    return scored_arm(
        f"frozen_{name}",
        table["label"].to_numpy().astype(str),
        table["speaker"].to_numpy().astype(str),
        wide[list(repeats)].to_numpy(),
    )


def gate(all_uar: float, frozen_all: float, shuffled_uar: float) -> dict:
    return {
        "all_passes": bool(all_uar >= frozen_all - GATE_MARGIN),
        "shuffled_passes": bool(SHUFFLED_RANGE[0] <= shuffled_uar <= SHUFFLED_RANGE[1]),
    }


def reading(ii: float, best_screen_ii: float, gap: float, gap_low: float) -> str:
    if ii >= SHRINK_UAR or gap < GAP_GONE or gap_low <= 0:
        return "shrinks"
    if max(ii, best_screen_ii) < ANSWERED_UAR and gap >= GAP_HOLDS:
        return "objection answered for the last two layers"
    return "middle"


def leans_more_on_place(i_ft: float, i_frozen: float, ii_ft: float, ii_frozen: float) -> bool:
    return bool(i_ft - i_frozen >= LEANS_MORE and abs(ii_ft - ii_frozen) < LEANS_MORE)


def contrast(first: list[Arm], second: Arm, seed: int, n: int) -> dict:
    test = batch.c1_bootstrap(first, second, n, seed)
    test.pop("samples")
    return test


def summary_of(arm: Arm) -> dict:
    return {
        "speakers": int(arm.y_true.size),
        "repeats": int(arm.predictions.shape[1]),
        "uar": arm.uar,
        "roc_auc": arm.roc_auc,
        "sensitivity": arm.sensitivity,
        "specificity": arm.specificity,
    }


def run_collect(plan: dict) -> None:
    lr, screen = chosen_rate(plan)
    tables = arm_tables()
    seed = plan["statistics"]["seeds"]["bootstrap"]
    n = plan["confirmatory_family"]["tests"]["C1"]["test"]["resamples"]
    ii = arm_from_fits("ii", tables["ii"], labels_of(tables["ii"], "ii", plan), II_REPEATS, lr)
    arms_i = [
        arm_from_fits(name, tables[name], labels_of(tables[name], name, plan), [0], lr)
        for name in tables
        if name.startswith("i_")
    ]
    every = arm_from_fits("all", tables["all"], labels_of(tables["all"], "all", plan), [0], lr)
    shuffled = arm_from_fits(
        "shuffled", tables["ii"], labels_of(tables["ii"], "shuffled", plan), [0], lr
    )
    frozen_ii = frozen_arm("ii", tables["ii"], II_REPEATS)
    frozen_i = [frozen_arm(name, tables[name], [0]) for name in tables if name.startswith("i_")]
    frozen_all = json.loads(
        (HERE / "results" / "exploratory" / "avfad_all.json").read_text(encoding="utf-8")
    )["all_hubert"]["uar"]
    gap = contrast(arms_i, ii, seed, n)
    i_mean = float(np.mean([a.uar for a in arms_i]))
    frozen_i_mean = float(np.mean([a.uar for a in frozen_i]))
    report = {
        "learning_rate": lr,
        "screen_uar_ii_repeat0": screen,
        "ii": summary_of(ii),
        "i": {
            "draws": len(arms_i),
            "uar_mean": i_mean,
            "uar_min": float(min(a.uar for a in arms_i)),
            "uar_max": float(max(a.uar for a in arms_i)),
        },
        "all": summary_of(every),
        "shuffled": summary_of(shuffled),
        "frozen": {
            "ii_same_repeats": summary_of(frozen_ii),
            "i_same_draws_mean": frozen_i_mean,
            "all": frozen_all,
        },
        "gap_i_minus_ii": gap,
        "ii_fine_tuned_minus_frozen": contrast([ii], frozen_ii, seed, n),
        "gate": gate(every.uar, frozen_all, shuffled.uar),
        "fit_seconds": [
            json.loads(p.read_text(encoding="utf-8"))["seconds"]
            for p in sorted(FITS.glob("*.json"))
        ],
    }
    report["reading"] = reading(ii.uar, max(screen.values()), gap["difference"], gap["ci_low"])
    report["leans_more_on_place"] = leans_more_on_place(
        i_mean, frozen_i_mean, ii.uar, frozen_ii.uar
    )
    report["readable"] = all(report["gate"].values())
    extra = [
        arm_from_fits(name, tables["ii"], labels_of(tables["ii"], name, plan), [0], lr)
        for name in (f"shuffled_{k}" for k in EXTRA_SHUFFLES)
        if all((FITS / f"{fit_name(name, 0, f, lr)}.json").exists() for f in range(5))
    ]
    if len(extra) == len(EXTRA_SHUFFLES):
        uars = [shuffled.uar] + [a.uar for a in extra]
        report["shuffled_five"] = {
            "uar": uars,
            "roc_auc": [shuffled.roc_auc] + [a.roc_auc for a in extra],
            "uar_mean": float(np.mean(uars)),
        }
        report["gate_post_hoc"] = {
            "all_passes": report["gate"]["all_passes"],
            "shuffled_mean_passes": shuffled_mean_passes(uars),
        }
        report["readable_post_hoc"] = all(report["gate_post_hoc"].values())
    report["recheck"] = recheck()
    write_json(RESULTS / "summary.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "fit_seconds"}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("cache", "screen", "main", "shuffles", "collect"))
    args = parser.parse_args()
    verify_freeze()
    verify_svd_freeze()
    RESULTS.mkdir(parents=True, exist_ok=True)
    plan = load_plan()
    if args.step == "cache":
        run_cache()
    elif args.step == "screen":
        run_screen(plan)
    elif args.step == "main":
        run_main(plan)
    elif args.step == "shuffles":
        run_shuffles(plan)
    else:
        run_collect(plan)


if __name__ == "__main__":
    main()
