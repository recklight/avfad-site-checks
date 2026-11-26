"""C2 again with the phonation-to-noise ratio of ESSUA brought to HSA's; exploratory.

Definitions and readings are in experiments/deviations.md, written before this ran.

    noisy     paper venv: writes the noisy recordings and one index per condition and seed
    (segment) Kudio venv: ../Kudio/.venv/Scripts/python.exe -I -B s1_segments/segment.py
              --index data/derived_noise/<set>/index.csv --out data/derived_noise/<set>/s1
    features  paper venv: handcrafted features and HuBERT on the new spans
    run       paper venv: the arms, the paired intervals and the checks

Recordings stay under data/, outside every repository.
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.io import wavfile  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import exploratory_avfad as ex  # noqa: E402
import run_first_batch as batch  # noqa: E402
from voice_severity_bench.bootstrap import paired_bootstrap  # noqa: E402
from arms import evaluate_columns, evaluate_layers, merge_features, write_json  # noqa: E402
from common import load_signal  # noqa: E402
from extract_handcrafted import FEATURE_COLUMNS, extract_file  # noqa: E402
from freeze import verify_freeze  # noqa: E402
from harness import ROOT, load_plan, make_config, predictions_table  # noqa: E402
from joblib import Parallel, delayed, parallel_config  # noqa: E402
from sensitivity_avfad import CHOSEN_LAYERS, shuffled_uar  # noqa: E402

DERIVED = ROOT / "data" / "derived_noise"
HUBERT_OUT = ROOT / "features" / "out_hubert"
RESULTS = HERE / "results" / "exploratory" / "redo_c2"
C2_SPEAKERS = HERE / "results" / "first_batch" / "c2_speakers.csv"
RATE = 16_000
REDO_SEED = 20261203
MIN_QUIET_S = 1.0
D_EXTRA_NOISE_DB = 6.0
N_SHUFFLES = 5


@dataclass(frozen=True)
class Condition:
    name: str
    who: tuple[str, ...]  # places that get noise
    room: str
    extra_noise_db: float
    seeds: tuple[int, ...]


CONDITIONS = (
    Condition("B", ("ESSUA",), "HSA", 0.0, (0, 1, 2)),
    Condition("C", ("ESSUA",), "ESSUA", 0.0, (0,)),
    Condition("D", ("ESSUA", "HSA"), "HSA", D_EXTRA_NOISE_DB, (0,)),
)


def set_name(condition: str, seed: int) -> str:
    return f"{condition}_s{seed}"


def added_noise_gain_db(phonation_db: float, target_snr_db: float, noise_level_db: float) -> float:
    """Gain that puts the noise ``target_snr_db`` below this speaker's phonation."""
    return float(phonation_db - target_snr_db - noise_level_db)


def needs_noise(own_snr_db: float, target_snr_db: float) -> bool:
    """A speaker already noisier than the target stays as recorded; noise cannot raise an SNR."""
    return bool(np.isnan(own_snr_db) or own_snr_db > target_snr_db)


def speaker_table() -> pd.DataFrame:
    """The 158 speakers of C2, with phonation level, quiet level and their SNR."""
    speakers = batch.load_speakers()
    chosen = pd.read_csv(C2_SPEAKERS, dtype={"speaker": str})
    table = speakers[speakers["speaker"].isin(chosen["speaker"])].copy()
    phonation = batch.load_features("phonation_norm")[["file_id", "phonation_rms_dbfs"]]
    quiet = batch.load_features("nonphon_g100")[["file_id", "level_mean", "kept_s"]]
    table = table.merge(phonation, on="file_id", validate="one_to_one")
    table = table.merge(quiet, on="file_id", how="left", validate="one_to_one")
    enough = table["kept_s"] >= MIN_QUIET_S
    table["own_snr"] = (table["phonation_rms_dbfs"] - table["level_mean"]).where(enough)
    return table.sort_values("file_id").reset_index(drop=True)


def hsa_snr_pool() -> np.ndarray:
    """SNRs of every HSA normal speaker who reaches 1.0 s of quiet."""
    speakers = batch.load_speakers()
    hsa = speakers[(speakers["place"] == "HSA") & (speakers["label"] == "normal")]
    phonation = batch.load_features("phonation_norm")[["file_id", "phonation_rms_dbfs"]]
    quiet = batch.load_features("nonphon_g100")[["file_id", "level_mean", "kept_s"]]
    merged = hsa.merge(phonation, on="file_id").merge(quiet, on="file_id")
    merged = merged[merged["kept_s"] >= MIN_QUIET_S]
    return (merged["phonation_rms_dbfs"] - merged["level_mean"]).to_numpy(dtype=float)


def _write_noisy(job: dict) -> dict:
    x, rate = load_signal(job["wav"])
    if rate != RATE:
        raise ValueError(f"{job['file_id']}: expected {RATE} Hz, got {rate}")
    pieces, level = ex.room_quiet(*job["room"])
    total = int(sum(p.size for p in pieces))
    rng = np.random.default_rng(job["rng_key"])
    start = int(rng.integers(total))
    added = False
    if job["add"]:
        gain = added_noise_gain_db(job["phonation_db"], job["target_snr"], level)
        noise = ex.loop_to_length(pieces, x.size, start, rate) * 10.0 ** (gain / 20.0)
        x = x + noise
        added = True
    target = ROOT / job["out_wav"]
    target.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(target, rate, x.astype(np.float32))
    return {
        "file_id": job["file_id"],
        "database": "avfad",
        "material": "a",
        "wav": job["out_wav"],
        "noise_added": added,
        "target_snr": job["target_snr"],
        "room_file": job["room"][0],
    }


def run_noisy(jobs: int) -> None:
    table = speaker_table()
    pool = hsa_snr_pool()
    files = pd.read_csv(ex.S1_OUT / "files.csv").set_index("file_id")
    spans = pd.read_csv(ex.S1_OUT / "spans.csv")
    quiet = spans[spans["kind"] == "nonphonation_g100"]
    for condition in CONDITIONS:
        rooms = []
        for row in ex.noise_files(condition.room).itertuples():
            part = quiet[quiet["file_id"] == row.file_id]
            rooms.append(
                (row.file_id, row.wav, list(zip(part["start_s"], part["end_s"], strict=True)))
            )
        for seed in condition.seeds:
            name = set_name(condition.name, seed)
            chosen = table[table["place"].isin(condition.who)]
            tag = ord(condition.name)
            rng = np.random.default_rng([REDO_SEED, tag, seed])
            draws = rng.choice(pool, size=len(chosen), replace=True)
            picks = rng.integers(len(rooms), size=len(chosen))
            todo = []
            for i, row in enumerate(chosen.itertuples()):
                # Under D an HSA speaker keeps their own SNR, 6 dB lower; nobody else is drawn.
                own = row.own_snr
                base = own if row.place == "HSA" and not np.isnan(own) else draws[i]
                target = float(base - condition.extra_noise_db)
                todo.append(
                    {
                        "file_id": row.file_id,
                        "wav": files.loc[row.file_id, "wav"],
                        "room": rooms[picks[i]],
                        "phonation_db": float(row.phonation_rms_dbfs),
                        "target_snr": target,
                        "add": needs_noise(own, target),
                        "rng_key": [REDO_SEED, tag, seed, i],
                        "out_wav": f"data/derived_noise/{name}/{row.file_id}.wav",
                    }
                )
            with ProcessPoolExecutor(jobs) as executor:
                rows = list(executor.map(_write_noisy, todo, chunksize=4))
            index = pd.DataFrame(rows)
            folder = DERIVED / name
            index[["file_id", "database", "material", "wav"]].to_csv(
                folder / "index.csv", index=False, lineterminator="\n"
            )
            index.to_csv(folder / "noise_record.csv", index=False, lineterminator="\n")
            print(
                name, len(index), "files,", int(index["noise_added"].sum()), "got noise", flush=True
            )


def span_lookup(spans: pd.DataFrame) -> dict[str, dict[str, list[tuple[float, float]]]]:
    grouped: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for (file_id, kind), part in spans.groupby(["file_id", "kind"], sort=False):
        grouped.setdefault(file_id, {})[kind] = list(
            zip(part["start_s"].astype(float), part["end_s"].astype(float), strict=True)
        )
    return grouped


def handcrafted_rows(index, files, spans, label: str, jobs: int) -> dict[str, pd.DataFrame]:
    lookup = span_lookup(spans)
    conditions = ["phonation_norm", "nonphon_g100"]
    todo = [
        (
            row.file_id,
            row.wav,
            float(files.loc[row.file_id, "duration_s"]),
            lookup.get(row.file_id, {}),
            conditions,
        )
        for row in index.itertuples()
    ]
    with ProcessPoolExecutor(jobs) as executor:
        outputs = list(executor.map(extract_file, todo, chunksize=4))
    tables = {}
    for condition in conditions:
        rows = []
        for job, output in zip(todo, outputs, strict=True):
            values, info = output[condition]
            if values is not None:
                rows.append(
                    {"file_id": job[0], **info, **dict(zip(FEATURE_COLUMNS, values, strict=True))}
                )
        tables[condition] = pd.DataFrame(rows)
        path = DERIVED / label / f"{condition}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        tables[condition].to_csv(path, index=False, lineterminator="\n")
    return tables


def hubert_table(index, files, spans, label: str, jobs: int) -> None:
    from extract_hubert import OUT, assemble, extract_one

    lookup = span_lookup(spans[spans["kind"] == "phonation"])
    condition = f"redo_{label}"
    (OUT / "cache" / condition).mkdir(parents=True, exist_ok=True)
    todo = [
        (
            condition,
            row.file_id,
            row.wav,
            float(files.loc[row.file_id, "duration_s"]),
            lookup.get(row.file_id, {}).get("phonation", []),
        )
        for row in index.itertuples()
    ]
    with ProcessPoolExecutor(jobs) as executor:
        list(executor.map(extract_one, todo, chunksize=2))
    assemble("avfad", "a", condition, todo)


def run_features(jobs: int) -> None:
    original_spans = pd.read_csv(ex.S1_OUT / "spans.csv")
    for condition in CONDITIONS:
        for seed in condition.seeds:
            name = set_name(condition.name, seed)
            folder = DERIVED / name
            index = pd.read_csv(folder / "index.csv")
            files = pd.read_csv(folder / "s1" / "files.csv").set_index("file_id")
            bad = files[files["status"] != "ok"]
            if len(bad):
                raise RuntimeError(f"{name}: segmenter failed on {bad.index.tolist()[:5]}")
            spans = pd.read_csv(folder / "s1" / "spans.csv")
            handcrafted_rows(index, files, spans, name, jobs)
            hubert_table(index, files, spans, name, jobs)
            print(name, "features done", flush=True)
    # Control: the first B set cut with the original spans, not the new ones.
    name = set_name("B", 0)
    folder = DERIVED / name
    index = pd.read_csv(folder / "index.csv")
    files = pd.read_csv(folder / "s1" / "files.csv").set_index("file_id")
    original = original_spans[original_spans["file_id"].isin(index["file_id"])]
    handcrafted_rows(index, files, original, f"{name}_original_spans", jobs)
    hubert_table(index, files, original, f"{name}_original_spans", jobs)
    print("original spans control done", flush=True)


def replaced(base: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """The table of a condition: new rows where a speaker changed, original rows elsewhere."""
    kept = base[~base["file_id"].isin(new["file_id"])]
    return pd.concat([kept, new[base.columns]], ignore_index=True)


def hubert_features(label: str | None) -> pd.DataFrame:
    """Phonation HuBERT columns of a condition, or the original when ``label`` is None."""
    base = ex.load_hubert("phonation", CHOSEN_LAYERS)
    if label is None:
        return base
    new = ex.load_hubert(f"redo_{label}", CHOSEN_LAYERS)
    return replaced(base, new)


def handcrafted_features(label: str | None) -> pd.DataFrame:
    base = batch.load_features("phonation_norm")
    if label is None:
        return base
    return replaced(base, pd.read_csv(DERIVED / label / "phonation_norm.csv"))


def snr_check(table: pd.DataFrame, label: str | None) -> dict:
    """Medians of phonation minus quiet by place, on the spans of this condition."""
    quiet = batch.load_features("nonphon_g100")
    phonation = batch.load_features("phonation_norm")
    if label is not None:
        quiet = replaced(quiet, pd.read_csv(DERIVED / label / "nonphon_g100.csv"))
        phonation = replaced(phonation, pd.read_csv(DERIVED / label / "phonation_norm.csv"))
    merged = table[["file_id", "place"]].merge(
        phonation[["file_id", "phonation_rms_dbfs"]], on="file_id"
    )
    merged = merged.merge(quiet[["file_id", "level_mean", "kept_s"]], on="file_id", how="left")
    merged["snr"] = (merged["phonation_rms_dbfs"] - merged["level_mean"]).where(
        merged["kept_s"] >= MIN_QUIET_S
    )
    medians = merged.groupby("place")["snr"].median()
    return {
        "median_snr": {str(k): float(v) for k, v in medians.items()},
        "speakers_with_1s": {
            str(k): int(v) for k, v in merged.groupby("place")["snr"].count().items()
        },
        "difference": float(medians["ESSUA"] - medians["HSA"]),
    }


def _handcrafted_arm(name, meta, features, target, positive, plan):
    table = merge_features(meta, features[["file_id", *FEATURE_COLUMNS]], target)
    return evaluate_columns(name, table, FEATURE_COLUMNS, make_config(plan, positive))


def _hubert_arm(name, meta, features, target, positive, plan):
    table = merge_features(meta, features, target)
    return evaluate_layers(name, table, CHOSEN_LAYERS, make_config(plan, positive))


def _shuffle(meta, features, plan, seed):
    table = merge_features(meta, features, "place")
    return shuffled_uar(table, CHOSEN_LAYERS, make_config(plan, batch.PLACE_POSITIVE), seed)


def paired_row(arm, reference, seed: int) -> dict:
    """UAR of a condition against A on the same speakers, with a paired interval."""
    if not np.array_equal(arm.groups, reference.groups):
        raise ValueError(f"{arm.name}: the speakers differ from the reference")
    interval = paired_bootstrap(
        reference.y_true,
        reference.predictions,
        arm.predictions,
        reference.groups,
        metric="uar",
        n_resamples=10_000,
        seed=seed,
    )
    return {
        "arm": arm.name,
        "uar": arm.uar,
        "roc_auc": arm.roc_auc,
        "drop_from_a": interval.estimate,
        "drop_ci": [interval.low, interval.high],
    }


def arm_meta(table: pd.DataFrame) -> pd.DataFrame:
    """Who the speakers are, without the level columns, which would clash with the features."""
    older = table["age"] >= table["age"].median()
    kept = table[["file_id", "speaker", "place", "label", "age", "sex"]].copy()
    kept["age_group"] = np.where(older, "older", "younger")
    return kept


def run_arms(plan, jobs) -> None:
    seeds = plan["statistics"]["seeds"]
    meta = arm_meta(speaker_table())
    labels: list[str | None] = [None]
    for condition in CONDITIONS:
        labels += [set_name(condition.name, seed) for seed in condition.seeds]
    labels.append(f"{set_name('B', 0)}_original_spans")

    def name_of(label):
        return "A" if label is None else label

    handcrafted = {label: handcrafted_features(label) for label in labels}
    hubert = {label: hubert_features(label) for label in labels}
    tasks = []
    for label in labels:
        tasks.append(("hand", label, "place", batch.PLACE_POSITIVE))
        tasks.append(("hubert", label, "place", batch.PLACE_POSITIVE))
    first_b = set_name("B", 0)
    for label in (None, first_b):
        for target, positive in (("sex", "F"), ("age_group", "older")):
            tasks.append(("hand", label, target, positive))
            tasks.append(("hubert", label, target, positive))

    def job(kind, label, target, positive):
        arm_name = f"{kind}_{name_of(label)}_{target}"
        if kind == "hand":
            return delayed(_handcrafted_arm)(
                arm_name, meta, handcrafted[label], target, positive, plan
            )
        return delayed(_hubert_arm)(arm_name, meta, hubert[label], target, positive, plan)

    arms = Parallel(n_jobs=jobs)(job(*task) for task in tasks)
    by_key = {task: arm for task, arm in zip(tasks, arms, strict=True)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    predictions_table(arms).to_csv(RESULTS / "predictions.csv.gz", index=False)
    report: dict = {"manipulation_check": {}, "place": {}, "masking": {}}
    for label in labels:
        report["manipulation_check"][name_of(label)] = snr_check(meta, label)
    for kind in ("hand", "hubert"):
        reference = by_key[(kind, None, "place", batch.PLACE_POSITIVE)]
        report["place"][kind] = [
            paired_row(
                by_key[(kind, label, "place", batch.PLACE_POSITIVE)], reference, seeds["bootstrap"]
            )
            for label in labels
        ]
        report["masking"][kind] = {}
        for target, positive in (("sex", "F"), ("age_group", "older")):
            a = by_key[(kind, None, target, positive)]
            b = by_key[(kind, first_b, target, positive)]
            report["masking"][kind][target] = {
                "uar_a": a.uar,
                "uar_b": b.uar,
                **paired_row(b, a, seeds["bootstrap"]),
            }
    shuffled = Parallel(n_jobs=jobs)(
        delayed(_shuffle)(meta, hubert[first_b], plan, seeds["negative_control"] + k)
        for k in range(N_SHUFFLES)
    )
    report["shuffled_hubert_b"] = {"uar": shuffled, "mean": float(np.mean(shuffled))}
    write_json(RESULTS / "summary.json", report)
    print(json.dumps(report, indent=1), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=["noisy", "features", "run"])
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()
    verify_freeze()
    sys.path.insert(0, str(ROOT / "s0_audit"))
    from avfad_extract import require_access

    require_access()
    started = time.perf_counter()
    if args.what == "noisy":
        run_noisy(args.jobs)
    elif args.what == "features":
        run_features(args.jobs)
    else:
        with parallel_config(backend="loky", inner_max_num_threads=1):
            run_arms(load_plan(), args.jobs)
    print(f"{args.what} done in {time.perf_counter() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
