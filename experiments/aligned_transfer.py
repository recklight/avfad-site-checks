"""Mixed-place training with the ESSUA speakers moved onto the HSA normals first, scored on SVD.

Exploratory. Definitions and readings are in experiments/deviations.md (2026-11-10), written
before this run. Training sets, features, detectors and the SVD a_n test set are those of
cross_database.py; the alignment comes from all 356 AVFAD normals and moves only the ESSUA rows of
the training set. SVD is never moved.

    check     refit the unaligned all and (ii) and compare with the stored scores
    run       --detector hubert|handcrafted --kind location|scale|coral
    collect   writes results/aligned_transfer/summary.json

    .venv/Scripts/python.exe -I -X utf8 -B experiments/aligned_transfer.py run --detector hubert \
        --kind location --jobs 12
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
from joblib import Parallel, delayed  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "features"))

import coral_c1 as cc  # noqa: E402
import cross_database as cd  # noqa: E402
import run_first_batch as batch  # noqa: E402
import run_svd as svd  # noqa: E402
import site_erasure as se  # noqa: E402
from arms import write_json  # noqa: E402
from freeze import verify_freeze  # noqa: E402
from harness import load_plan, make_config  # noqa: E402

RESULTS = HERE / "results" / "aligned_transfer"
DETECTORS = ("hubert", "handcrafted")
KINDS = ("location", "scale", "coral")
FIXED = ("all", "essua_normals_hsa_patients")
# Readings, written before the run (deviations.md, 2026-11-10).
POINTS = 0.03


def groups_of(columns: list[str], blocks: dict | None) -> list[np.ndarray]:
    """Column indices aligned on their own: one HuBERT layer each, or all handcrafted columns."""
    if blocks is None:
        return [np.arange(len(columns))]
    index = {c: i for i, c in enumerate(columns)}
    return [np.array([index[c] for c in block]) for block in blocks.values()]


def moved_training(
    train: pd.DataFrame,
    normals: pd.DataFrame,
    columns: list[str],
    groups: list[np.ndarray],
    kind: str,
) -> pd.DataFrame:
    """ESSUA rows of ``train`` moved onto the HSA normals; every other row as it is."""
    essua = (normals["place"] == "ESSUA").to_numpy()
    reference = normals[columns].to_numpy(dtype=float)
    rows = (train["place"] == "ESSUA").to_numpy()
    values = train[columns].to_numpy(dtype=float)
    for idx in groups:
        block = reference[:, idx]
        values[np.ix_(rows, idx)] = cc.moved_by(
            block[essua], block[~essua], kind, values[np.ix_(rows, idx)]
        )
    out = train.copy()
    out[columns] = values
    return out


def inputs(detector: str):
    config = make_config(load_plan(), batch.LABEL_POSITIVE)
    avfad, svd_features, columns, blocks = cd.feature_tables(detector)
    test = cd.svd_test_set().merge(svd_features, on="file_id", how="left", validate="one_to_one")
    normals = se.normal_speakers().merge(avfad, on="file_id", how="left", validate="one_to_one")
    for table in (test, normals):
        if table[columns].isna().any().any():
            raise ValueError("a speaker has no features")
    return config, avfad, test, normals, columns, blocks


def _score(detector: str, kind: str | None, name: str) -> tuple[str, np.ndarray, float]:
    start = time.perf_counter()
    config, avfad, test, normals, columns, blocks = inputs(detector)
    members = cd.training_sets()[name]
    train = members.merge(avfad, on="file_id", how="left", validate="one_to_one")
    if train[columns].isna().any().any():
        raise ValueError(f"{name}: an AVFAD speaker has no features")
    if kind is not None:
        train = moved_training(train, normals, columns, groups_of(columns, blocks), kind)
    scores = cd.score_on_svd(train, test, columns, blocks, config)
    return name, scores, time.perf_counter() - start


def stored(detector: str) -> pd.DataFrame:
    return pd.read_csv(cd.RESULTS / f"scores_{detector}.csv.gz")


def check(jobs: int, detector: str) -> None:
    done = Parallel(n_jobs=jobs)(delayed(_score)(detector, None, n) for n in ("all", "ii"))
    old = stored(detector)
    for name, scores, seconds in done:
        diff = float(np.nanmax(np.abs(scores - old[name].to_numpy())))
        print(f"{detector} {name}: max score difference {diff:.3g} ({seconds:.0f} s)")


def path_of(detector: str, kind: str) -> Path:
    return RESULTS / f"aligned_scores_{detector}_{kind}.csv.gz"


def run(jobs: int, detector: str, kind: str) -> None:
    verify_freeze()
    names = [n for n in cd.training_sets() if n != "ii"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    done = Parallel(n_jobs=jobs, return_as="generator_unordered")(
        delayed(_score)(detector, kind, n) for n in names
    )
    frame = cd.svd_test_set()[["file_id", "speaker", "label", "age", "sex", "year"]].copy()
    got = {}
    for count, (name, scores, seconds) in enumerate(done, start=1):
        got[name] = scores
        if count % 10 == 0 or count == len(names):
            print(
                f"{count} of {len(names)} after {time.perf_counter() - start:.0f} s "
                f"(last {seconds:.0f} s)",
                flush=True,
            )
    for name in names:
        frame[name] = got[name]
    frame.to_csv(path_of(detector, kind), index=False)


def auc_with_samples(scores: np.ndarray, sick: np.ndarray, resamples: list[np.ndarray]):
    point = cd.auc(scores, sick)
    samples = np.array([cd.auc(scores[idx], sick[idx]) for idx in resamples])
    return point, samples


def draws_auc(frame: pd.DataFrame, sick: np.ndarray, resamples: list[np.ndarray]):
    draws = sorted(c for c in frame.columns if c.startswith("i_"))
    both = [auc_with_samples(frame[d].to_numpy(), sick, resamples) for d in draws]
    return float(np.mean([b[0] for b in both])), np.mean([b[1] for b in both], axis=0)


def difference(first, second) -> dict:
    diff = first[1] - second[1]
    low, high = np.quantile(diff, [0.025, 0.975])
    return {"difference": float(first[0] - second[0]), "ci_low": float(low), "ci_high": float(high)}


def change_reading(change: dict) -> str:
    if change["difference"] >= POINTS and change["ci_low"] > 0:
        return "the alignment makes mixed training better outside"
    if change["difference"] <= -POINTS and change["ci_high"] < 0:
        return "the alignment makes it worse"
    if change["ci_low"] <= 0 <= change["ci_high"]:
        return "the alignment does not change the outside score"
    return "as measured"


def versus_ii_reading(change: dict) -> str:
    if change["ci_low"] <= 0 <= change["ci_high"]:
        return "catches up with training inside HSA"
    if change["ci_high"] < 0:
        return "still below training inside HSA"
    return "above training inside HSA"


def collect() -> None:
    summary: dict = {}
    for detector in DETECTORS:
        plain = stored(detector)
        sick = (plain["label"] == "pathological").to_numpy()
        resamples = cd.resample_indices(sick, svd.stratum(plain), cd.N_RESAMPLES, cd.SEED)
        base = {n: auc_with_samples(plain[n].to_numpy(), sick, resamples) for n in ("ii", *FIXED)}
        base["i"] = draws_auc(plain, sick, resamples)
        part: dict = {"plain": {n: v[0] for n, v in base.items()}}
        for kind in KINDS:
            if not path_of(detector, kind).exists():
                continue
            moved = pd.read_csv(path_of(detector, kind))
            if not moved["file_id"].equals(plain["file_id"]):
                raise RuntimeError("the SVD rows are not in the stored order")
            got = {n: auc_with_samples(moved[n].to_numpy(), sick, resamples) for n in FIXED}
            got["i"] = draws_auc(moved, sick, resamples)
            all_change = difference(got["all"], base["all"])
            all_vs_ii = difference(got["all"], base["ii"])
            en_change = difference(
                got["essua_normals_hsa_patients"], base["essua_normals_hsa_patients"]
            )
            d_ext = difference(got["i"], base["ii"])
            part[kind] = {
                "auc": {n: v[0] for n, v in got.items()},
                "all_minus_plain_all": all_change,
                "all_reading": change_reading(all_change),
                "all_minus_ii": all_vs_ii,
                "versus_ii_reading": versus_ii_reading(all_vs_ii),
                "essua_normals_minus_plain": en_change,
                "essua_normals_reading": change_reading(en_change),
                "d_ext": d_ext,
                "d_ext_reading": cd.reading(d_ext["difference"], d_ext["ci_low"]),
            }
        summary[detector] = part
    write_json(RESULTS / "summary.json", summary)
    for detector, part in summary.items():
        print(detector, {k: round(v, 3) for k, v in part["plain"].items()})
        for kind in KINDS:
            if kind in part:
                print(" ", kind, {k: round(v, 3) for k, v in part[kind]["auc"].items()})
                print("   ", part[kind]["all_reading"], "|", part[kind]["versus_ii_reading"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("check", "run", "collect"))
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--detector", choices=DETECTORS, default="hubert")
    parser.add_argument("--kind", choices=KINDS, default="location")
    args = parser.parse_args()
    if args.step == "check":
        check(args.jobs, args.detector)
    elif args.step == "run":
        run(args.jobs, args.detector, args.kind)
    else:
        collect()


if __name__ == "__main__":
    main()
