"""Trained on AVFAD /i/ or /u/, tested on SVD i_n or u_n: does the /a/ result repeat?

Exploratory. Definitions and readings are in experiments/deviations.md (2026-11-09), written
before any /i/ or /u/ feature existed. Same detector and training sets as cross_database.py;
only the vowel changes. SVD files are read from data.zip in memory; no audio is written.

    features    HuBERT ``full`` of AVFAD 001/003 and SVD i_n/u_n into features/out_hubert/
    run         every training set scored on SVD; --vowel i|u
    collect     the readings; writes results/cross_vowels/summary.json

    .venv/Scripts/python.exe -I -X utf8 -B experiments/cross_vowels.py features --jobs 8
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "features"))

import cross_database as cd  # noqa: E402
import run_first_batch as batch  # noqa: E402
from arms import layer_columns, load_hubert, write_json  # noqa: E402
from common import load_signal, to_analysis_rate  # noqa: E402
from extract_hubert import REVISION, embed  # noqa: E402
from freeze import verify_freeze  # noqa: E402
from freeze_svd import verify_svd_freeze  # noqa: E402
from harness import ROOT, load_plan, make_config  # noqa: E402
from readers import check_nsp, parse_nsp  # noqa: E402  (s0_audit, put on the path by common)
from sensitivity_avfad import CHOSEN_LAYERS  # noqa: E402
from svd_eggcheck import f1_to_f0_db  # noqa: E402

VOWELS = ("i", "u")
AVFAD_TASK = {"i": "001", "u": "003"}
SVD_MATERIAL = {"i": "i_n", "u": "u_n"}
SEEDS = {"i": 20261115, "u": 20261116}
ZIP = ROOT / "data" / "svd" / "data.zip"
HUBERT_DIR = ROOT / "features" / "out_hubert"
RESULTS = HERE / "results" / "cross_vowels"


def avfad_file_id(speaker: str, vowel: str) -> str:
    return f"avfad__{speaker}-{vowel}"


def svd_member(session: int, vowel: str) -> str:
    return f"data/{session}/vowels/{session}-{SVD_MATERIAL[vowel]}.nsp"


def egg_member(session: int, vowel: str) -> str:
    return f"data/{session}/vowels/{session}-{SVD_MATERIAL[vowel]}-egg.egg"


# ------------------------------------------------------------------ features


def avfad_jobs(vowel: str) -> list[tuple[str, str]]:
    """Each speaker's file of the vowel, next to their /a/ file; the extension case varies."""
    files = pd.read_csv(ROOT / "s1_segments" / "out" / "files.csv")
    a = files[(files["database"] == "avfad") & (files["material"] == "a")].sort_values("file_id")
    jobs = []
    for file_id, wav in zip(a["file_id"], a["wav"], strict=True):
        speaker = file_id.removeprefix("avfad__")
        folder = (ROOT / wav).parent
        found = [
            p for p in folder.glob(f"{speaker}{AVFAD_TASK[vowel]}.*") if p.suffix.lower() == ".wav"
        ]
        if len(found) != 1:
            raise FileNotFoundError(f"{speaker}{AVFAD_TASK[vowel]}: {len(found)} files")
        jobs.append((avfad_file_id(speaker, vowel), found[0].relative_to(ROOT).as_posix()))
    return jobs


def _avfad_one(job: tuple[str, str]) -> tuple[str, np.ndarray, int]:
    file_id, wav = job
    x, rate = load_signal(wav)
    if rate != 16_000:
        raise ValueError(f"{file_id}: expected 16 kHz after loading, got {rate}")
    pooled, n_frames = embed([x])
    return file_id, pooled, n_frames


def svd_sessions() -> pd.DataFrame:
    """The 1,459 sessions of the /a/ cross-database test set."""
    return cd.svd_test_set()


_ZIP: zipfile.ZipFile | None = None


def _zip() -> zipfile.ZipFile:
    """One open handle per worker; the central directory has 64k entries."""
    global _ZIP
    if _ZIP is None:
        _ZIP = zipfile.ZipFile(ZIP)
    return _ZIP


SWAP_PROBLEMS = ("speech_egg_swapped", "speech_egg_swapped_partway", "swap_screen")


def swapped_sessions() -> set[int]:
    """Sessions with a speech/EGG swap, or a borderline one, in any material of the S0 audit.

    The F1/F0 margin of s0_audit/svd_eggcheck.py cannot be used on /i/ and /u/: their F1 is
    below 500 Hz, so most good speech files score below their EGG (deviations.md, 2026-11-10).
    """
    status = pd.read_csv(
        ROOT / "s0_audit" / "out" / "svd_file_status.csv", dtype=str, keep_default_na=False
    )
    return set(status.loc[status["file_problem"].isin(SWAP_PROBLEMS), "session"].astype(int))


def drop_reason(problems: list[str], session_swapped: bool) -> str:
    """Why a file is left out, or "" to keep it."""
    if problems:
        return "; ".join(problems)
    if session_swapped:
        return "a speech/EGG swap elsewhere in the session (S0 audit)"
    return ""


def _svd_one(job: tuple[int, str, bool]) -> dict:
    """HuBERT of one session's vowel, read from the zip in memory; the margin is only reported."""
    session, vowel, session_swapped = job
    row = {"session": session, "file_id": f"svd__{session}-{SVD_MATERIAL[vowel]}"}
    zf = _zip()
    try:
        speech = parse_nsp(zf.read(svd_member(session, vowel)))
    except (KeyError, ValueError) as err:
        return {**row, "problem": f"speech: {err}"}
    try:
        egg = parse_nsp(zf.read(egg_member(session, vowel)))
        margin = f1_to_f0_db(speech.samples, speech.rate) - f1_to_f0_db(egg.samples, egg.rate)
    except (KeyError, ValueError):
        margin = float("nan")
    row["margin_db"] = margin
    reason = drop_reason(check_nsp(speech), session_swapped)
    if reason:
        return {**row, "problem": reason}
    x, rate = to_analysis_rate(speech.samples.astype(np.float64) / 32768.0, speech.rate)
    pooled, n_frames = embed([x])
    return {**row, "problem": "", "pooled": pooled, "n_frames": n_frames, "rate": rate}


def save_npz(path: Path, ids: list[str], pooled: list[np.ndarray], frames: list[int]) -> None:
    np.savez(
        path,
        file_id=np.array(ids),
        pooled=np.stack(pooled),
        n_frames=np.array(frames),
        revision=np.array(REVISION),
    )


def avfad_features(pool: ProcessPoolExecutor, vowel: str) -> dict:
    done = list(pool.map(_avfad_one, avfad_jobs(vowel), chunksize=4))
    save_npz(
        HUBERT_DIR / f"avfad_{vowel}_full.npz",
        [f for f, _, _ in done],
        [p for _, p, _ in done],
        [n for _, _, n in done],
    )
    return {"avfad_files": len(done)}


def svd_features(pool: ProcessPoolExecutor, vowel: str) -> dict:
    sessions = svd_sessions()["session"].astype(int).tolist()
    swapped = swapped_sessions()
    rows = list(pool.map(_svd_one, [(s, vowel, s in swapped) for s in sessions], chunksize=8))
    kept = [r for r in rows if not r["problem"]]
    save_npz(
        HUBERT_DIR / f"svd_{SVD_MATERIAL[vowel]}_full.npz",
        [r["file_id"] for r in kept],
        [r["pooled"] for r in kept],
        [r["n_frames"] for r in kept],
    )
    margins = np.array([r.get("margin_db", np.nan) for r in rows], dtype=float)
    return {
        "svd_sessions": len(rows),
        "svd_kept": len(kept),
        "svd_dropped": [
            {k: r.get(k) for k in ("session", "file_id", "problem", "margin_db")}
            for r in rows
            if r["problem"]
        ],
        "margin_db_speech_minus_egg": {
            "median": float(np.nanmedian(margins)),
            "below_zero": int(np.sum(margins < 0)),
            "missing_egg": int(np.sum(np.isnan(margins))),
        },
    }


def run_features(
    jobs: int, databases: tuple[str, ...] = ("avfad", "svd"), vowels: tuple[str, ...] = VOWELS
) -> None:
    """One report file per database and vowel, so parts can run side by side."""
    started = time.perf_counter()
    RESULTS.mkdir(parents=True, exist_ok=True)
    make = {"avfad": avfad_features, "svd": svd_features}
    with ProcessPoolExecutor(jobs) as pool:
        for vowel in vowels:
            for database in databases:
                report = make[database](pool, vowel)
                write_json(RESULTS / f"features_{database}_{vowel}.json", report)
                print(database, vowel, report, round(time.perf_counter() - started), "s")


# ------------------------------------------------------------------ scoring


def training_sets(vowel: str) -> dict[str, pd.DataFrame]:
    """The /a/ training sets with each speaker's file swapped for the same speaker's vowel."""
    return {
        name: members.assign(
            file_id=[avfad_file_id(s, vowel) for s in members["speaker"].astype(str)]
        )
        for name, members in cd.training_sets().items()
    }


def run_scores(vowel: str, jobs: int) -> None:
    plan = load_plan()
    config = make_config(plan, batch.LABEL_POSITIVE)
    avfad = load_hubert("full", CHOSEN_LAYERS, material=f"avfad_{vowel}")
    svd_features = load_hubert("full", CHOSEN_LAYERS, material=f"svd_{SVD_MATERIAL[vowel]}")
    blocks = {f"l{layer:02d}": layer_columns(layer) for layer in CHOSEN_LAYERS}
    columns = [c for block in blocks.values() for c in block]
    sessions = svd_sessions()
    test = sessions.assign(
        file_id=[f"svd__{s}-{SVD_MATERIAL[vowel]}" for s in sessions["session"]]
    ).merge(svd_features, on="file_id", how="inner", validate="one_to_one")
    sets = {}
    for name, members in training_sets(vowel).items():
        train = members.merge(avfad, on="file_id", how="left", validate="one_to_one")
        if train[columns[0]].isna().any():
            raise ValueError(f"{name}: an AVFAD speaker has no features")
        sets[name] = train
    started = time.perf_counter()
    done = Parallel(n_jobs=jobs)(
        delayed(cd._one)(name, train, test, columns, blocks, config) for name, train in sets.items()
    )
    print(vowel, len(done), "training sets in", round(time.perf_counter() - started), "s")
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = test[["file_id", "speaker", "label", "age", "sex", "year"]].copy()
    for name, scores in done:
        frame[name] = scores
    frame.to_csv(RESULTS / f"scores_{vowel}.csv.gz", index=False)


# ------------------------------------------------------------------ readings


def both_vowels(readings: dict[str, str]) -> str:
    """Repeats only when both vowels do not travel; fails when either partly travels."""
    values = list(readings.values())
    if all(r.startswith("does not travel") for r in values):
        return "repeats"
    if any(r == "part of it travels" for r in values):
        return "does not repeat"
    return "no conclusion"


def run_collect() -> None:
    report = {}
    for vowel in VOWELS:
        path = RESULTS / f"scores_{vowel}.csv.gz"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        record = cd.evaluate(frame, seed=SEEDS[vowel])
        record["reading"] = cd.reading(record["d_ext"]["auc"], record["d_ext"]["auc_ci"][0])
        record["reversed"] = bool(record["d_ext"]["auc_ci"][1] < 0)
        low, high = record["essua_normals_hsa_patients"]["auc_ci"]
        record["mixed_contrast_stays_inside"] = bool(low <= 0.5 <= high)
        record["all_minus_ii"] = record["all"]["auc"] - record["ii"]["auc"]
        report[vowel] = record
    if set(report) == set(VOWELS):
        report["both"] = {
            "reading": both_vowels({v: report[v]["reading"] for v in VOWELS}),
            "reversed_in_both": all(report[v]["reversed"] for v in VOWELS),
            "mixed_contrast_stays_inside_in_both": all(
                report[v]["mixed_contrast_stays_inside"] for v in VOWELS
            ),
        }
    write_json(RESULTS / "summary.json", report)
    print(json.dumps(report, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("features", "run", "collect"))
    parser.add_argument("--vowel", choices=VOWELS)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--databases", nargs="*", choices=("avfad", "svd"), default=["avfad", "svd"]
    )
    args = parser.parse_args()
    verify_freeze()
    verify_svd_freeze()
    if args.step == "features":
        run_features(args.jobs, tuple(args.databases), (args.vowel,) if args.vowel else VOWELS)
    elif args.step == "run":
        for vowel in [args.vowel] if args.vowel else VOWELS:
            run_scores(vowel, args.jobs)
    else:
        run_collect()


if __name__ == "__main__":
    main()
