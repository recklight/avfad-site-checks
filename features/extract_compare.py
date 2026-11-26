"""ComParE 2016 functionals, the features of Ribas et al. 2023, written to a CSV.

Runs only in opensmile_env, which holds opensmile and nothing of the study's own code, so the
frozen .venv never changes. The job list comes from experiments/ribas_method.py (step
``jobs``); the SVM and every score are computed back in .venv.

Their files were 16 kHz; how they converted them is not known. Here the first channel of
the original file is read and, when it is not 16 kHz already, resampled with audresample.

    opensmile_env/Scripts/python.exe -I -B features/extract_compare.py a --workers 16
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import csv  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "out_compare"
RATE = 16_000

_SMILE = None


def read_jobs(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return [(row["stem"], row["path"]) for row in csv.DictReader(f)]


def _smile():
    global _SMILE
    if _SMILE is None:
        import opensmile

        _SMILE = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    return _SMILE


def features_of(job: tuple[str, str]) -> tuple[str, list[float] | None, list[str] | None, str]:
    """One file's 6373 functionals, or the reason it has none."""
    import audiofile
    import audresample
    import numpy as np

    stem, relative = job
    try:
        signal, rate = audiofile.read(str(ROOT / relative), always_2d=True)
        signal = np.asarray(signal[0], dtype=np.float32)
        if signal.size == 0:
            return stem, None, None, "empty file"
        if rate != RATE:
            signal = audresample.resample(signal, rate, RATE)[0]
        table = _smile().process_signal(signal, RATE)
    except Exception as err:  # every failure is written down, not raised
        return stem, None, None, f"{type(err).__name__}: {err}"
    return stem, table.iloc[0].tolist(), list(table.columns), ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("material", choices=("a", "sentences"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, help="first N only, to check speed; writes no table")
    args = parser.parse_args()
    jobs = read_jobs(OUT / f"jobs_{args.material}.csv")
    if args.limit:
        jobs = jobs[: args.limit]
    started = time.perf_counter()
    with Pool(args.workers) as pool:
        done = pool.map(features_of, jobs, chunksize=4)
    seconds = time.perf_counter() - started
    print(len(done), "files in", round(seconds, 1), "s", flush=True)
    if args.limit:
        return
    columns = next(names for _, values, names, _ in done if names is not None)
    with (OUT / f"compare_{args.material}.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["stem", *columns])
        for stem, values, names, _ in done:
            if values is not None:
                if names != columns:
                    raise ValueError(f"{stem}: columns differ from the first file")
                writer.writerow([stem, *(repr(float(v)) for v in values)])
    with (OUT / f"compare_{args.material}_failed.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["stem", "reason"])
        for stem, values, _, reason in done:
            if values is None:
                writer.writerow([stem, reason])
    import opensmile

    (OUT / f"compare_{args.material}.txt").write_text(
        f"opensmile {opensmile.__version__}, ComParE_2016 functionals, {len(columns)} columns, "
        f"{RATE} Hz, first channel, {seconds:.0f} s\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
