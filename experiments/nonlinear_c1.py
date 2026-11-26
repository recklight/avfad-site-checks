"""C1 with two non-linear detectors on the handcrafted features, raw and with both moments aligned.

Exploratory. Definitions and readings are in experiments/deviations.md (2026-11-10), written
before this run. The fold-wise path, the alignment and the speakers are those of harmonize_c1.py;
only ``model.name`` changes (RBF SVM, random forest) with the library's own grids.

    run       --model svc|random_forest --kind plain|scale
    collect   writes results/nonlinear_c1/summary.json

    .venv/Scripts/python.exe -I -X utf8 -B experiments/nonlinear_c1.py run --model svc \
        --kind plain --jobs 12
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import harmonize_c1 as hc  # noqa: E402
import run_first_batch as batch  # noqa: E402
import vowel_fusion as vf  # noqa: E402
from arms import write_json  # noqa: E402
from freeze import verify_freeze  # noqa: E402
from harness import load_plan, make_config  # noqa: E402

RESULTS = HERE / "results" / "nonlinear_c1"
MODELS = ("svc", "random_forest")
KINDS = ("plain", "scale")
LOGISTIC_SCALE = HERE / "results" / "harmonize_c1" / "c1_predictions_handcrafted_scale.csv.gz"
# Readings, written before the run (deviations.md, 2026-11-10).
GAP_POINTS = 0.10
VERSUS_POINTS = 0.03


def path_of(model: str, kind: str) -> Path:
    return RESULTS / f"c1_predictions_{model}_{kind}.csv.gz"


def run(jobs: int, model: str, kind: str) -> None:
    verify_freeze()
    plan = load_plan()
    config = make_config(plan, batch.LABEL_POSITIVE).override({"model.name": model})
    wanted = ("ii", "i_") if kind == "plain" else ("i_",)
    sets = [(n, m) for n, m in vf.arm_sets(plan) if n.startswith(wanted)]
    todo = [(n, m, r) for n, m in sets for r in range(config.evaluation.n_repeats)]
    RESULTS.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    done = Parallel(n_jobs=jobs, return_as="generator_unordered")(
        delayed(hc._fit)("handcrafted", kind, n, m, r, config) for n, m, r in todo
    )
    frames = []
    for count, (frame, seconds) in enumerate(done, start=1):
        frames.append(frame)
        if count % 50 == 0 or count == len(todo):
            print(
                f"{count} of {len(todo)} after {time.perf_counter() - start:.0f} s "
                f"(last {seconds:.0f} s)",
                flush=True,
            )
    pd.concat(frames).sort_values(["arm", "speaker", "repeat"]).to_csv(
        path_of(model, kind), index=False
    )


def plain_reading(gap: dict) -> str:
    if gap["difference"] >= GAP_POINTS and gap["ci_low"] > 0:
        return "the non-linear detector has the gap too"
    if gap["ci_low"] <= 0:
        return "the non-linear detector has no gap"
    return "a smaller gap"


def versus_reading(change: dict) -> str:
    if change["difference"] >= VERSUS_POINTS and change["ci_low"] > 0:
        return "learns more than logistic"
    if change["difference"] <= -VERSUS_POINTS and change["ci_high"] < 0:
        return "learns less than logistic"
    if change["ci_low"] <= 0 <= change["ci_high"]:
        return "about the same as logistic"
    return "as measured"


def aligned_versus_reading(change: dict) -> str:
    if change["difference"] >= VERSUS_POINTS and change["ci_low"] > 0:
        return "with both moments aligned, the non-linear detector still finds more place"
    if change["ci_low"] <= 0 <= change["ci_high"]:
        return "aligned, the same as logistic"
    return "as measured"


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"speaker": str})


def collect() -> None:
    plan = load_plan()
    logistic_plain = read(hc.PLAIN["handcrafted"])
    logistic = {"plain": hc.gap_of(logistic_plain, logistic_plain, plan)}
    if LOGISTIC_SCALE.exists():
        logistic["scale"] = hc.gap_of(read(LOGISTIC_SCALE), logistic_plain, plan)
    summary: dict = {"logistic": {k: v[0]["difference"] for k, v in logistic.items()}}
    for model in MODELS:
        if not path_of(model, "plain").exists():
            continue
        plain = read(path_of(model, "plain"))
        base = hc.gap_of(plain, plain, plan)
        versus = hc.paired(base, logistic["plain"])
        part: dict = {
            "plain": {
                "gap_i_minus_ii": base[0],
                "gap_reading": plain_reading(base[0]),
                "minus_logistic": versus,
                "versus_reading": versus_reading(versus),
            }
        }
        if path_of(model, "scale").exists():
            got = hc.gap_of(read(path_of(model, "scale")), plain, plan)
            part["scale"] = {
                "gap_i_minus_ii": got[0],
                "gap_reading": hc.gap_reading(got[0]),
                "minus_plain": hc.paired(got, base),
            }
            if "scale" in logistic:
                change = hc.paired(got, logistic["scale"])
                part["scale"]["minus_logistic"] = change
                part["scale"]["versus_reading"] = aligned_versus_reading(change)
        summary[model] = part
    write_json(RESULTS / "summary.json", summary)
    for model in MODELS:
        if model in summary:
            for kind, v in summary[model].items():
                g = v["gap_i_minus_ii"]
                print(
                    model, kind, v["gap_reading"], round(g["difference"], 4), round(g["ci_low"], 4)
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("run", "collect"))
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--model", choices=MODELS, default="svc")
    parser.add_argument("--kind", choices=KINDS, default="plain")
    args = parser.parse_args()
    if args.step == "run":
        run(args.jobs, args.model, args.kind)
    else:
        collect()


if __name__ == "__main__":
    main()
