"""The handcrafted arms of redo_c2.py again, in a process that fits no HuBERT arm.

A check, not a new analysis (deviations.md, 2026-11-10): joblib reuses its workers, and a worker
that fitted a HuBERT arm keeps the HuBERT C grid. redo_c2.py had both kinds in one Parallel. The
original files under results/exploratory/redo_c2/ are left as they are.

    .venv/Scripts/python.exe -I -X utf8 -B experiments/redo_recheck.py --jobs 2
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed, parallel_config  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import redo_c2 as rc  # noqa: E402
import run_first_batch as batch  # noqa: E402
from arms import write_json  # noqa: E402
from harness import load_plan, predictions_table  # noqa: E402

RESULTS = HERE / "results" / "redo_recheck"
ORIGINAL = HERE / "results" / "exploratory" / "redo_c2" / "predictions.csv.gz"
SAME = 1e-9


def tasks() -> list[tuple]:
    labels: list[str | None] = [None]
    for condition in rc.CONDITIONS:
        labels += [rc.set_name(condition.name, seed) for seed in condition.seeds]
    labels.append(f"{rc.set_name('B', 0)}_original_spans")
    out = [(label, "place", batch.PLACE_POSITIVE) for label in labels]
    for label in (None, rc.set_name("B", 0)):
        out += [(label, "sex", "F"), (label, "age_group", "older")]
    return out


def run(jobs: int) -> None:
    plan = load_plan()
    meta = rc.arm_meta(rc.speaker_table())
    todo = tasks()
    features = {label: rc.handcrafted_features(label) for label, _, _ in todo}
    with parallel_config(backend="loky", inner_max_num_threads=1):
        arms = Parallel(n_jobs=jobs)(
            delayed(rc._handcrafted_arm)(
                f"hand_{'A' if label is None else label}_{target}",
                meta,
                features[label],
                target,
                positive,
                plan,
            )
            for label, target, positive in todo
        )
    RESULTS.mkdir(parents=True, exist_ok=True)
    new = predictions_table(arms)
    new.to_csv(RESULTS / "predictions_handcrafted.csv.gz", index=False)
    old = pd.read_csv(ORIGINAL, dtype={"speaker": str})
    new = pd.read_csv(RESULTS / "predictions_handcrafted.csv.gz", dtype={"speaker": str})
    report: dict = {"arms": {}}
    for arm in arms:
        a = new[new["arm"] == arm.name].set_index(["speaker", "repeat"])["score"].sort_index()
        b = old[old["arm"] == arm.name].set_index(["speaker", "repeat"])["score"].sort_index()
        change = float(np.abs(a.to_numpy() - b.loc[a.index].to_numpy()).max())
        report["arms"][arm.name] = {"uar_new": arm.uar, "max_abs_score_change": change}
    report["same"] = all(r["max_abs_score_change"] <= SAME for r in report["arms"].values())
    write_json(RESULTS / "summary.json", report)
    for name, r in report["arms"].items():
        print(name, round(r["uar_new"], 4), f"{r['max_abs_score_change']:.2g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=2)
    run(parser.parse_args().jobs)


if __name__ == "__main__":
    main()
