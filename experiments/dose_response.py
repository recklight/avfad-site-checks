"""C1 with the normal group's ESSUA share stepped from 0 to about 1 (dose and response).

Exploratory. Definitions and readings are in experiments/deviations.md (2026-11-10), written
before this run. 82 normals throughout: k of the HSA normals are swapped for ESSUA normals of
the same age-by-sex cell; 0% is the stored (ii).

    run       40 arms per detector (--detector hubert or handcrafted, one process each)
    collect   writes results/dose_response/summary.json

Added after /a/: ``--variant i|u`` (HuBERT on the same 40 groups, /i/ or /u/) and
``--variant erased`` (HuBERT /a/ with the place direction taken out fold by fold).

    .venv/Scripts/python.exe -I -X utf8 -B experiments/dose_response.py run --detector hubert
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import itertools  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from voice_severity_bench import models as vsb_models  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import diagnosis_breakdown as db  # noqa: E402
import run_first_batch as batch  # noqa: E402
import vowel_fusion as vf  # noqa: E402
from arms import evaluate_layers, load_hubert, merge_features, write_json  # noqa: E402
from freeze import verify_freeze  # noqa: E402
from harness import (  # noqa: E402
    analysis_table,
    evaluate_arm,
    load_plan,
    make_config,
    predictions_table,
)
from sensitivity_avfad import CHOSEN_LAYERS, hsa_groups  # noqa: E402

RESULTS = HERE / "results" / "dose_response"
STORED = {
    "hubert": HERE / "results" / "sensitivity" / "hubert" / "c1_predictions.csv.gz",
    "handcrafted": HERE / "results" / "first_batch" / "c1_predictions.csv.gz",
}
OUTSIDE = HERE / "results" / "cross_database" / "reverse_scores_hubert.csv.gz"
VARIANT_BASE = {
    "i": HERE / "results" / "vowel_fusion" / "predictions_i.csv.gz",
    "u": HERE / "results" / "vowel_fusion" / "predictions_u.csv.gz",
    "erased": HERE / "results" / "erasure_folds" / "c1_predictions.csv.gz",
}
VARIANT_OUTSIDE = {
    "i": HERE / "results" / "outside_view" / "outside_scores_i.csv.gz",
    "u": HERE / "results" / "outside_view" / "outside_scores_u.csv.gz",
    "erased": OUTSIDE,
}
LEVELS = (20, 41, 61, 82)
DRAWS = 10
NORMALS = 82
SEED = 20261139
N_SLOPE = 1000
LOGISTIC = ("classification", "logistic")
# Taken at import, before any HuBERT fit in this process can replace it.
HANDCRAFTED_GRID = {k: list(v) for k, v in vsb_models.SEARCH_SPACES[LOGISTIC].items()}
# Readings, written before the run (deviations.md, 2026-11-10).
SLOPE_POINTS = 0.10
ERASED_FLAT = 0.10


def cell_of(age: float, sex: str) -> str:
    return f"{int(age) // 5}_{sex}"


def swapped_normals(
    hsa_normals: pd.DataFrame, essua_normals: pd.DataFrame, k: int, rng: np.random.Generator
) -> pd.DataFrame:
    """k HSA normals, chosen at random, each swapped for an unused ESSUA normal of its cell."""
    hsa_normals = hsa_normals.sort_values("speaker").reset_index(drop=True)
    essua_normals = essua_normals.sort_values("speaker").reset_index(drop=True)
    chosen = np.sort(rng.choice(len(hsa_normals), size=k, replace=False))
    pool_cells = np.array(
        [cell_of(a, s) for a, s in zip(essua_normals["age"], essua_normals["sex"], strict=True)]
    )
    used: set[str] = set()
    rows = [hsa_normals.drop(index=chosen)]
    for index in chosen:
        person = hsa_normals.loc[index]
        free = np.flatnonzero(
            (pool_cells == cell_of(person["age"], person["sex"]))
            & ~essua_normals["speaker"].isin(used).to_numpy()
        )
        if free.size == 0:
            rows.append(hsa_normals.loc[[index]])
            continue
        pick = essua_normals.loc[[free[rng.integers(free.size)]]]
        used.add(str(pick["speaker"].iloc[0]))
        rows.append(pick)
    out = pd.concat(rows, ignore_index=True)
    if len(out) != len(hsa_normals) or out["speaker"].duplicated().any():
        raise RuntimeError("the swapped normal group is not 82 different speakers")
    return out


def arm_sets() -> list[tuple[str, pd.DataFrame, float]]:
    patients, hsa_normals, _ = hsa_groups(batch.load_speakers())
    speakers = batch.load_speakers()
    essua_normals = speakers[(speakers["place"] == "ESSUA") & (speakers["label"] == "normal")]
    rng = np.random.default_rng(SEED)
    sets = []
    for k in LEVELS:
        for d in range(DRAWS):
            normals = swapped_normals(hsa_normals, essua_normals, k, rng)
            share = float((normals["place"] == "ESSUA").mean())
            sets.append((f"s{k:02d}_{d}", pd.concat([patients, normals]), share))
    return sets


def _fit(detector: str, name: str, members: pd.DataFrame, config):
    start = time.perf_counter()
    if detector == "hubert":
        table = merge_features(members, load_hubert("full", CHOSEN_LAYERS), "label")
        arm = evaluate_layers(name, table, CHOSEN_LAYERS, config)
    else:
        vsb_models.SEARCH_SPACES[LOGISTIC] = {k: list(v) for k, v in HANDCRAFTED_GRID.items()}
        table = analysis_table(members, batch.load_features("full_original"), "label")
        arm = evaluate_arm(name, table, config)
    return arm, time.perf_counter() - start


def _fit_vowel(vowel: str, name: str, members: pd.DataFrame, config):
    start = time.perf_counter()
    features = load_hubert("full", CHOSEN_LAYERS, material=f"avfad_{vowel}")
    table = merge_features(vf.with_vowel(members, vowel), features, "label")
    return evaluate_layers(name, table, CHOSEN_LAYERS, config), time.perf_counter() - start


def run_variant(variant: str, jobs: int) -> None:
    """HuBERT on the same 40 groups: another vowel, or /a/ with the place direction out."""
    import erasure_folds as ef

    verify_freeze()
    config = make_config(load_plan(), batch.LABEL_POSITIVE)
    sets = arm_sets()
    start = time.perf_counter()
    if variant == "erased":
        todo = [(n, m, r) for n, m, _ in sets for r in range(config.evaluation.n_repeats)]
        done = Parallel(n_jobs=jobs, return_as="generator_unordered")(
            delayed(ef._fit)(n, m, r, config) for n, m, r in todo
        )
        frames = []
        for count, (frame, _) in enumerate(done, start=1):
            frames.append(frame)
            if count % 40 == 0 or count == len(todo):
                print(f"erased {count} of {len(todo)} after {time.perf_counter() - start:.0f} s")
        table = pd.concat(frames).sort_values(["arm", "speaker", "repeat"])
    else:
        done = Parallel(n_jobs=jobs, return_as="generator_unordered")(
            delayed(_fit_vowel)(variant, n, m, config) for n, m, _ in sets
        )
        arms = []
        for count, (arm, _) in enumerate(done, start=1):
            arms.append(arm)
            if count % 5 == 0 or count == len(sets):
                print(f"{variant} {count} of {len(sets)} after {time.perf_counter() - start:.0f} s")
        arms.sort(key=lambda a: a.name)
        table = predictions_table(arms)
    table.to_csv(RESULTS / f"predictions_hubert_{variant}.csv.gz", index=False)


def run(detector: str, jobs: int) -> None:
    verify_freeze()
    config = make_config(load_plan(), batch.LABEL_POSITIVE)
    sets = arm_sets()
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = [
        {"arm": name, "speaker": s, "place": p, "label": lab}
        for name, members, _ in sets
        for s, p, lab in zip(members["speaker"], members["place"], members["label"], strict=True)
    ]
    pd.DataFrame(rows).to_csv(RESULTS / "arms_members.csv.gz", index=False)
    start = time.perf_counter()
    done = Parallel(n_jobs=jobs, return_as="generator_unordered")(
        delayed(_fit)(detector, name, members, config) for name, members, _ in sets
    )
    arms = []
    for count, (arm, seconds) in enumerate(done, start=1):
        arms.append(arm)
        if count % 5 == 0 or count == len(sets):
            print(
                f"{detector} {count} of {len(sets)} after {time.perf_counter() - start:.0f} s "
                f"(last {seconds:.0f} s)",
                flush=True,
            )
    arms.sort(key=lambda a: a.name)
    predictions_table(arms).to_csv(RESULTS / f"predictions_{detector}.csv.gz", index=False)


def slope(shares: np.ndarray, values: np.ndarray) -> float:
    return float(np.polyfit(shares, values, 1)[0])


def slope_interval(levels: dict[int, list[tuple[float, float]]], base: tuple[float, float]) -> dict:
    """Least squares over every arm; the draws resample within each level, 0% stays fixed."""
    rng = np.random.default_rng(SEED)
    points = [base] + [p for part in levels.values() for p in part]
    x, y = np.array(points).T
    estimate = slope(x, y)
    samples = []
    for _ in range(N_SLOPE):
        picked = [base]
        for part in levels.values():
            picked += [part[i] for i in rng.integers(len(part), size=len(part))]
        px, py = np.array(picked).T
        samples.append(slope(px, py))
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"slope": estimate, "ci_low": float(low), "ci_high": float(high)}


def rise_reading(s: dict) -> str:
    if s["slope"] >= SLOPE_POINTS and s["ci_low"] > 0:
        return "the score rises with the ESSUA share"
    return "as measured"


def outside_reading(s: dict) -> str:
    if s["ci_low"] <= 0 <= s["ci_high"]:
        return "flat for the outside detector"
    if s["ci_high"] < 0:
        return "the outside detector goes down"
    return "as measured"


def erased_reading(s: dict) -> str:
    if s["slope"] <= ERASED_FLAT:
        return "with the direction out, most of the dose response is gone"
    if s["ci_low"] > ERASED_FLAT:
        return "with one direction out, the dose response stays"
    return "as measured"


def collect(variant: str = "a") -> None:
    members = pd.read_csv(RESULTS / "arms_members.csv.gz", dtype={"speaker": str})
    shares = (
        members[members["label"] == "normal"]
        .groupby("arm")["place"]
        .apply(lambda p: float((p == "ESSUA").mean()))
    )
    outside_path = OUTSIDE if variant == "a" else VARIANT_OUTSIDE[variant]
    outside = pd.read_csv(outside_path, dtype={"speaker": str}).set_index("speaker")["score"]
    patients = members.loc[members["label"] == "pathological", "speaker"].unique()
    _, hsa_normals, _ = hsa_groups(batch.load_speakers())

    def outside_auc(normals) -> float:
        return float(
            db.comparisons(outside.loc[patients].to_numpy(), outside.loc[normals].to_numpy()).mean()
        )

    base_outside = outside_auc(hsa_normals["speaker"])
    out_levels: dict[int, list] = {k: [] for k in LEVELS}
    for arm, part in members[members["label"] == "normal"].groupby("arm"):
        out_levels[int(arm[1:3])].append((shares[arm], outside_auc(part["speaker"])))
    summary: dict = {
        "shares_by_level": {
            str(k): [round(s, 3) for s, _ in sorted(v)] for k, v in out_levels.items()
        },
        "outside_auc": {
            "levels": {"0": base_outside}
            | {str(k): float(np.mean([a for _, a in v])) for k, v in out_levels.items()},
            "slope": slope_interval(out_levels, (0.0, base_outside)),
        },
    }
    summary["outside_auc"]["reading"] = outside_reading(summary["outside_auc"]["slope"])
    sources = STORED.items() if variant == "a" else [(f"hubert_{variant}", VARIANT_BASE[variant])]
    for detector, stored in sources:
        path = RESULTS / f"predictions_{detector}.csv.gz"
        if not path.exists():
            continue
        table = pd.read_csv(path, dtype={"speaker": str})
        base = vf.arm_from_table(pd.read_csv(stored, dtype={"speaker": str}), "ii")
        levels: dict[int, list] = {k: [] for k in LEVELS}
        for name in sorted(set(table["arm"])):
            arm = vf.arm_from_table(table, name)
            levels[int(name[1:3])].append((shares[name], arm.uar))
        means = {"0": base.uar} | {
            str(k): float(np.mean([u for _, u in v])) for k, v in levels.items()
        }
        values = list(means.values())
        s = slope_interval(levels, (0.0, base.uar))
        summary[detector] = {
            "uar_levels": means,
            "uar_range_by_level": {
                str(k): [min(u for _, u in v), max(u for _, u in v)] for k, v in levels.items()
            },
            "slope": s,
            "monotonic": bool(all(b > a for a, b in itertools.pairwise(values))),
            "reading": rise_reading(s) if variant != "erased" else erased_reading(s),
        }
        print(detector, summary[detector]["reading"], {k: round(v, 3) for k, v in means.items()})
    print("outside", summary["outside_auc"]["reading"], summary["outside_auc"]["slope"])
    summary["variant"] = variant
    write_json(RESULTS / ("summary.json" if variant == "a" else f"summary_{variant}.json"), summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("run", "collect"))
    parser.add_argument("--detector", choices=("hubert", "handcrafted"), default="hubert")
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--variant", choices=("a", "i", "u", "erased"), default="a")
    args = parser.parse_args()
    if args.step == "run" and args.variant != "a":
        run_variant(args.variant, args.jobs)
    elif args.step == "run":
        run(args.detector, args.jobs)
    else:
        collect(args.variant)


if __name__ == "__main__":
    main()
