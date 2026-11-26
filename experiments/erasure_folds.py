"""C1 with the place direction taken out, the direction fitted without the test fold's speakers.

Exploratory. Definitions and readings are in experiments/deviations.md (2026-11-10), written
before this run. Arms, speakers, detector and outer folds are those of the HuBERT /a/ C1; the
fold-wise path is the one pool_training.py checked against the stored (ii) scores.

    run       510 jobs (51 arms x 10 repeats, five folds each)
    collect   writes results/erasure_folds/summary.json (/a/) or summary_{i,u}.json

/i/ and /u/ (``--vowel``) were added after /a/, before they ran.

    .venv/Scripts/python.exe -I -X utf8 -B experiments/erasure_folds.py run --jobs 10
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

import pool_training as pt  # noqa: E402
import published_splits as ps  # noqa: E402
import run_first_batch as batch  # noqa: E402
import site_erasure as se  # noqa: E402
import vowel_fusion as vf  # noqa: E402
from arms import layer_columns, load_hubert, merge_features, write_json  # noqa: E402
from freeze import verify_freeze  # noqa: E402
from harness import load_plan, make_config  # noqa: E402
from sensitivity_avfad import CHOSEN_LAYERS  # noqa: E402

RESULTS = HERE / "results" / "erasure_folds"
PLAIN_C1 = HERE / "results" / "sensitivity" / "hubert" / "c1_predictions.csv.gz"
LEAKY = HERE / "results" / "site_erasure" / "summary.json"
PAIRED_SEED = 20261128
VOWEL_PAIRED_SEEDS = {"i": 20261132, "u": 20261133}
VOWEL_PLAIN = {v: HERE / "results" / "vowel_fusion" / f"predictions_{v}.csv.gz" for v in ("i", "u")}
VOWEL_PLAIN_SUMMARY = HERE / "results" / "vowel_fusion" / "summary.json"
# Phonation spans only (/a/), added after the full-file runs.
PHONATION_PLAIN = HERE / "results" / "phonation_c1" / "c1_predictions_hubert.csv.gz"
PHONATION_PLAIN_SUMMARY = HERE / "results" / "phonation_c1" / "summary.json"
PHONATION_PAIRED_SEED = 20261135


def suffix(vowel: str, condition: str = "full") -> str:
    if condition == "phonation":
        return "_phonation"
    return "" if vowel == "a" else f"_{vowel}"


# Readings, written before the run (deviations.md, 2026-11-10).
GAP_POINTS = 0.10
II_LOSS_POINTS = 0.03


def held_out_directions(normals: pd.DataFrame, test_speakers: set[str]) -> dict[int, np.ndarray]:
    """ESSUA-minus-HSA direction per layer from the normals outside the test fold."""
    kept = normals[~normals["speaker"].isin(test_speakers)]
    essua = (kept["place"] == "ESSUA").to_numpy()
    if essua.all() or not essua.any():
        raise RuntimeError("a place has no normal speaker outside the test fold")
    return se.place_directions(se.layer_blocks(kept), essua, ~essua)


def _fit(
    name: str, members: pd.DataFrame, repeat: int, config, vowel: str = "a", condition: str = "full"
) -> tuple[pd.DataFrame, float]:
    start = time.perf_counter()
    avfad = load_hubert(condition, CHOSEN_LAYERS, material=f"avfad_{vowel}")
    table = merge_features(vf.with_vowel(members, vowel), avfad, "label")
    table = table.assign(fold=pt.fold_ids(table, config)[:, repeat])
    normals = vf.with_vowel(se.normal_speakers(), vowel).merge(
        avfad, on="file_id", how="left", validate="one_to_one"
    )
    blocks = {f"l{layer:02d}": layer_columns(layer) for layer in CHOSEN_LAYERS}
    columns = [c for block in blocks.values() for c in block]
    tuned = config.override({"model.block_selection": True})
    score = np.full(len(table), np.nan)
    predicted = np.empty(len(table), dtype=object)
    for k in range(config.evaluation.n_splits):
        test_speakers = set(table.loc[table["fold"] == k, "speaker"])
        directions = held_out_directions(normals, test_speakers)
        output = ps.fit_listed_fold(se.erased(table, directions), columns, blocks, tuned, k)
        score[output["items"]] = output["scores"]
        predicted[output["items"]] = output["predicted"]
    frame = pd.DataFrame(
        {
            "arm": name,
            "speaker": table["speaker"].to_numpy(),
            "label": table["label"].to_numpy(),
            "repeat": repeat,
            "predicted": predicted,
            "score": score,
        }
    )
    return frame, time.perf_counter() - start


def run(jobs: int, vowel: str = "a", condition: str = "full") -> None:
    verify_freeze()
    plan = load_plan()
    config = make_config(plan, batch.LABEL_POSITIVE)
    sets = [(n, m) for n, m in vf.arm_sets(plan) if not n.startswith("shuffled")]
    todo = [(n, m, r) for n, m in sets for r in range(config.evaluation.n_repeats)]
    RESULTS.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    done = Parallel(n_jobs=jobs, return_as="generator_unordered")(
        delayed(_fit)(n, m, r, config, vowel, condition) for n, m, r in todo
    )
    frames = []
    for count, (frame, seconds) in enumerate(done, start=1):
        frames.append(frame)
        if count % 10 == 0 or count == len(todo):
            print(
                f"{count} of {len(todo)} after {time.perf_counter() - start:.0f} s "
                f"(last {seconds:.0f} s)",
                flush=True,
            )
    table = pd.concat(frames).sort_values(["arm", "speaker", "repeat"])
    table.to_csv(RESULTS / f"c1_predictions{suffix(vowel, condition)}.csv.gz", index=False)


def gap_reading(gap: dict) -> str:
    if gap["difference"] >= GAP_POINTS and gap["ci_low"] > 0:
        return "the gap stays: one direction is not enough"
    if gap["ci_low"] <= 0:
        return "the gap is gone: it sits on this one direction"
    return "the gap shrinks"


def collect(vowel: str = "a", condition: str = "full") -> None:
    import json

    plan = load_plan()
    tag = suffix(vowel, condition)
    table = pd.read_csv(RESULTS / f"c1_predictions{tag}.csv.gz", dtype={"speaker": str})
    arms = {name: vf.arm_from_table(table, name) for name in sorted(set(table["arm"]))}
    draws = [arms[f"i_{i:02d}"] for i in range(plan["confirmatory_family"]["tests"]["C1"]["draws"])]
    boot_seed = plan["statistics"]["seeds"]["bootstrap"]
    gap = batch.c1_bootstrap(draws, arms["ii"], vf.N_RESAMPLES, boot_seed)
    samples = gap.pop("samples")
    if condition == "phonation":
        plain_path = PHONATION_PLAIN
    else:
        plain_path = PLAIN_C1 if vowel == "a" else VOWEL_PLAIN[vowel]
    plain_table = pd.read_csv(plain_path, dtype={"speaker": str})
    plain_ii = vf.arm_from_table(plain_table, "ii")
    # Same seed, same speakers: the resample weights match, so the samples pair.
    plain_draws = [vf.arm_from_table(plain_table, a.name) for a in draws]
    plain_gap = batch.c1_bootstrap(plain_draws, plain_ii, vf.N_RESAMPLES, boot_seed)
    change = samples - plain_gap.pop("samples")
    low, high = np.quantile(change[np.isfinite(change)], [0.025, 0.975])
    seed = PAIRED_SEED if vowel == "a" else VOWEL_PAIRED_SEEDS.get(vowel)
    if condition == "phonation":
        seed = PHONATION_PAIRED_SEED
    paired = vf.paired_difference(arms["ii"], plain_ii, "uar", seed)
    summary = {
        "ii": {"uar": arms["ii"].uar, "roc_auc": arms["ii"].roc_auc},
        "i": {
            "uar_mean": float(np.mean([a.uar for a in draws])),
            "uar_range": [min(a.uar for a in draws), max(a.uar for a in draws)],
        },
        "gap_i_minus_ii": gap,
        "gap_reading": gap_reading(gap),
        "ii_erased_minus_plain": paired,
        "ii_loses_pathology": paired["difference"] <= -II_LOSS_POINTS and paired["ci_high"] < 0,
        "erased_minus_plain_gap": {
            "difference": gap["difference"] - plain_gap["difference"],
            "ci_low": float(low),
            "ci_high": float(high),
        },
    }
    if condition == "phonation":
        plain = json.loads(PHONATION_PLAIN_SUMMARY.read_text(encoding="utf-8"))["hubert"]
        summary["plain_gap"] = plain["gap_i_minus_ii"]["difference"]
    elif vowel == "a":
        leaky = json.loads(LEAKY.read_text(encoding="utf-8"))["avfad"]
        summary["leaky_gap"] = leaky["gap_i_minus_ii"]["difference"]
    else:
        plain = json.loads(VOWEL_PLAIN_SUMMARY.read_text(encoding="utf-8"))[vowel]
        summary["plain_gap"] = plain["gap_i_minus_ii"]["difference"]
    write_json(RESULTS / f"summary{tag}.json", summary)
    print(summary["gap_reading"], gap["difference"], gap["ci_low"], gap["ci_high"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("run", "collect"))
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--vowel", choices=("a", "i", "u"), default="a")
    parser.add_argument("--condition", choices=("full", "phonation"), default="full")
    args = parser.parse_args()
    if args.condition == "phonation" and args.vowel != "a":
        parser.error("the phonation features exist for /a/ only")
    if args.step == "run":
        run(args.jobs, args.vowel, args.condition)
    else:
        collect(args.vowel, args.condition)


if __name__ == "__main__":
    main()
