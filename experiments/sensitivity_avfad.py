"""Planned sensitivity analyses on AVFAD: C1 without laryngeal surgery, C1 and C2 on HuBERT.

Details of the HuBERT runs are in experiments/deviations.md, written before they ran.

    .venv/Scripts/python.exe -I -B experiments/sensitivity_avfad.py surgery --jobs 4
    .venv/Scripts/python.exe -I -B experiments/sensitivity_avfad.py hubert-layers --jobs 16
    .venv/Scripts/python.exe -I -B experiments/sensitivity_avfad.py hubert-c1 --jobs 16
    .venv/Scripts/python.exe -I -B experiments/sensitivity_avfad.py hubert-c2 --jobs 16
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
from voice_severity_bench.bootstrap import bootstrap_metric  # noqa: E402
from voice_severity_bench.dataset import feature_set_from_table  # noqa: E402
from voice_severity_bench.permutation import permutation_test  # noqa: E402
from joblib import Parallel, delayed, parallel_config  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_first_batch as batch  # noqa: E402
from arms import (  # noqa: E402
    ALL_LAYERS,
    _use_hubert_grid,
    evaluate_layers,
    layer_columns,
    load_hubert,
    merge_features,
    write_json,
)
from freeze import FreezeError, verify_freeze  # noqa: E402
from harness import load_plan, make_config, match_cells, predictions_table  # noqa: E402

RESULTS = HERE / "results" / "sensitivity"
CHOSEN_LAYERS = (0, 3, 6, 9, 12)
N_SHUFFLES = 5


def shuffled_uar(table: pd.DataFrame, layers, config, seed: int) -> float:
    """One label shuffle with the full CV; runs inside one worker, so the grid is set there."""
    _use_hubert_grid()
    blocks = {f"l{layer:02d}": layer_columns(layer) for layer in layers}
    columns = [c for block in blocks.values() for c in block]
    features = feature_set_from_table(
        table[["path", "group", "label", *columns]],
        target="label",
        blocks=blocks,
        task="classification",
    )
    tuned = config.override({"model.block_selection": True})
    return float(permutation_test(features, tuned, n_permutations=1, seed=seed, n_jobs=1).null[0])


def hsa_groups(speakers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patients = speakers[(speakers["place"] == "HSA") & (speakers["label"] == "pathological")]
    hsa_normals = speakers[(speakers["place"] == "HSA") & (speakers["label"] == "normal")]
    all_normals = speakers[speakers["label"] == "normal"]
    return patients, hsa_normals, all_normals


def run_surgery(plan: dict, jobs: int) -> None:
    speakers = batch.load_speakers()
    if speakers["laryngeal_surgery"].isna().any():
        raise ValueError("surgery unknown for some speakers")
    kept = speakers[speakers["laryngeal_surgery"] == 0]
    out = RESULTS / "c1_no_surgery"
    out.mkdir(parents=True, exist_ok=True)
    result = batch.run_c1(kept, batch.load_features("full_original"), plan, jobs, out)
    result["removed"] = {
        "patients": int(
            ((speakers["place"] == "HSA") & (speakers["label"] == "pathological")).sum()
            - result["patients"]
        ),
        "hsa_normals": int(
            ((speakers["place"] == "HSA") & (speakers["label"] == "normal")).sum()
            - result["normals_per_arm"]
        ),
    }
    write_json(out / "summary.json", result)


def run_hubert_layers(plan: dict, jobs: int) -> None:
    """Arm (ii) with the layer chosen inside, and with every layer fixed."""
    speakers = batch.load_speakers()
    patients, hsa_normals, _ = hsa_groups(speakers)
    meta = pd.concat([patients, hsa_normals])
    out = RESULTS / "hubert"
    out.mkdir(parents=True, exist_ok=True)
    config = make_config(plan, batch.LABEL_POSITIVE)
    chosen = merge_features(meta, load_hubert("full", CHOSEN_LAYERS), "label")
    started = time.perf_counter()
    arm = evaluate_layers("ii_chosen", chosen, CHOSEN_LAYERS, config)
    seconds = time.perf_counter() - started
    print(f"ii chosen layer: UAR {arm.uar:.4f}, one worker {seconds:.0f} s", flush=True)
    full = merge_features(meta, load_hubert("full", ALL_LAYERS), "label")
    fixed = Parallel(n_jobs=jobs)(
        delayed(evaluate_layers)(f"ii_l{layer:02d}", full, (layer,), config) for layer in ALL_LAYERS
    )
    rows = pd.DataFrame([a.row() for a in [arm, *fixed]])
    rows.to_csv(out / "ii_layers.csv", index=False)
    predictions_table([arm, *fixed]).to_csv(out / "ii_predictions.csv.gz", index=False)
    interval = bootstrap_metric(
        arm.y_true,
        arm.predictions,
        arm.groups,
        metric="uar",
        n_resamples=10_000,
        seed=plan["statistics"]["seeds"]["bootstrap"],
    )
    write_json(
        out / "ii_summary.json",
        {
            **arm.row(),
            "seconds_one_worker": round(seconds, 1),
            "uar_interval": interval.as_dict(),
            "fixed_layer_uar": {a.name: a.uar for a in fixed},
            "fixed_layer_auc": {a.name: a.roc_auc for a in fixed},
        },
    )


def run_hubert_c1(plan: dict, jobs: int, n_draws: int | None) -> None:
    spec = plan["confirmatory_family"]["tests"]["C1"]
    seeds = plan["statistics"]["seeds"]
    speakers = batch.load_speakers()
    patients, hsa_normals, all_normals = hsa_groups(speakers)
    rng = np.random.default_rng(seeds["c1_draws"])
    # Always walk all 50 so a shorter run still holds the same people as the handcrafted one.
    draws = [match_cells(hsa_normals, all_normals, rng)[0] for _ in range(spec["draws"])]
    draws = draws[: n_draws or spec["draws"]]
    features = load_hubert("full", CHOSEN_LAYERS)
    config = make_config(plan, batch.LABEL_POSITIVE)
    jobs_list = [("ii", merge_features(pd.concat([patients, hsa_normals]), features, "label"))]
    jobs_list += [
        (f"i_{i:02d}", merge_features(pd.concat([patients, drawn]), features, "label"))
        for i, drawn in enumerate(draws)
    ]
    arms = Parallel(n_jobs=jobs)(
        delayed(evaluate_layers)(name, table, CHOSEN_LAYERS, config) for name, table in jobs_list
    )
    arm_ii, arms_i = arms[0], arms[1:]
    out = RESULTS / "hubert"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([a.row() for a in arms]).to_csv(out / "c1_arms.csv", index=False)
    predictions_table(arms).to_csv(out / "c1_predictions.csv.gz", index=False)
    test = batch.c1_bootstrap(arms_i, arm_ii, spec["test"]["resamples"], seeds["bootstrap"])
    test.pop("samples")
    shuffles = Parallel(n_jobs=jobs)(
        delayed(shuffled_uar)(jobs_list[0][1], CHOSEN_LAYERS, config, seeds["negative_control"] + k)
        for k in range(N_SHUFFLES)
    )
    uars = [a.uar for a in arms_i]
    write_json(
        out / "c1_summary.json",
        {
            **test,
            "draws": len(arms_i),
            "uar_i_mean": float(np.mean(uars)),
            "uar_i_min": float(np.min(uars)),
            "uar_i_max": float(np.max(uars)),
            "uar_ii": arm_ii.uar,
            "roc_auc_i_mean": float(np.mean([a.roc_auc for a in arms_i])),
            "roc_auc_ii": arm_ii.roc_auc,
            "sensitivity_i_mean": float(np.mean([a.sensitivity for a in arms_i])),
            "specificity_i_mean": float(np.mean([a.specificity for a in arms_i])),
            "sensitivity_ii": arm_ii.sensitivity,
            "specificity_ii": arm_ii.specificity,
            "chosen_ii": arm_ii.chosen_c,
            "shuffled_ii_mean_uar": float(np.mean(shuffles)),
            "shuffled_ii_uar": shuffles,
        },
    )


def run_hubert_c2(plan: dict, jobs: int) -> None:
    seeds = plan["statistics"]["seeds"]
    normal = batch.load_speakers().query("label == 'normal'")
    hsa = normal[normal["place"] == "HSA"]
    essua, _ = match_cells(
        hsa, normal[normal["place"] == "ESSUA"], np.random.default_rng(seeds["c2_essua_draw"])
    )
    table = merge_features(
        pd.concat([hsa, essua]), load_hubert("phonation", CHOSEN_LAYERS), "place"
    )
    config = make_config(plan, batch.PLACE_POSITIVE)
    arm = evaluate_layers("c2", table, CHOSEN_LAYERS, config)
    out = RESULTS / "hubert"
    out.mkdir(parents=True, exist_ok=True)
    predictions_table([arm]).to_csv(out / "c2_predictions.csv.gz", index=False)
    interval = bootstrap_metric(
        arm.y_true,
        arm.predictions,
        arm.groups,
        metric="uar",
        n_resamples=10_000,
        seed=seeds["bootstrap"],
    )
    shuffles = Parallel(n_jobs=jobs)(
        delayed(shuffled_uar)(table, CHOSEN_LAYERS, config, seeds["negative_control"] + k)
        for k in range(N_SHUFFLES)
    )
    write_json(
        out / "c2_summary.json",
        {
            **arm.row(),
            "uar_interval": interval.as_dict(),
            "shuffled_mean_uar": float(np.mean(shuffles)),
            "shuffled_uar": shuffles,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=["surgery", "hubert-layers", "hubert-c1", "hubert-c2"])
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--draws", type=int, help="hubert-c1: only the first N of the 50 draws")
    args = parser.parse_args()
    try:
        verify_freeze()
    except FreezeError as err:
        sys.exit(str(err))
    plan = load_plan()
    started = time.perf_counter()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        if args.what == "surgery":
            run_surgery(plan, args.jobs)
        elif args.what == "hubert-layers":
            run_hubert_layers(plan, args.jobs)
        elif args.what == "hubert-c1":
            run_hubert_c1(plan, args.jobs, args.draws)
        else:
            run_hubert_c2(plan, args.jobs)
    print(f"{args.what} done in {time.perf_counter() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
