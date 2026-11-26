"""Frozen HuBERT with the ESSUA-minus-HSA direction of the normal speakers projected out.

Exploratory. Definitions and readings are in experiments/deviations.md (2026-11-10), written
before any erased score existed. With equal class means no linear predictor beats a constant
(Belrose et al. 2023, LEACE), so one direction per layer is enough to hide the mean
difference between the two places from a linear detector; whatever it learns beyond that is
not a mean shift. The directions use the place of the 356 AVFAD normal speakers only.

    check     held-out place AUC among the normal speakers, before and after, per layer
    svd       every cross-database training set on erased features, scored on SVD a_n
    avfad     C1 on erased features: (ii) and the 50 draws of (i)
    collect   the readings; writes results/site_erasure/summary.json

    .venv/Scripts/python.exe -I -X utf8 -B experiments/site_erasure.py svd --jobs 10
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "features"))

import cross_database as cd  # noqa: E402
import published_splits as ps  # noqa: E402
import run_first_batch as batch  # noqa: E402
import run_svd as svd  # noqa: E402
import vowel_fusion as vf  # noqa: E402
from arms import evaluate_layers, layer_columns, load_hubert, merge_features, write_json  # noqa: E402
from freeze import verify_freeze  # noqa: E402
from freeze_svd import verify_svd_freeze  # noqa: E402
from harness import load_plan, make_config, predictions_table  # noqa: E402
from sensitivity_avfad import CHOSEN_LAYERS  # noqa: E402

RESULTS = HERE / "results" / "site_erasure"
PLAIN_SVD = HERE / "results" / "cross_database" / "scores_hubert.csv.gz"
PLAIN_C1 = HERE / "results" / "sensitivity" / "hubert" / "c1_predictions.csv.gz"
SVD_SEED = 20261118
AVFAD_SEED = 20261120
N_SVD = 1000
CHECK_FOLDS, CHECK_REPEATS, CHECK_C = 5, 10, 0.01
# Readings, written before the run (deviations.md, 2026-11-10).
HELPS_POINTS = 0.05
GAP_POINTS = 0.10
II_LOSS_POINTS = 0.03
STILL_THERE_AUC = 0.75


# ------------------------------------------------------------------ the projection


def place_directions(
    values: dict[int, np.ndarray], essua: np.ndarray, hsa: np.ndarray
) -> dict[int, np.ndarray]:
    """Unit vector of the ESSUA-minus-HSA mean difference, one per layer."""
    directions = {}
    for layer, block in values.items():
        d = block[essua].mean(axis=0) - block[hsa].mean(axis=0)
        directions[layer] = d / np.linalg.norm(d)
    return directions


def project_out(block: np.ndarray, direction: np.ndarray) -> np.ndarray:
    return block - np.outer(block @ direction, direction)


def layer_blocks(table: pd.DataFrame) -> dict[int, np.ndarray]:
    return {layer: table[layer_columns(layer)].to_numpy() for layer in CHOSEN_LAYERS}


def erased(table: pd.DataFrame, directions: dict[int, np.ndarray]) -> pd.DataFrame:
    out = table.copy()
    for layer, block in layer_blocks(table).items():
        out[layer_columns(layer)] = project_out(block, directions[layer])
    return out


def normal_speakers() -> pd.DataFrame:
    speakers = batch.load_speakers()
    normal = speakers[(speakers["label"] == "normal") & speakers["place"].isin(["ESSUA", "HSA"])]
    return normal.sort_values("file_id").reset_index(drop=True)


def fitted_directions(avfad: pd.DataFrame) -> dict[int, np.ndarray]:
    normal = normal_speakers().merge(avfad, on="file_id", how="left", validate="one_to_one")
    essua = (normal["place"] == "ESSUA").to_numpy()
    directions = place_directions(layer_blocks(normal), essua, ~essua)
    after = layer_blocks(erased(normal, directions))
    for layer, block in after.items():
        gap = np.abs(block[essua].mean(axis=0) - block[~essua].mean(axis=0)).max()
        if gap > 1e-8 * np.abs(block).max():
            raise RuntimeError(f"layer {layer}: means still differ by {gap}")
    return directions


def features() -> tuple[pd.DataFrame, pd.DataFrame, dict[int, np.ndarray]]:
    avfad = load_hubert("full", CHOSEN_LAYERS)
    test = load_hubert("full", CHOSEN_LAYERS, material="svd_a_n")
    directions = fitted_directions(avfad)
    return erased(avfad, directions), erased(test, directions), directions


# ------------------------------------------------------------------ steps


def check() -> dict:
    """Place AUC on held-out normal speakers; the direction is fitted on the training folds."""
    normal = normal_speakers().merge(
        load_hubert("full", CHOSEN_LAYERS), on="file_id", how="left", validate="one_to_one"
    )
    essua = (normal["place"] == "ESSUA").to_numpy()
    blocks = layer_blocks(normal)
    report = {}
    for layer, block in blocks.items():
        before, after = [], []
        for repeat in range(CHECK_REPEATS):
            folds = StratifiedKFold(CHECK_FOLDS, shuffle=True, random_state=AVFAD_SEED + repeat)
            for train, test in folds.split(block, essua):
                for erase, sink in ((False, before), (True, after)):
                    x_train, x_test = block[train], block[test]
                    if erase:
                        d = place_directions({0: x_train}, essua[train], ~essua[train])[0]
                        x_train, x_test = project_out(x_train, d), project_out(x_test, d)
                    model = make_pipeline(
                        StandardScaler(), LogisticRegression(C=CHECK_C, max_iter=5000)
                    )
                    model.fit(x_train, essua[train])
                    sink.append(roc_auc_score(essua[test], model.predict_proba(x_test)[:, 1]))
        report[f"l{layer:02d}"] = {"before": float(np.mean(before)), "after": float(np.mean(after))}
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / "check.json", report)
    return report


def run_svd(jobs: int) -> None:
    config = make_config(load_plan(), batch.LABEL_POSITIVE)
    avfad, svd_features, _ = features()
    columns = [c for layer in CHOSEN_LAYERS for c in layer_columns(layer)]
    blocks = {f"l{layer:02d}": layer_columns(layer) for layer in CHOSEN_LAYERS}
    test = cd.svd_test_set().merge(svd_features, on="file_id", how="left", validate="one_to_one")
    if test[columns[0]].isna().any():
        raise ValueError("an SVD speaker has no features")
    sets = {
        name: members.merge(avfad, on="file_id", how="left", validate="one_to_one")
        for name, members in cd.training_sets().items()
    }
    started = time.perf_counter()
    done = Parallel(n_jobs=jobs)(
        delayed(cd._one)(name, train, test, columns, blocks, config) for name, train in sets.items()
    )
    print(len(done), "training sets in", round(time.perf_counter() - started), "s", flush=True)
    frame = test[["file_id", "speaker", "label", "age", "sex", "year"]].copy()
    for name, scores in done:
        frame[name] = scores
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS / "scores_svd.csv.gz", index=False)


def _arm(name: str, table: pd.DataFrame, config):
    return evaluate_layers(name, table, CHOSEN_LAYERS, config)


def run_avfad(jobs: int) -> None:
    plan = load_plan()
    config = make_config(plan, batch.LABEL_POSITIVE)
    avfad, _, _ = features()
    sets = [(name, m) for name, m in vf.arm_sets(plan) if not name.startswith("shuffled")]
    started = time.perf_counter()
    arms = Parallel(n_jobs=jobs)(
        delayed(_arm)(name, merge_features(members, avfad, "label"), config)
        for name, members in sets
    )
    print(len(arms), "arms in", round(time.perf_counter() - started), "s", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    predictions_table(arms).to_csv(RESULTS / "c1_predictions.csv.gz", index=False)
    pd.DataFrame([a.row() for a in arms]).to_csv(RESULTS / "c1_arms.csv", index=False)


# ------------------------------------------------------------------ collect


def remedy_reading(change: dict) -> str:
    if change["ci_low"] <= 0 <= change["ci_high"]:
        return "does not help"
    if change["difference"] >= HELPS_POINTS and change["ci_low"] > 0:
        return "helps"
    return "small"


def mixed_reading(difference: dict) -> str:
    if difference["ci_high"] < 0:
        return "mixed training still loses"
    if difference["ci_low"] <= 0:
        return "mixed training no longer loses"
    return "mixed training wins"


def gap_reading(gap: dict) -> str:
    if gap["difference"] >= GAP_POINTS and gap["ci_low"] > 0:
        return "the gap stays"
    return "the gap is smaller; not evidence for the remedy (the erasure saw the test folds)"


def ii_loss(paired: dict) -> bool:
    return paired["difference"] <= -II_LOSS_POINTS and paired["ci_high"] < 0


def paired_svd(plain: pd.DataFrame, erased_scores: pd.DataFrame) -> dict:
    if not plain["file_id"].equals(erased_scores["file_id"]):
        raise ValueError("the two SVD score tables hold different rows")
    sick = (plain["label"] == "pathological").to_numpy()
    cells = svd.stratum(plain)
    resamples = cd.resample_indices(sick, cells, N_SVD, SVD_SEED)

    def contrast(a: np.ndarray, b: np.ndarray) -> dict:
        point = cd.auc(a, sick) - cd.auc(b, sick)
        samples = np.array([cd.auc(a[i], sick[i]) - cd.auc(b[i], sick[i]) for i in resamples])
        low, high = ps.interval(samples)
        return {"difference": float(point), "ci_low": low, "ci_high": high}

    e_all, p_all = erased_scores["all"].to_numpy(), plain["all"].to_numpy()
    e_ii, p_ii = erased_scores["ii"].to_numpy(), plain["ii"].to_numpy()
    return {
        "all_erased_minus_all_plain": contrast(e_all, p_all),
        "ii_erased_minus_ii_plain": contrast(e_ii, p_ii),
        "all_minus_ii_erased": contrast(e_all, e_ii),
        "all_minus_ii_plain": contrast(p_all, p_ii),
    }


def collect() -> dict:
    plan = load_plan()
    report: dict = {}
    if (RESULTS / "check.json").exists():
        report["check"] = json.loads((RESULTS / "check.json").read_text(encoding="utf-8"))
        best = max(v["after"] for v in report["check"].values())
        report["check_reading"] = (
            "one direction is not enough; the place stays in other directions"
            if best >= STILL_THERE_AUC
            else "little place information left for held-out speakers"
        )
    if (RESULTS / "scores_svd.csv.gz").exists():
        plain = pd.read_csv(PLAIN_SVD)
        scores = pd.read_csv(RESULTS / "scores_svd.csv.gz")
        record = cd.evaluate(scores, seed=SVD_SEED)
        record["reading"] = cd.reading(record["d_ext"]["auc"], record["d_ext"]["auc_ci"][0])
        record["paired"] = paired_svd(plain, scores)
        record["remedy"] = remedy_reading(record["paired"]["all_erased_minus_all_plain"])
        record["mixed"] = mixed_reading(record["paired"]["all_minus_ii_erased"])
        report["svd"] = record
    if (RESULTS / "c1_predictions.csv.gz").exists():
        table = pd.read_csv(RESULTS / "c1_predictions.csv.gz", dtype={"speaker": str})
        names = sorted(set(table["arm"]))
        arms = {name: vf.arm_from_table(table, name) for name in names}
        draws = [
            arms[f"i_{i:02d}"] for i in range(plan["confirmatory_family"]["tests"]["C1"]["draws"])
        ]
        gap = batch.c1_bootstrap(
            draws, arms["ii"], vf.N_RESAMPLES, plan["statistics"]["seeds"]["bootstrap"]
        )
        gap.pop("samples")
        plain_ii = vf.arm_from_table(pd.read_csv(PLAIN_C1, dtype={"speaker": str}), "ii")
        paired = vf.paired_difference(arms["ii"], plain_ii, "uar", AVFAD_SEED)
        report["avfad"] = {
            "ii": {"uar": arms["ii"].uar, "roc_auc": arms["ii"].roc_auc},
            "i": {"uar_mean": float(np.mean([a.uar for a in draws]))},
            "gap_i_minus_ii": gap,
            "gap_reading": gap_reading(gap),
            "ii_erased_minus_plain": paired,
            "ii_loses_pathology": ii_loss(paired),
        }
    write_json(RESULTS / "summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("check", "svd", "avfad", "collect"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    verify_freeze()
    verify_svd_freeze()
    if args.step == "check":
        print(json.dumps(check(), indent=1))
    elif args.step == "svd":
        run_svd(args.jobs)
    elif args.step == "avfad":
        run_avfad(args.jobs)
    else:
        print(json.dumps(collect(), indent=1))


if __name__ == "__main__":
    main()
