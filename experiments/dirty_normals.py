"""How unclean would HSA's normals have to be to give AVFAD's picture without any recording change?

Exploratory. Definitions and readings are in experiments/deviations.md (2026-11-10), written
before this ran and before any result of the simulated second place. The people are those of
``simulated_place`` and nobody's recording is changed. Of the 82 "A" normals, the first 21, 41
or 62 of a fixed random order are replaced by SVD patients outside the 329, labelled normal;
each replacement has the replaced normal's sex and the nearest age.

    extra                     handcrafted features of the replacements
    c1 handcrafted|hubert [n ...]
    place                     place-only detectors for every n
    collect                   writes results/dirty_normals/summary.json

    .venv/Scripts/python.exe -I -X utf8 -B experiments/dirty_normals.py c1 hubert 41 --jobs 12
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
from voice_severity_bench import models as vsb_models  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "features"))

import diagnosis_breakdown as db  # noqa: E402
import phonation_c1 as pc  # noqa: E402
import placebo_c1 as placebo  # noqa: E402
import pool_training as pt  # noqa: E402
import published_splits as ps  # noqa: E402
import run_first_batch as batch  # noqa: E402
import simulated_place as sim  # noqa: E402
from arms import layer_columns, load_hubert, write_json  # noqa: E402
from extract_handcrafted import FEATURE_COLUMNS, features_of  # noqa: E402
from freeze import verify_freeze  # noqa: E402
from harness import load_plan, make_config  # noqa: E402
from sensitivity_avfad import CHOSEN_LAYERS  # noqa: E402

RESULTS = HERE / "results" / "dirty_normals"
SEED = 20261158
DIRTY = (21, 41, 62)
# HuBERT in this order; whatever is left at the stop time is not started.
HUBERT_DIRTY = (41, 21, 62)
# AVFAD inside HSA, reported next to the simulation only (place_only/summary.json, first batch).
AVFAD_INSIDE = {"handcrafted": 0.513, "hubert": 0.643}
AVFAD_II_UAR = {"handcrafted": 0.553, "hubert": 0.621}


def replacements(who: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """A's fixed order and, for its first max(DIRTY), the patient that replaces each one."""
    rng = np.random.default_rng(SEED)
    a = who[who["role"] == "A"].sort_values("speaker")
    order = a.iloc[rng.permutation(len(a))].reset_index(drop=True)
    pool = meta[(meta["label"] == "pathological") & ~meta["speaker"].isin(who["speaker"])]
    pool = pool.sort_values("speaker").reset_index(drop=True)
    ties = np.random.default_rng([SEED, 1])
    used: set[int] = set()
    rows = []
    for _, normal in order.head(max(DIRTY)).iterrows():
        free = pool[(pool["sex"] == normal["sex"]) & ~pool.index.isin(used)]
        gap = (free["age"] - normal["age"]).abs()
        best = free[gap == gap.min()]
        chosen = best.index[int(ties.integers(len(best)))]
        used.add(chosen)
        rows.append({"replaced": normal["speaker"], **pool.loc[chosen].to_dict()})
    return pd.DataFrame(rows)


def with_dirty(who: pd.DataFrame, swaps: pd.DataFrame, n: int) -> pd.DataFrame:
    """``who`` with the first ``n`` swaps made; the newcomers are labelled normal and sit in A."""
    if n == 0:
        return who.assign(dirty=False)
    first = swaps.head(n)
    kept = who[~who["speaker"].isin(first["replaced"])].assign(dirty=False)
    newcomers = first.drop(columns="replaced").assign(
        label="normal", site="A", role="A", dirty=True
    )
    out = pd.concat([kept, newcomers[kept.columns]], ignore_index=True)
    return out.sort_values("file_id").reset_index(drop=True)


def load_swaps() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "swaps.csv", dtype={"speaker": str, "file_id": str})


def extra() -> None:
    verify_freeze()
    who = sim.load_people()
    swaps = replacements(who, placebo.svd_meta())
    RESULTS.mkdir(parents=True, exist_ok=True)
    swaps.to_csv(RESULTS / "swaps.csv", index=False)
    values = np.stack([features_of([sim.recorded(f)], sim.RATE) for f in swaps["file_id"]])
    np.savez(
        RESULTS / "handcrafted_extra.npz",
        file_id=swaps["file_id"].to_numpy(dtype=str),
        values=values,
    )
    replaced = who.set_index("speaker").loc[swaps["replaced"]]
    print(
        len(swaps),
        "swaps; age difference median",
        float(np.median(np.abs(swaps["age"].to_numpy() - replaced["age"].to_numpy()))),
        "max",
        float(np.max(np.abs(swaps["age"].to_numpy() - replaced["age"].to_numpy()))),
        "same sex",
        bool((swaps["sex"].to_numpy() == replaced["sex"].to_numpy()).all()),
    )


_TABLES: dict = {}


def feature_table(kind: str) -> pd.DataFrame:
    if kind not in _TABLES:
        if kind == "hubert":
            _TABLES[kind] = load_hubert("full", CHOSEN_LAYERS, material="svd_a_n")
        else:
            parts = []
            for path in (
                sim.FEATURES / "handcrafted_none.npz",
                RESULTS / "handcrafted_extra.npz",
            ):
                data = np.load(path)
                part = pd.DataFrame(data["values"], columns=FEATURE_COLUMNS)
                part.insert(0, "file_id", data["file_id"].astype(str))
                parts.append(part)
            _TABLES[kind] = pd.concat(parts, ignore_index=True)
    return _TABLES[kind]


def _fit(detector: str, name: str, members: pd.DataFrame, repeat: int, config):
    start = time.perf_counter()
    feats = feature_table(detector)
    table = members.merge(feats, on="file_id", how="left", validate="one_to_one")
    if table[feats.columns[1]].isna().any():
        raise RuntimeError("someone has no features")
    table = table.assign(
        path=table["file_id"], group=table["speaker"], label=table["label"].astype(str)
    )
    table = table.sort_values("file_id").reset_index(drop=True)
    if detector == "hubert":
        blocks = {f"l{layer:02d}": layer_columns(layer) for layer in CHOSEN_LAYERS}
        columns = sim.HUBERT_COLUMNS
        tuned = config.override({"model.block_selection": True})
    else:
        vsb_models.SEARCH_SPACES[pc.LOGISTIC] = {k: list(v) for k, v in pc.HANDCRAFTED_GRID.items()}
        blocks, columns, tuned = None, list(FEATURE_COLUMNS), config
    table = table.assign(fold=pt.fold_ids(table, config)[:, repeat])
    score = np.full(len(table), np.nan)
    predicted = np.empty(len(table), dtype=object)
    for k in range(config.evaluation.n_splits):
        output = ps.fit_listed_fold(table, columns, blocks, tuned, k)
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


def path_of(detector: str, n: int) -> Path:
    return RESULTS / f"c1_{detector}_{n:02d}.csv.gz"


def c1(detector: str, counts: list[int], jobs: int, stop_at: str | None) -> None:
    verify_freeze()
    plan = load_plan()
    config = make_config(plan, batch.LABEL_POSITIVE)
    who, swaps = sim.load_people(), load_swaps()
    for n in counts:
        if path_of(detector, n).exists():
            print(detector, n, "already done", flush=True)
            continue
        if stop_at and time.strftime("%Y-%m-%d %H:%M") >= stop_at:
            print(detector, n, f"not started: past {stop_at}", flush=True)
            break
        sets = sim.arm_sets(plan, with_dirty(who, swaps, n))
        todo = [(name, m, r) for name, m in sets for r in range(config.evaluation.n_repeats)]
        start = time.perf_counter()
        done = Parallel(n_jobs=jobs, return_as="generator_unordered")(
            delayed(_fit)(detector, name, m, r, config) for name, m, r in todo
        )
        frames = []
        for count, (frame, seconds) in enumerate(done, start=1):
            frames.append(frame)
            if count % 100 == 0 or count == len(todo):
                print(
                    f"{detector} {n}: {count} of {len(todo)} after "
                    f"{time.perf_counter() - start:.0f} s (last {seconds:.0f} s)",
                    flush=True,
                )
        out = pd.concat(frames).sort_values(["arm", "speaker", "repeat"])
        out.to_csv(path_of(detector, n), index=False)


def _place(detector: str, n: int, repeat: int):
    who = with_dirty(sim.load_people(), load_swaps(), n)
    table = who.merge(feature_table(detector), on="file_id", how="left", validate="one_to_one")
    columns = sim.HUBERT_COLUMNS if detector == "hubert" else list(FEATURE_COLUMNS)
    values = table[columns].to_numpy(dtype=float)
    normal = (table["label"] == "normal").to_numpy()
    score = sim.place_scores(values, table["site"].to_numpy(), normal, repeat)
    return detector, n, table["file_id"].to_numpy(), score


def place(jobs: int) -> None:
    verify_freeze()
    todo = [
        (d, n, r) for d in ("handcrafted", "hubert") for n in DIRTY for r in range(sim.N_REPEATS)
    ]
    start = time.perf_counter()
    done = Parallel(n_jobs=jobs)(delayed(_place)(d, n, r) for d, n, r in todo)
    rows = [
        pd.DataFrame({"detector": d, "dirty": n, "file_id": ids, "score": s})
        for d, n, ids, s in done
    ]
    out = pd.concat(rows).groupby(["detector", "dirty", "file_id"], as_index=False)["score"]
    out.mean().to_csv(RESULTS / "place_scores.csv.gz", index=False)
    print(f"{len(todo)} fits in {time.perf_counter() - start:.0f} s")


def gap_reading(gap: dict, detector: str) -> str:
    _, low, _ = sim.AVFAD_GAP[detector]
    if gap["difference"] >= low:
        return "this unclean, the normals alone make AVFAD's gap"
    return "this unclean, the normals alone do not make AVFAD's gap"


def place_reading(record: dict, detector: str) -> str:
    _, low, _ = sim.AVFAD_PLACE[detector]
    if record["auc"] >= low:
        return "this unclean, the place detector reaches AVFAD's"
    return "this unclean, the place detector does not reach AVFAD's"


def overall_reading(both: dict[int, bool]) -> str:
    reached = sorted(n for n, ok in both.items() if ok)
    if reached:
        return f"without a recording change, at least {reached[0]} of HSA's 82 normals"
    if max(DIRTY) in both:
        return (
            "three quarters of HSA's normals being patients does not give AVFAD's picture "
            "without a recording change"
        )
    return "as measured"


def place_record(scores: pd.DataFrame, who: pd.DataFrame, rng) -> dict:
    table = who.merge(scores, on="file_id", validate="one_to_one")
    score = table["score"].to_numpy()
    sick = (table["label"] == "pathological").to_numpy()
    record = sim.db_auc(score, sick, rng)
    a = (table["site"] == "A").to_numpy()
    record["inside_a"] = float(db.comparisons(score[sick & a], score[~sick & a]).mean())
    dirty = table["dirty"].to_numpy(dtype=bool)
    clean_a = a & ~sick & ~dirty
    if dirty.any():
        record["dirty_vs_clean_a"] = float(db.comparisons(score[dirty], score[clean_a]).mean())
    return record


def collect() -> None:
    plan = load_plan()
    who, swaps = sim.load_people(), load_swaps()
    summary: dict = {}
    both: dict[str, dict[int, bool]] = {"handcrafted": {}, "hubert": {}}
    scores_path = RESULTS / "place_scores.csv.gz"
    scores = pd.read_csv(scores_path) if scores_path.exists() else None
    rng = np.random.default_rng(SEED)
    for detector in ("handcrafted", "hubert"):
        part: dict = {}
        for n in DIRTY:
            record: dict = {}
            if path_of(detector, n).exists():
                table = pd.read_csv(path_of(detector, n), dtype={"speaker": str})
                gap, _, extra_ = pc.gap_with_samples(table, plan)
                record.update({**extra_, "gap_i_minus_ii": gap})
                record["gap_reading"] = gap_reading(gap, detector)
            if scores is not None:
                mine = scores[(scores["detector"] == detector) & (scores["dirty"] == n)]
                if len(mine):
                    place_part = place_record(
                        mine[["file_id", "score"]], with_dirty(who, swaps, n), rng
                    )
                    place_part["reading"] = place_reading(place_part, detector)
                    record["place"] = place_part
            if "gap_i_minus_ii" in record and "place" in record:
                both[detector][n] = (
                    record["gap_i_minus_ii"]["difference"] >= sim.AVFAD_GAP[detector][1]
                    and record["place"]["auc"] >= sim.AVFAD_PLACE[detector][1]
                )
            if record:
                part[str(n)] = record
        part["overall"] = overall_reading(both[detector])
        part["avfad"] = {
            "inside_hsa_place_auc": AVFAD_INSIDE[detector],
            "ii_uar": AVFAD_II_UAR[detector],
        }
        summary[detector] = part
    write_json(RESULTS / "summary.json", summary)
    for detector in ("handcrafted", "hubert"):
        for n in DIRTY:
            r = summary[detector].get(str(n))
            if not r:
                continue
            g = r.get("gap_i_minus_ii")
            p = r.get("place")
            line = f"{detector:11s} {n:2d} dirty"
            if g:
                line += (
                    f" | gap {100 * g['difference']:5.1f} [{100 * g['ci_low']:5.1f}, "
                    f"{100 * g['ci_high']:5.1f}] ii {r['ii']['uar']:.3f}"
                )
            if p:
                line += (
                    f" | place {p['auc']:.3f} [{p['ci_low']:.3f}, {p['ci_high']:.3f}]"
                    f" inside A {p['inside_a']:.3f} dirty-vs-clean {p['dirty_vs_clean_a']:.3f}"
                )
            print(line)
        print(detector, summary[detector]["overall"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("extra", "c1", "place", "collect"))
    parser.add_argument("detector", nargs="?", choices=("handcrafted", "hubert"))
    parser.add_argument("counts", nargs="*", type=int)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--stop-at", help="c1: start no count after this local time")
    args = parser.parse_args()
    if args.step == "extra":
        extra()
    elif args.step == "c1":
        if args.detector is None:
            parser.error("c1 needs a detector")
        default = HUBERT_DIRTY if args.detector == "hubert" else DIRTY
        c1(args.detector, list(args.counts or default), args.jobs, args.stop_at)
    elif args.step == "place":
        place(args.jobs)
    else:
        collect()


if __name__ == "__main__":
    main()
