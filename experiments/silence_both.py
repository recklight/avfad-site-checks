"""The simulated second place with silence on both sides: A padded to HSA's share, B to ESSUA's.

Exploratory, added after the handcrafted results of ``simulated_place``. Definitions and readings
are in experiments/deviations.md (2026-11-10), written before this ran. Same people, draws,
detectors and padding as there, except that every file is padded, the patients' too, so (ii) is
fitted again.

    features handcrafted|hubert
    c1 handcrafted|free|hubert
    place
    collect                      writes results/simulated_place/silence_both/summary.json

    .venv/Scripts/python.exe -I -X utf8 -B experiments/silence_both.py c1 handcrafted --jobs 12
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
from voice_severity_bench import models as vsb_models  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "features"))

import phonation_c1 as pc  # noqa: E402
import pool_training as pt  # noqa: E402
import published_splits as ps  # noqa: E402
import run_first_batch as batch  # noqa: E402
import simulated_place as sim  # noqa: E402
from arms import layer_columns, write_json  # noqa: E402
from common import cut  # noqa: E402
from extract_handcrafted import FEATURE_COLUMNS, features_of  # noqa: E402
from freeze import verify_freeze  # noqa: E402
from harness import load_plan, make_config  # noqa: E402
from sensitivity_avfad import CHOSEN_LAYERS  # noqa: E402

RESULTS = sim.RESULTS / "silence_both"


def shares(measured: dict) -> dict[str, float]:
    """Silence share of each site: A as HSA's normals, B as ESSUA's."""
    return {
        "A": measured["silence_share"]["HSA"],
        "B": measured["silence_share"]["ESSUA"],
    }


def padded(file_id: str, session: int, share: float) -> np.ndarray:
    """The file padded with its session's iau background; same start as ``simulated_place``."""
    quiet = sim.quiet_spans(session)
    if quiet.empty:
        raise RuntimeError(f"session {session}: no quiet span in the iau file")
    files = sim.listing("files")
    signal, rate = sim.load_signal(files.loc[f"svd__{session}-iau", "wav"])
    background = cut(signal, rate, list(zip(quiet["start_s"], quiet["end_s"], strict=True)))
    start = int(
        np.random.default_rng([sim.SEED, session]).integers(sum(p.size for p in background))
    )
    return sim.pad_with_background(sim.recorded(file_id), share, background, start)


def _features(kind: str, file_id: str, session: int, share: float):
    y = padded(file_id, session, share)
    if kind == "hubert":
        import extract_hubert as eh

        pooled, _ = eh.embed([y])
        return file_id, np.concatenate([pooled[layer] for layer in CHOSEN_LAYERS])
    return file_id, features_of([y], sim.RATE)


def features(kind: str, jobs: int) -> None:
    verify_freeze()
    who = sim.load_people()
    share = shares(sim.load_measured())
    start = time.perf_counter()
    done = Parallel(n_jobs=jobs)(
        delayed(_features)(kind, r.file_id, int(r.session), share[r.site]) for r in who.itertuples()
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez(
        RESULTS / f"features_{kind}.npz",
        file_id=np.array([f for f, _ in done]),
        values=np.stack([v for _, v in done]).astype(np.float64),
    )
    print(f"{len(done)} files in {time.perf_counter() - start:.0f} s", flush=True)


_TABLES: dict = {}


def feature_table(kind: str) -> pd.DataFrame:
    if kind not in _TABLES:
        data = np.load(RESULTS / f"features_{kind}.npz")
        columns = sim.HUBERT_COLUMNS if kind == "hubert" else list(FEATURE_COLUMNS)
        table = pd.DataFrame(data["values"], columns=columns)
        table.insert(0, "file_id", data["file_id"].astype(str))
        _TABLES[kind] = table
    return _TABLES[kind]


def _fit(detector: str, name: str, members: pd.DataFrame, repeat: int, config):
    start = time.perf_counter()
    feats = feature_table("hubert" if detector == "hubert" else "handcrafted")
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
        blocks, columns, tuned = None, sim.COLUMNS[detector], config
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


def path_of(detector: str) -> Path:
    return RESULTS / f"c1_{detector}.csv.gz"


def c1(detector: str, jobs: int) -> None:
    verify_freeze()
    plan = load_plan()
    config = make_config(plan, batch.LABEL_POSITIVE)
    sets = sim.arm_sets(plan, sim.load_people())
    todo = [(n, m, r) for n, m in sets for r in range(config.evaluation.n_repeats)]
    start = time.perf_counter()
    done = Parallel(n_jobs=jobs, return_as="generator_unordered")(
        delayed(_fit)(detector, n, m, r, config) for n, m, r in todo
    )
    frames = []
    for count, (frame, seconds) in enumerate(done, start=1):
        frames.append(frame)
        if count % 100 == 0 or count == len(todo):
            print(
                f"{detector}: {count} of {len(todo)} after {time.perf_counter() - start:.0f} s "
                f"(last {seconds:.0f} s)",
                flush=True,
            )
    out = pd.concat(frames).sort_values(["arm", "speaker", "repeat"])
    out.to_csv(path_of(detector), index=False)


def _place(detector: str, repeat: int):
    who = sim.load_people()
    kind = "hubert" if detector == "hubert" else "handcrafted"
    table = who.merge(feature_table(kind), on="file_id", how="left", validate="one_to_one")
    columns = sim.HUBERT_COLUMNS if detector == "hubert" else sim.COLUMNS[detector]
    values = table[columns].to_numpy(dtype=float)
    normal = (table["label"] == "normal").to_numpy()
    score = sim.place_scores(values, table["site"].to_numpy(), normal, repeat)
    return detector, table["file_id"].to_numpy(), score


def place(jobs: int) -> None:
    verify_freeze()
    detectors = ["handcrafted", "free"]
    if (RESULTS / "features_hubert.npz").exists():
        detectors.append("hubert")
    todo = [(d, r) for d in detectors for r in range(sim.N_REPEATS)]
    done = Parallel(n_jobs=jobs)(delayed(_place)(d, r) for d, r in todo)
    rows = [pd.DataFrame({"detector": d, "file_id": ids, "score": s}) for d, ids, s in done]
    out = pd.concat(rows).groupby(["detector", "file_id"], as_index=False)["score"].mean()
    out.to_csv(RESULTS / "place_scores.csv.gz", index=False)


def reading(gap: dict, detector: str) -> str:
    text = sim.alone_reading(gap, detector)
    return {
        "alone it reaches half of AVFAD's gap or more": (
            "the measured silence difference alone reaches half of AVFAD's gap or more"
        ),
        "alone it makes a gap, less than half": "a gap, less than half",
        "alone it makes no gap": (
            "with silence on both sides, the measured difference makes no gap"
        ),
    }.get(text, text)


def collect() -> None:
    plan = load_plan()
    who = sim.load_people()
    earlier = json.loads((sim.RESULTS / "summary.json").read_text(encoding="utf-8"))
    scores_path = RESULTS / "place_scores.csv.gz"
    scores = pd.read_csv(scores_path) if scores_path.exists() else None
    rng = np.random.default_rng(sim.SEED)
    summary: dict = {"shares": shares(sim.load_measured())}
    for detector in sim.DETECTORS:
        record: dict = {}
        reference = "hubert" if detector == "hubert" else "handcrafted"
        if path_of(detector).exists():
            table = pd.read_csv(path_of(detector), dtype={"speaker": str})
            gap, _, extra = pc.gap_with_samples(table, plan)
            record.update({**extra, "gap_i_minus_ii": gap})
            if detector != "free":
                record["reading"] = reading(gap, reference)
        if scores is not None and (scores["detector"] == detector).any():
            mine = scores[scores["detector"] == detector][["file_id", "score"]]
            place_part = sim.place_record(mine, who, rng)
            if detector != "free":
                place_part["reading"] = sim.place_reading(place_part, reference)
            record["place"] = place_part
        only_b = earlier.get(detector, {}).get("silence_1x")
        if only_b:
            record["only_b_padded_gap"] = only_b["gap_i_minus_ii"]["difference"]
        if record:
            summary[detector] = record
    write_json(RESULTS / "summary.json", summary)
    for detector in sim.DETECTORS:
        r = summary.get(detector)
        if not r:
            continue
        line = detector
        if "gap_i_minus_ii" in r:
            g = r["gap_i_minus_ii"]
            line += (
                f" gap {100 * g['difference']:.1f} [{100 * g['ci_low']:.1f}, "
                f"{100 * g['ci_high']:.1f}] ii {r['ii']['uar']:.3f} i {r['i_uar_mean']:.3f}"
            )
        if "place" in r:
            p = r["place"]
            line += (
                f" | place {p['auc']:.3f} [{p['ci_low']:.3f}, {p['ci_high']:.3f}]"
                f" vis {p['visibility_a_vs_b']:.3f}"
            )
        print(line, "|", r.get("reading"), "| only B padded", r.get("only_b_padded_gap"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("features", "c1", "place", "collect"))
    parser.add_argument("detector", nargs="?", choices=sim.DETECTORS)
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()
    if args.step == "features":
        if args.detector not in ("handcrafted", "hubert"):
            parser.error("features needs handcrafted or hubert")
        features(args.detector, args.jobs)
    elif args.step == "c1":
        if args.detector is None:
            parser.error("c1 needs a detector")
        c1(args.detector, args.jobs)
    elif args.step == "place":
        place(args.jobs)
    else:
        collect()


if __name__ == "__main__":
    main()
