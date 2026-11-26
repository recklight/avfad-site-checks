"""A second place simulated on SVD a_n: which recording difference alone makes a C1 gap?

Exploratory. Definitions and readings are in experiments/deviations.md (2026-11-10), written
before this ran. SVD is one place. 329 patients and 82 "A" normals stay as they are; 274 "B"
normals get one recording change at a time, at three doses, and then all four together. The
numbers of people follow AVFAD (HSA patients, HSA normals, ESSUA normals). Each condition gets
the C1 design ((ii) = patients + A, (i) = patients + 82 drawn from A and B, 50 draws) and the
place-only detector (fitted on the A and B normals only).

    measure                      AVFAD's measured differences, the people and the draws
    check                        re-extraction against the stored SVD features
    features handcrafted|hubert  features of every condition (B files only where changed)
    c1 handcrafted|free|hubert [condition ...]
    place                        place-only detectors of every condition with features
    collect                      writes results/simulated_place/summary.json

    .venv/Scripts/python.exe -I -X utf8 -B experiments/simulated_place.py c1 hubert all --jobs 12
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
from scipy.signal import fftconvolve, firwin2  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "features"))
sys.path.insert(0, str(ROOT / "s0_audit"))

import channel_free_c1 as cf  # noqa: E402
import diagnosis_breakdown as db  # noqa: E402
import exploratory_avfad as ex  # noqa: E402
import phonation_c1 as pc  # noqa: E402
import placebo_c1 as placebo  # noqa: E402
import pool_training as pt  # noqa: E402
import published_splits as ps  # noqa: E402
import run_first_batch as batch  # noqa: E402
from arms import layer_columns, load_hubert, write_json  # noqa: E402
from common import cut, load_signal, normalise_to_phonation  # noqa: E402
from extract_handcrafted import (  # noqa: E402
    BAND_CENTRES_HZ,
    FEATURE_COLUMNS,
    LTAS_COLUMNS,
    features_of,
)
from freeze import verify_freeze  # noqa: E402
from harness import load_plan, make_config, match_cells  # noqa: E402
from sensitivity_avfad import CHOSEN_LAYERS  # noqa: E402
from stats import frame_energy_db  # noqa: E402

RESULTS = HERE / "results" / "simulated_place"
FEATURES = RESULTS / "features"
S0_OUT = ROOT / "s0_audit" / "out"
S1_OUT = ROOT / "s1_segments" / "out"
SEED = 20261157
RATE = 16_000
N_PATIENTS = 329
N_A = 82
N_B = 274
N_PARTS = 5
N_REPEATS = 10
N_RESAMPLES = 10_000
# Reverb: statistical room model, 50 m3 room, microphone 0.3 m from the mouth.
ROOM_VOLUME_M3 = 50.0
MIC_DISTANCE_M = 0.3
DIRECT_GAP_S = 0.0025
# Channel: below this the phonation LTAS difference follows the noise floor, which (a) covers.
CHANNEL_LOWEST_HZ = 150.0
CHANNEL_TAPS = 257

CONDITIONS: dict[str, dict[str, float]] = {
    "none": {},
    "noise_6": {"noise_db": 6.0},
    "noise_12": {"noise_db": 12.0},
    "noise_24": {"noise_db": 24.0},
    "silence_0.5x": {"silence": 0.5},
    "silence_1x": {"silence": 1.0},
    "silence_2x": {"silence": 2.0},
    "reverb_0.25": {"rt60": 0.25},
    "reverb_0.5": {"rt60": 0.5},
    "reverb_1.0": {"rt60": 1.0},
    "channel_0.5x": {"channel": 0.5},
    "channel_1x": {"channel": 1.0},
    "channel_2x": {"channel": 2.0},
    "all": {"rt60": 0.5, "silence": 1.0, "noise_db": 12.0, "channel": 1.0},
}
MIDDLE = {
    "noise": "noise_12",
    "silence": "silence_1x",
    "reverb": "reverb_0.5",
    "channel": "channel_1x",
}
# HuBERT runs in this order; whatever is left at 07:30 is not started.
HUBERT_CONDITIONS = (
    "all",
    "noise_12",
    "reverb_0.5",
    "none",
    "channel_1x",
    "silence_1x",
)
DETECTORS = ("handcrafted", "free", "hubert")
COLUMNS = {"handcrafted": list(FEATURE_COLUMNS), "free": list(cf.FREE)}
HUBERT_COLUMNS = [c for layer in CHOSEN_LAYERS for c in layer_columns(layer)]
# AVFAD references, for the readings only (summary and deviations.md, 2026-11-10).
AVFAD_GAP = {
    "handcrafted": (0.2217, 0.1748, 0.2710),
    "hubert": (0.1797, 0.1345, 0.2277),
}
AVFAD_PLACE = {"handcrafted": (0.853, 0.823, 0.882), "hubert": (0.907, 0.883, 0.928)}
AVFAD_FREE_GAP = (0.135, 0.089, 0.181)
# Readings, written before the run (deviations.md, 2026-11-10).
FREE_POINTS = 0.05
CHECK_VISIBILITY = 0.60


# ------------------------------------------------------------------ AVFAD's measured differences


def measure_avfad() -> dict:
    """Silence share, phonation spectrum and phonation level: ESSUA normals minus HSA normals."""
    speakers = batch.load_speakers()
    normals = speakers[speakers["label"] == "normal"]
    files = pd.read_csv(S1_OUT / "files.csv", dtype={"file_id": str})
    files = files[(files["database"] == "avfad") & (files["material"] == "a")]
    table = normals.merge(files, on="file_id", how="left", validate="one_to_one")
    share = 1.0 - table["phonation_s"] / table["duration_s"]
    phonation = normals.merge(batch.load_features("phonation_norm"), on="file_id")
    essua = phonation["place"] == "ESSUA"
    hsa = phonation["place"] == "HSA"
    spectrum = phonation.loc[essua, LTAS_COLUMNS].mean() - phonation.loc[hsa, LTAS_COLUMNS].mean()
    level = (
        phonation.loc[essua, "phonation_rms_dbfs"].median()
        - phonation.loc[hsa, "phonation_rms_dbfs"].median()
    )
    return {
        "silence_share": {
            "ESSUA": float(share[table["place"] == "ESSUA"].median()),
            "HSA": float(share[table["place"] == "HSA"].median()),
            "difference": float(
                share[table["place"] == "ESSUA"].median() - share[table["place"] == "HSA"].median()
            ),
        },
        "band_difference_db": {c: float(v) for c, v in spectrum.items()},
        "level_difference_db": float(level),
    }


def channel_gains(measured: dict) -> np.ndarray:
    """Band gains of the channel in dB; the lowest bands take the first band above 150 Hz."""
    gains = np.array([measured["band_difference_db"][c] for c in LTAS_COLUMNS], dtype=float)
    first = int(np.argmax(BAND_CENTRES_HZ >= CHANNEL_LOWEST_HZ))
    gains[:first] = gains[first]
    return gains


# ------------------------------------------------------------------ recording changes


def room_response(rt60: float, rate: int = RATE) -> np.ndarray:
    """Direct impulse, then exponentially decaying noise from 2.5 ms; tail energy from the DRR."""
    rng = np.random.default_rng([SEED, round(rt60 * 1000)])
    n = round(rt60 * rate)
    start = round(DIRECT_GAP_S * rate)
    t = np.arange(start, n) / rate
    tail = rng.standard_normal(t.size) * np.exp(-3.0 * np.log(10.0) * t / rt60)
    tail *= np.sqrt(10.0 ** (-direct_to_reverberant_db(rt60) / 10.0) / np.sum(tail**2))
    h = np.zeros(n)
    h[0] = 1.0
    h[start:] = tail
    return h


def direct_to_reverberant_db(rt60: float) -> float:
    critical = 0.057 * np.sqrt(ROOM_VOLUME_M3 / rt60)
    return float(20.0 * np.log10(critical / MIC_DISTANCE_M))


def reverberate(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Convolved, cut back to the input length and scaled back to the input RMS."""
    y = fftconvolve(x, h)[: x.size]
    return y * np.sqrt(np.mean(x**2) / np.mean(y**2))


def pad_with_background(
    x: np.ndarray, share: float, background: list[np.ndarray], start: int
) -> np.ndarray:
    """Half the padding before, half after, so the padding is ``share`` of the new file."""
    total = round(x.size * share / (1.0 - share))
    if total == 0:
        return x
    filler = ex.loop_to_length(background, total, start, RATE)
    head = total // 2
    return np.concatenate([filler[:head], x, filler[head:]])


def floor_db(x: np.ndarray, rate: int = RATE) -> float:
    """S0's floor: 10th percentile of the 30 ms frame energies."""
    return float(np.percentile(frame_energy_db(x, rate), 10))


def added_noise(
    n: int, own_floor_db: float, rise_db: float, room: list[np.ndarray], start: int
) -> np.ndarray:
    """Room noise at the level that lifts a floor of ``own_floor_db`` by ``rise_db``."""
    noise = ex.loop_to_length(room, n, start, RATE)
    target = own_floor_db + 10.0 * np.log10(10.0 ** (rise_db / 10.0) - 1.0)
    return noise * 10.0 ** ((target - floor_db(noise)) / 20.0)


def channel_filter(gains_db: np.ndarray, rate: int = RATE) -> np.ndarray:
    """Linear-phase FIR through the band gains, linear in log frequency between band centres."""
    freqs = np.linspace(0.0, rate / 2.0, 513)
    level = np.interp(np.log10(np.maximum(freqs, 1.0)), np.log10(BAND_CENTRES_HZ), gains_db)
    return firwin2(CHANNEL_TAPS, freqs, 10.0 ** (level / 20.0), fs=rate)


def through_channel(x: np.ndarray, h: np.ndarray, gain_db: float) -> np.ndarray:
    return np.convolve(x, h, mode="same") * 10.0 ** (gain_db / 20.0)


def changed(x: np.ndarray, knobs: dict[str, float], context: dict) -> np.ndarray:
    """Reverb, padding, noise, then channel and gain; ``context`` holds the session's choices."""
    y = x
    if "rt60" in knobs:
        y = reverberate(y, room_response(knobs["rt60"]))
    if "silence" in knobs:
        share = knobs["silence"] * context["silence_share"]
        y = pad_with_background(y, share, context["background"], context["background_start"])
    if "noise_db" in knobs:
        y = y + added_noise(
            y.size,
            context["floor_db"],
            knobs["noise_db"],
            context["room"],
            context["room_start"],
        )
    if "channel" in knobs:
        k = knobs["channel"]
        y = through_channel(y, channel_filter(k * context["channel_gains"]), k * context["gain_db"])
    return y


# ------------------------------------------------------------------ people and sessions


def people() -> pd.DataFrame:
    """Patients, A and B, drawn once from the SVD a_n analysis set."""
    meta = placebo.svd_meta()
    rng = np.random.default_rng(SEED)
    patients = meta[meta["label"] == "pathological"].sort_values("speaker")
    normals = meta[meta["label"] == "normal"].sort_values("speaker")
    patients = patients.iloc[np.sort(rng.choice(len(patients), N_PATIENTS, replace=False))]
    a = np.sort(rng.choice(len(normals), N_A, replace=False))
    rest = np.setdiff1d(np.arange(len(normals)), a)
    b = np.sort(rng.choice(rest, N_B, replace=False))
    out = pd.concat(
        [
            patients.assign(site="A", role="patient"),
            normals.iloc[a].assign(site="A", role="A"),
            normals.iloc[b].assign(site="B", role="B"),
        ]
    )
    return out.sort_values("file_id").reset_index(drop=True)


def arm_sets(plan: dict, who: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """(ii) and the 50 draws of (i); the same people in every condition."""
    patients = who[who["role"] == "patient"]
    a = who[who["role"] == "A"]
    pool = who[who["role"] != "patient"]
    rng = np.random.default_rng(plan["statistics"]["seeds"]["c1_draws"])
    sets = [("ii", pd.concat([patients, a]))]
    for index in range(plan["confirmatory_family"]["tests"]["C1"]["draws"]):
        drawn, cells = match_cells(a, pool, rng)
        if (cells["found"] < cells["wanted"]).any():
            raise RuntimeError("a cell of the pool came up short")
        sets.append((f"i_{index:02d}", pd.concat([patients, drawn])))
    return sets


_LISTS: dict[str, pd.DataFrame] = {}


def listing(name: str) -> pd.DataFrame:
    """S0 and S1 tables, read once per process."""
    if name not in _LISTS:
        if name == "stats":
            _LISTS[name] = pd.read_csv(S0_OUT / "svd_stats.csv")
        elif name == "files":
            files = pd.read_csv(S1_OUT / "files.csv", dtype={"file_id": str})
            _LISTS[name] = files.set_index("file_id")
        else:
            _LISTS[name] = pd.read_csv(S1_OUT / "spans.csv")
    return _LISTS[name]


def quiet_spans(session: int) -> pd.DataFrame:
    """Quiet spans of the session's iau file: g100 where there are any, else the raw ones."""
    spans = listing("spans")
    own = spans[spans["file_id"] == f"svd__{session}-iau"]
    quiet = own[own["kind"] == "nonphonation_g100"]
    return quiet if not quiet.empty else own[own["kind"] == "nonphonation_raw"]


def session_context(session: int, measured: dict, rooms: list) -> dict:
    """Everything a B file's changes need; the random choices depend on the session only."""
    rng = np.random.default_rng([SEED, session])
    stats = listing("stats")
    row = stats[(stats["session"] == session) & (stats["material"] == "iau")]
    if len(row) != 1:
        raise RuntimeError(f"session {session}: no iau floor")
    files = listing("files")
    iau = f"svd__{session}-iau"
    quiet = quiet_spans(session)
    if quiet.empty:
        raise RuntimeError(f"session {session}: no quiet span in the iau file")
    signal, rate = load_signal(files.loc[iau, "wav"])
    background = cut(signal, rate, list(zip(quiet["start_s"], quiet["end_s"], strict=True)))
    room_index = int(rng.integers(len(rooms)))
    room, _ = ex.room_quiet(*rooms[room_index])
    return {
        "floor_db": float(row["noise_floor_dbfs"].iloc[0]),
        "background": background,
        "background_start": int(rng.integers(sum(p.size for p in background))),
        "room": room,
        "room_file": rooms[room_index][0],
        "room_start": int(rng.integers(sum(p.size for p in room))),
        "silence_share": measured["silence_share"]["difference"],
        "channel_gains": channel_gains(measured),
        "gain_db": measured["level_difference_db"],
    }


def hsa_rooms() -> list:
    spans = pd.read_csv(S1_OUT / "spans.csv")
    quiet = spans[spans["kind"] == "nonphonation_g100"]
    return [
        (
            row.file_id,
            row.wav,
            list(
                zip(
                    quiet.loc[quiet["file_id"] == row.file_id, "start_s"],
                    quiet.loc[quiet["file_id"] == row.file_id, "end_s"],
                    strict=True,
                )
            ),
        )
        for row in ex.noise_files("HSA").itertuples()
    ]


# ------------------------------------------------------------------ steps: measure, check, features


def load_measured() -> dict:
    return json.loads((RESULTS / "measured.json").read_text(encoding="utf-8"))


def load_people() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "people.csv", dtype={"speaker": str, "file_id": str})


def measure() -> None:
    verify_freeze()
    plan = load_plan()
    who = people()
    sets = arm_sets(plan, who)
    RESULTS.mkdir(parents=True, exist_ok=True)
    measured = measure_avfad()
    measured["reverb_drr_db"] = {
        name: direct_to_reverberant_db(k["rt60"]) for name, k in CONDITIONS.items() if "rt60" in k
    }
    write_json(RESULTS / "measured.json", measured)
    who.to_csv(RESULTS / "people.csv", index=False)
    b_sessions = who.loc[who["role"] == "B", "session"].astype(int)
    quiet = {
        s: float((quiet_spans(s)["end_s"] - quiet_spans(s)["start_s"]).sum()) for s in b_sessions
    }
    stats = listing("stats")
    floors = stats[(stats["material"] == "iau") & stats["session"].isin(b_sessions)]
    if len(floors) != N_B or min(quiet.values()) <= 0:
        raise RuntimeError("a B session has no iau floor or no quiet span")
    b = set(who.loc[who["role"] == "B", "speaker"])
    shares = [float(m["speaker"].isin(b).sum()) / N_A for n, m in sets if n != "ii"]
    print(
        json.dumps(
            {
                "people": who["role"].value_counts().to_dict(),
                "b_share_in_draws": [
                    round(min(shares), 3),
                    round(float(np.mean(shares)), 3),
                ],
                "silence_share": measured["silence_share"],
                "level_difference_db": round(measured["level_difference_db"], 2),
                "b_quiet_s": [
                    round(min(quiet.values()), 2),
                    round(float(np.median(list(quiet.values()))), 2),
                ],
            }
        )
    )


def _check_hubert(file_id: str) -> float:
    import extract_hubert as eh

    pooled, _ = eh.embed([recorded(file_id)])
    stored = load_hubert("full", CHOSEN_LAYERS, material="svd_a_n").set_index("file_id")
    mine = np.concatenate([pooled[layer] for layer in CHOSEN_LAYERS])
    return float(np.abs(mine - stored.loc[file_id, HUBERT_COLUMNS].to_numpy()).max())


def check(jobs: int) -> None:
    """Untouched files through this script's path against the stored tables."""
    who = load_people()
    spans = listing("spans")
    chosen = list(who.sort_values("file_id")["file_id"].iloc[::34][:20])
    # Two of the files whose resampled length is one sample over the listed duration.
    chosen += ["svd__100-a_n", "svd__1003-a_n"]
    stored = pd.read_csv(ROOT / "features" / "out" / "svd_a_n_full_norm.csv").set_index("file_id")
    worst = 0.0
    for file_id in chosen:
        x = recorded(file_id)
        own = spans[(spans["file_id"] == file_id) & (spans["kind"] == "phonation")]
        scaled, _ = normalise_to_phonation(
            x, RATE, list(zip(own["start_s"], own["end_s"], strict=True))
        )
        values = features_of([scaled], RATE)
        worst = max(worst, float(np.abs(values - stored.loc[file_id, FEATURE_COLUMNS]).max()))
    print(f"handcrafted: {len(chosen)} files scaled as full_norm, largest difference {worst:.3g}")
    hubert = Parallel(n_jobs=min(jobs, 3))(delayed(_check_hubert)(f) for f in chosen[-3:])
    print(f"HuBERT: 3 files against the stored table, largest difference {max(hubert):.3g}")


def recorded(file_id: str) -> np.ndarray:
    """The a_n file as the stored features saw it: cut at the listed duration."""
    row = listing("files").loc[file_id]
    x, rate = load_signal(row["wav"])
    if rate != RATE:
        raise ValueError(f"{file_id}: expected {RATE} Hz, got {rate}")
    return cut(x, rate, [(0.0, float(row["duration_s"]))])[0]


def _session_features(detector, file_id, session, site, names, measured, rooms):
    x = recorded(file_id)
    out = {}
    context = session_context(session, measured, rooms) if site == "B" else None
    for name in names:
        if name != "none" and site != "B":
            continue
        y = x if name == "none" else changed(x, CONDITIONS[name], context)
        if detector == "hubert":
            import extract_hubert as eh

            pooled, _ = eh.embed([y])
            out[name] = np.concatenate([pooled[layer] for layer in CHOSEN_LAYERS])
        else:
            out[name] = features_of([y], RATE)
    return file_id, out, ({"room_file": context["room_file"]} if context else {})


def features(detector: str, jobs: int) -> None:
    verify_freeze()
    who = load_people()
    measured = load_measured()
    rooms = hsa_rooms()
    if detector == "hubert":
        names = [n for n in HUBERT_CONDITIONS if n != "none"]
        rows = who[who["role"] == "B"]
    else:
        names = list(CONDITIONS)
        rows = who
    FEATURES.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    done = Parallel(n_jobs=jobs)(
        delayed(_session_features)(
            detector,
            r.file_id,
            int(r.session),
            r.site,
            names,
            measured,
            rooms,
        )
        for r in rows.itertuples()
    )
    print(f"{len(done)} files in {time.perf_counter() - start:.0f} s", flush=True)
    rooms_used = pd.DataFrame(
        [{"file_id": f, "room_file": info["room_file"]} for f, _, info in done if info]
    )
    rooms_used.to_csv(FEATURES / "rooms_used.csv", index=False)
    for name in names:
        part = [(f, out[name]) for f, out, _ in done if name in out]
        ids = np.array([f for f, _ in part])
        values = np.stack([v for _, v in part]).astype(np.float64)
        np.savez(FEATURES / f"{detector}_{name}.npz", file_id=ids, values=values)


_TABLES: dict = {}


def condition_table(detector: str, condition: str) -> pd.DataFrame:
    """Features of all 685 people in one condition: B's rows changed, the rest as recorded."""
    key = (detector, condition)
    if key in _TABLES:
        return _TABLES[key]
    if detector == "hubert":
        who = load_people()
        base = load_hubert("full", CHOSEN_LAYERS, material="svd_a_n")
        base = base[base["file_id"].isin(who["file_id"])].set_index("file_id")
        columns = HUBERT_COLUMNS
    else:
        data = np.load(FEATURES / "handcrafted_none.npz")
        base = pd.DataFrame(data["values"], columns=FEATURE_COLUMNS, index=data["file_id"])
        columns = list(FEATURE_COLUMNS)
    if condition != "none":
        kind = "hubert" if detector == "hubert" else "handcrafted"
        data = np.load(FEATURES / f"{kind}_{condition}.npz")
        new = pd.DataFrame(data["values"], columns=columns, index=data["file_id"])
        base = base.copy()
        base.loc[new.index, columns] = new.to_numpy()
    table = base.rename_axis("file_id").reset_index()
    _TABLES.clear()
    _TABLES[key] = table
    return table


# ------------------------------------------------------------------ steps: C1 and place


def _fit(detector: str, condition: str, name: str, members: pd.DataFrame, repeat: int, config):
    start = time.perf_counter()
    features_kind = "hubert" if detector == "hubert" else "handcrafted"
    feats = condition_table(features_kind, condition)
    table = members.merge(feats, on="file_id", how="left", validate="one_to_one")
    if table[feats.columns[1]].isna().any():
        raise RuntimeError("someone has no features")
    table = table.assign(
        path=table["file_id"], group=table["speaker"], label=table["label"].astype(str)
    )
    table = table.sort_values("file_id").reset_index(drop=True)
    if detector == "hubert":
        blocks = {f"l{layer:02d}": layer_columns(layer) for layer in CHOSEN_LAYERS}
        columns = HUBERT_COLUMNS
        tuned = config.override({"model.block_selection": True})
    else:
        vsb_models.SEARCH_SPACES[pc.LOGISTIC] = {k: list(v) for k, v in pc.HANDCRAFTED_GRID.items()}
        blocks, columns, tuned = None, COLUMNS[detector], config
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


def path_of(detector: str, condition: str) -> Path:
    return RESULTS / f"c1_{detector}_{condition}.csv.gz"


def c1(detector: str, conditions: list[str], jobs: int, stop_at: str | None = None) -> None:
    verify_freeze()
    plan = load_plan()
    config = make_config(plan, batch.LABEL_POSITIVE)
    sets = arm_sets(plan, load_people())
    repeats = range(config.evaluation.n_repeats)
    if not path_of(detector, "ii").exists():
        # (ii) is the same in every condition: patients and A are never changed.
        frames = Parallel(n_jobs=jobs)(
            delayed(_fit)(detector, "none", "ii", sets[0][1], r, config) for r in repeats
        )
        pd.concat([f for f, _ in frames]).to_csv(path_of(detector, "ii"), index=False)
    for condition in conditions:
        if path_of(detector, condition).exists():
            print(detector, condition, "already done", flush=True)
            continue
        if stop_at and time.strftime("%Y-%m-%d %H:%M") >= stop_at:
            print(detector, condition, f"not started: past {stop_at}", flush=True)
            break
        todo = [(n, m, r) for n, m in sets[1:] for r in repeats]
        start = time.perf_counter()
        done = Parallel(n_jobs=jobs, return_as="generator_unordered")(
            delayed(_fit)(detector, condition, n, m, r, config) for n, m, r in todo
        )
        frames = []
        for count, (frame, seconds) in enumerate(done, start=1):
            frames.append(frame)
            if count % 100 == 0 or count == len(todo):
                print(
                    f"{detector} {condition}: {count} of {len(todo)} after "
                    f"{time.perf_counter() - start:.0f} s (last {seconds:.0f} s)",
                    flush=True,
                )
        out = pd.concat(frames).sort_values(["arm", "speaker", "repeat"])
        out.to_csv(path_of(detector, condition), index=False)


def place_scores(values: np.ndarray, site: np.ndarray, normal: np.ndarray, repeat: int):
    """A-vs-B decision values, every row scored by a fit on the other parts' normals."""
    strata = pd.Series(site).astype(str) + "_" + pd.Series(normal).astype(str)
    folds = StratifiedKFold(N_PARTS, shuffle=True, random_state=SEED + repeat)
    score = np.full(len(site), np.nan)
    for train_index, test_index in folds.split(values, strata):
        train = np.zeros(len(site), dtype=bool)
        train[train_index] = True
        train &= normal
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        clf.fit(values[train], site[train] == "A")
        score[test_index] = clf.decision_function(values[test_index])
    return score


def _place(detector: str, condition: str, repeat: int):
    kind = "hubert" if detector == "hubert" else "handcrafted"
    who = load_people()
    table = who.merge(condition_table(kind, condition), on="file_id", validate="one_to_one")
    columns = HUBERT_COLUMNS if detector == "hubert" else COLUMNS[detector]
    values = table[columns].to_numpy(dtype=float)
    normal = (table["label"] == "normal").to_numpy()
    return (
        detector,
        condition,
        table["file_id"].to_numpy(),
        place_scores(values, table["site"].to_numpy(), normal, repeat),
    )


def available(detector: str) -> list[str]:
    kind = "hubert" if detector == "hubert" else "handcrafted"
    names = HUBERT_CONDITIONS if detector == "hubert" else tuple(CONDITIONS)
    return [n for n in names if n == "none" or (FEATURES / f"{kind}_{n}.npz").exists()]


def place(jobs: int) -> None:
    verify_freeze()
    todo = [(d, c, r) for d in DETECTORS for c in available(d) for r in range(N_REPEATS)]
    start = time.perf_counter()
    done = Parallel(n_jobs=jobs)(delayed(_place)(d, c, r) for d, c, r in todo)
    rows = []
    for detector, condition, ids, score in done:
        rows.append(
            pd.DataFrame(
                {
                    "detector": detector,
                    "condition": condition,
                    "file_id": ids,
                    "score": score,
                }
            )
        )
    out = pd.concat(rows).groupby(["detector", "condition", "file_id"], as_index=False)["score"]
    out.mean().to_csv(RESULTS / "place_scores.csv.gz", index=False)
    print(f"{len(todo)} fits in {time.perf_counter() - start:.0f} s")


# ------------------------------------------------------------------ collect and readings


def alone_reading(gap: dict, detector: str) -> str:
    half = AVFAD_GAP["hubert" if detector == "hubert" else "handcrafted"][0] / 2.0
    if gap["difference"] >= half and gap["ci_low"] > 0:
        return "alone it reaches half of AVFAD's gap or more"
    if gap["ci_low"] > 0:
        return "alone it makes a gap, less than half"
    if gap["ci_low"] <= 0 <= gap["ci_high"]:
        return "alone it makes no gap"
    return "as measured"


def all_reading(gap: dict, detector: str) -> str:
    _, low, _ = AVFAD_GAP[detector]
    if gap["difference"] >= low:
        return "all four together reach AVFAD's range"
    if gap["ci_high"] < low:
        return "all four together fall short of AVFAD"
    return "as measured"


def place_reading(record: dict, detector: str) -> str:
    _, low, _ = AVFAD_PLACE[detector]
    if record["auc"] >= low:
        return "as far as AVFAD's place detector"
    if record["ci_high"] < low:
        return "short of AVFAD's place detector"
    return "as measured"


def free_reading(gap: dict) -> str:
    if gap["difference"] >= FREE_POINTS and gap["ci_low"] > 0:
        if gap["difference"] >= AVFAD_FREE_GAP[1]:
            return "the 40 columns respond to reverb; reverb alone reaches AVFAD's 13.5 points"
        return "the 40 columns respond to reverb"
    if gap["ci_low"] <= 0 <= gap["ci_high"]:
        return "the 40 columns do not respond to reverb"
    return "as measured"


def none_reading(gap: dict) -> str:
    if gap["ci_low"] <= 0 <= gap["ci_high"]:
        return "the design makes no gap by itself"
    return "the design makes a gap by itself: read the differences from dose 0"


def gap_of(detector: str, condition: str, plan: dict):
    table = pd.concat(
        [
            pd.read_csv(path_of(detector, "ii"), dtype={"speaker": str}),
            pd.read_csv(path_of(detector, condition), dtype={"speaker": str}),
        ]
    )
    return pc.gap_with_samples(table, plan)


def place_record(scores: pd.DataFrame, who: pd.DataFrame, rng) -> dict:
    table = who.merge(scores, on="file_id", validate="one_to_one")
    score = table["score"].to_numpy()
    sick = (table["label"] == "pathological").to_numpy()
    record = db_auc(score, sick, rng)
    normal = ~sick
    a = (table["site"] == "A").to_numpy()
    record["visibility_a_vs_b"] = float(
        db.comparisons(score[normal & a], score[normal & ~a]).mean()
    )
    record["ceiling"] = N_B / (N_A + N_B) + 0.5 * N_A / (N_A + N_B)
    return record


def db_auc(score: np.ndarray, sick: np.ndarray, rng) -> dict:
    kernel = db.comparisons(score[sick], score[~sick])
    w_sick = db.resample_counts(int(sick.sum()), N_RESAMPLES, rng)
    w_well = db.resample_counts(int((~sick).sum()), N_RESAMPLES, rng)
    low, high = np.percentile(db.weighted_auc(kernel, w_sick, w_well), [2.5, 97.5])
    return {"auc": float(kernel.mean()), "ci_low": float(low), "ci_high": float(high)}


def snr_of_noise(who: pd.DataFrame) -> dict:
    """Median phonation-minus-floor of the B files before and after each noise dose, in dB."""
    stats = pd.read_csv(S0_OUT / "svd_stats.csv")
    an = stats[stats["material"] == "a_n"].set_index("session")["rms_dbfs"]
    iau = stats[stats["material"] == "iau"].set_index("session")["noise_floor_dbfs"]
    b = who[who["role"] == "B"]["session"].astype(int)
    own = an.loc[b].to_numpy() - iau.loc[b].to_numpy()
    out = {"as_recorded": float(np.median(own))}
    for name, knobs in CONDITIONS.items():
        if set(knobs) == {"noise_db"}:
            out[name] = float(np.median(own - knobs["noise_db"]))
    return out


def collect() -> None:
    plan = load_plan()
    who = load_people()
    summary: dict = {"measured": load_measured(), "snr_db": snr_of_noise(who)}
    for detector in DETECTORS:
        if not path_of(detector, "ii").exists():
            continue
        part: dict = {}
        base_samples = None
        if path_of(detector, "none").exists():
            base, base_samples, _ = gap_of(detector, "none", plan)
        for condition in CONDITIONS:
            if not path_of(detector, condition).exists():
                continue
            gap, samples, extra = gap_of(detector, condition, plan)
            record = {**extra, "gap_i_minus_ii": gap}
            if base_samples is not None and condition != "none":
                diff = samples - base_samples
                low, high = np.quantile(diff[np.isfinite(diff)], [0.025, 0.975])
                record["minus_none"] = {
                    "difference": gap["difference"] - base["difference"],
                    "ci_low": float(low),
                    "ci_high": float(high),
                }
            part[condition] = record
        readings: dict = {}
        if "none" in part:
            readings["none"] = none_reading(part["none"]["gap_i_minus_ii"])
        reference = "hubert" if detector == "hubert" else "handcrafted"
        if detector != "free":
            for kind, name in MIDDLE.items():
                if name in part:
                    readings[name] = alone_reading(part[name]["gap_i_minus_ii"], reference)
            if "all" in part:
                readings["all"] = all_reading(part["all"]["gap_i_minus_ii"], reference)
        elif "reverb_0.5" in part:
            readings["reverb_0.5"] = free_reading(part["reverb_0.5"]["gap_i_minus_ii"])
        part["readings"] = readings
        summary[detector] = part
    scores_path = RESULTS / "place_scores.csv.gz"
    if scores_path.exists():
        scores = pd.read_csv(scores_path)
        rng = np.random.default_rng(SEED)
        place_part: dict = {}
        for (detector, condition), group in scores.groupby(["detector", "condition"], sort=False):
            record = place_record(group[["file_id", "score"]], who, rng)
            if detector != "free":
                ref = "hubert" if detector == "hubert" else "handcrafted"
                record["reading"] = place_reading(record, ref)
            place_part.setdefault(detector, {})[condition] = record
        free = place_part.get("free", {})
        if "channel_1x" in free:
            seen = free["channel_1x"]["visibility_a_vs_b"]
            free["check"] = {
                "visibility_channel_1x": seen,
                "passed": bool(seen <= CHECK_VISIBILITY),
            }
        summary["place"] = place_part
    write_json(RESULTS / "summary.json", summary)
    for detector in DETECTORS:
        for condition, record in summary.get(detector, {}).items():
            if condition == "readings":
                continue
            g = record["gap_i_minus_ii"]
            m = record.get("minus_none", {})
            pl = summary.get("place", {}).get(detector, {}).get(condition, {})
            print(
                f"{detector:11s} {condition:12s} gap {100 * g['difference']:6.1f} "
                f"[{100 * g['ci_low']:6.1f}, {100 * g['ci_high']:6.1f}]"
                + (f" | -none {100 * m['difference']:6.1f}" if m else "")
                + (
                    f" | place {pl['auc']:.3f} [{pl['ci_low']:.3f}, {pl['ci_high']:.3f}]"
                    f" vis {pl['visibility_a_vs_b']:.3f}"
                    if pl
                    else ""
                )
            )
        print(detector, summary.get(detector, {}).get("readings"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("measure", "check", "features", "c1", "place", "collect"))
    parser.add_argument("detector", nargs="?", choices=(*DETECTORS,))
    parser.add_argument("conditions", nargs="*")
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--stop-at", help="c1: start no condition after this local time")
    args = parser.parse_args()
    if args.step == "measure":
        measure()
    elif args.step == "check":
        check(args.jobs)
    elif args.step == "features":
        if args.detector not in ("handcrafted", "hubert"):
            parser.error("features needs handcrafted or hubert")
        features(args.detector, args.jobs)
    elif args.step == "c1":
        if args.detector is None:
            parser.error("c1 needs a detector")
        default = HUBERT_CONDITIONS if args.detector == "hubert" else tuple(CONDITIONS)
        c1(args.detector, list(args.conditions or default), args.jobs, args.stop_at)
    elif args.step == "place":
        place(args.jobs)
    else:
        collect()


if __name__ == "__main__":
    main()
