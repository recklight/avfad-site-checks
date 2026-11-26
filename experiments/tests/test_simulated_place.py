"""Recording changes and people of the simulated second place, on synthetic signals only."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "features"))

import simulated_place as sim  # noqa: E402
from extract_handcrafted import BAND_CENTRES_HZ  # noqa: E402

RATE = sim.RATE


def tone(seconds: float = 1.3, f0: float = 150.0, seed: int = 0) -> np.ndarray:
    t = np.arange(round(seconds * RATE)) / RATE
    rng = np.random.default_rng(seed)
    voice = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 20))
    return 0.1 * voice + 1e-4 * rng.standard_normal(t.size)


@pytest.mark.parametrize("rt60", [0.25, 0.5, 1.0])
def test_room_response_has_the_planned_direct_to_reverberant_ratio(rt60):
    h = sim.room_response(rt60)
    assert h.size == round(rt60 * RATE)
    assert h[0] == 1.0
    assert np.all(h[1 : round(sim.DIRECT_GAP_S * RATE)] == 0.0)
    drr = 10 * np.log10(1.0 / np.sum(h[1:] ** 2))
    assert drr == pytest.approx(sim.direct_to_reverberant_db(rt60), abs=1e-9)
    # Longer rooms are wetter.
    assert sim.direct_to_reverberant_db(1.0) < sim.direct_to_reverberant_db(0.25)


def test_reverb_keeps_length_and_level():
    x = tone()
    y = sim.reverberate(x, sim.room_response(0.5))
    assert y.size == x.size
    assert np.sqrt(np.mean(y**2)) == pytest.approx(np.sqrt(np.mean(x**2)), rel=1e-12)
    assert not np.allclose(y, x)


@pytest.mark.parametrize("share", [0.03, 0.06, 0.12])
def test_padding_is_the_planned_share_and_leaves_the_voice_in_the_middle(share):
    x = tone()
    background = [np.full(500, 0.01), np.full(700, -0.01)]
    y = sim.pad_with_background(x, share, background, start=123)
    added = y.size - x.size
    assert added / y.size == pytest.approx(share, abs=1.0 / y.size)
    head = added // 2
    assert np.array_equal(y[head : head + x.size], x)
    assert sim.pad_with_background(x, 0.0, background, 0) is x


@pytest.mark.parametrize("rise", [6.0, 12.0, 24.0])
def test_added_noise_lifts_a_white_floor_by_the_dose(rise):
    rng = np.random.default_rng(1)
    base = 10 ** (-50 / 20) * rng.standard_normal(RATE * 4)
    room = [rng.standard_normal(RATE * 3)]
    noise = sim.added_noise(base.size, sim.floor_db(base), rise, room, start=0)
    assert sim.floor_db(base + noise) - sim.floor_db(base) == pytest.approx(rise, abs=0.5)


def test_channel_filter_follows_the_band_gains():
    gains = 3.0 * np.sin(np.linspace(0, np.pi, BAND_CENTRES_HZ.size))
    h = sim.channel_filter(gains)
    assert h.size == sim.CHANNEL_TAPS
    freqs = np.fft.rfftfreq(1 << 14, 1 / RATE)
    response = 20 * np.log10(np.abs(np.fft.rfft(h, 1 << 14)))
    above = BAND_CENTRES_HZ >= 300
    at = np.interp(BAND_CENTRES_HZ[above], freqs, response)
    assert np.max(np.abs(at - gains[above])) < 0.5
    x = tone()
    y = sim.through_channel(x, sim.channel_filter(np.zeros(BAND_CENTRES_HZ.size)), -6.0)
    assert y.size == x.size
    assert np.max(np.abs(y - x * 10 ** (-6 / 20))) < 1e-3


def test_channel_gains_hold_the_lowest_bands():
    measured = {"band_difference_db": {c: float(i) for i, c in enumerate(sim.LTAS_COLUMNS)}}
    gains = sim.channel_gains(measured)
    first = int(np.argmax(BAND_CENTRES_HZ >= sim.CHANNEL_LOWEST_HZ))
    assert BAND_CENTRES_HZ[first] > 150 > BAND_CENTRES_HZ[first - 1]
    assert np.all(gains[:first] == gains[first])
    assert np.array_equal(gains[first:], np.arange(first, BAND_CENTRES_HZ.size))


def test_all_four_changes_in_order():
    x = tone()
    context = {
        "silence_share": 0.06,
        "background": [1e-3 * np.random.default_rng(2).standard_normal(4000)],
        "background_start": 0,
        "floor_db": -60.0,
        "room": [np.random.default_rng(3).standard_normal(8000)],
        "room_start": 5,
        "channel_gains": np.zeros(BAND_CENTRES_HZ.size),
        "gain_db": -1.0,
    }
    y = sim.changed(x, sim.CONDITIONS["all"], context)
    assert (y.size - x.size) / y.size == pytest.approx(0.06, abs=1e-3)
    assert sim.changed(x, sim.CONDITIONS["none"], context) is x
    for name, knobs in sim.CONDITIONS.items():
        assert set(knobs) <= {"rt60", "silence", "noise_db", "channel"}, name


def synthetic_meta(n_patients=500, n_normals=600, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_patients + n_normals
    return pd.DataFrame(
        {
            "file_id": [f"svd__{i}-a_n" for i in range(n)],
            "session": np.arange(n),
            "speaker": [f"svd{i:04d}" for i in range(n)],
            "label": ["pathological"] * n_patients + ["normal"] * n_normals,
            "age": rng.integers(18, 70, n).astype(float),
            "sex": rng.choice(["m", "w"], n),
            "year": 1999.0,
        }
    )


def test_people_and_draws(monkeypatch):
    monkeypatch.setattr(sim.placebo, "svd_meta", lambda: synthetic_meta())
    who = sim.people()
    counts = who["role"].value_counts().to_dict()
    assert counts == {"patient": sim.N_PATIENTS, "A": sim.N_A, "B": sim.N_B}
    assert who["speaker"].is_unique
    assert set(who.loc[who["role"] == "patient", "label"]) == {"pathological"}
    assert set(who.loc[who["role"] != "patient", "label"]) == {"normal"}
    assert set(who.loc[who["site"] == "B", "role"]) == {"B"}
    plan = {
        "statistics": {"seeds": {"c1_draws": 7}},
        "confirmatory_family": {"tests": {"C1": {"draws": 5}}},
    }
    sets = sim.arm_sets(plan, who)
    assert [n for n, _ in sets] == ["ii", "i_00", "i_01", "i_02", "i_03", "i_04"]
    patients = set(who.loc[who["role"] == "patient", "speaker"])
    for name, members in sets:
        normals = members[members["label"] == "normal"]
        assert len(normals) == sim.N_A
        assert patients <= set(members["speaker"])
        if name == "ii":
            assert set(normals["role"]) == {"A"}
    assert sim.people().equals(who)


def test_readings():
    assert sim.alone_reading(
        {"difference": 0.12, "ci_low": 0.05, "ci_high": 0.2}, "handcrafted"
    ) == ("alone it reaches half of AVFAD's gap or more")
    assert sim.alone_reading({"difference": 0.10, "ci_low": 0.05, "ci_high": 0.2}, "hubert") == (
        "alone it reaches half of AVFAD's gap or more"
    )
    assert sim.alone_reading(
        {"difference": 0.05, "ci_low": 0.01, "ci_high": 0.1}, "handcrafted"
    ) == ("alone it makes a gap, less than half")
    assert sim.alone_reading({"difference": 0.01, "ci_low": -0.02, "ci_high": 0.04}, "hubert") == (
        "alone it makes no gap"
    )
    assert sim.all_reading({"difference": 0.18, "ci_low": 0.1, "ci_high": 0.25}, "handcrafted") == (
        "all four together reach AVFAD's range"
    )
    assert sim.all_reading({"difference": 0.05, "ci_low": 0.0, "ci_high": 0.12}, "hubert") == (
        "all four together fall short of AVFAD"
    )
    assert sim.place_reading({"auc": 0.83, "ci_low": 0.8, "ci_high": 0.86}, "handcrafted") == (
        "as far as AVFAD's place detector"
    )
    assert sim.place_reading({"auc": 0.7, "ci_low": 0.65, "ci_high": 0.75}, "hubert") == (
        "short of AVFAD's place detector"
    )
    assert sim.free_reading({"difference": 0.10, "ci_low": 0.04, "ci_high": 0.16}).endswith(
        "AVFAD's 13.5 points"
    )
    assert sim.free_reading({"difference": 0.06, "ci_low": 0.01, "ci_high": 0.1}) == (
        "the 40 columns respond to reverb"
    )
    assert sim.free_reading({"difference": 0.01, "ci_low": -0.03, "ci_high": 0.05}) == (
        "the 40 columns do not respond to reverb"
    )
    assert sim.none_reading({"difference": 0.01, "ci_low": -0.03, "ci_high": 0.05}) == (
        "the design makes no gap by itself"
    )
