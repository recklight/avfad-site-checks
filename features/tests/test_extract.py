"""Synthetic checks of the feature code; no recording is read."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import complement, cut, normalise_to_phonation, rms_dbfs, to_analysis_rate
from extract_handcrafted import (
    BAND_CENTRES_HZ,
    FEATURE_COLUMNS,
    LTAS_COLUMNS,
    band_bins,
    features_of,
    level_statistics,
    ltas_db,
    mfcc_statistics,
    spans_for,
)

RATE = 16_000


def tone(freq: float, seconds: float, amplitude: float = 0.1) -> np.ndarray:
    t = np.arange(round(seconds * RATE)) / RATE
    return amplitude * np.sin(2 * np.pi * freq * t)


def noise(seconds: float, amplitude: float = 0.01, seed: int = 0) -> np.ndarray:
    return amplitude * np.random.default_rng(seed).normal(size=round(seconds * RATE))


def test_column_count_is_the_planned_one():
    assert len(FEATURE_COLUMNS) == 78
    assert len(set(FEATURE_COLUMNS)) == 78
    assert BAND_CENTRES_HZ.size == 21
    assert LTAS_COLUMNS[0] == "ltas_0063" and LTAS_COLUMNS[-1] == "ltas_6310"


def test_every_band_holds_a_bin_at_16_khz():
    assert all(bins.size > 0 for bins in band_bins(RATE))


def test_bands_above_nyquist_are_empty_at_8_khz():
    sizes = [bins.size for bins in band_bins(8000)]
    assert sizes[-1] == 0 and sizes[0] > 0


def test_tone_lands_in_its_own_band():
    db = ltas_db([tone(1000.0, 1.0)], RATE)
    assert LTAS_COLUMNS[int(np.argmax(db))] == "ltas_1000"


def test_low_room_tone_lands_in_a_low_band():
    db = ltas_db([tone(123.0, 1.0)], RATE)
    assert LTAS_COLUMNS[int(np.argmax(db))] == "ltas_0126"


def test_level_of_a_known_sine():
    stats = level_statistics([tone(1000.0, 1.0, amplitude=0.1)], RATE)
    # RMS of a sine is amplitude / sqrt(2): -23.01 dBFS.
    assert stats[0] == pytest.approx(-23.01, abs=0.05)
    assert stats[1] < 0.1


def test_digital_zero_stays_finite():
    assert np.all(np.isfinite(features_of([np.zeros(RATE)], RATE)))


def test_samples_between_spans_do_not_matter():
    x = noise(2.0)
    spans = [(0.2, 0.8), (1.2, 1.8)]
    changed = x.copy()
    changed[round(0.8 * RATE) : round(1.2 * RATE)] = 0.5
    first = features_of(cut(x, RATE, spans), RATE)
    second = features_of(cut(changed, RATE, spans), RATE)
    assert np.array_equal(first, second)


def test_frames_do_not_cross_a_gap():
    # A level step across the gap would show in the deltas if the spans were joined.
    quiet, loud = noise(0.5, amplitude=0.001), noise(0.5, amplitude=0.1, seed=1)
    apart = mfcc_statistics([quiet, loud], RATE)
    joined = mfcc_statistics([np.concatenate([quiet, loud])], RATE)
    assert not np.allclose(apart, joined)
    delta_std_c0 = FEATURE_COLUMNS.index("dmfcc_std_00")
    assert apart[delta_std_c0] < joined[delta_std_c0]


def test_span_shorter_than_a_frame_adds_nothing():
    long_piece = noise(0.5)
    short_piece = noise(0.01, seed=2)  # 10 ms: shorter than every frame
    assert np.array_equal(
        features_of([long_piece, short_piece], RATE), features_of([long_piece], RATE)
    )


def test_only_short_spans_give_nan_not_a_crash():
    assert np.all(np.isnan(features_of([noise(0.01)], RATE)))


def test_normalise_puts_phonation_at_minus_26():
    x = np.concatenate([noise(1.0, amplitude=0.001), tone(200.0, 1.0, amplitude=0.3)])
    scaled, before = normalise_to_phonation(x, RATE, [(1.0, 2.0)])
    assert before == pytest.approx(rms_dbfs(x[RATE:]))
    assert rms_dbfs(scaled[RATE:]) == pytest.approx(-26.0, abs=1e-6)


def test_normalised_features_ignore_recording_gain():
    x = noise(1.0, amplitude=0.05)
    spans = [(0.0, 1.0)]
    a, _ = normalise_to_phonation(x, RATE, spans)
    b, _ = normalise_to_phonation(x * 3.0, RATE, spans)
    assert np.allclose(
        features_of(cut(a, RATE, spans), RATE), features_of(cut(b, RATE, spans), RATE)
    )


def test_original_level_keeps_recording_gain():
    x = noise(1.0, amplitude=0.05)
    a = features_of([x], RATE)
    b = features_of([x * 3.0], RATE)
    level = FEATURE_COLUMNS.index("level_mean")
    assert b[level] - a[level] == pytest.approx(20 * np.log10(3.0), abs=1e-6)


def test_normalise_needs_phonation():
    with pytest.raises(ValueError):
        normalise_to_phonation(noise(1.0), RATE, [])


def test_complement():
    assert complement([(1.0, 2.0), (3.0, 4.0)], 5.0) == [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)]
    assert complement([], 2.0) == [(0.0, 2.0)]
    assert complement([(0.0, 2.0)], 2.0) == []
    assert complement([(1.0, 3.0), (2.0, 4.0)], 4.0) == [(0.0, 1.0)]


def test_phonation_plus_breath_is_what_is_not_quiet():
    spans = {
        "nonphonation_raw": [(0.0, 1.0)],
        "digital_zero": [(4.0, 5.0)],
        "phonation": [(1.5, 3.0)],
    }
    assert spans_for("phonation_breath", 5.0, spans) == [(1.0, 4.0)]


def test_full_is_the_whole_file_and_unknown_kind_is_empty():
    assert spans_for("full", 3.0, {}) == [(0.0, 3.0)]
    assert spans_for("nonphonation_g100", 3.0, {}) == []


def test_resampling_keeps_duration_and_leaves_low_rates_alone():
    y, rate = to_analysis_rate(np.zeros(48_000), 48_000)
    assert rate == 16_000 and y.size == 16_000
    y, rate = to_analysis_rate(np.zeros(8_000), 8_000)
    assert rate == 8_000 and y.size == 8_000
