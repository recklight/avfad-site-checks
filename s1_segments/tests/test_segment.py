"""Synthetic recordings only. Run with the Kudio venv:

python -I -B -m pytest -q -p no:cacheprovider s1_segments/tests/test_segment.py
"""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import segment as sg
import synth

HERE = Path(__file__).resolve().parents[1]
RATE = 16000
ROOMS = list(synth.ROOMS)


def seg(rec, **kwargs):
    return sg.segment_signal(rec.x, rec.rate, **kwargs)


def spans(result, kind):
    return [(s["start_s"], s["end_s"]) for s in result.spans if s["kind"] == kind]


@pytest.mark.parametrize("room", ROOMS)
def test_boundaries_within_100_ms(room):
    rec = synth.recording("brief", room, RATE, seed=1)
    result = seg(rec)
    raw = spans(result, "nonphonation_raw")
    assert len(raw) == len(rec.pauses) == 3
    for (a, b), (ta, tb) in zip(raw, rec.pauses):
        assert abs(a - ta) < 0.1
        assert abs(b - tb) < 0.1
    phonation = spans(result, "phonation")
    assert len(phonation) == len(rec.phonation)
    for (a, b), (ta, tb) in zip(phonation, rec.phonation):
        assert abs(a - ta) < 0.1
        assert abs(b - tb) < 0.1


def test_the_rooms_give_similar_non_phonation():
    # S0: -68 dB broadband against -56 dB low-frequency, with and without a steady
    # tone. A threshold tied to the file's peak finds 0% of the pauses in the noisy
    # rooms and 96% in the quiet one. Here the gap is about 0.3 s out of 1.6 s: pYIN
    # still calls some band-edge noise voiced in the low-frequency room. That is
    # what availability.py watches per place on the real files.
    means = {}
    floors = {}
    for room in ROOMS:
        results = [seg(synth.recording("three_reps", room, RATE, seed=s)).summary for s in (3, 5)]
        means[room] = np.mean([r["nonphonation_g100_s"] for r in results])
        floors[room] = np.mean([r["floor_db"] for r in results])
    assert min(means.values()) > 0.75
    assert max(means.values()) - min(means.values()) < 0.45
    assert floors["quiet_broadband"] < floors["noisy_lowfreq"] - 5


@pytest.mark.parametrize("room", ROOMS)
@pytest.mark.parametrize("voice", ["modal", "tremor", "breathy", "breathy_tremor"])
def test_voice_without_pauses_has_no_non_phonation_after_the_guard(room, voice):
    rec = synth.recording("no_pause", room, RATE, seed=3, voice=voice)
    summary = seg(rec).summary
    assert summary["nonphonation_g100_s"] == 0.0
    assert summary["phonation_s"] > 0.9 * rec.duration_s


@pytest.mark.parametrize("room", ROOMS)
def test_breathy_phonation_stays_out_of_non_phonation(room):
    rec = synth.recording("brief", room, RATE, seed=4, voice="breathy")
    result = seg(rec)
    assert result.summary["phonation_s"] == pytest.approx(4.0, abs=0.3)
    times = result.tracks["times"]
    quiet = sg.spans_to_mask(spans(result, "nonphonation_raw"), times)
    inside = sg.spans_to_mask([(a + 0.05, b - 0.05) for a, b in rec.phonation], times)
    assert not (quiet & inside).any()


def test_quiet_voice_in_a_noisy_room_gives_no_non_phonation_rather_than_a_wrong_one():
    # Under 20 dB of range the file counts as one class. This costs usable files in a
    # noisy room, which availability.py reports per place.
    rec = synth.recording("brief", "noisy_lowfreq", RATE, seed=5, voice="quiet")
    summary = seg(rec).summary
    assert not summary["separable"]
    assert summary["nonphonation_raw_s"] == 0.0


@pytest.mark.parametrize("room", ROOMS)
def test_room_recording_is_non_phonation_only_when_declared_as_one(room):
    rec = synth.recording("silence_only", room, RATE, seed=6)
    control = seg(rec, contains_voice=False).summary
    assert control["nonphonation_raw_s"] > 0.95 * rec.duration_s
    assert control["phonation_s"] == 0.0
    assert seg(rec, contains_voice=True).summary["nonphonation_raw_s"] == 0.0


@pytest.mark.parametrize("room", ["noisy_lowfreq", "noisy_tonal"])
def test_band_limit_keeps_pyin_from_calling_room_hum_voiced(room):
    rec = synth.recording("silence_only", room, RATE, seed=7)
    assert seg(rec, contains_voice=False).summary["voiced_fraction"] < 0.05


@pytest.mark.parametrize("voice", ["vowel_i", "vowel_u", "vowel_u_low_f0", "low_f0"])
def test_voices_with_little_energy_above_the_band_edge_are_still_found(voice):
    # /i/ and /u/ in SVD's iau have F1 near 300 Hz, right on the lower band edge.
    rec = synth.recording("brief", "quiet_broadband", RATE, seed=12, voice=voice)
    summary = seg(rec).summary
    assert summary["phonation_s"] == pytest.approx(4.0, abs=0.3)
    assert summary["n_nonphonation_raw"] == 3


def test_digital_zeros_are_kept_apart_from_room_noise():
    # The VOICED layout: about 0.115 s of exact zeros, then phonation to the end.
    rate = 8000
    rec = synth.recording("no_pause", "quiet_broadband", rate, seed=8)
    x = rec.x.copy()[: 5 * rate]
    x[: round(0.115 * rate)] = 0.0
    summary = sg.segment_signal(x, rate).summary
    assert summary["digital_zero_s"] == pytest.approx(0.115, abs=0.002)
    assert summary["nonphonation_g100_s"] == 0.0
    assert summary["floor_db"] > -60
    assert summary["phonation_s"] > 4.5


@pytest.mark.parametrize("rate", [8000, 48000, 50000])
def test_every_database_rate_works(rate):
    rec = synth.recording("brief", "quiet_broadband", rate, seed=9)
    result = seg(rec)
    assert result.summary["rate"] == rate
    assert result.summary["analysis_rate"] == min(rate, 16000)
    assert result.summary["n_nonphonation_raw"] == 3
    assert result.summary["phonation_s"] == pytest.approx(4.0, abs=0.3)


def test_rough_voice_swinging_12_to_20_db_stays_phonation():
    # One VOICED file: a rough breathy voice, 12 dB of range, pYIN says unvoiced.
    rec = synth.recording("no_pause", "quiet_broadband", RATE, seed=16, voice="breathy")
    t = np.arange(rec.x.size) / RATE
    x = rec.x * 10 ** (8 * np.sin(2 * np.pi * 0.4 * t) / 20)
    summary = sg.segment_signal(x, RATE).summary
    assert 12 < summary["range_db"] < 20
    assert not summary["separable"]
    assert summary["nonphonation_raw_s"] == 0.0
    assert summary["phonation_s"] > 0.9 * rec.duration_s


def test_a_steady_tone_inside_the_band_does_not_eat_the_pauses():
    # Some SVD sessions have a 750 Hz line in the room: pYIN calls every pause voiced.
    rec = synth.recording("brief", "quiet_broadband", RATE, seed=14)
    t = np.arange(rec.x.size) / RATE
    x = rec.x + 3e-4 * np.sin(2 * np.pi * 750 * t)
    result = sg.segment_signal(x, RATE)
    assert result.summary["range_db"] > 25
    assert result.summary["n_nonphonation_raw"] == 3
    assert result.summary["nonphonation_raw_s"] > 1.0
    assert result.summary["phonation_s"] == pytest.approx(4.0, abs=0.3)


def test_phonation_and_non_phonation_never_overlap():
    for room, seed in (("noisy_lowfreq", 3), ("noisy_tonal", 4), ("quiet_broadband", 5)):
        rec = synth.recording("three_reps", room, RATE, seed=seed)
        result = seg(rec)
        times = result.tracks["times"]
        phonation = sg.spans_to_mask(spans(result, "phonation"), times)
        for kind in ("nonphonation_raw", "nonphonation_g050", "nonphonation_g100"):
            assert not (phonation & sg.spans_to_mask(spans(result, kind), times)).any()


def test_a_breath_between_tokens_is_neither_phonation_nor_non_phonation():
    rec = synth.recording("brief", "quiet_broadband", RATE, seed=15)
    rng = np.random.default_rng(15)
    x = rec.x.copy()
    a, b = round(2.4 * RATE), round(2.9 * RATE)  # inside the 0.8 s pause
    x[a:b] += 2e-3 * rng.standard_normal(b - a)
    result = sg.segment_signal(x, RATE)
    times = result.tracks["times"]
    breath = sg.spans_to_mask([(2.5, 2.8)], times)
    phonation = sg.spans_to_mask(spans(result, "phonation"), times)
    quiet = sg.spans_to_mask(spans(result, "nonphonation_raw"), times)
    assert not phonation[breath].any()
    assert not quiet[breath].any()


def test_a_clip_shorter_than_the_vad_window_still_segments():
    rec = synth.recording("no_pause", "quiet_broadband", RATE, seed=13)
    summary = sg.segment_signal(rec.x[: round(0.38 * RATE)], RATE).summary
    assert summary["duration_s"] == pytest.approx(0.38)
    assert summary["nonphonation_g100_s"] == 0.0
    assert summary["phonation_s"] > 0.3


def test_each_file_records_its_threshold_and_segment_counts():
    rec = synth.recording("brief", "noisy_lowfreq", RATE, seed=10)
    result = seg(rec)
    s = result.summary
    assert s["threshold_db"] == pytest.approx(s["floor_db"] + s["margin_db"])
    assert s["margin_db"] == 8.0 and s["floor_percentile"] == 2.0
    assert (s["band_low_hz"], s["band_high_hz"]) == (300.0, 3500.0)
    for guard in ("g050", "g100", "g200"):
        got = spans(result, f"nonphonation_{guard}")
        assert s[f"n_nonphonation_{guard}"] == len(got)
        assert s[f"nonphonation_{guard}_s"] == pytest.approx(sum(b - a for a, b in got))
        assert s[f"longest_nonphonation_{guard}_s"] == pytest.approx(max(b - a for a, b in got))
    assert s["nonphonation_g050_s"] > s["nonphonation_g100_s"] > s["nonphonation_g200_s"]


def test_guard_leaves_the_file_edges_alone():
    spans_in = [(0.0, 0.5), (1.0, 2.0), (3.0, 3.2), (4.5, 5.0)]
    got = sg.apply_guard(spans_in, 0.1, 5.0, 0.05)
    assert got == [(0.0, 0.4), (1.1, 1.9), (4.6, 5.0)]


def test_index_must_hold_exactly_the_four_columns(tmp_path):
    good = pd.DataFrame({"file_id": ["a"], "database": ["x"], "material": ["a"], "wav": ["a.wav"]})
    good.to_csv(tmp_path / "good.csv", index=False)
    assert len(sg.load_index(tmp_path / "good.csv")) == 1
    for extra in ("label", "place", "age", "anything_else"):
        good.assign(**{extra: "x"}).to_csv(tmp_path / "bad.csv", index=False)
        with pytest.raises(SystemExit, match="exactly the columns"):
            sg.load_index(tmp_path / "bad.csv")
    pd.concat([good, good]).to_csv(tmp_path / "twice.csv", index=False)
    with pytest.raises(SystemExit, match="not unique"):
        sg.load_index(tmp_path / "twice.csv")


def test_segmenter_code_never_touches_labels_or_places():
    tree = ast.parse((HERE / "segment.py").read_text(encoding="utf-8"))
    words = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            words.add(node.id)
        elif isinstance(node, ast.Attribute):
            words.add(node.attr)
        elif isinstance(node, ast.arg):
            words.add(node.arg)
        elif isinstance(node, ast.alias):
            words.add(node.name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "\n" not in node.value:  # the module docstring explains the blinding
                words.add(node.value)
    text = " ".join(words).lower()
    forbidden = "label pathological normal cmvd diagnos place essua hsa age sex speakers sessions"
    for word in [*forbidden.split(), "voice_data", "voiced_meta", "xlsx"]:
        hits = [w for w in words if word in w.lower() and w != "standard_normal"]
        assert not hits, (word, hits, len(text))


@pytest.fixture
def small_project(tmp_path, monkeypatch):
    monkeypatch.setattr(sg, "ROOT", tmp_path)
    rec = synth.recording("brief", "quiet_broadband", RATE, seed=11)
    pcm = np.round(rec.x * 32768).astype(np.int16)
    wavfile.write(tmp_path / "good.wav", RATE, pcm)
    wavfile.write(tmp_path / "empty.wav", RATE, np.zeros(0, dtype=np.int16))
    (tmp_path / "broken.wav").write_bytes(b"not audio")
    row = {"database": "x", "material": "a"}
    return tmp_path, [
        {"file_id": "good", "wav": "good.wav", **row},
        {"file_id": "empty", "wav": "empty.wav", **row},
        {"file_id": "broken", "wav": "broken.wav", **row},
    ]


def test_a_bad_file_gets_a_status_and_the_rest_go_on(small_project):
    _, rows = small_project
    results = {r["file_id"]: sg.segment_row(r) for r in rows}
    assert results["good"][0]["status"] == "ok"
    assert results["good"][0]["n_nonphonation_raw"] == 3
    assert {s["file_id"] for s in results["good"][1]} == {"good"}
    assert results["empty"] == ({**rows[1], "status": "empty"}, [])
    assert results["broken"][0]["status"].startswith("error")


def test_tracks_are_kept_when_asked(small_project, tmp_path):
    _, rows = small_project
    (tmp_path / "t").mkdir()
    sg.segment_row(rows[0], str(tmp_path / "t"))
    saved = np.load(tmp_path / "t" / "good.npz")
    assert set(saved.files) == {"times", "level_db", "voiced", "voiced_prob", "in_vad"}


def test_a_run_writes_tables_and_resumes(small_project, tmp_path, monkeypatch, capsys):
    _, rows = small_project
    calls = []

    class SameProcess:
        def __init__(self, workers):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, fn, todo, targets, chunksize=1):
            calls.append([r["file_id"] for r in todo])
            return map(fn, todo, targets)

    monkeypatch.setattr(sg, "ProcessPoolExecutor", SameProcess)
    index = tmp_path / "index.csv"
    pd.DataFrame(rows)[sg.INDEX_COLUMNS].iloc[:2].to_csv(index, index=False)
    out = tmp_path / "out"
    sg.main(["--index", str(index), "--out", str(out), "--workers", "1"])
    files = pd.read_csv(out / "files.csv")
    assert files.status.tolist() == ["ok", "empty"]
    assert json.loads((out / "params.json").read_text())["margin_db"] == 8.0
    table = pd.read_csv(out / "spans.csv")
    assert set(table.kind) >= {"phonation", "nonphonation_raw", "nonphonation_g100"}

    pd.DataFrame(rows)[sg.INDEX_COLUMNS].to_csv(index, index=False)
    sg.main(["--index", str(index), "--out", str(out), "--workers", "1"])
    assert calls == [["good", "empty"], ["broken"]]
    assert len(pd.read_csv(out / "files.csv")) == 3
    assert len(pd.read_csv(out / "spans.csv")) == len(table)


def test_a_real_process_pool_runs_from_an_empty_jit_cache(small_project, tmp_path):
    # Runs the script as a user would; workers starting from an empty numba cache
    # once corrupted it and crashed every later run.
    root, rows = small_project
    index = root / "index.csv"
    frame = pd.DataFrame([rows[0], {**rows[0], "file_id": "again"}])[sg.INDEX_COLUMNS]
    frame.assign(wav=str(root / "good.wav")).to_csv(index, index=False)
    env = {**os.environ, "NUMBA_CACHE_DIR": str(tmp_path / "numba")}
    out = tmp_path / "pool_out"
    done = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(HERE / "segment.py"),
            "--index",
            str(index),
            "--out",
            str(out),
            "--workers",
            "2",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr[-800:]
    assert pd.read_csv(out / "files.csv").status.tolist() == ["ok", "ok"]
