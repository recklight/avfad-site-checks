"""The index builder and the availability report, on small made-up tables."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "s0_audit"))
import availability as av
import avfad_extract
import make_index


@pytest.fixture
def s0_tables(tmp_path, monkeypatch):
    out = tmp_path / "s0"
    out.mkdir()
    pd.DataFrame(
        {
            "record": ["voice001"],
            "wav": ["data/derived/voiced/voice001.wav"],
            "reader_problems": [""],
        }
    ).to_csv(out / "voiced_files.csv", index=False)
    pd.DataFrame(
        {
            "session": [1, 1, 1, 2, 3, 4],
            "material": ["a_n", "iau", "phrase", "a_n", "a_n", "iau"],
            "wav": ["1a.wav", "1i.wav", "1p.wav", "2a.wav", "3a.wav", ""],
            "file_problem": ["", "", "", "", "speech_egg_swapped", ""],
            "label": ["normal", "normal", "normal", "pathological", "normal", "normal"],
            "eligible": [True, True, True, True, True, True],
            "primary": [True, True, True, False, True, True],
        }
    ).to_csv(out / "svd_file_status.csv", index=False)
    pd.DataFrame(
        {
            "path": ["raw/AAC/AAC002.WAV", "raw/AAC/AAC001.wav", "raw/ABC/ABC002.wav"],
            "speaker": ["AAC", "AAC", "ABC"],
            "task": ["002", "001", "002"],
        }
    ).to_csv(out / "avfad_file_status.csv", index=False, float_format="%g")
    pd.DataFrame({"path": ["raw/README/2014_HSA_silence.wav"], "place": ["HSA"]}).to_csv(
        out / "avfad_silence.csv", index=False
    )
    monkeypatch.setattr(avfad_extract, "ACCESS", tmp_path / "access.json")
    monkeypatch.setattr(avfad_extract, "ROOT", tmp_path)
    return out, tmp_path / "access.json"


def test_index_has_no_label_and_picks_the_right_files(s0_tables):
    out, access = s0_tables
    access.write_text('{"retention_off_confirmed_on": "2026-11-04"}')
    index = make_index.build_index(out)
    assert list(index.columns) == ["file_id", "database", "material", "wav"]
    assert index.file_id.tolist() == [
        "voiced__voice001",
        "svd__1-a_n",
        "svd__1-iau",
        "avfad__AAC",
        "avfad__ABC",
        "avfad_silence__0001",
    ]
    assert index.set_index("file_id").loc["avfad_silence__0001", "material"] == "silence"
    assert index.file_id.is_unique


def test_eligible_scope_adds_the_later_sessions(s0_tables):
    out, access = s0_tables
    access.write_text('{"retention_off_confirmed_on": "2026-11-04"}')
    assert "svd__2-a_n" in set(make_index.build_index(out, "eligible").file_id)


def test_index_needs_the_retention_date_for_avfad(s0_tables):
    out, _ = s0_tables
    with pytest.raises(SystemExit, match="retention_off_confirmed_on"):
        make_index.build_index(out)


def file_rows(ids, lengths, database="avfad", material="a"):
    n = len(ids)
    table = {
        "file_id": ids,
        "database": [database] * n,
        "material": [material] * n,
        "wav": [f"{i}.wav" for i in ids],
        "status": ["ok"] * n,
        "duration_s": [10.0] * n,
        "floor_db": [-60.0] * n,
        "threshold_db": [-52.0] * n,
        "separable": [True] * n,
        "voiced_fraction": [0.8] * n,
        "low_level_voiced_s": [0.0] * n,
        "other_s": [0.1] * n,
        "nonphonation_raw_s": [v + 0.2 for v in lengths],
    }
    for guard in av.GUARDS:
        table[f"nonphonation_{guard}_s"] = lengths
        table[f"n_nonphonation_{guard}"] = [2] * n
        table[f"longest_nonphonation_{guard}_s"] = [v / 2 for v in lengths]
    return pd.DataFrame(table)


def test_availability_counts_shares_per_place():
    files = file_rows(["a", "b", "c", "d"], [0.1, 0.6, 1.5, 3.0]).assign(place=["P"] * 4)
    row = av.availability(files, ["place"]).query("guard == 'g100'").iloc[0]
    assert row.n_files == 4
    assert row["share_at_least_0.5s"] == 0.75
    assert row.share_at_least_1s == 0.5
    assert row.share_at_least_5s == 0.0
    assert row.median_total_s == pytest.approx(1.05)


@pytest.mark.parametrize(
    ("essua", "hsa", "kept"),
    [
        ([2.0] * 5, [2.0] * 5, True),
        ([2.0] * 4 + [0.0], [2.0] * 4 + [0.0], True),  # exactly 80% in both
        ([2.0] * 5, [2.0] * 3 + [0.0] * 2, False),  # one place short is enough
        ([0.0] * 5, [2.0] * 5, False),
    ],
)
def test_h1_rule_is_judged_per_place(essua, hsa, kept):
    files = pd.DataFrame(
        {
            "place": ["ESSUA"] * len(essua) + ["HSA"] * len(hsa),
            "nonphonation_g100_s": essua + hsa,
        }
    )
    rule = av.h1_avfad_rule(files, min_total_s=1.0, min_share=0.8)
    assert rule["h1_avfad_stays_confirmatory"] is kept
    assert set(rule["places"]) == {"ESSUA", "HSA"}


def test_report_reads_only_speaker_and_place(tmp_path):
    s1 = tmp_path / "s1"
    s0 = tmp_path / "s0"
    s1.mkdir()
    s0.mkdir()
    a = file_rows(["avfad__AAA", "avfad__BBB"], [2.0, 2.0])
    room = file_rows(["avfad_silence__0001"], [50.0], material="silence").assign(
        wav="raw/x_silence.wav"
    )
    voiced = file_rows(["voiced__v1"], [0.0], database="voiced")
    pd.concat([a, room, voiced]).to_csv(s1 / "files.csv", index=False)
    # A row with a broken "label" column would raise if the script tried to use it.
    pd.DataFrame(
        {"speaker": ["AAA", "BBB"], "place": ["ESSUA", "HSA"], "age": [1, 2], "label": ["x", "y"]}
    ).to_csv(s0 / "avfad_speakers.csv", index=False)
    pd.DataFrame({"path": ["raw/x_silence.wav"], "place": ["HSA"]}).to_csv(
        s0 / "avfad_silence.csv", index=False
    )
    places = av.load_places(s0)
    assert list(places.columns) == ["file_id", "place"]
    av.main(["--s1-out", str(s1), "--s0-out", str(s0)])
    rule = json.loads((s1 / "availability_rule.json").read_text())
    assert rule["h1_avfad_stays_confirmatory"] is True
    table = pd.read_csv(s1 / "availability_avfad_by_place.csv")
    assert "label" not in table.columns
    assert set(table.place) == {"ESSUA", "HSA"}
    control = pd.read_csv(s1 / "availability_silence_by_place.csv")
    assert control.place.tolist() == ["HSA"]
    by_db = pd.read_csv(s1 / "availability_by_database.csv")
    assert {"voiced", "avfad"} <= set(by_db.database)
