import hashlib
import json
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import avfad_extract
import avfad_meta
import avfad_stats
from readers import check_wav, read_wav


def wav_bytes(frames, rate=48000, width=2, fmt="pcm", extensible=False, before_data=()):
    """A RIFF/WAVE file built by hand; ``frames`` is (n, channels) of stored integers or floats."""
    frames = np.asarray(frames)
    if frames.ndim == 1:
        frames = frames[:, None]
    channels = frames.shape[1]
    if fmt == "float":
        data = frames.astype("<f4" if width == 4 else "<f8").tobytes()
    elif width == 3:
        v = frames.astype(np.int64).ravel() & 0xFFFFFF
        data = np.stack([v & 0xFF, (v >> 8) & 0xFF, v >> 16], axis=1).astype(np.uint8).tobytes()
    else:
        data = frames.astype({1: "u1", 2: "<i2", 4: "<i4"}[width]).tobytes()
    block = channels * width
    tag = 3 if fmt == "float" else 1
    if extensible:
        guid = struct.pack("<I", tag) + bytes.fromhex("00001000800000aa00389b71")
        fmt_body = struct.pack(
            "<HHIIHHHHI", 0xFFFE, channels, rate, rate * block, block, 8 * width, 22, 8 * width, 0
        )
        fmt_body += guid
    else:
        fmt_body = struct.pack("<HHIIHH", tag, channels, rate, rate * block, block, 8 * width)
    body = b"WAVE"
    for cid, chunk in ((b"fmt ", fmt_body), *before_data, (b"data", data)):
        body += cid + struct.pack("<I", len(chunk)) + chunk + (b"\x00" if len(chunk) & 1 else b"")
    return b"RIFF" + struct.pack("<I", len(body)) + body


def test_wav_reader_matches_scipy_for_16_bit_mono(tmp_path):
    x = np.random.default_rng(0).integers(-32768, 32767, 4800).astype(np.int16)
    wavfile.write(tmp_path / "a.wav", 48000, x)
    wav = read_wav(tmp_path / "a.wav")
    assert (wav.rate, wav.channels, wav.bits, wav.sample_format) == (48000, 1, 16, "pcm")
    assert np.array_equal(wav.samples[:, 0], x)
    assert wav.duration_s == pytest.approx(0.1)
    assert check_wav(wav) == []


def test_wav_reader_24_bit_stereo(tmp_path):
    left = np.array([-(1 << 23), -1, 0, 1, (1 << 23) - 1])
    frames = np.stack([left, -left // 2], axis=1)
    (tmp_path / "b.wav").write_bytes(wav_bytes(frames, width=3))
    wav = read_wav(tmp_path / "b.wav")
    assert (wav.channels, wav.bits, wav.n_frames) == (2, 24, 5)
    assert np.array_equal(wav.samples, frames)
    y = wav.as_float()
    assert y[0, 0] == -1.0 and y.max() < 1.0
    assert check_wav(wav) == []


def test_wav_reader_float_extensible_with_an_odd_chunk_before_data(tmp_path):
    frames = np.array([[0.5, -0.25], [-1.0, 0.75], [0.0, 0.125]])
    data = wav_bytes(frames, width=4, fmt="float", extensible=True, before_data=[(b"LIST", b"x")])
    (tmp_path / "c.wav").write_bytes(data)
    wav = read_wav(tmp_path / "c.wav")
    assert wav.extensible and wav.sample_format == "float"
    assert [c for c, _ in wav.chunks] == ["fmt ", "LIST", "data"]
    assert np.array_equal(wav.as_float(), frames)
    assert check_wav(wav) == []


def test_header_read_leaves_the_samples_alone(tmp_path):
    (tmp_path / "d.wav").write_bytes(wav_bytes(np.zeros((480, 2)), width=2))
    wav = read_wav(tmp_path / "d.wav", load=False)
    assert wav.samples is None and wav.n_frames == 480
    with pytest.raises(ValueError, match="header-only"):
        wav.as_float()


def test_wav_checks_report_a_truncated_file(tmp_path):
    full = wav_bytes(np.arange(100), width=2)
    (tmp_path / "e.wav").write_bytes(full[:-11])
    wav = read_wav(tmp_path / "e.wav")
    problems = check_wav(wav)
    assert wav.n_frames == 94
    assert any("RIFF length" in p for p in problems)
    assert any("declares 200 bytes, 189" in p for p in problems)
    assert any("mid-frame" in p for p in problems)
    assert np.array_equal(wav.samples[:, 0], np.arange(94))


def test_wav_reader_rejects_other_files(tmp_path):
    (tmp_path / "f.wav").write_bytes(b"FORMDS16" + b"\x00" * 40)
    with pytest.raises(ValueError, match="RIFF"):
        read_wav(tmp_path / "f.wav")
    adpcm = bytearray(wav_bytes(np.zeros(10)))
    struct.pack_into("<H", adpcm, 20, 2)
    (tmp_path / "g.wav").write_bytes(bytes(adpcm))
    with pytest.raises(ValueError, match="format tag"):
        read_wav(tmp_path / "g.wav")


@pytest.mark.parametrize(
    ("name", "ok"),
    [
        ("Folder/a.wav", True),
        ("Folder\\sub\\a.wav", True),
        ("../a.wav", False),
        ("Folder/../../a.wav", False),
        ("/etc/a.wav", False),
        ("C:/a.wav", False),
        ("a.wav:stream", False),
        ("", False),
    ],
)
def test_safe_member_path(name, ok):
    assert (avfad_extract.safe_member_path(name) is not None) == ok


def point_scripts_at(root, monkeypatch):
    avfad = root / "data" / "avfad"
    for name, value in {
        "ROOT": root,
        "AVFAD": avfad,
        "RAW": avfad / "raw",
        "FILES": avfad / "files",
        "MANIFEST": avfad / "manifest.json",
        "ACCESS": avfad / "access.json",
        "OUT": root / "out",
    }.items():
        monkeypatch.setattr(avfad_extract, name, value)
    monkeypatch.setattr(avfad_stats, "ROOT", root)
    monkeypatch.setattr(avfad_stats, "OUT", root / "out")


@pytest.fixture
def fake_avfad(tmp_path, monkeypatch):
    root = tmp_path / "paper"
    avfad = root / "data" / "avfad"
    raw = avfad / "raw"
    raw.mkdir(parents=True)
    rng = np.random.default_rng(2)
    one = rng.integers(-3000, 3000, 4800).astype(np.int16)
    two = np.stack([one, one // 2], axis=1)
    with zipfile.ZipFile(raw / "Recordings.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Recordings/", b"")
        zf.writestr("Recordings/S001/S001_a.wav", wav_bytes(one))
        zf.writestr("Recordings/S002/S002_a.wav", wav_bytes(two))
        zf.writestr("Recordings/broken.wav", b"not audio")
        zf.writestr("Recordings/metadata.xlsx", b"PK fake spreadsheet")
        zf.writestr("../escape.wav", wav_bytes(one))
    (raw / "notes.txt").write_text("loose file next to the zip\n")
    point_scripts_at(root, monkeypatch)
    return root, one, two


@pytest.mark.parametrize("content", [None, "{}", '{"retention_off_confirmed_on": "soon"}'])
def test_reading_recordings_needs_the_retention_date(fake_avfad, content):
    root, _, _ = fake_avfad
    if content is not None:
        (root / "data" / "avfad" / "access.json").write_text(content)
    # Hashes and headers only: these two run without the date.
    avfad_extract.unpack()
    avfad_extract.inventory()
    manifest = json.loads((root / "data" / "avfad" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["access"]["retention_off_confirmed_on"] is None
    with pytest.raises(SystemExit, match="retention_off_confirmed_on"):
        avfad_stats.main([])
    assert not (root / "out" / "avfad_stats.csv").exists()


def test_unpack_and_inventory(fake_avfad, tmp_path):
    root, one, _ = fake_avfad
    avfad = root / "data" / "avfad"
    (avfad / "access.json").write_text('{"retention_off_confirmed_on": "2026-11-04"}')
    zip_path = avfad / "raw" / "Recordings.zip"
    zip_sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()

    avfad_extract.unpack()
    manifest = json.loads((avfad / "manifest.json").read_text(encoding="utf-8"))
    assert {d["path"]: d["sha256"] for d in manifest["downloads"]}[
        "data/avfad/raw/Recordings.zip"
    ] == zip_sha
    assert hashlib.sha256(zip_path.read_bytes()).hexdigest() == zip_sha
    assert manifest["unpack"]["members"] == 4
    assert [r["member"] for r in manifest["unpack"]["rejected_members"]] == ["../escape.wav"]
    assert not list(tmp_path.rglob("escape.wav"))
    unpacked = avfad / "files" / "Recordings" / "Recordings"
    assert read_wav(unpacked / "S001" / "S001_a.wav").samples[:, 0].tolist() == one.tolist()

    avfad_extract.inventory()
    files = pd.read_csv(root / "out" / "avfad_files.csv").set_index("path")
    assert len(files) == 5  # four members plus the loose text file
    for path, row in files.iterrows():
        assert row.sha256 == hashlib.sha256((root / path).read_bytes()).hexdigest()
    s002 = files.loc["data/avfad/files/Recordings/Recordings/S002/S002_a.wav"]
    assert (s002.rate, s002.channels, s002.bits, s002.n_frames) == (48000, 2, 16, 4800)
    assert files.loc["data/avfad/files/Recordings/Recordings/broken.wav"].reader_error == (
        "not a RIFF/WAVE file"
    )
    manifest = json.loads((avfad / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["access"]["retention_off_confirmed_on"] == "2026-11-04"
    assert "downloads_note" not in manifest
    inv = manifest["inventory"]
    assert inv["by_ext"] == {".wav": 3, ".xlsx": 1, ".txt": 1}
    assert inv["wav_formats"] == {"48000 Hz, 1 ch, 16-bit pcm": 1, "48000 Hz, 2 ch, 16-bit pcm": 1}
    assert inv["wav_reader_errors"] == 1 and inv["wav_with_problems"] == 0
    assert len(manifest["files"]) == 5

    stats = avfad_stats.one("data/avfad/files/Recordings/Recordings/S002/S002_a.wav")
    assert stats["duration_s"] == pytest.approx(0.1)
    rms = [float(v) for v in stats["rms_dbfs_by_channel"].split()]
    assert rms[0] - rms[1] == pytest.approx(20 * np.log10(2), abs=0.1)
    assert stats["max_abs_diff_ch1_ch2"] > 0


def add_to_zip(zip_path, name, data):
    with zipfile.ZipFile(zip_path, "a") as zf:
        zf.writestr(name, data)


@pytest.mark.parametrize(
    "error_name",
    ["Recordings/___All_Errors.txt", "Recordings/__S003/S003_Error.txt"],
)
def test_onedrive_error_files_stop_before_any_hashing(fake_avfad, error_name):
    root, _, _ = fake_avfad
    avfad = root / "data" / "avfad"
    (avfad / "access.json").write_text('{"retention_off_confirmed_on": "2026-11-04"}')
    add_to_zip(avfad / "raw" / "Recordings.zip", error_name, "TooManyRequestsMeTAException\n")
    for step in (avfad_extract.unpack, avfad_extract.inventory):
        with pytest.raises(SystemExit, match=r"Recordings\.zip: 1 OneDrive error file"):
            step()
    assert not (avfad / "files").exists()
    assert not (avfad / "manifest.json").exists()


def test_error_files_in_an_unpacked_folder_also_stop(fake_avfad):
    root, _, _ = fake_avfad
    avfad = root / "data" / "avfad"
    (avfad / "access.json").write_text('{"retention_off_confirmed_on": "2026-11-04"}')
    synced = avfad / "raw" / "4_M"
    synced.mkdir()
    (synced / "___All_Errors.txt").write_text("WebException\n")
    with pytest.raises(SystemExit, match="4_M: 1 OneDrive error file"):
        avfad_extract.unpack()
    (synced / "___All_Errors.txt").unlink()
    avfad_extract.unpack()
    (avfad / "files" / "Recordings" / "Recordings" / "S001" / "S001_Error.txt").write_text("x")
    with pytest.raises(SystemExit, match="files/Recordings: 1 OneDrive error file"):
        avfad_extract.inventory()


@pytest.mark.parametrize("prefix", ["", "Recordings/"])
def test_copies_unpacked_into_raw_stop(fake_avfad, prefix):
    root, one, _ = fake_avfad
    avfad = root / "data" / "avfad"
    (avfad / "access.json").write_text('{"retention_off_confirmed_on": "2026-11-04"}')
    copy = avfad / "raw" / f"{prefix}Recordings" / "S001" / "S001_a.wav"
    copy.parent.mkdir(parents=True)
    copy.write_bytes(wav_bytes(one))
    with pytest.raises(SystemExit, match=r"1 file\(s\) unpacked from its own zips"):
        avfad_extract.unpack()
    assert not (avfad / "manifest.json").exists()


def test_inventory_reads_folders_that_arrive_unpacked(tmp_path, monkeypatch):
    root = tmp_path / "paper"
    raw = root / "data" / "avfad" / "raw"
    point_scripts_at(root, monkeypatch)
    x = np.zeros(480, dtype=np.int16)
    layout = {
        "AVFAD_01_00_00_2_A_to_C/S001/S001_a.wav": wav_bytes(x),
        "AVFAD_01_00_00_2_A_to_C/S001/S001_e.WAV": wav_bytes(x, rate=44100),
        "AVFAD_01_00_00_2_A_to_C/S001/S001.prt": b"p",
        "AVFAD_01_00_00_2_A_to_C/S002/S002_a.wav": wav_bytes(np.zeros((480, 2))),
        "AVFAD_01_00_00_2_A_to_C/S002/S002_e.wav": wav_bytes(x),
        "AVFAD_01_00_00_2_A_to_C/S002/S002.prt": b"p",
        "AVFAD_01_00_00_2_A_to_C/S002/S002_b.prt": b"p",
        "AVFAD_01_00_00_1_README/room_silence.wav": wav_bytes(x),
        "AVFAD_01_00_00_1_README/table.xlsx": b"PK",
    }
    for rel, data in layout.items():
        (raw / rel).parent.mkdir(parents=True, exist_ok=True)
        (raw / rel).write_bytes(data)

    avfad_extract.unpack()  # nothing to unpack
    assert not (root / "data" / "avfad" / "files").exists()
    avfad_extract.inventory()
    manifest = json.loads((root / "data" / "avfad" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["downloads_note"] == "zip 未保留；OneDrive 不提供 checksum"  # noqa: RUF001
    assert len(manifest["files"]) == len(layout)
    inv = manifest["inventory"]
    assert inv["speakers_by_folder"] == {"AVFAD_01_00_00_2_A_to_C": 2}
    assert inv["files_per_speaker"]["AVFAD_01_00_00_2_A_to_C"] == {
        ".prt": {"1": 1, "2": 1},
        ".wav": {"2": 2},
    }
    assert inv["suffix_not_lower_case"] == {"AVFAD_01_00_00_2_A_to_C": 1}
    assert inv["wav_formats_by_folder"]["AVFAD_01_00_00_2_A_to_C"] == {
        "44100 Hz, 1 ch, 16-bit pcm": 1,
        "48000 Hz, 1 ch, 16-bit pcm": 2,
        "48000 Hz, 2 ch, 16-bit pcm": 1,
    }
    assert inv["by_folder"]["AVFAD_01_00_00_1_README"] == 2


def silence_rows(names, hashes=None):
    hashes = hashes or [f"h{i}" for i in range(len(names))]
    return pd.DataFrame(
        {
            "path": [f"data/avfad/raw/README/README/{n}" for n in names],
            "ext": ".wav",
            "sha256": hashes,
        }
    )


def test_silence_names_in_every_spelling_found_in_the_folder():
    names = [
        "2014_3_18_HSA_silence60s.wav",
        "2014_4_29_HSA_silence_60s_room1.wav",
        "2014_7_15_HSA_silence_60s.wav.wav",
        "2014_8_28_ESSUA_silence60s_AAWraw_ICVraw.wav",
        "2015_3_5_ESSUA_silence_60_seconds.wav",
        "2015_8_18_ESSUA_silence_60s_AJO_COX.wav",
    ]
    sil = avfad_meta.parse_silence(silence_rows(names, ["a", "b", "c", "d", "e", "a"]))
    assert sil.parsed.all()
    assert sil.place.tolist() == ["HSA", "HSA", "HSA", "ESSUA", "ESSUA", "ESSUA"]
    assert sil.room.tolist()[1] == "room1"
    assert sil.listed_speakers.tolist()[3] == "AAW ICV"
    assert sil.listed_speakers.tolist()[5] == "AJO COX"
    assert (sil.unparsed_tokens == "").all()
    assert sil.identical_files.tolist() == [2, 1, 1, 1, 1, 2]


def test_each_speaker_gets_the_silence_file_of_its_visit():
    names = [
        "2014_8_28_ESSUA_silence60s_AAWraw.wav",
        "2014_8_28_ESSUA_silence60s.wav",
        "2014_4_29_HSA_silence_60s_room1.wav",
        "2014_4_29_HSA_silence_60s_room2.wav",
        "2015_3_31_ULSM_silence_60s_FMX.wav",
        "2015_9_28_ESSUA_silence_60s_AJJ.wav",
        "2015_9_28_ESSUA_silence_60s_DCC.wav",
    ]
    sil = avfad_meta.parse_silence(silence_rows(names))
    d = pd.Timestamp
    speakers = pd.DataFrame(
        {
            "speaker": ["AAW", "BBB", "CCC", "FMX", "GGG", "HHH", "JJJ"],
            "visit_date": [
                d(2014, 8, 28).date(),
                d(2014, 8, 28).date(),
                d(2014, 4, 29).date(),
                d(2015, 3, 31).date(),
                d(2015, 3, 31).date(),
                d(2015, 9, 28).date(),
                d(2013, 1, 1).date(),
            ],
            "place": ["ESSUA", "ESSUA", "HSA", "ULSM", "ULSM", "ESSUA", "HSA"],
        }
    )
    got = avfad_meta.match_silence(speakers, sil).set_index("speaker")
    assert got.silence_match.to_dict() == {
        "AAW": "listed",
        "BBB": "visit",  # the one file of that day that lists nobody
        "CCC": "several_files",  # room1 and room2
        "FMX": "listed",
        "GGG": "visit",  # the day's only file, even though it lists FMX
        "HHH": "several_files",
        "JJJ": "no_file_for_visit",
    }
    assert got.loc["BBB", "silence_file"] == "2014_8_28_ESSUA_silence60s.wav"
    assert got.loc["CCC", "silence_file"] == ""
