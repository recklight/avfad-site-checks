import struct
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from readers import (
    check_nsp,
    check_wfdb,
    parse_nsp,
    read_wfdb32,
    to_int16_grid,
    wfdb_checksum,
    write_wav16,
)
from stats import recording_stats


def nsp_bytes(samples, rate=50000, header="HEDR", peak=None, extra=(), pad_note=False):
    samples = np.asarray(samples, dtype="<i2")
    peak = int(np.max(np.abs(samples.astype(np.int32)))) if peak is None else peak
    date = b"Mar 04 10:11:12 1999"
    body = date + struct.pack("<IIhh", rate, samples.size, peak, -1)
    if header == "HDR8":
        body += b"\x00" * 12
    chunks = [(header.encode(), body)]
    if pad_note:
        chunks.append((b"NOTE", b"odd"))  # 3 bytes, padded to 4
    chunks.extend(extra)
    chunks.append((b"SDA_", samples.tobytes()))
    payload = b""
    for cid, data in chunks:
        payload += cid + struct.pack("<I", len(data)) + data + (b"\x00" if len(data) & 1 else b"")
    return b"FORMDS16" + struct.pack("<I", len(payload)) + payload


@pytest.mark.parametrize("header", ["HEDR", "HDR8"])
def test_nsp_round_trip(header):
    rng = np.random.default_rng(0)
    x = rng.integers(-30000, 30000, 1234).astype(np.int16)
    nsp = parse_nsp(nsp_bytes(x, header=header, pad_note=True))
    assert nsp.rate == 50000
    assert nsp.n_samples == 1234
    assert np.array_equal(nsp.samples, x)
    assert nsp.date == "Mar 04 10:11:12 1999"
    assert check_nsp(nsp) == []


def test_nsp_checks_report_mismatches():
    x = np.arange(100, dtype=np.int16)
    # A header peak that differs from the file's own peak is normal in SVD.
    data = bytearray(nsp_bytes(x, peak=7))
    assert check_nsp(parse_nsp(bytes(data))) == []
    # Header claims more samples than the chunk holds.
    struct.pack_into("<I", data, 12 + 8 + 24, 101)
    assert any("length" in p for p in check_nsp(parse_nsp(bytes(data))))


def test_nsp_rejects_other_files():
    with pytest.raises(ValueError, match="FORMDS16"):
        parse_nsp(b"RIFF" + b"\x00" * 40)
    stereo = nsp_bytes([1, 2], extra=[(b"SDAB", b"\x00" * 8)])
    with pytest.raises(ValueError):
        parse_nsp(stereo)
    truncated = nsp_bytes(np.arange(50, dtype=np.int16))[:-10]
    with pytest.raises(ValueError, match="truncated"):
        parse_nsp(truncated)


def write_wfdb32(folder, name, physical, gain=4079702243.3775, baseline=-260023747):
    digital = np.round(np.asarray(physical) * gain).astype(np.int64) + baseline
    (folder / f"{name}.dat").write_bytes(digital.astype("<i4").tobytes())
    header = (
        f"{name} 1 8000 {digital.size}\n"
        f"{name}.dat 32 {gain}({baseline})/NU 0 0 {digital[0]} {wfdb_checksum(digital)} 0 voice\n"
        f"#<age>: 40 <sex>: F <diagnoses>: healthy <medications>: none\n"
    )
    (folder / f"{name}.hea").write_text(header)
    return digital


def test_wfdb32_round_trip(tmp_path):
    x = np.arange(-50, 50) / 32768.0
    digital = write_wfdb32(tmp_path, "rec1", x)
    rec = read_wfdb32(tmp_path / "rec1.hea")
    assert np.array_equal(rec.digital, digital)
    assert np.allclose(rec.physical, x, atol=1e-5)
    assert check_wfdb(rec) == []
    assert np.array_equal(to_int16_grid(rec.physical), np.arange(-50, 50))
    assert rec.comments[0].startswith("<age>: 40")


def test_wfdb32_checksum_mismatch(tmp_path):
    write_wfdb32(tmp_path, "rec2", np.zeros(10))
    dat = tmp_path / "rec2.dat"
    raw = bytearray(dat.read_bytes())
    raw[4] ^= 1
    dat.write_bytes(bytes(raw))
    assert any("checksum" in p for p in check_wfdb(read_wfdb32(tmp_path / "rec2.hea")))


def test_int16_grid_refuses_off_grid_values():
    with pytest.raises(ValueError, match="grid"):
        to_int16_grid(np.array([0.1234567]))


def test_wav_round_trip(tmp_path):
    x = np.array([-32768, 0, 32767], dtype=np.int16)
    write_wav16(tmp_path / "a" / "x.wav", 50000, x)
    rate, y = wavfile.read(tmp_path / "a" / "x.wav")
    assert rate == 50000 and np.array_equal(x, y)


def test_stats_on_a_tone_with_silent_edges():
    rate = 8000
    t = np.arange(rate) / rate
    tone = 0.5 * np.sin(2 * np.pi * 200 * t)
    x = np.concatenate([np.zeros(rate // 2), tone, np.zeros(rate // 4)])
    s = recording_stats(x, rate)
    assert s["duration_s"] == pytest.approx(1.75)
    assert s["lead_s"] == pytest.approx(0.5, abs=0.03)
    assert s["trail_s"] == pytest.approx(0.25, abs=0.03)
    # Sine RMS is peak / sqrt(2): 20*log10(0.5/sqrt(2)) = -9.03 dB on the tone part.
    assert s["peak_dbfs"] == pytest.approx(20 * np.log10(0.5), abs=0.01)
    assert s["rolloff99_hz"] < 400
    assert s["clip_fraction"] == 0.0
