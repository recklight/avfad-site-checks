"""Readers for the raw formats in this study, and a 16-bit WAV writer.

VOICED ships WFDB records in format 32 (32-bit little-endian integers) with a
per-record gain and baseline. SVD ships Kay CSL ``.nsp`` files: an IFF-like
container ("FORM" + "DS16") with a HEDR or HDR8 header chunk and one sample
chunk (SDA_, SD_B or SDAB) of 16-bit little-endian samples. AVFAD is expected
to be RIFF/WAVE; its bit depth and channel count are not documented, so the
WAV reader takes 8/16/24/32-bit PCM and 32/64-bit float, plain or
WAVE_FORMAT_EXTENSIBLE, with any number of channels.

The readers check everything the files let them check (sample counts, the
WFDB checksum, the RIFF and data chunk lengths) and report mismatches instead
of silently fixing them.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.io import wavfile


@dataclass
class WfdbRecord:
    name: str
    rate: int
    n_samples: int
    digital: np.ndarray  # int64, as stored
    gain: float
    baseline: int
    units: str
    init_value: int
    checksum: int
    comments: list[str] = field(default_factory=list)

    @property
    def physical(self) -> np.ndarray:
        return (self.digital - self.baseline) / self.gain


_SIGNAL_SPEC = re.compile(r"^(?P<gain>[-+\d.eE]+)(\((?P<base>-?\d+)\))?(/(?P<units>\S+))?$")


def read_wfdb32(header_path: str | Path) -> WfdbRecord:
    """Read a single-signal WFDB record stored in format 32."""
    header_path = Path(header_path)
    lines = header_path.read_text(encoding="latin-1").splitlines()
    record_line = lines[0].split()
    name, n_sig, rate, n_samples = (
        record_line[0],
        int(record_line[1]),
        record_line[2],
        record_line[3],
    )
    if n_sig != 1:
        raise ValueError(f"{header_path.name}: expected 1 signal, found {n_sig}")
    signal = lines[1].split()
    if signal[1] != "32":
        raise ValueError(f"{header_path.name}: format {signal[1]}, only 32 is handled")
    spec = _SIGNAL_SPEC.match(signal[2])
    if spec is None:
        raise ValueError(f"{header_path.name}: cannot parse gain field {signal[2]!r}")
    gain = float(spec["gain"])
    # WFDB: a missing baseline means it equals the ADC zero (field 5).
    baseline = int(spec["base"]) if spec["base"] is not None else int(signal[4])
    digital = np.fromfile(header_path.with_name(signal[0]), dtype="<i4").astype(np.int64)
    return WfdbRecord(
        name=name,
        rate=int(float(rate)),
        n_samples=int(n_samples),
        digital=digital,
        gain=gain,
        baseline=baseline,
        units=spec["units"] or "",
        init_value=int(signal[5]),
        checksum=int(signal[6]),
        comments=[line[1:].strip() for line in lines[2:] if line.startswith("#")],
    )


def wfdb_checksum(digital: np.ndarray) -> int:
    """WFDB's 16-bit checksum: the sum of all samples as a signed short."""
    total = int(np.sum(digital, dtype=np.int64)) % 65536
    return total - 65536 if total > 32767 else total


def check_wfdb(record: WfdbRecord) -> list[str]:
    problems = []
    if record.digital.size != record.n_samples:
        problems.append(f"length {record.digital.size} != header {record.n_samples}")
    if record.digital.size and record.digital[0] != record.init_value:
        problems.append(f"first sample {record.digital[0]} != header {record.init_value}")
    if wfdb_checksum(record.digital) != record.checksum:
        problems.append(f"checksum {wfdb_checksum(record.digital)} != header {record.checksum}")
    return problems


def to_int16_grid(x: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    """Map samples in [-1, 1) back to int16 when they sit on the 1/32768 grid.

    VOICED's physical values are 16-bit samples divided by 32768 (every one
    of the 208 records lies on that grid to well within 1e-3 of a step), so
    the WAV copy can be 16-bit and still lossless.
    """
    scaled = x * 32768.0
    rounded = np.round(scaled)
    off_grid = float(np.max(np.abs(scaled - rounded))) if x.size else 0.0
    if off_grid > tol:
        raise ValueError(f"samples are off the int16 grid by up to {off_grid:.4f} steps")
    if rounded.min(initial=0) < -32768 or rounded.max(initial=0) > 32767:
        raise ValueError("samples exceed the int16 range")
    return rounded.astype(np.int16)


@dataclass
class NspFile:
    rate: int
    n_samples: int
    samples: np.ndarray  # int16, mono
    date: str
    header_chunk: str
    header_peaks: tuple[int, int]
    form_length: int
    file_length: int
    chunks: list[tuple[str, int]]
    channel_chunk: str


_DATA_CHUNKS = ("SDA_", "SD_B", "SDAB")


def parse_nsp(data: bytes) -> NspFile:
    """Parse a Kay CSL ``.nsp`` file held in memory. Only mono files are accepted."""
    if len(data) < 12 or data[:8] != b"FORMDS16":
        raise ValueError("not a Kay NSP file (no FORMDS16 signature)")
    form_length = struct.unpack_from("<I", data, 8)[0]
    pos = 12
    chunks: list[tuple[str, int]] = []
    header = None
    samples = None
    channel_chunk = ""
    while pos + 8 <= len(data):
        cid = data[pos : pos + 4].decode("latin-1")
        size = struct.unpack_from("<I", data, pos + 4)[0]
        body = data[pos + 8 : pos + 8 + size]
        if len(body) < size:
            raise ValueError(f"chunk {cid} is truncated ({len(body)} of {size} bytes)")
        chunks.append((cid, size))
        if cid in ("HEDR", "HDR8"):
            if size < 32:
                raise ValueError(f"{cid} chunk too short ({size} bytes)")
            date = body[:20].decode("latin-1").rstrip("\x00 ")
            rate, n_samples = struct.unpack_from("<II", body, 20)
            peak_a, peak_b = struct.unpack_from("<hh", body, 28)
            header = (cid, date, rate, n_samples, (peak_a, peak_b))
        elif cid in _DATA_CHUNKS:
            if samples is not None:
                raise ValueError("more than one sample chunk")
            if cid == "SDAB":
                raise ValueError("stereo SDAB chunk; this reader expects mono files")
            if size % 2:
                raise ValueError(f"{cid} chunk has an odd byte count ({size})")
            samples = np.frombuffer(body, dtype="<i2").copy()
            channel_chunk = cid
        # Chunks are padded to an even length.
        pos += 8 + size + (size & 1)
    if header is None:
        raise ValueError("no HEDR/HDR8 header chunk")
    if samples is None:
        raise ValueError("no sample chunk")
    cid, date, rate, n_samples, peaks = header
    return NspFile(
        rate=rate,
        n_samples=n_samples,
        samples=samples,
        date=date,
        header_chunk=cid,
        header_peaks=peaks,
        form_length=form_length,
        file_length=len(data),
        chunks=chunks,
        channel_chunk=channel_chunk,
    )


def check_nsp(nsp: NspFile) -> list[str]:
    problems = []
    if nsp.samples.size != nsp.n_samples:
        problems.append(f"length {nsp.samples.size} != header {nsp.n_samples}")
    if nsp.form_length != nsp.file_length - 12:
        problems.append(
            f"FORM length {nsp.form_length} != file length - 12 ({nsp.file_length - 12})"
        )
    # The header peak is not checked: in SVD it is the peak of the whole
    # session recording the file was cut from, so it rarely equals the
    # file's own peak.
    return problems


def write_wav16(path: str | Path, rate: int, samples: np.ndarray) -> None:
    if samples.dtype != np.int16:
        raise TypeError("write_wav16 takes int16 samples")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    wavfile.write(tmp, rate, samples)
    tmp.replace(path)


@dataclass
class WavFile:
    rate: int
    channels: int
    bits: int  # bits per sample from the fmt chunk
    valid_bits: int  # WAVE_FORMAT_EXTENSIBLE's wValidBitsPerSample, else = bits
    sample_format: str  # "pcm" or "float"
    extensible: bool
    byte_rate: int
    block_align: int
    n_frames: int  # whole frames actually present in the file
    data_declared: int  # bytes, from the data chunk header
    data_present: int  # bytes of the data chunk that are in the file
    riff_length: int
    file_length: int
    chunks: list[tuple[str, int]]
    samples: np.ndarray | None = None  # (n_frames, channels) as stored; None for a header read

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.rate if self.rate else 0.0

    def as_float(self) -> np.ndarray:
        """Samples as float64 in [-1, 1), shape (n_frames, channels)."""
        if self.samples is None:
            raise ValueError("header-only read; use read_wav(path, load=True)")
        if self.sample_format == "float":
            return self.samples.astype(np.float64)
        width = self.block_align // self.channels
        return self.samples.astype(np.float64) / float(1 << (8 * width - 1))


# Every KSDATAFORMAT sub-format GUID ends in these 12 bytes; the first 4 hold
# the plain format tag (1 = PCM, 3 = IEEE float).
_KSDATAFORMAT_TAIL = bytes.fromhex("00001000800000aa00389b71")
_FORMATS = {1: "pcm", 3: "float"}


def read_wav(path: str | Path, load: bool = True) -> WavFile:
    """Read a RIFF/WAVE file. With ``load=False`` only the chunks are walked.

    A data chunk that runs past the end of the file is read as far as it goes
    and reported by ``check_wav``.
    """
    path = Path(path)
    file_length = path.stat().st_size
    with path.open("rb") as f:
        head = f.read(12)
        if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            raise ValueError("not a RIFF/WAVE file")
        riff_length = struct.unpack_from("<I", head, 4)[0]
        chunks: list[tuple[str, int]] = []
        fmt = None
        data_pos = data_declared = None
        pos = 12
        while pos + 8 <= file_length:
            f.seek(pos)
            cid_raw, size = struct.unpack("<4sI", f.read(8))
            cid = cid_raw.decode("latin-1")
            chunks.append((cid, size))
            if cid == "fmt ":
                fmt = f.read(size)
                if len(fmt) < size:
                    raise ValueError("fmt chunk is truncated")
            elif cid == "data":
                if data_pos is not None:
                    raise ValueError("more than one data chunk")
                data_pos, data_declared = pos + 8, size
            # Chunks are padded to an even length.
            pos += 8 + size + (size & 1)
        if fmt is None or len(fmt) < 16:
            raise ValueError("no fmt chunk")
        if data_pos is None:
            raise ValueError("no data chunk")
        tag, channels, rate, byte_rate, block_align, bits = struct.unpack_from("<HHIIHH", fmt)
        extensible = tag == 0xFFFE
        valid_bits = bits
        if extensible:
            if len(fmt) < 40:
                raise ValueError("WAVE_FORMAT_EXTENSIBLE fmt chunk is too short")
            valid_bits = struct.unpack_from("<H", fmt, 18)[0]
            if fmt[28:40] != _KSDATAFORMAT_TAIL:
                raise ValueError("unknown WAVE_FORMAT_EXTENSIBLE sub-format GUID")
            tag = struct.unpack_from("<I", fmt, 24)[0]
        if tag not in _FORMATS:
            raise ValueError(f"format tag {tag:#06x} is neither PCM nor IEEE float")
        sample_format = _FORMATS[tag]
        if channels < 1 or block_align < channels or block_align % channels:
            raise ValueError(f"{channels} channels with block align {block_align}")
        width = block_align // channels
        allowed = (4, 8) if sample_format == "float" else (1, 2, 3, 4)
        if width not in allowed:
            raise ValueError(f"{8 * width}-bit {sample_format} samples are not handled")
        data_present = min(data_declared, file_length - data_pos)
        n_frames = data_present // block_align
        samples = None
        if load:
            f.seek(data_pos)
            samples = _decode(f.read(n_frames * block_align), width, sample_format, channels)
    return WavFile(
        rate=rate,
        channels=channels,
        bits=bits,
        valid_bits=valid_bits,
        sample_format=sample_format,
        extensible=extensible,
        byte_rate=byte_rate,
        block_align=block_align,
        n_frames=n_frames,
        data_declared=data_declared,
        data_present=data_present,
        riff_length=riff_length,
        file_length=file_length,
        chunks=chunks,
        samples=samples,
    )


def _decode(raw: bytes, width: int, sample_format: str, channels: int) -> np.ndarray:
    if sample_format == "float":
        x = np.frombuffer(raw, dtype="<f4" if width == 4 else "<f8")
    elif width == 1:
        # 8-bit WAV is unsigned with 128 as zero.
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128
    elif width == 2:
        x = np.frombuffer(raw, dtype="<i2")
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        x = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = np.where(x >= 1 << 23, x - (1 << 24), x)
    else:
        x = np.frombuffer(raw, dtype="<i4")
    return x.reshape(-1, channels).copy()


def check_wav(wav: WavFile) -> list[str]:
    problems = []
    if wav.riff_length != wav.file_length - 8:
        problems.append(f"RIFF length {wav.riff_length} != file length - 8 ({wav.file_length - 8})")
    if wav.data_present < wav.data_declared:
        problems.append(
            f"data chunk declares {wav.data_declared} bytes, {wav.data_present} are in the file"
        )
    if wav.data_present % wav.block_align:
        problems.append(f"data chunk ends mid-frame ({wav.data_present % wav.block_align} bytes)")
    if wav.bits % 8 == 0 and wav.block_align != wav.channels * wav.bits // 8:
        problems.append(f"block align {wav.block_align} != channels x bits / 8")
    if wav.byte_rate != wav.rate * wav.block_align:
        problems.append(f"byte rate {wav.byte_rate} != rate x block align")
    if wav.valid_bits > wav.bits:
        problems.append(f"valid bits {wav.valid_bits} > container bits {wav.bits}")
    return problems
