"""S0 for SVD: verify data.zip, then pull out the speech files this study uses.

Usage:
  python -I s0_audit/svd_extract.py verify    # size + MD5 against the Zenodo record
  python -I s0_audit/svd_extract.py extract   # a_n, iau, phrase speech files + metadata files

Only the speech channel is taken (``*.nsp``); EGG files (``*.egg``) and the
other eleven vowel files stay in the zip. Each NSP is parsed and checked,
kept as is under data/svd/nsp/, and written as a 16-bit WAV at its own rate
under data/derived/svd/. Non-audio members (overview tables, remarks) go to
data/svd/meta/. Every file gets a SHA-256 in data/svd/manifest.json, so the
zip can be deleted afterwards and the extraction still checked.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from readers import check_nsp, parse_nsp, write_wav16

ROOT = Path(__file__).resolve().parents[1]
SVD = ROOT / "data" / "svd"
ZIP = SVD / "data.zip"
NSP_DIR = SVD / "nsp"
META_DIR = SVD / "meta"
WAV_DIR = ROOT / "data" / "derived" / "svd"
MANIFEST = SVD / "manifest.json"
OUT = Path(__file__).resolve().parent / "out"

ZENODO = {
    "dataset": "Saarbruecken Voice Database (Zenodo record 16874898)",
    "record": "https://zenodo.org/records/16874898",
    "doi": "10.5281/zenodo.16874898",
    "licence": "CC BY 4.0",
    "file": "data.zip",
    "size": 17881176281,
    "md5": "8314013ccf4830ca732486206fbed9e6",
    "url": "https://zenodo.org/api/records/16874898/files/data.zip/content",
}

SPEECH = re.compile(r"^(?P<session>\d+)-(?P<material>a_n|iau|phrase)\.nsp$")


def file_hash(path: Path, algo: str, chunk: int = 1 << 24) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}


def save_manifest(manifest: dict) -> None:
    tmp = MANIFEST.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(MANIFEST)


def verify() -> None:
    size = ZIP.stat().st_size
    md5 = file_hash(ZIP, "md5")
    ok = size == ZENODO["size"] and md5 == ZENODO["md5"]
    manifest = load_manifest()
    manifest["source"] = ZENODO
    manifest["zip_check"] = {
        "checked_on": dt.date.today().isoformat(),
        "size": size,
        "md5": md5,
        "matches_record": ok,
    }
    save_manifest(manifest)
    print(f"size {size} md5 {md5} -> {'match' if ok else 'MISMATCH'}")
    if not ok:
        sys.exit(1)


def extract() -> None:
    manifest = load_manifest()
    if not manifest.get("zip_check", {}).get("matches_record"):
        sys.exit("run 'verify' first; the zip has not been matched against the record")
    OUT.mkdir(parents=True, exist_ok=True)
    listing, rows, meta_files = [], [], []
    with zipfile.ZipFile(ZIP) as zf:
        for info in zf.infolist():
            name = PurePosixPath(info.filename)
            listing.append(
                {
                    "member": info.filename,
                    "size": info.file_size,
                    "compressed": info.compress_size,
                    "crc": f"{info.CRC:08x}",
                    "date_time": "{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}".format(
                        *info.date_time
                    ),
                }
            )
            if info.is_dir():
                continue
            m = SPEECH.match(name.name)
            if m is None:
                if name.suffix.lower() in (".nsp", ".egg"):
                    continue
                # Anything that is not audio is small metadata; keep all of it.
                target = META_DIR / Path(*name.parts[1:] if len(name.parts) > 1 else name.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                data = zf.read(info)  # zipfile checks the CRC here
                target.write_bytes(data)
                meta_files.append(
                    {
                        "member": info.filename,
                        "path": target.relative_to(ROOT).as_posix(),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "size": len(data),
                    }
                )
                continue
            session, material = m["session"], m["material"]
            data = zf.read(info)
            nsp_path = NSP_DIR / session / name.name
            nsp_path.parent.mkdir(parents=True, exist_ok=True)
            nsp_path.write_bytes(data)
            row = {
                "session": int(session),
                "material": material,
                "member": info.filename,
                "folder_session": name.parts[1] if len(name.parts) > 2 else "",
                "crc": f"{info.CRC:08x}",
                "nsp": nsp_path.relative_to(ROOT).as_posix(),
                "nsp_size": len(data),
                "nsp_sha256": hashlib.sha256(data).hexdigest(),
            }
            try:
                nsp = parse_nsp(data)
            except ValueError as err:
                row["reader_error"] = str(err)
                rows.append(row)
                continue
            wav_path = WAV_DIR / session / f"{session}-{material}.wav"
            write_wav16(wav_path, nsp.rate, nsp.samples)
            row.update(
                {
                    "rate": nsp.rate,
                    "n_samples": int(nsp.samples.size),
                    "n_samples_header": nsp.n_samples,
                    "duration_s": nsp.samples.size / nsp.rate,
                    "nsp_date": nsp.date,
                    "header_chunk": nsp.header_chunk,
                    "channel_chunk": nsp.channel_chunk,
                    "chunks": " ".join(c for c, _ in nsp.chunks),
                    "header_peak": nsp.header_peaks[0],
                    "reader_problems": "; ".join(check_nsp(nsp)),
                    "wav": wav_path.relative_to(ROOT).as_posix(),
                    "wav_sha256": file_hash(wav_path, "sha256"),
                }
            )
            rows.append(row)
    pd.DataFrame(listing).to_csv(OUT / "svd_zip_listing.csv", index=False)
    files = pd.DataFrame(rows).sort_values(["session", "material"])
    files.to_csv(OUT / "svd_files.csv", index=False)
    manifest["extraction"] = {
        "done_on": dt.date.today().isoformat(),
        "zip_members": len(listing),
        "speech_files": len(files),
        "by_material": files.material.value_counts().to_dict(),
        "sessions": int(files.session.nunique()),
        "reader_errors": int(files.get("reader_error", pd.Series(dtype=str)).notna().sum()),
        "files_table": "s0_audit/out/svd_files.csv (member, CRC, NSP and WAV SHA-256 per file)",
        "listing_table": "s0_audit/out/svd_zip_listing.csv",
        "meta_files": meta_files,
    }
    manifest["files"] = (
        files[["nsp", "nsp_sha256", "wav", "wav_sha256"]].fillna("").to_dict("records")
    )
    save_manifest(manifest)
    print(
        json.dumps({k: v for k, v in manifest["extraction"].items() if k != "meta_files"}, indent=2)
    )
    print("meta files:", [f["member"] for f in meta_files])


if __name__ == "__main__":
    {"verify": verify, "extract": extract}[sys.argv[1]]()
