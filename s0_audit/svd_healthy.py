"""Fill in the normal sessions that data.zip lacks, from healthy.zip.

Usage:
  python -I s0_audit/svd_healthy.py verify    # size + MD5 against the Zenodo record
  python -I s0_audit/svd_healthy.py extract

voice_data.csv lists 869 normal sessions; data.zip holds 687 of them. For
every a_n, iau and phrase speech file in healthy.zip:
- sessions already taken from data.zip are compared byte for byte (SHA-256),
  which checks the two archives against each other;
- the other sessions are extracted the same way as from data.zip and appended
  to s0_audit/out/svd_files.csv with source_zip = healthy.zip.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import sys
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import svd_extract as base
from readers import check_nsp, parse_nsp, write_wav16

ZIP = base.SVD / "healthy.zip"
ZENODO = {
    **base.ZENODO,
    "file": "healthy.zip",
    "size": 6020018616,
    "md5": "67789ec89a161ce26d562719bc8d1aaa",
    "url": "https://zenodo.org/api/records/16874898/files/healthy.zip/content",
}


def verify() -> None:
    size = ZIP.stat().st_size
    md5 = base.file_hash(ZIP, "md5")
    ok = size == ZENODO["size"] and md5 == ZENODO["md5"]
    manifest = base.load_manifest()
    manifest["healthy_source"] = ZENODO
    manifest["healthy_zip_check"] = {
        "checked_on": dt.date.today().isoformat(),
        "size": size,
        "md5": md5,
        "matches_record": ok,
    }
    base.save_manifest(manifest)
    print(f"size {size} md5 {md5} -> {'match' if ok else 'MISMATCH'}")
    if not ok:
        sys.exit(1)


def extract() -> None:
    manifest = base.load_manifest()
    if not manifest.get("healthy_zip_check", {}).get("matches_record"):
        sys.exit("run 'verify' first; healthy.zip has not been matched against the record")
    files_csv = base.OUT / "svd_files.csv"
    files = pd.read_csv(files_csv)
    if "source_zip" not in files:
        files["source_zip"] = "data.zip"
    files = files[files.source_zip == "data.zip"]  # rerunnable: drop earlier healthy rows
    known = {(int(s), m): h for s, m, h in zip(files.session, files.material, files.nsp_sha256)}
    listing, new_rows, overlap, meta_files = [], [], [], []
    with zipfile.ZipFile(ZIP) as zf:
        for info in zf.infolist():
            listing.append(
                {
                    "member": info.filename,
                    "size": info.file_size,
                    "compressed": info.compress_size,
                    "crc": f"{info.CRC:08x}",
                }
            )
            if info.is_dir():
                continue
            name = PurePosixPath(info.filename)
            m = base.SPEECH.match(name.name)
            if m is None:
                if name.suffix.lower() in (".nsp", ".egg"):
                    continue
                target = base.META_DIR / "healthy_zip" / Path(*name.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                data = zf.read(info)
                target.write_bytes(data)
                meta_files.append(
                    {
                        "member": info.filename,
                        "path": target.relative_to(base.ROOT).as_posix(),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "size": len(data),
                    }
                )
                continue
            session, material = int(m["session"]), m["material"]
            data = zf.read(info)
            digest = hashlib.sha256(data).hexdigest()
            if (session, material) in known:
                overlap.append(
                    {
                        "session": session,
                        "material": material,
                        "same_bytes": digest == known[(session, material)],
                    }
                )
                continue
            nsp_path = base.NSP_DIR / str(session) / name.name
            nsp_path.parent.mkdir(parents=True, exist_ok=True)
            nsp_path.write_bytes(data)
            row = {
                "session": session,
                "material": material,
                "member": info.filename,
                "folder_session": name.parts[-3] if len(name.parts) >= 3 else "",
                "crc": f"{info.CRC:08x}",
                "nsp": nsp_path.relative_to(base.ROOT).as_posix(),
                "nsp_size": len(data),
                "nsp_sha256": digest,
                "source_zip": "healthy.zip",
            }
            try:
                nsp = parse_nsp(data)
            except ValueError as err:
                row["reader_error"] = str(err)
                new_rows.append(row)
                continue
            wav_path = base.WAV_DIR / str(session) / f"{session}-{material}.wav"
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
                    "wav": wav_path.relative_to(base.ROOT).as_posix(),
                    "wav_sha256": base.file_hash(wav_path, "sha256"),
                }
            )
            new_rows.append(row)
    pd.DataFrame(listing).to_csv(base.OUT / "svd_healthy_listing.csv", index=False)
    overlap = pd.DataFrame(overlap)
    overlap.to_csv(base.OUT / "svd_healthy_overlap.csv", index=False)
    added = pd.DataFrame(new_rows)
    files = pd.concat([files, added], ignore_index=True).sort_values(["session", "material"])
    files.to_csv(files_csv, index=False)
    summary = {
        "done_on": dt.date.today().isoformat(),
        "zip_members": len(listing),
        "overlap_files": len(overlap),
        "overlap_identical": int(overlap.same_bytes.sum()) if len(overlap) else 0,
        "added_files": len(added),
        "added_sessions": int(added.session.nunique()) if len(added) else 0,
        "added_by_material": added.material.value_counts().to_dict() if len(added) else {},
        "reader_errors": int(added.get("reader_error", pd.Series(dtype=str)).notna().sum()),
        "meta_files": meta_files,
    }
    manifest["healthy_extraction"] = summary
    manifest["files"] = (
        files[["nsp", "nsp_sha256", "wav", "wav_sha256"]].fillna("").to_dict("records")
    )
    base.save_manifest(manifest)
    print({k: v for k, v in summary.items() if k != "meta_files"})
    print("meta files:", [f["member"] for f in meta_files][:20], len(meta_files))


if __name__ == "__main__":
    {"verify": verify, "extract": extract}[sys.argv[1]]()
