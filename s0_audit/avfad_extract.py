"""S0 for AVFAD: unpack the downloaded folders and record a SHA-256 for every file.

Usage:
  python -I s0_audit/avfad_extract.py unpack      # zips in data/avfad/raw/ -> data/avfad/files/
  python -I s0_audit/avfad_extract.py inventory   # hash and WAV header of every file

The shared OneDrive folders are downloaded by hand into data/avfad/raw/. A
browser download of a folder comes as a zip; if the folders arrive already
unpacked (the zips were not kept), `unpack` has nothing to do and `inventory`
reads the folders in raw/ directly. Aveiro publishes no checksums and
OneDrive gives none, so the SHA-256 taken here is the reference from now on.
Nothing in raw/ is changed or moved.

A zip that tests fine can still be incomplete: when OneDrive skips items
(SharePoint throttling, TooManyRequests or WebException), it puts
___All_Errors.txt and one <name>_Error.txt per folder into the download
instead. Both steps stop before hashing anything if such a file is in raw/
or files/, or if raw/ also holds copies unpacked from its own zips.

The analysis scripts shared with the data belong to Aveiro. They are hashed
like every other file, read for reference only, and never copied into a
repository.

Both steps only look at the structure: hashes, file names and WAV headers.
Anything that reads the recordings or the metadata (avfad_stats.py, the
label tables) stops unless data/avfad/access.json holds the date on which
data retention was confirmed to be off (TalkBank Ground Rules).
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
from readers import check_wav, read_wav

ROOT = Path(__file__).resolve().parents[1]
AVFAD = ROOT / "data" / "avfad"
RAW = AVFAD / "raw"
FILES = AVFAD / "files"
MANIFEST = AVFAD / "manifest.json"
ACCESS = AVFAD / "access.json"
OUT = Path(__file__).resolve().parent / "out"

SOURCE = {
    "dataset": "Advanced Voice Function Assessment Databases (AVFAD)",
    "provider": "University of Aveiro (Luís M. T. Jesus)",
    "approved": "2026-11-03, by e-mail; shared as five OneDrive folders",
    "conditions": [
        "non-commercial research only",
        "kept locally only",
        "not redistributed",
        "cite Jesus et al. (2017)",
        "CC BY-NC-ND 4.0, the licence of the TalkBank copy, is followed as well",
    ],
    "citation": (
        "Jesus, L. M. T., Belo, I., Machado, J., & Hall, A. (2017). The Advanced Voice "
        "Function Assessment Databases (AVFAD): Tools for voice clinicians and speech "
        "research. In Advances in Speech-language Pathology. InTech."
    ),
    "doi": "10.5772/intechopen.69643",
}

WAV_COLUMNS = [
    "rate",
    "channels",
    "bits",
    "valid_bits",
    "sample_format",
    "extensible",
    "n_frames",
    "duration_s",
    "chunks",
    "reader_problems",
    "reader_error",
]


def file_hash(path: Path, algo: str = "sha256", chunk: int = 1 << 24) -> str:
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


def access_date() -> str | None:
    """The confirmation date in access.json, or None if it is missing or malformed."""
    try:
        access = json.loads(ACCESS.read_text(encoding="utf-8"))
        return dt.date.fromisoformat(access["retention_off_confirmed_on"]).isoformat()
    except (OSError, KeyError, TypeError, ValueError):
        return None


def require_access() -> str:
    confirmed = access_date()
    if confirmed is None:
        sys.exit(
            f"{ACCESS.relative_to(ROOT).as_posix()} needs a valid "
            '"retention_off_confirmed_on" date before the recordings or metadata are read'
        )
    return confirmed


def is_zip(path: Path) -> bool:
    # Checking the suffix first keeps zipfile from opening thousands of WAVs.
    return path.suffix.lower() == ".zip" and zipfile.is_zipfile(path)


# OneDrive's markers for skipped items: ___All_Errors.txt and <name>_Error.txt.
ERROR_FILE = re.compile(r"_Errors?\.txt$", re.IGNORECASE)


def check_complete(downloads: list[Path], unpacked: list[Path] | None = None) -> None:
    """Exit if a download carries OneDrive error files or raw/ holds unpacked copies.

    Counts are given per zip or per top-level folder of raw/; file names are
    not printed.
    """
    errors: dict[str, int] = {}
    member_paths: set[str] = set()
    loose = []
    for path in downloads:
        rel = path.relative_to(RAW)
        if is_zip(path):
            with zipfile.ZipFile(path) as zf:
                names = [n.replace("\\", "/").rstrip("/") for n in zf.namelist()]
            n_bad = sum(bool(ERROR_FILE.search(PurePosixPath(n).name)) for n in names)
            if n_bad:
                errors[rel.as_posix()] = n_bad
            stem = rel.with_suffix("").as_posix()
            member_paths.update(names)
            member_paths.update(f"{stem}/{n}" for n in names)
        else:
            loose.append(rel)
            if ERROR_FILE.search(path.name):
                errors[rel.parts[0]] = errors.get(rel.parts[0], 0) + 1
    for path in unpacked or []:
        if ERROR_FILE.search(path.name):
            top = path.relative_to(FILES).parts[0]
            errors[f"files/{top}"] = errors.get(f"files/{top}", 0) + 1
    copies = sum(rel.as_posix() in member_paths for rel in loose)
    problems = [f"{where}: {n} OneDrive error file(s)" for where, n in sorted(errors.items())]
    if copies:
        problems.append(f"raw/ holds {copies} file(s) unpacked from its own zips")
    if problems:
        sys.exit("AVFAD download is not complete; nothing was hashed.\n  " + "\n  ".join(problems))


def safe_member_path(name: str) -> PurePosixPath | None:
    """The member's path below the target folder, or None if it would land elsewhere."""
    p = PurePosixPath(name.replace("\\", "/"))
    if p.is_absolute() or not p.parts:
        return None
    if any(part in ("", ".", "..") or ":" in part for part in p.parts):
        return None
    return p


def unpack() -> None:
    downloads = sorted(p for p in RAW.rglob("*") if p.is_file())
    if not downloads:
        sys.exit(f"{RAW.relative_to(ROOT).as_posix()} is empty")
    check_complete(downloads)
    if not any(is_zip(p) for p in downloads):
        print("no zip in raw/; run 'inventory', which reads the folders there directly")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    rows, members, rejected = [], [], []
    for path in downloads:
        stat = path.stat()
        zipped = is_zip(path)
        rel = path.relative_to(ROOT).as_posix()
        rows.append(
            {
                "path": rel,
                "size": stat.st_size,
                "sha256": file_hash(path),
                "modified": dt.datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                "zip": zipped,
            }
        )
        if not zipped:
            continue
        target_root = FILES / path.relative_to(RAW).with_suffix("")
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                inner = safe_member_path(info.filename)
                if inner is None:
                    rejected.append({"zip": rel, "member": info.filename})
                    continue
                target = target_root / Path(*inner.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(target.name + ".part")
                h = hashlib.sha256()
                # zipfile checks the member's CRC when the last block is read.
                with zf.open(info) as src, tmp.open("wb") as dst:
                    while block := src.read(1 << 24):
                        h.update(block)
                        dst.write(block)
                tmp.replace(target)
                members.append(
                    {
                        "zip": rel,
                        "member": info.filename,
                        "path": target.relative_to(ROOT).as_posix(),
                        "size": info.file_size,
                        "crc": f"{info.CRC:08x}",
                        "sha256": h.hexdigest(),
                        "nested_zip": inner.suffix.lower() == ".zip",
                    }
                )
    pd.DataFrame(members).to_csv(OUT / "avfad_unpacked.csv", index=False)
    manifest = load_manifest()
    manifest["source"] = SOURCE
    manifest["downloads"] = rows
    manifest["unpack"] = {
        "done_on": dt.date.today().isoformat(),
        "downloads": len(rows),
        "zips": sum(r["zip"] for r in rows),
        "members": len(members),
        "nested_zips": sum(m["nested_zip"] for m in members),
        "rejected_members": rejected,
        "members_table": "s0_audit/out/avfad_unpacked.csv",
    }
    save_manifest(manifest)
    print(json.dumps(manifest["unpack"], indent=2, ensure_ascii=False))


def inventory() -> None:
    unpacked = sorted(p for p in FILES.rglob("*") if p.is_file() and not p.name.endswith(".part"))
    downloads = sorted(p for p in RAW.rglob("*") if p.is_file())
    check_complete(downloads, unpacked)
    zips = [p for p in downloads if is_zip(p)]
    paths = unpacked + [p for p in downloads if p not in zips]
    if not paths:
        sys.exit("nothing to inventory; run 'unpack' first")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in paths:
        if path.is_relative_to(FILES):
            # files/<zip name>/<the zip's own folders>: drop the zip name so both
            # sources give <shared folder>/<speaker>/<file>.
            parts = path.relative_to(FILES).parts[1:]
        else:
            parts = path.relative_to(RAW).parts
        row = {
            "path": path.relative_to(ROOT).as_posix(),
            "folder": parts[0] if len(parts) > 1 else "",
            "speaker": parts[1] if len(parts) > 2 else "",
            "ext": path.suffix.lower(),
            "ext_as_named": path.suffix,
            "size": path.stat().st_size,
            "sha256": file_hash(path),
        }
        if row["ext"] == ".wav":
            try:
                wav = read_wav(path, load=False)
            except ValueError as err:
                row["reader_error"] = str(err)
            else:
                row.update(
                    {
                        "rate": wav.rate,
                        "channels": wav.channels,
                        "bits": wav.bits,
                        "valid_bits": wav.valid_bits,
                        "sample_format": wav.sample_format,
                        "extensible": wav.extensible,
                        "n_frames": wav.n_frames,
                        "duration_s": wav.duration_s,
                        "chunks": " ".join(c for c, _ in wav.chunks),
                        "reader_problems": "; ".join(check_wav(wav)),
                    }
                )
        rows.append(row)
    files = pd.DataFrame(rows)
    for col in WAV_COLUMNS:
        if col not in files:
            files[col] = pd.NA
    files.to_csv(OUT / "avfad_files.csv", index=False)
    wavs = files[files.ext == ".wav"]
    read_ok = wavs[wavs.rate.notna()]
    formats = (
        read_ok.rate.astype(int).astype(str)
        + " Hz, "
        + read_ok.channels.astype(int).astype(str)
        + " ch, "
        + read_ok.bits.astype(int).astype(str)
        + "-bit "
        + read_ok.sample_format.astype(str)
    )
    # Per folder: how many speakers hold n files of each type (0 included).
    in_speaker = files[files.speaker.fillna("") != ""]
    counts = in_speaker.groupby(["folder", "speaker"]).ext.value_counts().unstack(fill_value=0)
    per_speaker = {
        folder: {
            ext: {int(n): int(k) for n, k in grp[ext].value_counts().sort_index().items()}
            for ext in grp.columns
        }
        for folder, grp in counts.groupby(level="folder")
    }
    by_folder_format = pd.crosstab(read_ok.folder, formats)
    manifest = load_manifest()
    manifest["source"] = SOURCE
    manifest["access"] = {"retention_off_confirmed_on": access_date()}
    if not zips and not unpacked:
        manifest["downloads_note"] = "zip 未保留；OneDrive 不提供 checksum"  # noqa: RUF001
    else:
        manifest.pop("downloads_note", None)
    manifest["inventory"] = {
        "done_on": dt.date.today().isoformat(),
        "files": len(files),
        "bytes": int(files["size"].sum()),
        "by_ext": files.ext.value_counts().to_dict(),
        "by_folder": files.folder.value_counts().to_dict(),
        "suffix_not_lower_case": files[files.ext_as_named != files.ext]
        .groupby("folder")
        .size()
        .to_dict(),
        "wav_formats": formats.value_counts().to_dict(),
        "wav_formats_by_folder": {
            folder: {fmt: int(n) for fmt, n in row.items() if n}
            for folder, row in by_folder_format.iterrows()
        },
        "speakers_by_folder": in_speaker.groupby("folder").speaker.nunique().to_dict(),
        "files_per_speaker": per_speaker,
        "wav_hours": round(float(read_ok.duration_s.sum()) / 3600, 2),
        "wav_reader_errors": len(wavs) - len(read_ok),
        "wav_with_problems": int((read_ok.reader_problems.fillna("") != "").sum()),
        "files_table": "s0_audit/out/avfad_files.csv",
    }
    manifest["files"] = files[["path", "size", "sha256"]].to_dict("records")
    save_manifest(manifest)
    print(json.dumps(manifest["inventory"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    {"unpack": unpack, "inventory": inventory}[sys.argv[1]]()
