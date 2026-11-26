"""The freeze record of the experiments and the check every run starts with.

``build`` writes experiments/freeze.json: the SHA-256 of the plan, the inputs, the feature
tables and the scripts, plus the package versions. ``verify`` recomputes all of it and also
reads which commit of voice-severity-bench is installed. pip records that commit in
``direct_url.json`` when it installs from git, and RECORD holds a hash per installed file,
so a run notices an edit inside site-packages as well. Any difference stops the run; a
change made on purpose goes to experiments/deviations.md and a new freeze.

    .venv/Scripts/python.exe -I -B experiments/freeze.py build
    .venv/Scripts/python.exe -I -B experiments/freeze.py verify
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import platform
import sys
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = ROOT / "experiments" / "freeze.json"

PLAN_SHA256 = {
    "analysis_plan.md": "2de9e87f5243606ff7b23cbd7d710a99ee5019d537c41aa5208b178232b204ee",
    "analysis_plan.yaml": "fa2866d62a6132a02f84f274b9e3bc0710fecf53475b0bbba5f068a0cbf97764",
}
VSB_DISTRIBUTION = "voice-severity-bench"
VSB_COMMIT = "35de45dee783cc7c5b635e723d988efa47b2a214"
PINNED_VERSIONS = {"scikit-learn": "1.7.2", "speechdsp": "0.2.0"}
RECORDED_PACKAGES = ("joblib", "numpy", "pandas", "PyYAML", "scikit-learn", "scipy", "speechdsp")

TRACKED_FILES = (
    "analysis_plan.md",
    "analysis_plan.yaml",
    "s0_audit/readers.py",
    "s0_audit/out/avfad_speakers.csv",
    "s0_audit/out/svd_sessions.csv",
    "s1_segments/segment.py",
    "s1_segments/out/index.csv",
    "s1_segments/out/files.csv",
    "s1_segments/out/spans.csv",
    "s1_segments/out/params.json",
    "features/common.py",
    "features/extract_handcrafted.py",
    "features/check_features.py",
    "features/out/check.json",
    "experiments/freeze.py",
    "experiments/harness.py",
    "experiments/run_first_batch.py",
)
TRACKED_GLOBS = ("features/out/*.csv",)


class FreezeError(RuntimeError):
    """The frozen state and the working state differ."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tracked_paths(root: Path = ROOT) -> list[str]:
    names = set(TRACKED_FILES)
    for pattern in TRACKED_GLOBS:
        names.update(p.relative_to(root).as_posix() for p in root.glob(pattern))
    return sorted(names)


def file_hashes(names: Iterable[str], root: Path = ROOT) -> dict[str, str]:
    return {name: sha256(root / name) for name in names}


def package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    versions.update({name: metadata.version(name) for name in RECORDED_PACKAGES})
    return versions


def compare_hashes(frozen: dict[str, str], names: Iterable[str], root: Path = ROOT) -> list[str]:
    """Differences between the recorded hashes and the files as they are now."""
    problems = []
    now = set(names)
    for name in sorted(now - set(frozen)):
        problems.append(f"{name} is not in the freeze")
    for name, expected in sorted(frozen.items()):
        path = root / name
        if not path.exists():
            problems.append(f"{name} is gone")
        elif sha256(path) != expected:
            problems.append(f"{name} changed")
    return problems


def check_direct_url(raw: str | None, commit: str = VSB_COMMIT) -> list[str]:
    """``direct_url.json`` has to name the frozen commit of a git install."""
    if raw is None:
        return ["voice-severity-bench has no direct_url.json; it was not installed from git"]
    info = json.loads(raw)
    vcs = info.get("vcs_info")
    if vcs is None:
        return [f"voice-severity-bench came from {info.get('url')}, not from a git commit"]
    if vcs.get("vcs") != "git" or vcs.get("commit_id") != commit:
        return [f"voice-severity-bench is at {vcs.get('commit_id')}, the freeze says {commit}"]
    return []


def check_record(files: Iterable[metadata.PackagePath]) -> list[str]:
    """Installed files against the hashes pip wrote at install time."""
    problems = []
    for entry in files:
        if entry.hash is None:  # scripts and bytecode that pip made after unpacking
            continue
        if entry.hash.mode != "sha256":
            problems.append(f"{entry}: unexpected hash {entry.hash.mode}")
            continue
        path = Path(str(entry.locate()))
        if not path.exists():
            problems.append(f"{entry} is gone")
            continue
        value = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest())
        if value.rstrip(b"=").decode() != entry.hash.value:
            problems.append(f"{entry} changed after install")
    return problems


def check_installed_vsb() -> list[str]:
    try:
        dist = metadata.distribution(VSB_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return ["voice-severity-bench is not installed in this environment"]
    problems = check_direct_url(dist.read_text("direct_url.json"))
    problems += check_record(dist.files or [])
    spec = importlib.util.find_spec("voice_severity_bench")
    expected = Path(str(dist.locate_file("voice_severity_bench/__init__.py"))).resolve()
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != expected:
        origin = None if spec is None else spec.origin
        problems.append(f"voice_severity_bench imports from {origin}, not from {expected}")
    return problems


def check_versions(frozen: dict[str, str]) -> list[str]:
    now = package_versions()
    problems = [
        f"{name} is {now.get(name)}, the freeze says {version}"
        for name, version in frozen.items()
        if now.get(name) != version
    ]
    problems += [
        f"{name} is {now.get(name)}, the plan pins {version}"
        for name, version in PINNED_VERSIONS.items()
        if now.get(name) != version
    ]
    return problems


def check_plan_hashes(root: Path = ROOT) -> list[str]:
    return [
        f"{name} is not the frozen plan"
        for name, expected in PLAN_SHA256.items()
        if sha256(root / name) != expected
    ]


def build_freeze(path: Path = FREEZE_PATH, root: Path = ROOT) -> dict:
    problems = check_plan_hashes(root)
    if problems:
        raise FreezeError("; ".join(problems))
    record = {
        "voice_severity_bench_commit": VSB_COMMIT,
        "files": file_hashes(tracked_paths(root), root),
        "versions": package_versions(),
    }
    path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return record


def verify_freeze(path: Path = FREEZE_PATH, root: Path = ROOT) -> dict:
    """Stop with every difference listed; return the record when nothing differs."""
    if not path.exists():
        raise FreezeError(f"{path} does not exist; nothing is frozen yet")
    record = json.loads(path.read_text(encoding="utf-8"))
    problems = check_plan_hashes(root)
    problems += compare_hashes(record["files"], tracked_paths(root), root)
    problems += check_versions(record["versions"])
    problems += check_installed_vsb()
    if problems:
        raise FreezeError("the frozen state differs:\n" + "\n".join(problems))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["build", "verify"])
    args = parser.parse_args()
    try:
        if args.action == "build":
            record = build_freeze()
            print(f"{len(record['files'])} files frozen in {FREEZE_PATH.name}")
        else:
            record = verify_freeze()
            print(f"freeze holds: {len(record['files'])} files, commit {VSB_COMMIT}")
    except FreezeError as err:
        sys.exit(str(err))


if __name__ == "__main__":
    main()
