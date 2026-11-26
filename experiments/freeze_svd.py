"""The second freeze: the C3 and C4 scripts, written after the first batch had run.

The first freeze stays as it was; this record also holds its hash, so a new first freeze
would stop these runs too. The scripts were written after C1 and C2 were seen, but C3 and C4
are on another database and their definitions were fixed in the frozen plan.

    .venv/Scripts/python.exe -I -B experiments/freeze_svd.py build
    .venv/Scripts/python.exe -I -B experiments/freeze_svd.py verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from freeze import ROOT, FreezeError, compare_hashes, file_hashes  # noqa: E402

FREEZE_SVD_PATH = HERE / "freeze_svd.json"
TRACKED = (
    "experiments/freeze.json",
    "experiments/freeze_svd.py",
    "experiments/run_svd.py",
    "experiments/arms.py",
    "experiments/tests/test_svd.py",
    "s0_audit/out/svd_file_status.csv",
)


def build_svd_freeze(path: Path = FREEZE_SVD_PATH, root: Path = ROOT) -> dict:
    record = {"files": file_hashes(TRACKED, root)}
    path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return record


def verify_svd_freeze(path: Path = FREEZE_SVD_PATH, root: Path = ROOT) -> dict:
    if not path.exists():
        raise FreezeError(f"{path} does not exist; the C3 and C4 scripts are not frozen yet")
    record = json.loads(path.read_text(encoding="utf-8"))
    problems = compare_hashes(record["files"], TRACKED, root)
    if problems:
        raise FreezeError("the second freeze differs:\n" + "\n".join(problems))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["build", "verify"])
    args = parser.parse_args()
    try:
        record = build_svd_freeze() if args.action == "build" else verify_svd_freeze()
    except FreezeError as err:
        sys.exit(str(err))
    print(f"{args.action}: {len(record['files'])} files")


if __name__ == "__main__":
    main()
