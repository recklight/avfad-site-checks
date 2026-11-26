"""Which HuBERT layer the inner loop chose, (ii) against the 50 draws of (i). Description only.

Exploratory. The reading is in experiments/deviations.md (2026-11-10), written before counting.
Each arm stores, in ``chosen_c``, how often each layer won across its 50 outer folds.

    .venv/Scripts/python.exe -I -X utf8 -B experiments/chosen_layers.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from arms import write_json  # noqa: E402

RESULTS = HERE / "results" / "chosen_layers"
SOURCES = {
    "a_full": HERE / "results" / "sensitivity" / "hubert" / "c1_arms.csv",
    "a_phonation": HERE / "results" / "phonation_c1" / "c1_arms_hubert.csv",
    "i_full": HERE / "results" / "vowel_fusion" / "arms_i.csv",
    "u_full": HERE / "results" / "vowel_fusion" / "arms_u.csv",
}
LAYERS = ("l00", "l03", "l06", "l09", "l12")
# Reading, written before counting (deviations.md, 2026-11-10).
SHIFT = 0.20


def shares(records: list[str]) -> dict[str, float]:
    counts = pd.Series(0.0, index=list(LAYERS))
    for record in records:
        for layer, n in json.loads(record)["layer"].items():
            counts[layer] += n
    return (counts / counts.sum()).round(4).to_dict()


def reading(ii: dict[str, float], i: dict[str, float]) -> str:
    shift = i["l00"] - ii["l00"]
    if shift >= SHIFT:
        return "(i) leans to the shallowest layer"
    if shift <= -SHIFT:
        return "(i) leans to deeper layers"
    return "both choose about the same layers"


def main() -> None:
    summary = {}
    for name, path in SOURCES.items():
        arms = pd.read_csv(path)
        arms = arms[~arms["arm"].str.startswith("shuffled")]
        ii = shares(arms.loc[arms["arm"] == "ii", "chosen_c"].tolist())
        i = shares(arms.loc[arms["arm"].str.startswith("i_"), "chosen_c"].tolist())
        summary[name] = {"ii": ii, "i": i, "reading": reading(ii, i)}
        print(name, summary[name])
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / "summary.json", summary)


if __name__ == "__main__":
    main()
