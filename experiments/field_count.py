"""Smoking and laryngeal surgery among the HSA normals, against the ESSUA normals and HSA patients.

Post hoc, counts only, no model. The readings are in experiments/deviations.md (2026-11-11),
written before this ran. The public AVFAD spreadsheet has no SLT Intervention or Notes column,
so voice therapy cannot be counted; smoking is coded 0 non-smoker, 1 former, 2 current.

    .venv/Scripts/python.exe -I -X utf8 -B experiments/field_count.py
"""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEAKERS = ROOT / "s0_audit" / "out" / "avfad_speakers.csv"
OUT = ROOT / "experiments" / "results" / "field_count" / "summary.json"

Z = 1.959963984540054
THRESHOLD = 0.10
GROUPS = {
    "hsa_normal": ("HSA", "normal"),
    "essua_normal": ("ESSUA", "normal"),
    "hsa_patient": ("HSA", "pathological"),
}
OTHER_NORMALS = ("HSJ", "ULSM")
# AVFAD chapter bands: 18-45, 46-65, over 65.
AGE_EDGES = (45, 65)


def wilson(k: int, n: int) -> tuple[float, float]:
    """Wilson score interval for k of n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + Z**2 / (2 * n)) / (1 + Z**2 / n)
    half = Z * sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / (1 + Z**2 / n)
    return (centre - half, centre + half)


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]:
    """p1 - p2 with Newcombe's hybrid score interval (his method 10, 1998)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lower = d - sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    upper = d + sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (d, lower, upper)


def age_band(age: pd.Series) -> pd.Series:
    low, high = AGE_EDGES
    return pd.Series(
        np.select([age <= low, age <= high], ["18-45", "46-65"], "66+"), index=age.index
    )


def standardized(target: pd.DataFrame, other: pd.DataFrame, flag: str) -> dict:
    """Share of `flag` in `other`, weighted by the target's age-band x sex cells.

    Cells where `other` has nobody are left out; `n_target_used` says how many target people
    the weights still cover.
    """
    weights = target.groupby(["band", "sex"]).size()
    rates = other.groupby(["band", "sex"])[flag].mean()
    used = weights[weights.index.isin(rates.index)]
    value = float((used * rates.loc[used.index]).sum() / used.sum())
    return {"share": value, "n_target_used": int(used.sum())}


def compare(target: pd.DataFrame, other: pd.DataFrame, flag: str, adjust: bool) -> dict:
    k1, n1 = int(target[flag].sum()), len(target)
    k2, n2 = int(other[flag].sum()), len(other)
    d, lower, upper = newcombe(k1, n1, k2, n2)
    result = {"difference": d, "ci": [lower, upper]}
    if adjust:
        std = standardized(target, other, flag)
        result["standardized_other"] = std["share"]
        result["standardized_difference"] = k1 / n1 - std["share"]
        result["n_target_used"] = std["n_target_used"]
    return result


def read_vs_essua(c: dict) -> str:
    """Reading written in deviations.md before the counts were seen."""
    lower, upper = c["ci"]
    crude_up = c["difference"] >= THRESHOLD and lower > 0
    crude_down = c["difference"] <= -THRESHOLD and upper < 0
    std_up = c["standardized_difference"] >= THRESHOLD
    std_down = c["standardized_difference"] <= -THRESHOLD
    if crude_up and std_up:
        return "HSA 正常人的比例比 ESSUA 高"
    if crude_down and std_down:
        return "比 ESSUA 低"
    if crude_up or crude_down:
        return "差別大多來自年齡和性別"
    return "看不出兩地正常人有差"


def read_vs_patients(c: dict) -> str:
    lower, upper = c["ci"]
    if abs(c["difference"]) < THRESHOLD and lower <= 0 <= upper:
        return "和同院病人相近"
    if abs(c["difference"]) >= THRESHOLD and (lower > 0 or upper < 0):
        return "和同院病人不同"
    return "照實"


def describe(group: pd.DataFrame) -> dict:
    n = len(group)
    out = {"n": n, "age_median": float(group["age"].median())}
    for flag in ("never", "former", "current", "ever", "surgery"):
        k = int(group[flag].sum())
        out[flag] = {"k": k, "share": k / n if n else float("nan"), "ci": list(wilson(k, n))}
    out["bands"] = {
        band: {
            "n": int((group["band"] == band).sum()),
            "former": int(group.loc[group["band"] == band, "former"].sum()),
            "ever": int(group.loc[group["band"] == band, "ever"].sum()),
        }
        for band in ("18-45", "46-65", "66+")
    }
    return out


def prepare(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    smoking = table["smoking"].astype(int)
    if not set(smoking.unique()) <= {0, 1, 2}:
        raise ValueError(f"unexpected smoking codes: {sorted(smoking.unique())}")
    table["never"] = smoking == 0
    table["former"] = smoking == 1
    table["current"] = smoking == 2
    table["ever"] = smoking >= 1
    table["surgery"] = table["laryngeal_surgery"].astype(int) == 1
    table["band"] = age_band(table["age"])
    return table


def main() -> None:
    table = prepare(pd.read_csv(SPEAKERS, dtype={"speaker": str}))
    groups = {
        name: table[(table["place"] == place) & (table["label"] == label)]
        for name, (place, label) in GROUPS.items()
    }
    others = table[table["place"].isin(OTHER_NORMALS) & (table["label"] == "normal")]
    target = groups["hsa_normal"]

    comparisons, readings = {}, {}
    for flag in ("former", "ever"):
        vs_essua = compare(target, groups["essua_normal"], flag, adjust=True)
        vs_patients = compare(target, groups["hsa_patient"], flag, adjust=True)
        comparisons[flag] = {"vs_essua_normal": vs_essua, "vs_hsa_patient": vs_patients}
        readings[flag] = {
            "vs_essua_normal": read_vs_essua(vs_essua),
            "vs_hsa_patient": read_vs_patients(vs_patients),
        }

    summary = {
        "source": SPEAKERS.relative_to(ROOT).as_posix(),
        "smoking_codes": {"0": "non-smoker", "1": "former smoker", "2": "smoker"},
        "not_in_public_release": ["SLT Intervention", "Notes"],
        "threshold_points": THRESHOLD * 100,
        "groups": {name: describe(g) for name, g in groups.items()},
        "other_normals": describe(others),
        "comparisons": comparisons,
        "readings": readings,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for name, g in summary["groups"].items():
        print(
            f"{name:13s} n={g['n']:3d} age {g['age_median']:.0f}  "
            f"former {g['former']['k']:3d} ({g['former']['share']:.1%})  "
            f"current {g['current']['k']:3d} ({g['current']['share']:.1%})  "
            f"surgery {g['surgery']['k']:3d} ({g['surgery']['share']:.1%})"
        )
    for flag, c in comparisons.items():
        for other, r in c.items():
            std = r.get("standardized_difference")
            print(
                f"{flag:6s} {other:15s} {r['difference']:+.3f} "
                f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] std {std:+.3f}  "
                f"{readings[flag][other]}"
            )
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
