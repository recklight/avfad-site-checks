"""The AVFAD scores reported the proposed way: inside HSA first, everyone only beside it.

Assembles numbers already computed; nothing is fitted except the AUC inside the age by sex
strata for the ComParE runs, from their stored predictions. Definition in
experiments/deviations.md (2026-11-09).

    .venv/Scripts/python.exe -I -X utf8 -B experiments/protocol_table.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_first_batch as batch  # noqa: E402
from arms import write_json  # noqa: E402
from harness import age_sex_cell  # noqa: E402
from within_strata_auc import within_strata  # noqa: E402

RESULTS = HERE / "results"
OUT = RESULTS / "protocol"
SEED = 20261204


def load(relative: str):
    path = RESULTS / relative
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def row(detector, material, ii_uar, ii_auc, strata, all_uar, all_auc, quiet=None) -> dict:
    return {
        "detector": detector,
        "material": material,
        "hsa_uar": ii_uar,
        "hsa_auc": ii_auc,
        "hsa_within_strata_auc": None if strata is None else strata["auc"],
        "hsa_within_strata_ci": None if strata is None else strata["ci"],
        "all_uar": all_uar,
        "all_auc": all_auc,
        "non_phonation_all_to_hsa": quiet,
    }


def compare_strata(path: Path) -> dict | None:
    """AUC inside the strata of the HSA arm; sentence scores averaged per speaker first."""
    if not path.exists():
        return None
    table = pd.read_csv(path, dtype={"speaker": str})
    part = table[table["arm"] == "ii"]
    wide = part.groupby(["speaker", "repeat"])["score"].mean().unstack("repeat")
    speakers = batch.load_speakers().set_index("speaker").loc[wide.index]
    sick = (speakers["label"] == "pathological").to_numpy()
    cells = age_sex_cell(speakers.reset_index()).to_numpy().astype(str)
    return within_strata(wide.to_numpy(), sick, cells, SEED)


def finetune_strata(tuned: dict) -> dict:
    """AUC inside the strata of the fine-tuned (ii) arm, from the stored fits."""
    fits = RESULTS / "finetune" / "fits"
    lr = tuned["learning_rate"]
    speakers = batch.load_speakers().set_index("file_id")
    columns = []
    for repeat in range(tuned["ii"]["repeats"]):
        scores = {}
        for fold in range(5):
            record = json.loads(
                (fits / f"ii_r{repeat}_f{fold}_lr{lr:.0e}.json").read_text(encoding="utf-8")
            )
            scores.update(zip(record["test_items"], record["scores"], strict=True))
        columns.append(pd.Series(scores))
    wide = pd.concat(columns, axis=1).sort_index()
    chosen = speakers.loc[wide.index].reset_index()
    sick = (chosen["label"] == "pathological").to_numpy()
    cells = age_sex_cell(chosen).to_numpy().astype(str)
    return within_strata(wide.to_numpy(), sick, cells, SEED)


def build() -> list[dict]:
    first = load("first_batch/summary.json")["c1"]
    every = load("exploratory/avfad_all.json")
    strata = load("exploratory/within_strata_auc.json")
    hub_ii = load("sensitivity/hubert/ii_summary.json")
    s8 = load("exploratory/section8.json")["avfad"]["non_phonation"]
    speech = {k: load(f"speech/c1_{k}.json") for k in ("handcrafted", "hubert")}
    inside = load("speech/summary.json")["inside_strata"]
    quiet = load("speech/nonphonation.json")
    rows = [
        row(
            "handcrafted",
            "/a/",
            first["uar_ii"],
            first["roc_auc_ii"],
            strata["avfad_hsa_handcrafted"],
            every["all_handcrafted"]["uar"],
            every["all_handcrafted"]["roc_auc"],
            [s8["all"]["uar"], s8["hsa"]["uar"]],
        ),
        row(
            "HuBERT (frozen)",
            "/a/",
            hub_ii["uar"],
            hub_ii["roc_auc"],
            strata["avfad_hsa_hubert"],
            every["all_hubert"]["uar"],
            every["all_hubert"]["roc_auc"],
        ),
    ]
    for kind, label in (("handcrafted", "handcrafted"), ("hubert", "HuBERT (frozen)")):
        for material, name in (("cape_v", "six sentences"), ("reading", "reading")):
            record = speech[kind][material]
            rows.append(
                row(
                    label,
                    name,
                    record["uar_ii"],
                    record["roc_auc_ii"],
                    inside[kind][material],
                    record["all"]["uar"],
                    record["all"]["roc_auc"],
                    [
                        quiet[material]["non_phonation"]["all"]["uar"],
                        quiet[material]["non_phonation"]["hsa"]["uar"],
                    ]
                    if kind == "handcrafted"
                    else None,
                )
            )
    for suffix, label in (
        ("", "ComParE + SVM (their setting)"),
        ("_balanced", "ComParE + SVM, class weight"),
    ):
        record = load(f"ribas_method/c1{suffix}.json")
        for material, name in (("a", "/a/"), ("sentences", "six sentences")):
            if record is None or material not in record:
                rows.append(row(label, name, None, None, None, None, None))
                continue
            r = record[material]
            rows.append(
                row(
                    label,
                    name,
                    r["ii"]["uar"],
                    r["ii"]["roc_auc"],
                    compare_strata(
                        RESULTS / "ribas_method" / f"c1_predictions_{material}{suffix}.csv.gz"
                    ),
                    r["all"]["uar"],
                    r["all"]["roc_auc"],
                )
            )
    tuned = load("finetune/summary.json")
    if tuned is not None:
        rows.append(
            row(
                "HuBERT, last 2 layers fine-tuned",
                "/a/",
                tuned["ii"]["uar"],
                tuned["ii"]["roc_auc"],
                finetune_strata(tuned),
                tuned["all"]["uar"],
                tuned["all"]["roc_auc"],
            )
        )
    else:
        rows.append(row("HuBERT, last 2 layers fine-tuned", "/a/", None, None, None, None, None))
    return rows


def number(value) -> str:
    return "not run" if value is None else f"{value:.3f}"


def markdown(rows: list[dict]) -> str:
    lines = [
        "| detector | material | HSA UAR | HSA AUC | HSA AUC inside strata | everyone UAR | everyone AUC | non-phonation, everyone -> HSA |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        strata = (
            "not run"
            if r["hsa_within_strata_auc"] is None
            else f"{r['hsa_within_strata_auc']:.3f} [{r['hsa_within_strata_ci'][0]:.3f}, {r['hsa_within_strata_ci'][1]:.3f}]"
        )
        quiet = (
            "-"
            if r["non_phonation_all_to_hsa"] is None
            else f"{r['non_phonation_all_to_hsa'][0]:.3f} -> {r['non_phonation_all_to_hsa'][1]:.3f}"
        )
        lines.append(
            f"| {r['detector']} | {r['material']} | {number(r['hsa_uar'])} | {number(r['hsa_auc'])} | "
            f"{strata} | {number(r['all_uar'])} | {number(r['all_auc'])} | {quiet} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = build()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "table.json", {"rows": rows, "split_seed": 20261104})
    text = markdown(rows)
    (OUT / "table.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
