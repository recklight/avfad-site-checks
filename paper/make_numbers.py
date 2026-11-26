"""Every number the paper uses, read again from the results and the S0 tables.

Reuses the readers of experiments/continue_numbers.py and adds what only the paper needs. Writes
paper/numbers.txt (each value with its source, then the check of the manuscript and of my notes
on it) and paper/latex/numbers.tex, where `\\val{name}` prints a value and stops the build on
an unknown name.

    .venv/Scripts/python.exe -I -X utf8 -B paper/make_numbers.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
sys.path.insert(0, str(ROOT / "experiments"))

import continue_numbers as cn  # noqa: E402

CLAIMS = PAPER / "claims.md"
README = PAPER / "README.md"
MAIN = PAPER / "latex" / "main.tex"
# The abstract and the cover letter as plain text.
SUBMISSION = (PAPER / "submission" / "abstract.txt", PAPER / "submission" / "cover_letter.txt")
TXT = PAPER / "numbers.txt"
TEX = PAPER / "latex" / "numbers.tex"
S0 = ROOT / "s0_audit" / "out"

# From the documents shipped with AVFAD, not from a result file. Belo (2015), section 3.2:
# 436 participants were confirmed by laryngostroboscopy and 273 by the case-history form.
BELO_STROBOSCOPY = 436
BELO_FORM = 273
# Citations of Jesus et al. (2017) in OpenAlex on 2026-11-11.
OPENALEX_CITES = 22

PAPER_FIXED = {
    "Ribas 2023": "citation year",
    "Jesus et al. 2017": "citation year",
    "Belo 2015": "citation year",
    "Belo (2015)": "citation year",
    "8 頁": "page count of the built PDF (pdflatex log)",
    "249 字": "abstract word count",
    "39 條": "a count in the working notes",
    "9 項": "plan section 8 items",
    "41 項": "a count in the working notes",
    "et al. 2016": "citation year",
    "et al. 2019": "citation year",
    "et al. 2005": "citation year",
    "et al. 2006": "citation year",
    "et al. 2023": "citation year",
    "et al. 2025": "citation year",
    "et al. 2026": "citation year",
    "Godino-Llorente 2005": "citation year",
    "§5.1（p. 334）": "where Ribas et al. 2023 say it",
    "handle 10773/15717": "repository handle",
    "10 篇": "count of studies in the related-work notes",
    "5 歲": "matching band of AVFAD (Jesus et al. 2017)",
    "門檻 10 點": "C1 and C2 thresholds (analysis_plan.md)",
    "2015 年": "calendar year",
    "2014 年": "calendar year",
    "95%": "95% intervals",
    "12 November 2026": "date of the editorial office reply",
}


def put(name: str, text: str, source: str) -> None:
    """Add a value; a name continue_numbers already has must agree with it."""
    if name in cn.values:
        if cn.values[name][0] != text:
            raise ValueError(f"{name}: {cn.values[name][0]} there, {text} here")
        return
    cn.put(name, text, source)


def u2(x: float) -> str:
    return f"{x:.2f}"


def s0() -> None:
    """Recruitment and procedure: counts, years, days, file length, noise floor, age."""
    table = pd.read_csv(S0 / "avfad_speakers.csv", dtype={"speaker": str})
    source = "s0_audit/out/avfad_speakers.csv"
    essua = table[(table["place"] == "ESSUA") & (table["label"] == "normal")]
    hsa_n = table[(table["place"] == "HSA") & (table["label"] == "normal")]
    hsa_p = table[(table["place"] == "HSA") & (table["label"] == "pathological")]
    others = table[table["place"].isin(["HSJ", "ULSM"]) & (table["label"] == "normal")]
    patients = int((table["label"] == "pathological").sum())
    put("n.other_normals", str(len(others)), source)
    put("n.patients", str(patients), source)
    put("n.normals_outside_essua", str(len(hsa_n) + len(others)), source)
    put("belo.stroboscopy", str(BELO_STROBOSCOPY), "Belo 2015, section 3.2")
    put("belo.form", str(BELO_FORM), "Belo 2015, section 3.2")
    put("cites.openalex", str(OPENALEX_CITES), "OpenAlex, 2026-11-11")
    put("belo.stroboscopy_normals", str(BELO_STROBOSCOPY - patients), "436 minus the patients")
    put("year.essua_2015", cn.pts((essua["year"] == 2015).mean()), source)
    put("year.hsa_patient_2014", cn.pts((hsa_p["year"] == 2014).mean()), source)
    put("days.hsa_patient", str(hsa_p["visit_date"].nunique()), source)

    describe = pd.read_csv(S0 / "avfad_describe_a_normal_by_place.csv").set_index("variable")
    source = "s0_audit/out/avfad_describe_a_normal_by_place.csv"
    for variable, name in (
        ("duration_s", "length"),
        ("noise_floor_dbfs", "floor"),
        ("age", "age_median"),
    ):
        for place in ("ESSUA", "HSA"):
            median = describe.loc[variable, place].split(" ", 1)[0]
            put(f"{name}.{place.lower()}", median, source)


def field_count() -> None:
    """Smoking and laryngeal surgery by group (post hoc count, readings written first)."""
    summary = cn.load("field_count/summary.json")
    source = "field_count/summary.json"
    groups = {"essua_normal": "essua", "hsa_normal": "hsa", "hsa_patient": "hsa_patient"}
    for group, name in groups.items():
        for flag in ("former", "current", "surgery"):
            put(f"{flag}.{name}", cn.pts(summary["groups"][group][flag]["share"]), source)


def procedure_table() -> None:
    """Table 1: recording years by group and the patients' side of the S0 description."""
    table = pd.read_csv(S0 / "avfad_speakers.csv", dtype={"speaker": str})
    source = "s0_audit/out/avfad_speakers.csv"
    groups = {
        "essua": ("ESSUA", "normal"),
        "hsa": ("HSA", "normal"),
        "hsa_patient": ("HSA", "pathological"),
    }
    for name, (place, label) in groups.items():
        years = table.loc[(table["place"] == place) & (table["label"] == label), "year"]
        for year in (2013, 2014, 2015):
            put(f"year.{name}.{year}", f"{100 * (years == year).mean():.0f}", source)
    hsa = pd.read_csv(S0 / "avfad_describe_a_hsa.csv").set_index("variable")
    source = "s0_audit/out/avfad_describe_a_hsa.csv"
    for variable, name in (
        ("duration_s", "length"),
        ("noise_floor_dbfs", "floor"),
        ("age", "age_median"),
    ):
        put(f"{name}.hsa_patient", hsa.loc[variable, "pathological"].split(" ", 1)[0], source)
        put(f"{name}.hsa", hsa.loc[variable, "normal"].split(" ", 1)[0], source)


def place_and_published() -> None:
    oracle = cn.load("place_only/summary_oracle.json")
    put("oracle.sensitivity", cn.u(oracle["sensitivity"]), "place_only/summary_oracle.json")
    put("oracle.specificity", cn.u(oracle["specificity"]), "place_only/summary_oracle.json")
    put(
        "oracle.real.compare", cn.u(oracle["real_uar"]["compare"]), "place_only/summary_oracle.json"
    )
    phrase = cn.load("published_splits/published_predictions.json")["phrase"]
    source = "published_splits/published_predictions.json"
    put("published.phrase.all", f"{100 * phrase['uar_all']:.1f}", source)
    put("published.phrase.hsa", f"{100 * phrase['uar_hsa']:.1f}", source)
    paper_a = cn.load("published_splits/published_predictions.json")["a_n"]["paper_uar_percent"]
    put("published.paper.a", f"{paper_a:.2f}", source + ", their Table 3")
    method = cn.load("ribas_method/hsa.json")
    for material in ("a_n", "phrase"):
        put(
            f"ribas_method.{material}.ii",
            cn.u(method[material]["ii"]["uar"]),
            "ribas_method/hsa.json",
        )


def inside_hsa() -> None:
    """UAR inside HSA, the same five detectors as the summary's first line."""
    sources = {
        "first_batch/summary.json": cn.load("first_batch/summary.json")["c1"]["uar_ii"],
        "sensitivity/hubert/ii_summary.json": cn.load("sensitivity/hubert/ii_summary.json")["uar"],
        "ribas_method/hsa.json": cn.load("ribas_method/hsa.json")["a_n"]["ii"]["uar"],
        "ribas_method/hsa_balanced.json": cn.load("ribas_method/hsa_balanced.json")["a_n"]["ii"][
            "uar"
        ],
        "finetune/summary.json": cn.load("finetune/summary.json")["ii"]["uar"],
    }
    source = ", ".join(sources)
    put("inside.min", cn.u(min(sources.values())), source)
    put("inside.max", cn.u(max(sources.values())), source)


def gap_range() -> None:
    """C1 gap over materials, detectors and the published method, as in the summary."""
    files = {
        "first_batch/summary.json": [cn.load("first_batch/summary.json")["c1"]],
        "sensitivity/hubert/c1_summary.json": [cn.load("sensitivity/hubert/c1_summary.json")],
        "speech/c1_handcrafted.json": [
            cn.load("speech/c1_handcrafted.json")[m] for m in ("cape_v", "reading")
        ],
        "speech/c1_hubert.json": [
            cn.load("speech/c1_hubert.json")[m] for m in ("cape_v", "reading")
        ],
        "ribas_method/c1.json": [
            r["gap_i_minus_ii"] for r in cn.load("ribas_method/c1.json").values()
        ],
        "ribas_method/c1_balanced.json": [
            r["gap_i_minus_ii"] for r in cn.load("ribas_method/c1_balanced.json").values()
        ],
        "finetune/summary.json": [cn.load("finetune/summary.json")["gap_i_minus_ii"]],
    }
    gaps = [r["difference"] for rows in files.values() for r in rows]
    source = ", ".join(files)
    put("gaps.min", cn.pts(min(gaps)), source)
    put("gaps.max", cn.pts(max(gaps)), source)


def normals_called_normal() -> None:
    """Share of HSA normals that the pooled detectors call normal (complements flagged.*)."""
    inside = cn.load("place_only/summary_inside_hsa.json")
    kept = [inside[d]["full_specificity_hsa_normals"] for d in ("handcrafted", "hubert")]
    source = "place_only/summary_inside_hsa.json"
    put("inside.normal_called.min", f"{round(100 * min(kept))}", source)
    put("inside.normal_called.max", f"{round(100 * max(kept))}", source)


def design_counts() -> None:
    """Counts and shares the methods section quotes."""
    first = cn.load("first_batch/summary.json")
    source = "first_batch/summary.json"
    put("C1.essua_share", cn.pts(first["c1"]["draw_place_share"]["ESSUA"]), source)
    put("C2.essua", str(first["c2"]["ESSUA"]), source)
    put("C2.n", str(first["c2"]["n_speakers"]), source)
    table = pd.read_csv(S0 / "avfad_speakers.csv")
    put(
        "n.normals", str(int((table["label"] == "normal").sum())), "s0_audit/out/avfad_speakers.csv"
    )
    room = cn.load("place_only/summary_room.json")
    put("room.n", str(room["speakers"]["all"]), "place_only/summary_room.json")
    put("room.auc", cn.u(room["all_avfad"]["auc"]), "place_only/summary_room.json")
    rep = cn.load("ribas_method/reproduce.json")["a_n"]
    gap = abs(rep["uar_mean_over_folds_percent"] - rep["paper_uar_percent"])
    put("ribas.reproduce.gap", f"{gap:.1f}", "ribas_method/reproduce.json")
    put(
        "ribas.reproduce.agree",
        cn.pts(rep["agreement_with_their_predictions"]),
        "ribas_method/reproduce.json",
    )
    everyone = {
        "exploratory/avfad_all.json": [
            cn.load("exploratory/avfad_all.json")[k]["uar"]
            for k in ("all_handcrafted", "all_hubert")
        ],
        "ribas_method/hsa.json": [cn.load("ribas_method/hsa.json")["a_n"]["everyone"]["uar"]],
        "finetune/summary.json": [cn.load("finetune/summary.json")["all"]["uar"]],
    }
    values = [v for rows in everyone.values() for v in rows]
    put("everyone.min", cn.u(min(values)), ", ".join(everyone))
    put("everyone.max", cn.u(max(values)), ", ".join(everyone))


def planned_analyses() -> None:
    """Section 8 and 9 of the plan (always reported), C4's own caveats, and a few details."""
    s8 = cn.load("exploratory/section8.json")
    src = "exploratory/section8.json"
    leak = s8["avfad"]["segmenter_leak"]
    put("s8.leak.all", cn.u(leak["all"]["uar"]), src)
    put("s8.leak.hsa", cn.u(leak["hsa"]["uar"]), src)
    put("s8.leak.hsa.low", cn.u(leak["hsa"]["uar_ci"][0]), src)
    put("s8.leak.hsa.high", cn.u(leak["hsa"]["uar_ci"][1]), src)
    put("s8.leak.hsa.p", f"{leak['hsa']['p_one_sided']:.3f}", src)
    quiet = s8["avfad"]["non_phonation"]
    put("s8.quiet.all", cn.u(quiet["all"]["uar"]), src)
    put("s8.quiet.hsa", cn.u(quiet["hsa"]["uar"]), src)
    year = s8["svd"]["recording_year"]
    put("s8.year.uar", cn.u(year["uar"]), src)
    put("s8.year.minus_null", cn.pts(year["uar_minus_null"]), src)
    put("s8.phrase.minus_null", cn.pts(s8["svd"]["phrase_length"]["uar_minus_null"]), src)
    put("s8.agesex.uar", cn.u(s8["svd"]["age_sex"]["random_forest"]["uar"]), src)
    rooms = s8["rooms"]
    put("s8.rooms.essua", f"{rooms['ESSUA']['median_rms_dbfs']:.1f}", src)
    put("s8.rooms.hsa", f"{rooms['HSA']['median_rms_dbfs']:.1f}", src)
    low = (
        rooms["HSA"]["median_band_db"]["ltas_0126"] - rooms["ESSUA"]["median_band_db"]["ltas_0126"]
    )
    put("s8.rooms.low_band", f"{low:.1f}", src)

    cells = cn.load("exploratory/intervention.json")["cells"]
    src = "exploratory/intervention.json"
    essua = cells["ESSUA_with_HSA_+0"]
    put("intervention.0db.low", cn.pts(essua["ci"][0]), src)
    put("intervention.0db.high", cn.pts(essua["ci"][1]), src)
    put("intervention.control", cn.signed(cells["HSA_with_HSA_+0"]["added_flag_rate"]), src)

    h2e = cn.load("exploratory/hsa_to_essua.json")
    src = "exploratory/hsa_to_essua.json"
    for name, row in h2e.items():
        gain = row["specificity_essua_matched"] - row["specificity_hsa_out_of_fold"]
        put(f"h2e.{name}", cn.signed(gain), src)

    sens = cn.load("svd/c4_sensitivity.json")
    src = "svd/c4_sensitivity.json"
    for name, short in (
        ("guard_50_ms", "guard50"),
        ("minimum_0.5_s", "min05"),
        ("age_bands_10_years", "bands"),
    ):
        put(f"c4s.{short}", cn.signed(sens[name]["uar_minus_null"]), src)
    put("c4s.guard200", cn.pts(abs(sens["guard_200_ms"]["uar_minus_null"])), src)
    put("c4s.guard200.n", str(sens["guard_200_ms"]["n_speakers"]), src)

    c4 = cn.load("svd/c4.json")
    put("C4.n", str(c4["n_speakers"]), "svd/c4.json")
    put("C4.left_out", str(c4["files_left_out"]), "svd/c4.json")
    put("C4.avail_alone", cn.u(c4["availability_alone"]["uar"]), "svd/c4.json")

    first = cn.load("first_batch/summary.json")
    shuffled = [first["gate"]["negative"]["mean_uar"], first["c2"]["negative"]["mean_uar"]]
    shuffled += [arm["mean_uar"] for arm in first["c1"]["negative"]]
    put("negative.min", cn.u(min(shuffled)), "first_batch/summary.json")
    put("negative.max", cn.u(max(shuffled)), "first_batch/summary.json")

    ribas = cn.load("ribas_method/hsa.json")["a_n"]["ii"]
    put("ribas_method.ii.sens", cn.u(ribas["sensitivity"]), "ribas_method/hsa.json")
    put("ribas_method.ii.spec", cn.u(ribas["specificity"]), "ribas_method/hsa.json")
    balanced = cn.load("ribas_method/hsa_balanced.json")["a_n"]["ii"]["uar"]
    put("ribas_method.balanced.ii", cn.u(balanced), "ribas_method/hsa_balanced.json")
    pub = cn.load("published_splits/published_predictions.json")
    src = "published_splits/published_predictions.json"
    put("published.spec_hsa", cn.u(pub["a_n"]["specificity_hsa"]["rate"]), src)
    for material, short in (("a_n", "a"), ("phrase", "phrase")):
        p = pub[material]
        put(f"pub.{short}.all", cn.u(p["uar_all"]), src)
        put(f"pub.{short}.hsa", cn.u(p["uar_hsa"]), src)
    put("pub.a.hsa.low", cn.u(pub["a_n"]["uar_hsa_ci"][0]), src)
    put("pub.a.hsa.high", cn.u(pub["a_n"]["uar_hsa_ci"][1]), src)
    put("pub.paper.a", cn.u(pub["a_n"]["paper_uar_percent"] / 100), src + ", their Table 3")

    view = cn.load("outside_view/summary.json")
    src = "outside_view/summary.json"
    put("outside.a.ii_auc", cn.u(view["outside"]["ii"]["auc"]), src)
    put("outside.a.i_auc", cn.u(view["outside"]["i"]["auc"]), src)
    put("inside_view.a.gap", cn.signed(view["inside"]["gap"]["difference"]), src)
    phon = cn.load("phonation_c1/summary.json")
    put(
        "phonation_c1.hubert",
        cn.pts(phon["hubert"]["gap_i_minus_ii"]["difference"]),
        "phonation_c1",
    )


def outside_u() -> None:
    u = cn.load("outside_view/summary_u.json")["outside"]
    put("outside.u.ii", cn.u(u["ii"]["auc"]), "outside_view/summary_u.json")
    put("outside.u.i", cn.u(u["i"]["auc"]), "outside_view/summary_u.json")


def diagnosis() -> None:
    chapters = cn.load("diagnosis_breakdown/summary.json")["hubert"]["chapters"]
    source = "diagnosis_breakdown/summary.json"
    for name, row in chapters.items():
        put(f"diag.{name}.n", str(row["patients"]), source)
        put(f"diag.{name}.ii", cn.u(row["ii"]["auc"]), source)
    table = pd.read_csv(S0 / "avfad_speakers.csv")
    hsa_p = table[(table["place"] == "HSA") & (table["label"] == "pathological")]
    reflux = int(hsa_p["cmvd_name"].str.contains("Reflux", case=False).sum())
    put("diag.reflux", str(reflux), "s0_audit/out/avfad_speakers.csv, CMVD-I name")


def svd_small() -> None:
    s = cn.load("svd_inside/summary_small.json")
    source = "svd_inside/summary_small.json"
    for detector in ("handcrafted", "hubert"):
        put(f"svd_small.{detector}", cn.u(s[detector]["within_strata"]["auc"]), source)
        put(f"svd_small.hsa.{detector}", cn.u(s[detector]["hsa_within_strata"]["auc"]), source)


def review_round() -> None:
    """Values added in a later round of checks; every one was already in a result file."""
    place = cn.load("place_only/summary.json")
    source = "place_only/summary.json"
    for detector in ("handcrafted", "hubert"):
        hsa = place[detector]["hsa_only"]
        put(f"place.{detector}.hsa", cn.u(hsa["auc"]), source)
        put(f"place.{detector}.hsa.low", cn.u(hsa["ci_low"]), source)
        put(f"place.{detector}.hsa.high", cn.u(hsa["ci_high"]), source)
    inside = cn.load("place_only/summary_inside_hsa.json")
    source = "place_only/summary_inside_hsa.json"
    for detector in ("handcrafted", "hubert"):
        put(f"pooled.hsa_auc.{detector}", cn.u(inside[detector]["full_auc"]), source)
        put(f"trained_hsa_auc.{detector}", cn.u(inside[detector]["ii_auc"]), source)
    tuned = cn.load("finetune/summary.json")
    put("finetune.shuffled", cn.u(tuned["shuffled"]["uar"]), "finetune/summary.json")
    put("finetune.shuffled_five", cn.u(tuned["shuffled_five"]["uar_mean"]), "finetune/summary.json")
    table = pd.read_csv(S0 / "avfad_speakers.csv", dtype={"speaker": str})
    patients = table[table["label"] == "pathological"]
    put(
        "n.other_patients",
        str(int((patients["place"] != "HSA").sum())),
        "s0_audit/out/avfad_speakers.csv",
    )


def presubmission() -> None:
    """Values for the final check; every one already in a result file."""
    cells = cn.load("exploratory/intervention.json")
    source = "exploratory/intervention.json"
    c = cells["cells"]
    put("iv.m6", f"+{100 * c['ESSUA_with_HSA_-6']['added_flag_rate']:.1f}", source)
    put("iv.p6", f"+{100 * c['ESSUA_with_HSA_+6']['added_flag_rate']:.1f}", source)
    put("iv.own", f"+{100 * c['ESSUA_with_ESSUA_+0']['added_flag_rate']:.1f}", source)
    put("iv.logit", f"{c['ESSUA_with_HSA_+0']['logit_shift']:.2f}", source)
    put("iv.logit.control", f"{c['HSA_with_HSA_+0']['logit_shift']:.2f}", source)
    put("iv.logit.own", f"{c['ESSUA_with_ESSUA_+0']['logit_shift']:.2f}", source)
    clean = cells["flagged_clean"]["HSA"]
    put("iv.hsa_clean", f"{100 * clean:.1f}", source)
    put("iv.headroom", f"{100 * (1 - clean):.1f}", source)

    c2 = cn.load("first_batch/summary.json")["c2"]
    source = "first_batch/summary.json"
    put("c2.draws.min", cn.u(c2["other_draws_uar_min"]), source)
    put("c2.draws.max", cn.u(c2["other_draws_uar_max"]), source)
    put("C2.permutations", str(c2["n_permutations"]), source)
    put("C4.permutations", str(cn.load("svd/c4.json")["n_permutations"]), "svd/c4.json")

    ii = cn.load("first_batch/post_hoc.json")["arm_ii_uar_interval"]
    put("C1.ii.low", cn.u(ii["low"]), "first_batch/post_hoc.json")
    put("C1.ii.high", cn.u(ii["high"]), "first_batch/post_hoc.json")
    hub = cn.load("sensitivity/hubert/ii_summary.json")["uar_interval"]
    put("C1.hubert.ii.low", cn.u(hub["low"]), "sensitivity/hubert/ii_summary.json")
    put("C1.hubert.ii.high", cn.u(hub["high"]), "sensitivity/hubert/ii_summary.json")

    inside = cn.load("place_only/summary_inside_hsa.json")
    spec = [inside[d]["full_specificity_hsa_normals"] for d in ("handcrafted", "hubert")]
    put("spec.pooled.min", u2(min(spec)), "place_only/summary_inside_hsa.json")
    put("spec.pooled.max", u2(max(spec)), "place_only/summary_inside_hsa.json")

    reproduce = cn.load("ribas_method/reproduce.json")["a_n"]
    put("ribas.reproduce.ours", f"{reproduce['uar_mean_over_folds_percent']:.2f}", "ribas_method/reproduce.json")
    composition = cn.load("published_splits/composition.json")["ribas"]
    put("ribas.list", str(composition["a_n"]["speakers_on_list"]), "published_splits/composition.json")
    both = set(composition["phrase"]["speakers_on_both_sides"])
    assert len(both) == 1
    put("ribas.phrase.both", str(both.pop()), "published_splits/composition.json")

    c3 = cn.load("svd/c3.json")
    put("svd.normal", str(c3["labels"]["normal"]), "svd/c3.json")
    put("svd.pathological", str(c3["labels"]["pathological"]), "svd/c3.json")


def dirty_normals() -> None:
    """V2 in the discussion: intervals at the largest swap, and who was swapped in."""
    dirty = cn.load("dirty_normals/summary.json")
    source = "dirty_normals/summary.json"
    for detector in ("handcrafted", "hubert"):
        gap = dirty[detector]["62"]["gap_i_minus_ii"]
        put(f"v2.{detector}.gap.low", cn.pts(gap["ci_low"]), source)
        put(f"v2.{detector}.gap.high", cn.pts(gap["ci_high"]), source)
    swaps = pd.read_csv(cn.RESULTS / "dirty_normals" / "swaps.csv")
    put("v2.swapped", str(len(swaps)), "dirty_normals/swaps.csv")
    put("age.v2_swapped", f"{swaps['age'].median():.0f}", "dirty_normals/swaps.csv, median age")


SECTIONS = (
    cn.confirmatory,
    cn.place_and_outside,
    cn.metadata,
    cn.simulations,
    cn.records,
    cn.overview,
    s0,
    field_count,
    procedure_table,
    place_and_published,
    inside_hsa,
    gap_range,
    design_counts,
    normals_called_normal,
    planned_analyses,
    outside_u,
    diagnosis,
    svd_small,
    review_round,
    presubmission,
    dirty_normals,
)


def collect() -> None:
    cn.values.clear()
    for section in SECTIONS:
        section()


def check(path: Path) -> tuple[list[str], list[str]]:
    saved = dict(cn.FIXED)
    cn.FIXED.update(PAPER_FIXED)
    try:
        return cn.check_text(path)
    finally:
        cn.FIXED.clear()
        cn.FIXED.update(saved)


# Settings and dates main.tex may state without a result file; each as a whole phrase.
TEX_FIXED = {
    r"95\%": "95% intervals",
    "4 November 2026": "freeze date of the analysis plan (analysis_plan.yaml)",
    "11 November 2026": "query date of the OpenAlex citation count",
    "over 55": "age group of Estevez et al. 2025",
    "48~kHz": "AVFAD sampling rate (Jesus et al. 2017)",
    "mostly in 2015": "Table I, year.essua.2015",
    "mostly in 2014": "Table I, year.hsa_patient.2014",
    r"$\alpha = 0.05$": "analysis_plan.md section 5",
    "78 handcrafted features": "analysis_plan.md section 4",
    "13 mel-frequency": "analysis_plan.md section 4",
    "52 values": "analysis_plan.md section 4",
    "21 one-third-octave": "analysis_plan.md section 4",
    "63 to 6300~Hz": "analysis_plan.md section 4",
    "inner 3-fold": "analysis_plan.md section 4",
    "300--3500~Hz": "analysis_plan.md section 3",
    "$-26$~dBFS": "analysis_plan.md section 4",
    "L2 logistic": "regularization type",
    "182 normal sessions": "SVD index sessions without recordings (analysis_plan.md section 2)",
    "687 normal sessions": "SVD normal sessions with recordings (analysis_plan.md section 2)",
    "more than 10 points": "C4 availability rule (analysis_plan.md section 5)",
    "200~ms": "C4 sensitivity setting (analysis_plan.md section 9)",
    "50~ms": "C4 sensitivity setting (analysis_plan.md section 9)",
    "0.5~s": "C4 sensitivity setting (analysis_plan.md section 9)",
    "(0 dB)": "intervention dose (deviations.md, before the run)",
    "includes 0.5": "VOICED negative-control rule (deviations.md, before the run)",
    "2013/14/15": "calendar years in a table row label",
    "in 2015": "calendar year",
    "CC BY-NC-ND 4.0": "licence name in the AVFAD release README",
    "in 2014": "calendar year",
    "at least 10 points": "reading threshold (deviations.md, before the run)",
    "Claude Opus 5.5": "name of the AI system (acknowledgment)",
    "Claude Haiku 5.5": "name of the AI system used for auxiliary steps (acknowledgment)",
    "30~cm": "microphone distance (Jesus et al. 2017)",
    "(AUC) of 0.861": "hospital-sorting AUC reported by Zech et al. 2018 (abstract)",
    "100~ms": "guard band around phonation (analysis_plan.md section 3)",
    "at 0~dB": "intervention dose (deviations.md, before the run)",
    "$-6$ and $+6$~dB": "intervention doses (analysis_plan.md section 8)",
    "CC BY 4.0": "licence of the SVD release (Zenodo)",
    "under 18": "SVD exclusion (analysis_plan.md section 2)",
    "AUC of 0.846 fell to 0.619": "unadjusted and matched AUCs reported by Coppock et al. 2024 (abstract)",
    "0~dB is the median": "intervention dose definition (deviations.md, before the run)",
}


def tex_body(path: Path) -> str:
    """main.tex without comments, macros' arguments and the preamble."""
    text = path.read_text(encoding="utf-8")
    text = text.split(r"\begin{document}", 1)[1]
    text = re.sub(r"(?<!\\)%.*", " ", text)
    for command in (
        "val", "cite", "ref", "label", "input", "bibliography", "bibliographystyle", "hspace",
    ):
        text = re.sub(r"\\" + command + r"\{[^}]*\}", " ", text)
    text = re.sub(r"\\(begin|end)\{[^}]*\}", " ", text)
    return text


# fig_design.tex holds only layout numbers besides its \val{} counts, so it is not scanned.
HAND_WRITTEN = ("tab_procedure.tex", "tab_confirmatory.tex", "tab_planned.tex")


def check_tex(path: Path) -> tuple[list[str], list[str]]:
    text = tex_body(path)
    for name in HAND_WRITTEN:
        extra = path.parent / name
        if extra.exists():
            body = re.sub(r"(?<!\\)%.*", " ", extra.read_text(encoding="utf-8"))
            for command in ("val", "cite", "ref", "label"):
                body = re.sub(r"\\" + command + r"\{[^}]*\}", " ", body)
            # Environment lines carry layout widths and coordinates, not results.
            body = re.sub(r"\\(begin|end)\{[^}]*\}.*", " ", body)
            body = re.sub(r"\\multicolumn\{\d+\}", " ", body)
            body = re.sub(r"\\(node|draw|tikzpicture)\b[^{\n]*", " ", body)
            text += "\n" + body
    matched = []
    for phrase, _ in TEX_FIXED.items():
        matched += [f"{phrase}: fixed"] * text.count(phrase)
        text = text.replace(phrase, " ")
    text = re.sub(r"\b(C[1-4]|S[0-9])\b", " ", text)
    tokens = re.findall(r"(?<![A-Za-z0-9_./-])[-+]?\d[\d,]*(?:\.\d+)?", text)
    return matched, tokens


def tex_value(text: str) -> str:
    """A value as LaTeX text: a real minus sign, and thousands commas kept in braces."""
    text = text.replace(",", "{,}")
    return "\\textminus{}" + text[1:] if text.startswith("-") else text


def write_tex() -> None:
    lines = [
        "% Written by paper/make_numbers.py from the result files; do not edit by hand.",
        "\\makeatletter",
        "\\newcommand{\\val}[1]{\\@ifundefined{pvc@#1}"
        "{\\PackageError{numbers}{unknown value #1}{}}{\\@nameuse{pvc@#1}}}",
    ]
    for name, (text, _) in cn.values.items():
        if not re.fullmatch(r"[A-Za-z0-9_.+-]+", name):
            raise ValueError(f"name not usable in LaTeX: {name}")
        lines.append(f"\\@namedef{{pvc@{name}}}{{{tex_value(text)}}}")
    lines.append("\\makeatother")
    TEX.parent.mkdir(parents=True, exist_ok=True)
    TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    collect()
    width = max(len(k) for k in cn.values)
    lines = ["# numbers of the paper, read again from the results", ""]
    lines += [f"{k:<{width}}  {t:>10}  {s}" for k, (t, s) in cn.values.items()]
    lines += ["", "allowed without a result file:"]
    lines += [f"  {k}: {v}" for k, v in {**cn.FIXED, **PAPER_FIXED, **TEX_FIXED}.items()]
    report = []
    for path in (CLAIMS, README, MAIN, *SUBMISSION):
        if not path.exists():
            continue
        matched, unmatched = check_tex(path) if path == MAIN else check(path)
        report += [
            f"numbers in {path.name}: {len(matched) + len(unmatched)}",
            f"matched: {len(matched)}",
            f"unmatched: {len(unmatched)} {unmatched}",
        ]
        lines += ["", *report[-3:], "each number in the order it appears:"]
        lines += [f"  {m}" for m in matched]
    TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_tex()
    print("\n".join(report))
    print(f"{len(cn.values)} values; wrote {TXT.name} and latex/{TEX.name}")
    print(json.dumps({"values": len(cn.values)}))


if __name__ == "__main__":
    main()
