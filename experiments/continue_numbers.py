"""Every number in two of my working notes, read again from results and records.

Read-only. The only file written is ``results/continue_numbers.txt``. The first part lists
each value under a name, with the file it came from, formatted the way the two notes print
it. The second part takes every number in those notes, which are not part of this record, and
looks it up among those values; a number that matches nothing is listed as unmatched, so a
figure typed by hand cannot slip through.

Settings that are not results (the doses written down before the runs, a citation year) are
in ``FIXED`` as whole phrases, with the reason they are allowed. Dates and clock times are
skipped.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"
ANALYSIS = ROOT / "continue_analysis.md"
OVERVIEW = ROOT / "overview.md"
OUT = RESULTS / "continue_numbers.txt"
SPEAKERS = ROOT / "s0_audit" / "out" / "avfad_speakers.csv"
CURRENT_SMOKER = 2

values: dict[str, tuple[str, str]] = {}


def load(relative: str) -> dict:
    return json.loads((RESULTS / relative).read_text(encoding="utf-8"))


def put(name: str, text: str, source: str) -> None:
    if name in values:
        raise KeyError(f"{name} defined twice")
    values[name] = (text, source)


def u(x: float) -> str:
    return f"{x:.3f}"


def pts(x: float) -> str:
    return f"{100 * x:.1f}"


def signed(x: float) -> str:
    return f"{100 * x:+.1f}"


def put_gap(name: str, record: dict, source: str, sign: bool = False) -> None:
    """A gap in points with its 95% interval, as three names."""
    show = signed if sign else pts
    put(name, show(record["difference"]), source)
    put(name + ".low", pts(record["ci_low"]), source)
    put(name + ".high", pts(record["ci_high"]), source)


def last_seconds(log: Path, prefix: str) -> int:
    """Seconds on the last progress line of one condition ('<prefix>: N of N after S s')."""
    pattern = re.compile(rf"^{re.escape(prefix)}: (\d+) of (\d+) after (\d+) s")
    seconds = None
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pattern.match(line)
        if m and m.group(1) == m.group(2):
            seconds = int(m.group(3))
    if seconds is None:
        raise ValueError(f"{prefix} not finished in {log}")
    return seconds


def files_line(log: Path) -> tuple[int, int]:
    m = re.search(r"(\d+) files in (\d+) s", log.read_text(encoding="utf-8", errors="replace"))
    if m is None:
        raise ValueError(f"no 'files in' line in {log}")
    return int(m.group(1)), int(m.group(2))


def record_in_plan(pattern: str) -> tuple[str, ...]:
    """A timing that only the work log kept: the groups of its one match."""
    plan = (ROOT / "PLAN.md").read_text(encoding="utf-8")
    found = re.findall(pattern, plan)
    if len(found) != 1:
        raise ValueError(f"{len(found)} matches in PLAN.md for {pattern}")
    return found[0] if isinstance(found[0], tuple) else (found[0],)


def worker_hours(seconds: float, workers: int) -> str:
    return f"{seconds * workers / 3600:.1f}"


# ---------------------------------------------------------------- pre-registered family


def confirmatory() -> None:
    family = load("final_family.json")
    first = load("first_batch/summary.json")
    c1, c2, gate = first["c1"], first["c2"], first["gate"]
    src = "first_batch/summary.json"
    put_gap("C1.gap", c1, src)
    put("C1.i", u(c1["uar_i_mean"]), src)
    put("C1.i.min", u(c1["uar_i_min"]), src)
    put("C1.i.max", u(c1["uar_i_max"]), src)
    put("C1.ii", u(c1["uar_ii"]), src)
    put("C1.patients", str(c1["patients"]), src)
    put("C1.normals", str(c1["normals_per_arm"]), src)
    put("C1.draws", str(c1["draws"]), src)
    put("C1.resamples", f"{c1['n_resamples']:,}", src)
    put("C2.uar", u(c2["uar"]), src)
    put("C2.above_chance", pts(c2["uar"] - 0.5), src + " (UAR - 0.5)")
    put("gate.uar", u(gate["uar"]), src)
    put("gate.min", f"{gate['min_uar']:.2f}", src)
    c3 = load("svd/c3.json")
    put_gap("C3.gap", c3, "svd/c3.json")
    c4 = load("svd/c4.json")
    put("C4.uar", u(c4["uar"]), "svd/c4.json")
    put("C4.null", u(c4["null_mean"]), "svd/c4.json")
    put("C4.minus_null", pts(c4["uar_minus_null"]), "svd/c4.json")
    if not c4["holds_only_for_files_with_enough_quiet"]:
        raise ValueError("C4 reading changed")
    avail = c4["availability_by_label"]
    put("C4.avail_diff", pts(-avail["within_strata_difference"]), "svd/c4.json (minus sign)")
    put("C4.avail_low", pts(-avail["within_strata_ci"][1]), "svd/c4.json")
    put("C4.avail_high", pts(-avail["within_strata_ci"][0]), "svd/c4.json")

    # Holm again from the one-sided p values, to check the stored adjusted ones.
    names = ["C1", "C2", "C3", "C4"]
    raw = {k: family[k]["p_one_sided"] for k in names}
    order = sorted(names, key=lambda k: raw[k])
    running = 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (len(names) - rank) * raw[k]))
        if abs(running - family[k]["p_holm"]) > 1e-12:
            raise ValueError(f"Holm for {k}: {running} against {family[k]['p_holm']}")
        shown = f"{running:.4f}" if running < 0.001 else f"{running:.3f}"
        put(f"{k}.p_holm", shown, "final_family.json, Holm recomputed")
        put(f"{k}.smallest", f"{100 * family[k]['smallest_gap']:.0f}", "final_family.json")

    surgery = load("sensitivity/c1_no_surgery/summary.json")
    put_gap("C1.no_surgery", surgery, "sensitivity/c1_no_surgery/summary.json")
    hub = load("sensitivity/hubert/c1_summary.json")
    put_gap("C1.hubert", hub, "sensitivity/hubert/c1_summary.json")
    put("C1.hubert.i", u(hub["uar_i_mean"]), "sensitivity/hubert/c1_summary.json")
    put("C1.hubert.ii", u(hub["uar_ii"]), "sensitivity/hubert/c1_summary.json")
    put("C2.hubert", u(load("sensitivity/hubert/c2_summary.json")["uar"]), "sensitivity/hubert")

    voiced = load("exploratory/section8.json")["voiced"]
    tech = voiced["m_tech"]["with_length"]
    put("voiced.tech", u(tech["uar"]), "exploratory/section8.json")
    lo, hi = tech["uar_ci"]
    put("voiced.tech.low", u(lo), "exploratory/section8.json")
    put("voiced.tech.high", u(hi), "exploratory/section8.json")
    put("voiced.normals", str(voiced["speakers"]["healthy"]), "exploratory/section8.json")


# ---------------------------------------------------------------- exploratory, rules first


def place_and_outside() -> None:
    p = load("place_only/summary.json")
    for d in ("handcrafted", "hubert"):
        a = p[d]["all_avfad"]
        put(f"place.{d}", u(a["auc"]), "place_only/summary.json")
        put(f"place.{d}.low", u(a["ci_low"]), "place_only/summary.json")
        put(f"place.{d}.high", u(a["ci_high"]), "place_only/summary.json")
    c1 = load("place_only/summary_c1.json")
    for d in ("handcrafted", "hubert"):
        put(f"place_c1.{d}.i", u(c1[d]["place_i_mean"]), "place_only/summary_c1.json")
        put(f"place_c1.{d}.real_i", u(c1[d]["real_i_mean"]), "place_only/summary_c1.json")
    put("oracle.uar", u(load("place_only/summary_oracle.json")["uar"]), "place_only")
    every = load("exploratory/avfad_all.json")
    for d in ("handcrafted", "hubert"):
        put(f"full.{d}.uar", u(every[f"all_{d}"]["uar"]), "exploratory/avfad_all.json")
        put(f"full.{d}.auc", u(every[f"all_{d}"]["roc_auc"]), "exploratory/avfad_all.json")
    put("room.uar", u(load("place_only/summary_room.json")["uar"]), "place_only/summary_room.json")
    inside = load("place_only/summary_inside_hsa.json")
    off = [1 - inside[d]["full_specificity_hsa_normals"] for d in ("handcrafted", "hubert")]
    put("flagged.min", f"{round(100 * min(off))}", "place_only/summary_inside_hsa.json")
    put("flagged.max", f"{round(100 * max(off))}", "place_only/summary_inside_hsa.json")

    out = load("outside_view/summary.json")
    put_gap("outside.a", out["outside"]["gap"], "outside_view/summary.json", sign=True)
    put_gap("learned.a", out["learned"], "outside_view/summary.json", sign=True)
    for v in ("i", "u"):
        o = load(f"outside_view/summary_{v}.json")
        put(f"outside.{v}", signed(o["outside"]["gap"]["difference"]), f"outside_view/summary_{v}")
    hand = load("outside_view/summary_a_handcrafted.json")
    put("learned.a.hand", signed(hand["learned"]["difference"]), "outside_view")
    for v in ("a", "i", "u"):
        e = load(f"outside_view/summary_{v}_outside_erased.json")["outside"]["gap"]
        put(
            f"outside_erased.{v}",
            signed(e["difference"]),
            f"outside_view/summary_{v}_outside_erased",
        )

    x = load("cross_database/summary.json")
    hub_x = x["hubert"]
    put("dext.hubert", signed(hub_x["d_ext"]["auc"]), "cross_database/summary.json")
    put("dext.hubert.low", pts(hub_x["d_ext"]["auc_ci"][0]), "cross_database/summary.json")
    put("dext.hubert.high", pts(hub_x["d_ext"]["auc_ci"][1]), "cross_database/summary.json")
    put("dext.hand", signed(x["handcrafted"]["d_ext"]["auc"]), "cross_database/summary.json")
    put("svd.speakers", f"{hub_x['svd']['speakers']:,}", "cross_database/summary.json")
    voiced_x = load("cross_database/voiced.json")
    put("dext.voiced", signed(voiced_x["d_ext"]["auc"]), "cross_database/voiced.json")
    put("dext.voiced.low", pts(voiced_x["d_ext"]["ci"][0]), "cross_database/voiced.json")
    put("dext.voiced.high", pts(voiced_x["d_ext"]["ci"][1]), "cross_database/voiced.json")
    put("voiced.speakers", str(voiced_x["speakers"]), "cross_database/voiced.json")
    vowels = load("cross_vowels/summary.json")
    for v in ("i", "u"):
        d = vowels[v]["d_ext"]
        put(f"dext.{v}", signed(d["auc"]), "cross_vowels/summary.json")
        put(f"dext.{v}.low", pts(d["auc_ci"][0]), "cross_vowels/summary.json")
        put(f"dext.{v}.high", pts(d["auc_ci"][1]), "cross_vowels/summary.json")
    external = [hub_x["ii"]["auc"], hub_x["i"]["auc"], hub_x["all"]["auc"]]
    external += [voiced_x["ii"]["auc"], voiced_x["i"]["auc"]]
    put("external.min", f"{min(external):.2f}", "cross_database (HuBERT and VOICED AUCs)")
    put("external.max", f"{max(external):.2f}", "cross_database (HuBERT and VOICED AUCs)")
    put("external.svd_ii", u(hub_x["ii"]["auc"]), "cross_database/summary.json")
    put("external.svd_all", u(hub_x["all"]["auc"]), "cross_database/summary.json")
    hand_x = x["handcrafted"]
    hand_aucs = [hand_x[k]["auc"] for k in ("ii", "i", "all", "essua_normals_hsa_patients")]
    put("external.hand.min", u(min(hand_aucs)), "cross_database/summary.json, handcrafted")
    put("external.hand.max", u(max(hand_aucs)), "cross_database/summary.json, handcrafted")
    for v in ("i", "u"):
        mixed = vowels[v]["essua_normals_hsa_patients"]["auc"]
        put(f"mixed_contrast.{v}", f"{mixed:.2f}", "cross_vowels, ESSUA normals vs HSA patients")

    pub = load("published_splits/published_predictions.json")["a_n"]
    put("published.all", f"{100 * pub['uar_all']:.1f}", "published_splits")
    put("published.hsa", f"{100 * pub['uar_hsa']:.1f}", "published_splits")
    put("published.hsa.low", f"{100 * pub['uar_hsa_ci'][0]:.1f}", "published_splits")
    put("published.hsa.high", f"{100 * pub['uar_hsa_ci'][1]:.1f}", "published_splits")

    dose = load("dose_response/summary.json")
    put("dose.slope", signed(dose["hubert"]["slope"]["slope"]), "dose_response/summary.json")
    erased = load("dose_response/summary_erased.json")
    put("dose.erased", signed(erased["hubert_erased"]["slope"]["slope"]), "dose_response")

    placebo = load("placebo_c1/summary.json")
    put("placebo.hand.min", signed(placebo["distribution"]["min"]), "placebo_c1/summary.json")
    put("placebo.hand.max", signed(placebo["distribution"]["max"]), "placebo_c1/summary.json")
    put("placebo.hubert.min", signed(placebo["hubert_range"]["min"]), "placebo_c1/summary.json")
    put("placebo.hubert.max", signed(placebo["hubert_range"]["max"]), "placebo_c1/summary.json")
    put("placebo.hand.splits", str(placebo["distribution"]["splits"]), "placebo_c1/summary.json")
    splits = [r for k, r in placebo["handcrafted"].items() if k.isdigit()]
    if len(splits) != placebo["distribution"]["splits"]:
        raise ValueError("placebo split count")
    above = sum(r["gap_i_minus_ii"]["ci_low"] > 0 for r in splits)
    put("placebo.hand.above_zero", str(above), "placebo_c1/summary.json, CI above 0")

    by_day = load("c1_by_day/summary.json")
    for d in ("handcrafted", "hubert"):
        put(f"by_day.{d}", pts(by_day[d]["gap_i_minus_ii"]["difference"]), "c1_by_day")

    within = load("exploratory/within_strata_auc.json")
    for d in ("handcrafted", "hubert"):
        w = within[f"avfad_hsa_{d}"]
        put(f"strata.{d}", u(w["auc"]), "exploratory/within_strata_auc.json")
    small = load("svd_inside/summary_small.json")
    for d in ("handcrafted", "hubert"):
        put(f"svd_small.{d}", u(small[d]["within_strata"]["auc"]), "svd_inside/summary_small")

    free = load("channel_free_c1/summary.json")["free"]
    put_gap("free.voiced", free["gap_i_minus_ii"], "channel_free_c1/summary.json")
    put("free.columns", str(free["columns"]), "channel_free_c1/summary.json")


def metadata() -> None:
    s = pd.read_csv(SPEAKERS)
    src = "s0_audit/out/avfad_speakers.csv"
    normal = s[s["label"] == "normal"]
    smoke = normal.groupby("place")["smoking"].apply(lambda x: float((x == CURRENT_SMOKER).mean()))
    put("smoke.essua", f"{100 * smoke['ESSUA']:.1f}", src)
    put("smoke.hsa", f"{100 * smoke['HSA']:.1f}", src)
    hsa = s[s["place"] == "HSA"]
    age = hsa.groupby("label")["age"].median()
    put("age.hsa_normal", f"{age['normal']:.0f}", src)
    put("age.hsa_patient", f"{age['pathological']:.0f}", src)
    people = pd.read_csv(RESULTS / "simulated_place" / "people.csv")
    sim_age = people.groupby("role")["age"].median()
    put("age.v1_a", f"{sim_age['A']:.0f}", "simulated_place/people.csv")
    sick = people[people["label"] == "pathological"]["age"].median()
    put("age.v1_patient", f"{sick:.0f}", "simulated_place/people.csv")
    surgery = hsa.groupby("label")["laryngeal_surgery"].apply(lambda x: float((x == 1).mean()))
    put("surgery.hsa_normal", f"{100 * surgery['normal']:.1f}", src)
    put("surgery.hsa_patient", f"{100 * surgery['pathological']:.1f}", src)
    counts = s.groupby(["place", "label"]).size()
    put("n.essua", str(counts[("ESSUA", "normal")]), src)
    put("n.hsa_normal", str(counts[("HSA", "normal")]), src)
    put("n.hsa_patient", str(counts[("HSA", "pathological")]), src)
    put("n.avfad", str(len(s)), src)


# ---------------------------------------------------------------- V1, V1b, V2


def simulations() -> None:
    sim = load("simulated_place/summary.json")
    src = "simulated_place/summary.json"
    for d in ("handcrafted", "hubert"):
        part = sim[d]
        for name in ("none", "noise_12", "silence_1x", "reverb_0.5", "channel_1x", "all"):
            put(f"v1.{d}.{name}", pts(part[name]["gap_i_minus_ii"]["difference"]), src)
        put(f"v1.{d}.all.low", pts(part["all"]["gap_i_minus_ii"]["ci_low"]), src)
        put(f"v1.{d}.all.high", pts(part["all"]["gap_i_minus_ii"]["ci_high"]), src)
        put(f"v1.{d}.ii", u(part["all"]["ii"]["uar"]), src)
        put(f"v1.{d}.i_all", u(part["all"]["i_uar_mean"]), src)
        put(f"v1.{d}.i_none", u(part["none"]["i_uar_mean"]), src)
    for d in ("handcrafted", "hubert"):
        put(f"v1.{d}.minus_none", pts(sim[d]["all"]["minus_none"]["difference"]), src)
    hand = sim["handcrafted"]
    put("v1.handcrafted.reverb_1.0", pts(hand["reverb_1.0"]["gap_i_minus_ii"]["difference"]), src)
    put("avfad.hand.gap.low", pts(load("first_batch/summary.json")["c1"]["ci_low"]), "first_batch")
    free = sim["free"]
    put("v1.free.reverb_0.5", pts(free["reverb_0.5"]["gap_i_minus_ii"]["difference"]), src)
    put("v1.free.reverb_1.0", pts(free["reverb_1.0"]["gap_i_minus_ii"]["difference"]), src)
    places = sim["place"]
    for d in ("handcrafted", "hubert"):
        put(f"v1.place.{d}.all", u(places[d]["all"]["auc"]), src)
    first = load("first_batch/summary.json")["c1"]
    hub = load("sensitivity/hubert/c1_summary.json")
    put("avfad.hubert.gap.low", pts(hub["ci_low"]), "sensitivity/hubert/c1_summary.json")
    put("avfad.half.hand", pts(first["difference"] / 2), "first_batch (half of the C1 gap)")
    put("avfad.half.hubert", pts(hub["difference"] / 2), "sensitivity/hubert (half of the gap)")

    both = load("simulated_place/silence_both/summary.json")
    for d in ("handcrafted", "hubert"):
        put(f"v1b.{d}", pts(both[d]["gap_i_minus_ii"]["difference"]), "silence_both/summary.json")

    dirty = load("dirty_normals/summary.json")
    for d in ("handcrafted", "hubert"):
        r = dirty[d]["62"]
        put(f"v2.{d}.gap", signed(r["gap_i_minus_ii"]["difference"]), "dirty_normals/summary.json")
        put(f"v2.{d}.place", u(r["place"]["auc"]), "dirty_normals/summary.json")
        put(f"v2.{d}.ii", u(r["ii"]["uar"]), "dirty_normals/summary.json")


# ---------------------------------------------------------------- time and CPU


def timings() -> None:
    first = load("first_batch/summary.json")
    put("time.first_c1", f"{first['c1']['seconds']:.1f}", "first_batch/summary.json")
    (workers,) = record_in_plan(r"第一批跑完（正向對照、C1、C2）\*\*，(\d+) workers")
    put("workers.first", workers, "work log, first batch")
    seconds = first["c1"]["seconds"]
    put("cpu.first_c1", worker_hours(seconds, int(workers)), "C1 seconds x workers")

    logs = RESULTS / "simulated_place" / "logs"
    v1 = [last_seconds(logs / "c1_hubert.log", f"hubert {n}") for n in ("all", "noise_12")]
    v1 += [last_seconds(logs / "c1_hubert.log", "hubert reverb_0.5")]
    v1 += [
        last_seconds(logs / "c1_hubert_rest.log", f"hubert {n}")
        for n in ("none", "channel_1x", "silence_1x")
    ]
    put("time.v1_hubert.min", str(round(min(v1) / 60)), "simulated_place/logs, minutes")
    put("time.v1_hubert.max", str(round(max(v1) / 60)), "simulated_place/logs, minutes")
    put("cpu.v1_hubert.min", worker_hours(min(v1), 12), "x 12 workers")
    put("cpu.v1_hubert.max", worker_hours(max(v1), 12), "x 12 workers")
    hand_v1 = last_seconds(logs / "c1_handcrafted.log", "handcrafted all")
    put("time.v1_hand", str(hand_v1), "simulated_place/logs/c1_handcrafted.log")

    dirty = RESULTS / "dirty_normals"
    d = [last_seconds(dirty / "c1_hubert_41.log", "hubert 41")]
    d += [last_seconds(dirty / "c1_hubert_rest.log", f"hubert {n}") for n in (21, 62)]
    put("time.v2_hubert.min", str(round(min(d) / 60)), "dirty_normals logs, minutes")
    put("time.v2_hubert.max", str(round(max(d) / 60)), "dirty_normals logs, minutes")
    put("cpu.v2_hubert.min", worker_hours(min(d), 12), "x 12 workers")
    put("cpu.v2_hubert.max", worker_hours(max(d), 12), "x 12 workers")

    n_files, sec = files_line(logs / "features_hubert.log")
    put("time.feat.files", str(n_files), "simulated_place/logs/features_hubert.log")
    put("time.feat.seconds", str(sec), "simulated_place/logs/features_hubert.log")

    # C1 by recording day is the latest HuBERT C1 on AVFAD; its time is only in the log entry.
    workers, hubert = record_in_plan(r"(\d+) workers：手工 \d+ s，HuBERT ([\d,]+) s")
    hubert_s = int(hubert.replace(",", ""))
    put("time.by_day", hubert, "work log, C1 by recording day")
    put("time.by_day.minutes", str(round(hubert_s / 60)), "same, in minutes")
    put("cpu.by_day", worker_hours(hubert_s, int(workers)), "same x workers")
    put("workers.v1", workers, "work log, C1 by recording day and V1 runs")
    i_min, u_min, feat_workers = record_in_plan(
        r"抽特徵 AVFAD /i/ (\d+) 分、/u/ (\d+) 分，打分數 \d+ s、\d+ s（(\d+) workers）"
    )
    put("time.avfad_i_feat", i_min, "work log, AVFAD /i/ HuBERT features")
    put("time.avfad_u_feat", u_min, "work log, AVFAD /u/ HuBERT features")
    put("workers.feat", feat_workers, "work log, same entry")
    put("cpu.avfad_feat.min", worker_hours(int(i_min) * 60, int(feat_workers)), "x workers")
    put("cpu.avfad_feat.max", worker_hours(int(u_min) * 60, int(feat_workers)), "x workers")
    fits, low, high, threads, together = record_in_plan(
        r"(\d+) 個 fit，每個 (\d+)–(\d+) 分鐘（(\d+) threads），(\d+) 個同時跑"
    )
    put("finetune.fits", fits, "work log, fine-tuning the last 2 layers")
    put("finetune.min", low, "work log, minutes per fit")
    put("finetune.max", high, "work log, minutes per fit")
    put("finetune.threads", threads, "work log, threads per fit")
    put("finetune.together", together, "work log, fits at once")
    put(
        "cpu.finetune.min",
        worker_hours(int(fits) * int(low) * 60, int(threads)),
        "fits x min x thr",
    )
    put(
        "cpu.finetune.max",
        worker_hours(int(fits) * int(high) * 60, int(threads)),
        "fits x min x thr",
    )
    # Two at a time: 55 fits are 28 rounds. The shuffled-label reruns are not counted.
    rounds = math.ceil(int(fits) / int(together))
    put("finetune.rounds", str(rounds), "fits run in rounds of the fits at once")
    wall = [rounds * int(m) / 60 for m in (low, high)]
    put("finetune.wall.min", f"{wall[0]:.1f}", "rounds x minutes, hours")
    put("finetune.wall.max", f"{wall[1]:.1f}", "rounds x minutes, hours")


# ---------------------------------------------------------------- records


def records() -> None:
    dev = (HERE / "deviations.md").read_text(encoding="utf-8")
    # The text counts the analyses up to the 2026-11-11 heading; that heading itself only
    # records the time, and sections after it came later.
    sprint = dev.split("\n## 2026-11-11 研究衝刺結束", 1)[0]
    heads = [h for h in sprint.splitlines() if h.startswith("## ")]
    put("dev.sections", str(len(heads)), "deviations.md headings")
    put("dev.exploratory", str(sum("探索性" in h for h in heads)), "deviations.md headings")
    put("dev.after", str(sum("事後" in h for h in heads)), "deviations.md headings with 事後")
    plan = (ROOT / "PLAN.md").read_text(encoding="utf-8")
    line = next(t for t in plan.splitlines() if "看過結果才做、判讀沒先寫的" in t)
    body = line.split("：", 1)[1].split("。", 1)[0]
    depth, items = 0, 1
    for ch in body:
        depth += ch in "（("
        depth -= ch in "）)"
        items += ch == "、" and depth == 0
    put("posthoc.items", str(items), "work log list of results-first items")
    put("plan.section8_items", str(sum(1 for t in plan_section8() if t)), "analysis_plan.md §8")


def overview() -> None:
    """Values that only the one-page overview uses."""
    cells = load("exploratory/intervention.json")["cells"]
    put("intervention.0db", signed(cells["ESSUA_with_HSA_+0"]["added_flag_rate"]), "intervention")
    h = load("exploratory/hsa_to_essua.json")["hubert_full"]
    gain = h["specificity_essua_matched"] - h["specificity_hsa_out_of_fold"]
    put("hsa_to_essua.hubert", signed(gain), "exploratory/hsa_to_essua.json, matched minus HSA")
    curve = load("exploratory/learning_curve.json")["essua_handcrafted"]["points"]
    first = next(p for p in curve if p["requested"] == "16")
    put("curve.essua16", u(first["uar"]), "exploratory/learning_curve.json")
    phon = load("phonation_c1/summary.json")["handcrafted"]["gap_i_minus_ii"]
    put("phonation_c1.hand", pts(phon["difference"]), "phonation_c1/summary.json")
    put("finetune.ii", u(load("finetune/summary.json")["ii"]["uar"]), "finetune/summary.json")
    put("erased.svd", u(load("site_erasure/summary.json")["svd"]["all"]["auc"]), "site_erasure")
    spec = load("diagnosis_breakdown/summary.json")["hubert"]["specificity"]
    put("diagnosis.flagged", f"{round(100 * (1 - spec['i_hsa_normals']))}", "diagnosis_breakdown")
    axis = load("within_place_axis/summary.json")["hubert"]
    moved = [
        v["i_minus_ii"]["difference"]
        for g in ("hsa_patients", "hsa_normals")
        for v in axis[g]["layers"].values()
    ]
    put("axis.min", f"{min(moved):+.2f}", "within_place_axis/summary.json, HuBERT /a/")
    put("axis.max", f"{max(moved):+.2f}", "within_place_axis/summary.json, HuBERT /a/")
    s = pd.read_csv(SPEAKERS)
    days = s[s["label"] == "normal"].groupby("place")["visit_date"].nunique()
    put("days.essua", str(days["ESSUA"]), "s0_audit/out/avfad_speakers.csv, recording days")
    put("days.hsa", str(days["HSA"]), "s0_audit/out/avfad_speakers.csv, recording days")


def plan_section8() -> list[str]:
    text = (ROOT / "analysis_plan.md").read_text(encoding="utf-8")
    block = text.split("## 8.", 1)[1].split("## 9.", 1)[0]
    return [t for t in block.splitlines() if t.startswith("- ")]


# ---------------------------------------------------------------- check the analysis text

# Dates and clock times are stripped before the check. Settings written down before any result
# are taken out as whole phrases, so a dose cannot pass by equalling an unrelated value (the
# 0.5 s reverb and the 0.5 upper bound of the outside gap). The listing at the end of the output
# says which names each remaining number matched.
FIXED = {
    "1 個 BLAS": "one BLAS thread per worker (analysis_plan.md §6)",
    "底噪高 12 dB": "V1 noise dose, the middle of 6, 12, 24 dB (deviations.md, before the run)",
    "迴響 0.5 s": "V1 reverb dose, RT60 0.5 s, an assumed value (deviations.md, before the run)",
    "0.5 s 迴響": "same V1 reverb dose",
    "1 s 才有": "V1 reverb dose, RT60 1.0 s (deviations.md, before the run)",
    "82 位正常人裡的 62 位": "V2 dose: 62 of the 82 A-site normals swapped (deviations.md)",
    "82 位裡 62 位": "same V2 dose",
    "最後 2 層": "fine-tuning the last two HuBERT layers (deviations.md, 2026-11-09)",
    "CI 在 0 以上": "an interval above zero",
    "都在 0 以下": "intervals below zero",
    "Ribas 2023": "citation year of the published system",
}


def check_text(path: Path | None = None) -> tuple[list[str], list[str]]:
    """Numbers in one text, each with the value names it matches, or 'fixed'."""
    text = (path or ANALYSIS).read_text(encoding="utf-8")
    text = re.sub(r"`[^`]*`", " ", text)  # paths, commits, code
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", text)
    text = re.sub(r"\b\d{1,2}-\d{2}\b", " ", text)  # 11-04
    text = re.sub(r"\b\d{1,2}:\d{2}Z?\b", " ", text)  # 15:22Z
    text = re.sub(r"(?m)^#+\s*\d+\.", " ", text)  # section numbers
    text = re.sub(r"(?m)^\s*\d+\.\s", " ", text)  # ordered lists
    fixed = []
    for phrase in FIXED:
        fixed += [f"{phrase}: fixed"] * text.count(phrase)
        text = text.replace(phrase, " ")
    tokens = re.findall(r"(?<![A-Za-z0-9_./-])[-+]?\d[\d,]*(?:\.\d+)?", text)
    names: dict[str, list[str]] = {}
    for name, (shown, _) in values.items():
        for form in {shown, shown.lstrip("+"), shown.lstrip("+-")}:
            names.setdefault(form, []).append(name)
    matched, unmatched = list(fixed), []
    for t in tokens:
        bare = t.lstrip("+")
        found = names.get(t) or names.get(bare) or names.get(bare.lstrip("-"))
        if found:
            matched.append(f"{t}: {', '.join(found)}")
        else:
            unmatched.append(t)
    return matched, unmatched


def main() -> None:
    confirmatory()
    place_and_outside()
    metadata()
    simulations()
    timings()
    records()
    overview()
    lines = ["# numbers of the two working notes, read again from the results", ""]
    width = max(len(k) for k in values)
    for name, (text, source) in values.items():
        lines.append(f"{name:<{width}}  {text:>10}  {source}")
    lines += ["", "allowed without a result file:"]
    lines += [f"  {k}: {v}" for k, v in FIXED.items()]
    for path in (ANALYSIS, OVERVIEW):
        if not path.exists():
            continue
        matched, unmatched = check_text(path)
        lines += ["", f"numbers in {path.name}: {len(matched) + len(unmatched)}"]
        lines.append(f"matched: {len(matched)}")
        lines.append(f"unmatched: {len(unmatched)} {unmatched}")
        lines += ["each number in the order it appears:"]
        lines += [f"  {m}" for m in matched]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    head = ("numbers in", "matched:", "unmatched:")
    print("\n".join(t for t in lines if t.startswith(head)) or f"{len(values)} values")


if __name__ == "__main__":
    main()
