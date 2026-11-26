"""Writes experiments/results/summary.md; every result is read from the result files.

The only figures written here by hand are definitions: thresholds fixed in the plan come from
the result files too, and the rest of the text is words.

    .venv/Scripts/python.exe -I -B experiments/make_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUTPUT = RESULTS / "summary.md"
SPEAKERS = HERE.parent / "s0_audit" / "out" / "avfad_speakers.csv"
CHECK_SHEET = HERE.parent / "s1_segments" / "out_speech" / "check" / "check_sheet.csv"
CURRENT_SMOKER = 2


def load(relative: str) -> dict:
    return json.loads((RESULTS / relative).read_text(encoding="utf-8"))


def optional(relative: str) -> dict | None:
    """A result file that a run still in progress has not written yet."""
    return load(relative) if (RESULTS / relative).exists() else None


def u(value: float) -> str:
    """A score to three decimals."""
    return f"{value:.3f}"


def pts(value: float) -> str:
    """A fraction as points, one decimal."""
    return f"{value * 100:.1f}"


def signed(value: float) -> str:
    text = f"{value * 100:+.1f}"
    return "0.0" if text in ("+0.0", "-0.0") else text


def bracket(low: float, high: float) -> str:
    return f"[{low * 100:.1f}, {high * 100:.1f}]"


def score_bracket(low: float, high: float) -> str:
    return f"[{low:.3f}, {high:.3f}]"


def p_text(value: float) -> str:
    return f"{value:.4f}" if value < 0.001 else f"{value:.3f}"


def inside(record: dict) -> str:
    return f"{u(record['auc'])} {score_bracket(*record['ci'])}"


MATERIAL_NAME = {"a": "/a/", "sentences": "句子"}
RIBAS_FAILS = 0.70  # ribas_method.py FAILS and DROP_HOLDS, written before the run
RIBAS_DROP_HOLDS = 0.10
TUNED_READING = {
    "objection answered for the last two layers": "微調最後 2 層也一樣",
    "shrinks": "結論要縮小",
    "middle": "中間",
}
TRAVEL_READING = {
    "does not travel: consistent with a place shortcut": "帶不過去",
    "part of it travels": "有一部分帶得過去",
    "middle": "中間",
}
VOWEL_READING = {"repeats": "重複了", "does not repeat": "沒有重複", "no conclusion": "不下結論"}
SVD_COLUMN = {"handcrafted": "handcrafted", "HuBERT (frozen)": "hubert"}
DETECTOR_NAME = {
    "handcrafted": "手工",
    "HuBERT (frozen)": "HuBERT 凍結",
    "HuBERT, last 2 layers fine-tuned": "HuBERT 微調最後 2 層",
    "ComParE + SVM (their setting)": "ComParE 加 SVM",
    "ComParE + SVM, class weight": "同上，加 class weight",
}
VOWEL_CLAUSE = {
    "repeats": "；換成沒用過的 /i/、/u/，(i) 多出來的分數也帶不過去",
    "does not repeat": "，但換成 /i/、/u/ 沒有重複，這一點只能說 /a/ 上如此",
    "no conclusion": "；/i/、/u/ 不下結論",
}


def reversed_vowels(vowels: dict) -> list[str]:
    return [f"/{v}/" for v in ("i", "u") if vowels[v]["reversed"]]


def mixed_note(vowels: dict) -> str:
    """The ESSUA-normal-versus-HSA-patient set on SVD, when it is not at chance."""
    away = [v for v in ("i", "u") if not vowels[v]["mixed_contrast_stays_inside"]]
    if not away:
        return ""
    return ""


def judgement_text(vowels, hub_x, voiced_x, diagnosis=None, erasure=None, plain_gap=0.0) -> str:
    both = None if vowels is None else vowels.get("both")
    clause = "" if both is None else VOWEL_CLAUSE[both["reading"]]
    if both is not None and both["reading"] == "repeats":
        turned = reversed_vowels(vowels)
        clause += (
            "，兩個都反過來"
            if len(turned) == 2
            else (f"，反過來的只有 {turned[0]}" if turned else "，但都沒有反過來")
        )
        clause += mixed_note(vowels)
    external = [hub_x["ii"]["auc"], hub_x["i"]["auc"], hub_x["all"]["auc"]]
    external += [voiced_x["ii"]["auc"], voiced_x["i"]["auc"]]
    normals = voiced_x["speakers"] - voiced_x["pathological"]
    if both is not None and both["reading"] == "repeats":
        claims = "只留 C1、C2 和不靠人聲的地點捷徑（含已發表系統在 HSA 內崩掉），外部偵測器當補充"
    else:
        claims = "跨資料庫降成 /a/ 上的觀察，主張只留非發聲段和只換正常組來源"
    flagged = ""
    if diagnosis is not None:
        flagged = ""
    aligned = ""
    transfer = optional("aligned_transfer/summary.json")
    harmonized = optional("harmonize_c1/summary.json")
    folds_now = optional("erasure_folds/summary.json")
    if None not in (transfer, harmonized, folds_now, erasure) and "svd" in erasure:
        aligned = (
            "；拿掉地點方向或只用正常人對齊平均，AVFAD 內縮到 "
            f"{pts(folds_now['gap_i_minus_ii']['difference'])}、"
            f"{pts(harmonized['hubert']['location']['gap_i_minus_ii']['difference'])} 點，"
            f"到 SVD 卻掉到 {u(erasure['svd']['all']['auc'])}、"
            f"{u(transfer['hubert']['location']['auc']['all'])}：內部檢查會被騙過去"
        )
    within = ""
    if optional("within_place_axis/summary_aligned.json") is not None:
        within = (
            "\n- 最直接的證據：只學地點、沒看過病人的偵測器分 C1 的 (i)，比用 (i) 訓練的還好"
            f"{direct_auc()}。只看 HSA 的完整檔，(i) 越像 ESSUA 的人越被當成正常，連病人也是，"
            "三個母音都這樣，(ii) 和外部偵測器沒有；發聲段看不出來。"
        )
    erased = ""
    if erasure is not None and "svd" in erasure:
        folds = optional("erasure_folds/summary.json")
        inside_gap = (
            "縮小"
            if folds is None
            else f"從 {pts(plain_gap)} 縮到 {pts(folds['gap_i_minus_ii']['difference'])} 點"
        )
        erased = (
            f"；拿掉兩地正常人的平均差方向，AVFAD 內差距{inside_gap}，到 SVD 混合訓練反而掉到 "
            f"{u(erasure['svd']['all']['auc'])}"
        )
    return f"""**值得繼續，做成一篇範圍小的稽核，不捨棄，也不再跑實驗。** 論文{claims}。

- 結論換了三種材料、六種偵測器和一個已發表的系統都沒變。「ESSUA 的人本來就比較健康」的回答：沒看過 AVFAD 的偵測器看同一組人沒有這個差距；混合地點多學到的分數到 SVD 也帶不過去。{inside_hsa_note()}{flagged}{within}
- 補救沒讓混合訓練在外部變好：加噪音、降噪、微調最後 2 層、加別處資料都沒拿掉差距{aligned if aligned else erased}。

**弱點**：四個檢定以外都是探索性的，7 項是看過結果才做的；VOICED 只有 {normals} 位正常人；外部 AUC 只有 {min(external):.2f}–{max(external):.2f}；{screening_note()}。"""


FUSION_HELPS = {
    "more material does not help": "多材料沒幫助",
    "more material helps": "多材料有幫助",
    "small": "小幅",
}
FUSION_GAP = {
    "the gap stays": "差距還在",
    "the gap shrinks": "差距縮小",
    "the gap is gone": "差距不見了",
}
DIAGNOSIS_CONTRAST = {
    "the least audible chapter gains no less": "不跟著病走",
    "the gain follows how audible the disorder is": "跟著病的明顯程度走",
    "unclear": "說不清",
}
OUTSIDE_GAP = {
    "the (i) pairing is harder for the outside detector": "(i) 的配法反而比較難分",
    "the outside detector sees no difference between the two sets of normals": "看不出差別",
    "the test populations differ this much": "測試人群本身就差這麼多",
    "the test populations differ a little": "測試人群有一點差別",
}
POOL_HELPS = {"helps": "有幫助", "does not help": "沒幫助", "worse": "變差", "small": "小幅"}
FOLDS_GAP = {
    "the gap stays: one direction is not enough": "差距還在",
    "the gap is gone: it sits on this one direction": "差距不見了",
    "the gap shrinks": "差距縮小",
}
CHAPTER_ORDER = ("structural", "inflammatory", "neurological")


def fusion_line(f: dict | None) -> str:
    if f is None:
        return ""
    avg, diff = f["average"], f["average_minus_a_ii"]["uar"]
    reading = f"{FUSION_HELPS[f['readings']['more_material']]}，{FUSION_GAP[f['readings']['gap']]}"
    return (
        f"| /a/、/i/、/u/ 每人平均（HuBERT） | (ii) /i/ {u(f['i']['ii']['uar'])}、"
        f"/u/ {u(f['u']['ii']['uar'])}、合起來 {u(avg['ii']['uar'])}，比 /a/ {signed(diff['difference'])} 點 "
        f"{bracket(diff['ci_low'], diff['ci_high'])}；差距 /i/ {pts(f['i']['gap_i_minus_ii']['difference'])}、"
        f"/u/ {pts(f['u']['gap_i_minus_ii']['difference'])}、合起來 {pts(avg['gap_i_minus_ii']['difference'])} "
        f"{bracket(avg['gap_i_minus_ii']['ci_low'], avg['gap_i_minus_ii']['ci_high'])} | {reading} |\n"
    )


def diagnosis_repeat(vowels: list[dict | None]) -> str:
    if any(v is None for v in vowels):
        return ""
    held = all(
        v["hubert"]["readings"]["gains_in_all_three"]
        and v["hubert"]["readings"]["contrast"] == "the least audible chapter gains no less"
        for v in vowels
    )
    return "（/i/、/u/ 重複了）" if held else "（/i/、/u/ 沒有都重複）"


def diagnosis_line(d: dict | None, repeat: str = "") -> str:
    if d is None:
        return ""
    h = d["hubert"]
    cells = h["chapters"]

    def three(key: str) -> str:
        return "、".join(u(cells[g][key]["auc"]) for g in CHAPTER_ORDER)

    return (
        f"| HSA 病人按診斷分 | 結構性、發炎（逆流）、神經的 AUC：(ii) {three('ii')}；"
        f"(i) {three('i')}；外部偵測器 {three('outside')}"
        " | "
        f"{DIAGNOSIS_CONTRAST[h['readings']['contrast']]}{repeat} |\n"
    )


def outside_line(views: tuple) -> str:
    a, i, u_, hand = views
    if a is None:
        return ""
    by_vowel = [(v, r) for v, r in (("/a/", a), ("/i/", i), ("/u/", u_)) if r is not None]
    gaps = "、".join(
        f"{v} {signed(r['outside']['gap']['difference'])}"
        + (
            f" {bracket(r['outside']['gap']['ci_low'], r['outside']['gap']['ci_high'])}"
            if v == "/a/"
            else ""
        )
        for v, r in by_vowel
    )
    learned = a["learned"]
    text = (
        f"(i)−(ii) 的 AUC：{gaps}；減掉之後學來的部分 /a/ "
        f"{signed(learned['difference'])} {bracket(learned['ci_low'], learned['ci_high'])}"
    )
    if hand is not None:
        text += f"（手工 {signed(hand['learned']['difference'])}）"
    blind = optional("outside_view/summary_a_outside_erased.json")
    if blind is not None:
        g = blind["outside"]["gap"]
        text += (
            f"；外部偵測器先拿掉地點方向（兩地正常人 {u(blind['outside_hsa_normals_vs_essua_normals'])}）"
            f"，/a/ {signed(g['difference'])} {bracket(g['ci_low'], g['ci_high'])}"
        )
        for v in ("i", "u"):
            other = optional(f"outside_view/summary_{v}_outside_erased.json")
            if other is not None:
                text += f"、/{v}/ {signed(other['outside']['gap']['difference'])}"
    text += ""
    learned_all = all(
        r["readings"]["learned"] == "most of the gap is learned in training" for _, r in by_vowel
    )
    no_gap = all(
        r["readings"]["outside_gap"]
        in (
            "the (i) pairing is harder for the outside detector",
            "the outside detector sees no difference between the two sets of normals",
        )
        for _, r in by_vowel
    )
    reading = (
        f"差距是訓練時學來的（{len(by_vowel)} 個母音都是）"
        if learned_all and no_gap
        else "；".join(f"{v} {OUTSIDE_GAP[r['readings']['outside_gap']]}" for v, r in by_vowel)
    )
    return f"| 沒看過 AVFAD 的偵測器（SVD 訓練）看 C1 同一組人 | {text} | {reading} |\n"


def dose_line(d: dict | None) -> str:
    if d is None or "hubert" not in d:
        return ""
    hub, hand, out = d["hubert"], d["handcrafted"], d["outside_auc"]
    shares = [0.0] + [max(v) for v in d["shares_by_level"].values()]
    span = f"0 → {max(shares) * 100:.0f}%"
    levels = "、".join(u(v) for v in hub["uar_levels"].values())
    outside = f"{u(next(iter(out['levels'].values())))} → {u(list(out['levels'].values())[-1])}"
    reading = (
        "成績跟著比例往上" if hub["reading"] == "the score rises with the ESSUA share" else "照實報"
    )
    if out["reading"] == "the outside detector goes down":
        reading += "，外部看反而往下"
    elif out["reading"] == "flat for the outside detector":
        reading += "，外部看是平的"
    extra = ""
    vowels = [optional(f"dose_response/summary_{v}.json") for v in ("i", "u")]
    if all(v is not None for v in vowels):
        extra += ""
        if all(
            v[f"hubert_{n}"]["reading"] == "the score rises with the ESSUA share"
            for n, v in zip("iu", vowels)
        ):
            reading += "（三個母音）"
    erased = optional("dose_response/summary_erased.json")
    if erased is not None:
        e = erased["hubert_erased"]
        extra += (
            f"；拿掉地點方向後斜率 {signed(e['slope']['slope'])} "
            f"{bracket(e['slope']['ci_low'], e['slope']['ci_high'])}"
        )
        if e["reading"] == "with the direction out, most of the dose response is gone":
            reading += "；拿掉方向大部分沒了"
        elif e["reading"] == "with one direction out, the dose response stays":
            reading += "；拿掉一個方向還在"
    drawn = optional("place_only/summary_dose.json")
    if drawn is not None and all(
        drawn[d]["reading"] == "the place detector draws the dose curve too"
        for d in ("handcrafted", "hubert")
    ):
        reading += "；地點偵測器也畫得出"
    return (
        f"| 正常組 82 人裡 ESSUA 的比例 {span}（每級 10 次） | HuBERT UAR {levels}，斜率 "
        f"{signed(hub['slope']['slope'])} {bracket(hub['slope']['ci_low'], hub['slope']['ci_high'])}"
        f"（手工 {signed(hand['slope']['slope'])}）{extra}；外部偵測器 AUC {outside} | {reading} |\n"
    )


def pool_line(p: dict | None) -> str:
    if p is None:
        return ""
    arms = p["arms"]
    e, v = p["ii_essua_minus_ii"]["roc_auc"], p["ii_svd_minus_ii"]["roc_auc"]
    readings = p["readings"]
    one, two = POOL_HELPS[readings["ii_essua"]["auc"]], POOL_HELPS[readings["ii_svd"]["auc"]]
    reading = f"都{one}" if one == two else f"加 ESSUA {one}、加 SVD {two}"
    return (
        f"| HSA 內訓練加別處的資料（HSA 內測） | AUC：加 ESSUA 正常人 {signed(e['difference'])} 點 "
        f"{bracket(e['ci_low'], e['ci_high'])}、加 SVD {signed(v['difference'])} 點 "
        f"{bracket(v['ci_low'], v['ci_high'])}；HSA 正常人特異度 {u(arms['ii']['specificity_hsa_normals'])} → "
        f"{u(arms['ii_essua']['specificity_hsa_normals'])}、{u(arms['ii_svd']['specificity_hsa_normals'])} | "
        f"{reading} |\n"
    )


def erasure_line(e: dict | None, folds: dict | None, plain_gap: float, plain_all: float) -> str:
    if e is None or "avfad" not in e or "svd" not in e:
        return ""
    worse = e["svd"]["paired"]["all_erased_minus_all_plain"]
    gap = f"{pts(e['avfad']['gap_i_minus_ii']['difference'])}（方向看過測試折）"
    reading = "AVFAD 內縮小"
    if folds is not None:
        g = folds["gap_i_minus_ii"]
        others, held = [], [folds["gap_reading"]]
        for v in ("i", "u"):
            r = optional(f"erasure_folds/summary_{v}.json")
            if r is not None:
                held.append(r["gap_reading"])
                others.append(
                    f"/{v}/ {pts(r['plain_gap'])} → {pts(r['gap_i_minus_ii']['difference'])}"
                )
        r = optional("erasure_folds/summary_phonation.json")
        if r is not None:
            others.append(
                f"只用發聲段 {pts(r['plain_gap'])} → {pts(r['gap_i_minus_ii']['difference'])}"
            )
        gap = f"{pts(g['difference'])} {bracket(g['ci_low'], g['ci_high'])}（方向只用訓練折" + "）"
        reading = f"AVFAD 內{FOLDS_GAP[folds['gap_reading']]}"
        if len(held) == 3:
            stays = "the gap stays: one direction is not enough"
            reading += "（三個母音）" if stays not in held else "（三個母音不一樣）"
    reading += "；SVD 上更差，不是補救" if worse["ci_high"] < 0 else "；SVD 上沒變好"
    return (
        f"| 從特徵拿掉地點方向（每層一個） | AVFAD C1 差距 {pts(plain_gap)} → {gap}；"
        f"SVD 上全體訓練 {u(plain_all)} → {u(e['svd']['all']['auc'])}，{signed(worse['difference'])} 點 "
        f"{bracket(worse['ci_low'], worse['ci_high'])} | {reading} |\n"
    )


TRANSFER_CHANGE = {
    "the alignment makes mixed training better outside": "SVD 上變好",
    "the alignment makes it worse": "SVD 上更差，不是補救",
    "the alignment does not change the outside score": "SVD 上沒變",
    "as measured": "SVD 上照實",
}
PLACEBO_READING = {
    "no place, no gap": "沒有地點就沒有差距",
    "the design makes a gap by itself": "設計本身造成差距",
    "the design puts (i) lower by itself": "設計本身讓 (i) 比較低",
    "as measured": "照實",
}


def align_line() -> str:
    h = optional("harmonize_c1/summary.json")
    nonlinear = optional("nonlinear_c1/summary.json")
    transfer = optional("aligned_transfer/summary.json")
    folds = optional("erasure_folds/summary.json")
    erased = optional("site_erasure/summary.json")
    if None in (h, nonlinear, transfer, folds, erased):
        return ""
    held = [folds["gap_reading"]]
    for v in ("i", "u"):
        r = optional(f"erasure_folds/summary_{v}.json")
        if r is not None:
            held.append(r["gap_reading"])
    vowels = "（三個母音）" if len(held) == 3 and len(set(held)) == 1 else ""
    g = folds["gap_i_minus_ii"]
    hub = h["hubert"]["location"]["gap_i_minus_ii"]
    hand = h["handcrafted"]["location"]["gap_i_minus_ii"]
    models = ("svc", "random_forest")
    plain = "、".join(pts(nonlinear[m]["plain"]["gap_i_minus_ii"]["difference"]) for m in models)
    moved = "、".join(pts(nonlinear[m]["scale"]["gap_i_minus_ii"]["difference"]) for m in models)
    t = transfer["hubert"]
    return (
        "| 拿掉地點方向（每層一個）；只用正常人把 ESSUA 每一維的平均對齊到 HSA | "
        f"C1 差距 HuBERT {pts(h['hubert']['plain_gap'])} → {pts(g['difference'])} "
        f"{bracket(g['ci_low'], g['ci_high'])}、{pts(hub['difference'])} "
        f"{bracket(hub['ci_low'], hub['ci_high'])}，手工 {pts(h['handcrafted']['plain_gap'])} → "
        f"{pts(hand['difference'])} {bracket(hand['ci_low'], hand['ci_high'])}（對齊）；"
        f"再對齊標準差、相關不再少；RBF SVM、隨機森林 {plain} → {moved}；"
        f"SVD 上全體訓練 {u(t['plain']['all'])} → {u(erased['svd']['all']['auc'])}、"
        f"{u(t['location']['auc']['all'])}（HSA 內訓練 {u(t['plain']['ii'])}） | "
        f"AVFAD 內差距縮小{vowels}；{TRANSFER_CHANGE[t['location']['all_reading']]} |\n"
    )


def within_line() -> str:
    a = optional("within_place_axis/summary.json")
    others = [optional(f"within_place_axis/summary_{v}.json") for v in ("i", "u")]
    moved = optional("within_place_axis/summary_aligned.json")
    position = optional("patient_position/summary.json")
    if a is None or moved is None or position is None or None in others:
        return ""

    def spread(parts: list[dict]) -> str:
        values = [v["i_minus_ii"]["difference"] for part in parts for v in part["layers"].values()]
        return f"{min(values):+.2f} 到 {max(values):+.2f}"

    groups = ("hsa_patients", "hsa_normals")
    own = spread([a["hubert"][g] for g in groups])
    vowels = spread([o["hubert"][g] for o in others for g in groups])
    after = spread([moved["hubert"]["location"][g] for g in groups])
    uses = "(i) uses the place direction inside one place too"
    every = all(r["hubert"][g]["overall"] == uses for r in (a, *others) for g in groups)
    places = [v["position"] for v in position["layers"].values()]
    reading = "(i) 在同一地點內也用地點方向" + ("（三個母音）" if every else "")
    reading += "，對齊後沒了"
    if position["overall"] == "the direction is place":
        reading += "；方向分的是地點"
    phonation = optional("within_place_axis/summary_phonation.json")
    if phonation is not None and all(phonation["hubert"][g]["overall"] != uses for g in groups):
        reading += "；發聲段看不出來（方向連外部也跟著）"
    return (
        f"| 只看 HSA：分數跟不跟著地點方向 | "
        f"投影（越像 ESSUA 越大）和分數的相關，(i) 減 (ii)：/a/ {own}（HuBERT 五層），"
        f"/i/、/u/ {vowels}，對齊後 {after}；HSA 病人在這個方向上的位置 "
        f"{min(places):+.2f} 到 {max(places):+.2f}（HSA 正常人 0，ESSUA 1） | {reading} |\n"
    )


def placebo_text() -> str:
    p = optional("placebo_c1/summary.json")
    if p is None or "hubert" not in p or "distribution" not in p:
        return ""
    d = p["distribution"]
    hub = p["hubert"]["0"]["gap_i_minus_ii"]
    hub_range = p.get("hubert_range")
    hub_text = (
        signed(hub["difference"])
        if hub_range is None
        else f"{hub_range['splits']} 組 {signed(hub_range['min'])} 到 {signed(hub_range['max'])}"
    )
    inside = d["reading"] == "what is left after the alignment is within the spread without place"
    left = [d["avfad"][k] for k in ("location", "scale", "coral")]
    span = f"{pts(min(left))}–{pts(max(left))}"
    tail = (
        f"；手工對齊後剩的 {span} 點在這個範圍內"
        if inside
        else f"；AVFAD 手工對齊後剩的 {span} 點不全在這個範圍內"
    )
    said = (
        "沒有地點就沒有差距" if p["overall"] == "the C1 design makes no gap by itself" else "照實"
    )
    return (
        f"C1 的安慰劑（SVD，一個地點）：手工 {d['splits']} 組 "
        f"{signed(d['min'])} 到 {signed(d['max'])} 點，HuBERT {hub_text}，"
        f"{said}{tail}。"
    )


SIM_ALONE = {
    "silence_1x": "alone it reaches half of AVFAD's gap or more",
    "noise_12": "alone it makes a gap, less than half",
    "reverb_0.5": "alone it makes no gap",
    "channel_1x": "alone it makes no gap",
    "all": "all four together fall short of AVFAD",
}


def simulated_text() -> str:
    d = optional("simulated_place/summary.json")
    if d is None or "readings" not in d.get("handcrafted", {}):
        return ""
    if any(d["handcrafted"]["readings"].get(k) != v for k, v in SIM_ALONE.items()):
        return ""
    h = d["handcrafted"]
    detectors = [h]
    hub = d.get("hubert", {})
    hub_read = hub.get("readings", {})
    same = ("silence_1x", "noise_12", "channel_1x")
    reverb_ok = (SIM_ALONE["reverb_0.5"], "alone it makes a gap, less than half")
    if (
        all(hub_read.get(k) == SIM_ALONE[k] for k in same)
        and hub_read.get("reverb_0.5") in reverb_ok
    ):
        detectors.append(hub)
    avfad = [load("first_batch/summary.json")["c1"], load("sensitivity/hubert/c1_summary.json")]
    if any(p["all"]["i_uar_mean"] < c["uar_i_mean"] for p, c in zip(detectors, avfad)):
        return ""

    def gap(part: dict, name: str) -> str:
        return pts(part[name]["gap_i_minus_ii"]["difference"])

    def gaps(name: str) -> str:
        return "、".join(gap(p, name) for p in detectors)

    both = optional("simulated_place/silence_both/summary.json")
    both_text = ""
    if both is not None and "gap_i_minus_ii" in both.get("handcrafted", {}):
        found = [both[k] for k in ("handcrafted", "hubert")[: len(detectors)] if k in both]
        both_text = "、".join(pts(p["gap_i_minus_ii"]["difference"]) for p in found)
        both_text = f"（兩邊都有 {both_text}）"
    reverb = "0.5 s 迴響和麥克風、增益沒有"
    if len(detectors) == 2 and hub_read["reverb_0.5"] != SIM_ALONE["reverb_0.5"]:
        reverb = f"0.5 s 迴響只 HuBERT 有 {gap(hub, 'reverb_0.5')}，麥克風、增益沒有"
    alls = gaps("all")
    sim_ii = "、".join(u(p["all"]["ii"]["uar"]) for p in detectors)
    return (
        f"單一地點 SVD 套上 AVFAD 錄音差異：靜音 {gaps('silence_1x')} 點{both_text}、"
        f"底噪 {gaps('noise_12')}，{reverb}；全開 {alls}，"
        f"(i) 比 AVFAD 高，但 SVD (ii) 本就到 {sim_ii}。"
    )


def simulated_free_note() -> str:
    d = optional("simulated_place/summary.json")
    if d is None or not channel_note():
        return ""
    f = d.get("free", {})
    if f.get("readings", {}).get("reverb_0.5") != "the 40 columns do not respond to reverb":
        return ""
    return f"，模擬裡迴響 1 s 才有 {pts(f['reverb_1.0']['gap_i_minus_ii']['difference'])}"


# Belo (2015), shipped with AVFAD, section 3.2: participants confirmed by laryngostroboscopy.
BELO_STROBOSCOPY = 436
BELO_FORM = 273


def screening_note() -> str:
    """How the normals were screened, inferred from totals; nobody is listed one by one."""
    speakers = pd.read_csv(SPEAKERS)
    normal = speakers["label"] == "normal"
    patients = int((~normal).sum())
    outside = int((normal & (speakers["place"] != "ESSUA")).sum())
    return (
        f"HSA 正常人的篩檢由總數推得（Belo 2015：喉頻閃鏡 {BELO_STROBOSCOPY} 人、問診表 "
        f"{BELO_FORM} 人，扣掉病人剩 {BELO_STROBOSCOPY - patients} 位，ESSUA 以外的正常人 "
        f"{outside} 位），就診原因和治療史不明，Aveiro 沒回信"
    )


def dirty_note() -> str:
    d = optional("dirty_normals/summary.json")
    unclean = "three quarters of HSA's normals being patients does not give AVFAD's picture"
    if d is None or not d["handcrafted"]["overall"].startswith(unclean):
        return ""
    worst = max(int(k) for k in d["handcrafted"] if k.isdigit())
    if any(d[k][str(worst)]["place"]["auc"] >= 0.75 for k in ("handcrafted", "hubert")):
        return ""
    return f"（模擬裡 {worst}／82 位是病人也造不出地點偵測器的成績）"


def not_needed() -> str:
    if optional("simulated_place/summary.json") is None:
        return "房間和噪音的細節、更大的微調"
    return "更細的錄音差異模擬、更大的微調"


def channel_note() -> str:
    c = optional("channel_free_c1/summary.json")
    if c is None or "free" not in c:
        return ""
    if c["free"]["reading"] != "a gap without a fixed channel or gain":
        return ""
    g = c["free"]["gap_i_minus_ii"]
    return f"；麥克風和增益也不夠：它們動不到的特徵在發聲段還有 {pts(g['difference'])} 點"


def place_only_row() -> str:
    p = optional("place_only/summary.json")
    if p is None:
        return ""
    cells = []
    for detector in ("handcrafted", "hubert"):
        a = p[detector]["all_avfad"]
        boundary = optional("place_only/summary_uar.json")
        if boundary is None:
            cells.append(f"AUC {u(a['auc'])} {score_bracket(a['ci_low'], a['ci_high'])}")
        else:
            cells.append(f"{u(boundary[detector]['uar'])}／{u(a['auc'])}")
    row = f"| 只學分兩地正常人、沒看過病人的偵測器，全體 | {cells[0]} | {cells[1]} |\n"
    rule = optional("place_only/summary_oracle.json")
    if rule is not None:
        both = f"{u(rule['uar'])}／{u(rule['uar'])}"
        row = f"| 不用聲音：不在 ESSUA 錄的就判病理，全體 | {both} | {both} |\n" + row
    c1 = optional("place_only/summary_c1.json")
    if c1 is None:
        return row
    pair = [
        f"{u(c1[d]['place_i_mean'])}／{u(c1[d]['place_ii'])}" for d in ("handcrafted", "hubert")
    ]
    return row + f"| 同一個偵測器分 C1 的 (i)／(ii) 的 AUC | {pair[0]} | {pair[1]} |\n"


def place_materials_note() -> str:
    m = optional("place_only/summary_materials.json")
    if m is None:
        return ""
    hub = [r["all_avfad"]["auc"] for r in m["hubert"].values()]
    hand = m["handcrafted"]["a_phonation"]["all_avfad"]["auc"]
    return (
        f"地點偵測器在其他材料和發聲段：HuBERT {u(min(hub))}–{u(max(hub))}，"
        f"手工發聲段 {u(hand)}{phonation_c1_note()}{compare_note()}。"
    )


def phonation_c1_note() -> str:
    c = optional("place_only/summary_c1_phonation.json")
    if c is None:
        return ""
    reproduced = "a detector that never saw a patient reproduces most of the C1 gap"
    if not all(c[d]["reading"] == reproduced for d in ("handcrafted", "hubert")):
        return ""
    m = optional("place_only/summary_c1_materials.json")
    if m is not None and m["overall"] == "the same on every material":
        return "，C1 差距也做得出來"
    return "，發聲段 C1 的差距它也做得出來"


def compare_note() -> str:
    c = optional("place_only/summary_compare.json")
    if c is None:
        return ""
    return f"；用 ComParE 特徵 {u(c['all_avfad']['auc'])}"


def place_only_lead() -> str:
    p = optional("place_only/summary.json")
    if p is None:
        return ""
    aucs = [p[d]["all_avfad"]["auc"] for d in ("handcrafted", "hubert")]
    return (
        f"只學分兩地正常人、沒看過病人的偵測器，在全體就有 AUC {u(min(aucs))}–{u(max(aucs))}"
        f"{rule_lead()}。"
    )


def direct_auc() -> str:
    c1 = optional("place_only/summary_c1.json")
    if c1 is None:
        return ""
    place = "、".join(u(c1[d]["place_i_mean"]) for d in ("handcrafted", "hubert"))
    real = "、".join(u(c1[d]["real_i_mean"]) for d in ("handcrafted", "hubert"))
    return f"（AUC {place} 對 {real}）"


def room_note() -> str:
    r = optional("place_only/summary_room.json")
    if r is None:
        return ""
    return f"；沒聽過人聲也沒看過病人的地點偵測器，全體就有 {u(r['uar'])}"


def svd_time_note() -> str:
    t = optional("time_only_svd/summary.json")
    if t is None or t["handcrafted"]["reading"] != "SVD has no shortcut in time":
        return ""
    aucs = "、".join(u(t[d]["auc"]) for d in ("handcrafted", "hubert"))
    return f"SVD 沒有這種捷徑：只學錄音年份的偵測器分不出病理（{aucs}）。"


def svd_inside_note() -> str:
    v = optional("svd_inside/summary_small.json")
    if v is None:
        return ""
    aucs = "、".join(u(v[d]["within_strata"]["auc"]) for d in ("handcrafted", "hubert"))
    return f"；單一地點 SVD 同樣人數下，層內 AUC 也只有 {aucs}"


def rule_lead() -> str:
    rule = optional("place_only/summary_oracle.json")
    if rule is None:
        return ""
    return f"；不聽聲音、只看是不是在 ESSUA 錄的，UAR 就有 {u(rule['uar'])}"


def inside_hsa_note() -> str:
    h = optional("place_only/summary_inside_hsa.json")
    if h is None:
        return ""
    off = [1 - h[d]["full_specificity_hsa_normals"] for d in ("handcrafted", "hubert")]
    return (
        f"全體訓練的偵測器拿到 HSA 裡，排序沒比 (ii) 好，HSA 的正常人卻有 "
        f"{round(100 * min(off))}–{round(100 * max(off))}% 被判成病理。"
    )


def by_day_text() -> str:
    d = optional("c1_by_day/summary.json")
    if d is None or set(d) != {"handcrafted", "hubert"}:
        return ""
    gaps = "、".join(pts(d[k]["gap_i_minus_ii"]["difference"]) for k in ("handcrafted", "hubert"))
    return f"按錄音日分折，C1 還有 {gaps} 點。"


def finetune_status(tuned: dict) -> str:
    first = tuned["shuffled"]["uar"]
    reading = TUNED_READING[tuned["reading"]]
    if tuned["readable"]:
        return reading
    five = tuned.get("shuffled_five")
    if five is None:
        return f"打亂標籤 {u(first)}，gate 沒過，補跑中"
    if tuned["readable_post_hoc"]:
        return f"{reading}（打亂標籤 {u(five['uar_mean'])}，事後補跑）"
    return f"不能解讀：打亂標籤 5 次平均 {u(five['uar_mean'])}，沒過"


def vowel_cells(vowels: dict | None) -> tuple[str, str]:
    if vowels is None or "both" not in vowels:
        return "還在跑", "還在跑"
    parts = []
    for v in ("i", "u"):
        r = vowels[v]
        parts.append(
            f"/{v}/：(ii) {u(r['ii']['auc'])}、(i) {u(r['i']['auc'])}，"
            f"D_ext {signed(r['d_ext']['auc'])} {bracket(*r['d_ext']['auc_ci'])}，"
            f"ESSUA 正常配 HSA 病人 {u(r['essua_normals_hsa_patients']['auc'])}"
        )
    both = vowels["both"]
    reading = VOWEL_READING[both["reading"]]
    turned = reversed_vowels(vowels)
    if turned:
        reading += "，" + ("兩個都反過來" if len(turned) == 2 else f"只有 {turned[0]} 反過來")
    if not both["mixed_contrast_stays_inside_in_both"]:
        reading += "；混合對比不是亂猜"
    return "；".join(parts), reading


def protocol_row(r: dict, cross: dict) -> str:
    def cell(value):
        return "沒跑" if value is None else u(value)

    strata = (
        "沒跑"
        if r["hsa_within_strata_auc"] is None
        else f"{u(r['hsa_within_strata_auc'])} {score_bracket(*r['hsa_within_strata_ci'])}"
    )
    column = SVD_COLUMN.get(r["detector"])
    svd = (
        "沒跑"
        if column is None
        else f"{u(cross[column]['ii']['auc'])}／{u(cross[column]['all']['auc'])}"
    )
    return (
        f"| {DETECTOR_NAME[r['detector']]} | {cell(r['hsa_uar'])}／{cell(r['hsa_auc'])} | "
        f"{strata} | {cell(r['all_uar'])} | {svd} |"
    )


def build() -> str:
    first = load("first_batch/summary.json")
    family = load("final_family.json")
    hub_c1 = load("sensitivity/hubert/c1_summary.json")
    hub_ii = load("sensitivity/hubert/ii_summary.json")
    every = load("exploratory/avfad_all.json")
    s8 = load("exploratory/section8.json")
    conditioned = load("exploratory/conditioned.json")
    strata = load("exploratory/within_strata_auc.json")
    c3 = load("svd/c3.json")
    c4 = load("svd/c4.json")
    ribas = load("published_splits/published_predictions.json")
    speech_hand = load("speech/c1_handcrafted.json")
    speech_hub = load("speech/c1_hubert.json")
    speech_quiet = load("speech/nonphonation.json")
    ribas_hsa = load("ribas_method/hsa.json")
    ribas_bal = load("ribas_method/hsa_balanced.json")
    ribas_rep = load("ribas_method/reproduce.json")
    ribas_c1 = load("ribas_method/c1.json")
    ribas_c1_bal = load("ribas_method/c1_balanced.json")
    tuned = load("finetune/summary.json")
    cross = load("cross_database/summary.json")
    voiced_x = load("cross_database/voiced.json")
    all_vs_ii = load("cross_database/all_vs_ii.json")
    vowels = optional("cross_vowels/summary.json")
    fusion = optional("vowel_fusion/summary.json")
    diagnosis = optional("diagnosis_breakdown/summary.json")
    views = tuple(
        optional(f"outside_view/{n}.json")
        for n in ("summary", "summary_i", "summary_u", "summary_a_handcrafted")
    )
    pooled = optional("pool_training/summary.json")
    erasure = optional("site_erasure/summary.json")
    erasure_folds = optional("erasure_folds/summary.json")
    phon = optional("phonation_c1/summary.json")
    phon_note = (
        ""
        if phon is None
        else "只用發聲段（沒有靜音和長度的差別），差距還有手工 "
        f"{pts(phon['handcrafted']['gap_i_minus_ii']['difference'])}、HuBERT "
        f"{pts(phon['hubert']['gap_i_minus_ii']['difference'])} 點。"
    )
    protocol = load("protocol/table.json")["rows"]

    c1, c2, gate = first["c1"], first["c2"], first["gate"]
    quiet_all = s8["avfad"]["non_phonation"]["all"]
    quiet_hsa = s8["avfad"]["non_phonation"]["hsa"]
    tech_all = s8["avfad"]["m_tech"]["all"]["with_length"]
    tech_hsa = s8["avfad"]["m_tech"]["hsa"]["with_length"]
    age_sex = s8["svd"]["age_sex"]["random_forest"]
    voiced = s8["voiced"]
    voiced_tech = voiced["m_tech"]["with_length"]
    speakers = pd.read_csv(SPEAKERS)
    normal = speakers[speakers["label"] == "normal"]
    smokes = normal.groupby("place")["smoking"].apply(lambda s: float((s == CURRENT_SMOKER).mean()))
    negatives = [
        gate["negative"]["mean_uar"],
        *[row["mean_uar"] for row in c1["negative"]],
        c2["negative"]["mean_uar"],
        *[row["mean_uar"] for row in c3["negative"]],
        c4["negative"]["mean_uar"],
    ]
    c_gap = [family[k]["smallest_gap"] * 100 for k in ("C1", "C2", "C3", "C4")]
    all_gaps = [
        c1["difference"],
        hub_c1["difference"],
        *[speech_hand[n]["difference"] for n in ("cape_v", "reading")],
        *[speech_hub[n]["difference"] for n in ("cape_v", "reading")],
    ]
    negatives += [
        row["mean_uar"] for name in ("cape_v", "reading") for row in speech_hand[name]["negative"]
    ]

    hub_x, hand_x = cross["hubert"], cross["handcrafted"]
    everyone_uars = [
        every["all_handcrafted"]["uar"],
        every["all_hubert"]["uar"],
        ribas_hsa["a_n"]["everyone"]["uar"],
        tuned["all"]["uar"],
    ]
    inside_uars = [
        c1["uar_ii"],
        hub_ii["uar"],
        ribas_hsa["a_n"]["ii"]["uar"],
        ribas_bal["a_n"]["ii"]["uar"],
        tuned["ii"]["uar"],
    ]
    gaps = [
        *all_gaps,
        *[r["gap_i_minus_ii"]["difference"] for r in ribas_c1.values()],
        *[r["gap_i_minus_ii"]["difference"] for r in ribas_c1_bal.values()],
        tuned["gap_i_minus_ii"]["difference"],
    ]
    ribas_c1_text = "放進 C1，差距 " + "、".join(
        f"{MATERIAL_NAME[m]} {pts(r['gap_i_minus_ii']['difference'])} "
        f"{bracket(r['gap_i_minus_ii']['ci_low'], r['gap_i_minus_ii']['ci_high'])}"
        for m, r in ribas_c1.items()
    )
    ribas_c1_reading = (
        "放進 C1 也一樣"
        if all(
            r["ii"]["uar"] < RIBAS_FAILS and r["gap_i_minus_ii"]["difference"] >= RIBAS_DROP_HOLDS
            for r in ribas_c1.values()
        )
        else "放進 C1 不一樣"
    )
    tuned_status = finetune_status(tuned)
    vowel_text, vowel_reading = vowel_cells(vowels)
    new_lines = {
        "fusion": fusion_line(fusion),
        "diagnosis": diagnosis_line(
            diagnosis,
            diagnosis_repeat(
                [optional(f"diagnosis_breakdown/summary_{v}.json") for v in ("i", "u")]
            ),
        ),
        "outside": outside_line(views),
        "pool": pool_line(pooled),
        "dose": dose_line(optional("dose_response/summary.json")),
        "erasure": erasure_line(erasure, erasure_folds, hub_c1["difference"], hub_x["all"]["auc"]),
        "align": align_line(),
        "within": within_line(),
    }
    outside_note = ""
    if views[0] is not None:
        outside_gaps = [r["outside"]["gap"]["difference"] for r in views[:3] if r is not None]
        outside_note = (
            f"換一個沒看過 AVFAD 的偵測器看同一組人，這個差距不存在（{len(outside_gaps)} 個母音 "
            f"{signed(max(outside_gaps))} 到 {signed(min(outside_gaps))} 點），是訓練時學到的地點。"
        )
    protocol_rows = "\n".join(
        protocol_row(r, cross)
        for r in protocol
        if r["material"] == "/a/" and r["hsa_uar"] is not None
    )
    judgement = judgement_text(vowels, hub_x, voiced_x, diagnosis, erasure, hub_c1["difference"])
    repeats = vowels is not None and vowels.get("both", {}).get("reading") == "repeats"
    vowel_note = "（/a/、/i/、/u/ 都是）" if repeats else "（/a/）"
    all_ii_note = (
        ""
        if vowels is None or "both" not in vowels
        else "；/i/、/u/ 是 "
        + "、".join(signed(vowels[v]["all_minus_ii"]) for v in ("i", "u"))
        + " 點"
        + ""
    )

    def speech_row(label: str, name: str) -> str:
        hand, hub = speech_hand[name], speech_hub[name]
        quiet = speech_quiet[name]["non_phonation"]
        return (
            f"| {label} | {u(hand['uar_ii'])} | {u(hand['uar_i_min'])}–{u(hand['uar_i_max'])} | "
            f"{pts(hand['difference'])} {bracket(hand['ci_low'], hand['ci_high'])} | "
            f"{u(hub['uar_ii'])} | {u(hub['uar_i_min'])}–{u(hub['uar_i_max'])} | "
            f"{pts(hub['difference'])} {bracket(hub['ci_low'], hub['ci_high'])} | "
            f"{u(quiet['all']['uar'])} → {u(quiet['hsa']['uar'])} |"
        )

    return f"""# AVFAD、SVD、VOICED 的地點與人群混淆：結果總結

2026-11-11。不是論文初稿。數字由 `make_summary.py` 從結果檔產生，細節見 `deviations.md`。第 1 節的四個檢定是事先定好的；其餘的探索，判讀都先寫好，標「事後」的除外。

**一句話**：AVFAD 全體 {u(min(everyone_uars))}–{u(max(everyone_uars))} 的 UAR，在同一家醫院（HSA）內只剩 {u(min(inside_uars))}–{u(max(inside_uars))}：手工特徵、凍結和微調的 HuBERT、已發表的 ComParE 加 SVM 都一樣。{place_only_lead()}同樣的病人換配另一個地點的正常人，成績就高 {pts(min(gaps))}–{pts(max(gaps))} 點；多出來的這部分帶不到 SVD{vowel_note}。{outside_note}

**名詞**：**UAR**：正常和病理各自答對率的平均；UAR、AUC 都是 0.5 等於亂猜。**ESSUA**：AVFAD 的大學端，{gate["normal_speakers"]["ESSUA"]} 位全是正常人。**HSA**：醫院端，{gate["normal_speakers"]["HSA"]} 位正常、{c1["patients"]} 位病人。**(i)／(ii)**：HSA 病人對「全體正常人配對抽樣」／對 HSA 自己的正常人。**手工**：MFCC、頻譜、音量統計接 logistic。**HuBERT**：語音模型的層特徵接 logistic。**非發聲段**：沒有人聲、只剩房間的片段。**D_ext**：AVFAD 訓練、外部資料庫測，(i) 的 AUC 減 (ii) 的 AUC。

## 1. 四個事先定好的檢定（手工特徵，Holm 校正）

| 檢定 | 結果 | Holm p | 結論（門檻） |
|---|---|---|---|
| C1 只換正常組來源 | (i) {u(c1["uar_i_mean"])}（{c1["draws"]} 次抽樣 {u(c1["uar_i_min"])}–{u(c1["uar_i_max"])}）對 (ii) {u(c1["uar_ii"])}，差 {pts(c1["difference"])} 點 {bracket(c1["ci_low"], c1["ci_high"])} | {p_text(family["C1"]["p_holm"])} | 有（{c_gap[0]:.0f} 點） |
| C2 只留發聲段分兩地正常人 | UAR {u(c2["uar"])}，比機率高 {pts(family["C2"]["estimate"])} 點 | {p_text(family["C2"]["p_holm"])} | 有（{c_gap[1]:.0f} 點） |
| C3 SVD iau 完整檔減只有發聲 | {pts(c3["difference"])} 點 {bracket(c3["ci_low"], c3["ci_high"])} | {p_text(family["C3"]["p_holm"])} | 不顯著（{c_gap[2]:.0f} 點） |
| C4 SVD 非發聲段，年齡性別年份之外 | {u(c4["uar"])} 減 null {u(c4["null_mean"])}，{pts(c4["difference"])} 點 | {p_text(family["C4"]["p_holm"])} | 存在但很小（{c_gap[3]:.0f} 點） |

正向對照 {u(gate["uar"])}，打亂標籤各臂平均 {u(min(negatives))}–{u(max(negatives))}。VOICED 負對照沒有乾淨地過（只用錄音技術特徵 {u(voiced_tech["uar"])} {score_bracket(*voiced_tech["uar_ci"])}）。{placebo_text()}{by_day_text()}

## 2. 未條件化對條件化

| AVFAD | 手工 UAR／AUC | HuBERT UAR／AUC |
|---|---|---|
| 全體 {every["all_handcrafted"]["n_speakers"]} 人，未條件化 | {u(every["all_handcrafted"]["uar"])}／{u(every["all_handcrafted"]["roc_auc"])} | {u(every["all_hubert"]["uar"])}／{u(every["all_hubert"]["roc_auc"])} |
{place_only_row()}| 同地點（HSA 內） | {u(c1["uar_ii"])}／{u(c1["roc_auc_ii"])} | {u(hub_ii["uar"])}／{u(hub_ii["roc_auc"])} |
| 同地點、同年齡性別層內的 AUC | {inside(strata["avfad_hsa_handcrafted"])} | {inside(strata["avfad_hsa_hubert"])} |
| 只用長度、靜音、音量、底噪：全體 → HSA 內 | {u(tech_all["uar"])} → {u(tech_hsa["uar"])} | 不適用 |

SVD（手工）只用年齡性別 {u(age_sex["uar"])}，iau 完整檔 {u(conditioned["svd_iau_full"]["uar"])}，iau 在年齡性別年份層內 AUC {inside(strata["svd_iau_full"])}。

**換材料**：六句、朗讀的 (ii) 手工 {u(speech_hand["cape_v"]["uar_ii"])}、{u(speech_hand["reading"]["uar_ii"])}，HuBERT {u(speech_hub["cape_v"]["uar_ii"])}、{u(speech_hub["reading"]["uar_ii"])}；(i) − (ii) 的差距和 /a/ 合起來是 {pts(min(all_gaps))}–{pts(max(all_gaps))} 點。{phon_note}{place_materials_note()}

## 3. 之後的探索：反駁和補救

| 問題和做法 | 結果 | 判讀 |
|---|---|---|
| Ribas 2023（ComParE 加 SVM） | 公開的逐檔預測只看 HSA：/a/ {pts(ribas["a_n"]["uar_all"])} → {pts(ribas["a_n"]["uar_hsa"])} {bracket(*ribas["a_n"]["uar_hsa_ci"])}，句子 {pts(ribas["phrase"]["uar_all"])} → {pts(ribas["phrase"]["uar_hsa"])}；他們的方法（重現差 {max(abs(r["uar_mean_over_folds_percent"] - r["paper_uar_percent"]) for r in ribas_rep.values()):.1f} 點內）只在 HSA 內訓練，/a/、句子 {u(ribas_hsa["a_n"]["ii"]["uar"])}、{u(ribas_hsa["phrase"]["ii"]["uar"])}；{ribas_c1_text} | 也分不開；{ribas_c1_reading} |
| 微調 HuBERT 最後 2 層，/a/ | (ii) {u(tuned["ii"]["uar"])}，比凍結 {signed(tuned["ii_fine_tuned_minus_frozen"]["difference"])} 點 {bracket(tuned["ii_fine_tuned_minus_frozen"]["ci_low"], tuned["ii_fine_tuned_minus_frozen"]["ci_high"])}；差距 {pts(tuned["gap_i_minus_ii"]["difference"])} {bracket(tuned["gap_i_minus_ii"]["ci_low"], tuned["gap_i_minus_ii"]["ci_high"])} | {tuned_status} |
| AVFAD /a/ 訓練，SVD a_n 測（{hub_x["svd"]["speakers"]:,} 人） | HuBERT AUC：(ii) {u(hub_x["ii"]["auc"])}、(i) {u(hub_x["i"]["auc"])}、全體 {u(hub_x["all"]["auc"])}、ESSUA 正常配 HSA 病人 {u(hub_x["essua_normals_hsa_patients"]["auc"])}；D_ext {signed(hub_x["d_ext"]["auc"])} {bracket(*hub_x["d_ext"]["auc_ci"])}（手工 {signed(hand_x["d_ext"]["auc"])}）；測 VOICED（{voiced_x["speakers"]} 人，8 kHz）D_ext {signed(voiced_x["d_ext"]["auc"])} {bracket(*voiced_x["d_ext"]["ci"])}；換成 /i/、/u/，D_ext {signed(vowels["i"]["d_ext"]["auc"])} {bracket(*vowels["i"]["d_ext"]["auc_ci"])}、{signed(vowels["u"]["d_ext"]["auc"])} {bracket(*vowels["u"]["d_ext"]["auc_ci"])} | 帶不過去（/i/、/u/ {VOWEL_READING[vowels["both"]["reading"]]}） |
{new_lines["outside"]}{new_lines["within"]}{new_lines["dose"]}{new_lines["align"]}
## 4. 建議的評估方式

先只用正常人訓練地點偵測器，看它在全體拿多少；報同地點內 UAR、AUC、年齡 × 性別層內 AUC 和非發聲段；至少一個外部資料庫，同地點內和混合訓練並排；再用沒看過此資料庫的偵測器看同一組人。/a/：

| 偵測器 | HSA 內 UAR／AUC | 層內 AUC | 全體 UAR | SVD AUC：HSA 內訓練／全體訓練 |
|---|---|---|---|---|
{protocol_rows}

## 5. 能說到什麼程度

- **事先登錄的**：只換正常組來源，C1 差 {pts(c1["difference"])} 點 {bracket(c1["ci_low"], c1["ci_high"])}；只留發聲段分兩地正常人，C2 {u(c2["uar"])}（第 1 節）。計畫裡列為探索性的：AVFAD 只用非發聲段，全體 {u(quiet_all["uar"])}，同地點內 {u(quiet_hsa["uar"])} {score_bracket(*quiet_hsa["uar_ci"])}。
- **探索性（事後加，判讀先寫）**：非發聲段換三種材料都一樣{room_note()}。換材料、偵測器和已發表系統，C1 的差距 {pts(min(gaps))}–{pts(max(gaps))} 點，人數補不上。(i) 多出來的分數帶不到 SVD{vowel_note}，沒看過 AVFAD 的偵測器也看不到這個差距。{svd_time_note()}混合訓練在 SVD 上比 HSA 內訓練低，只限 /a/（{signed(all_vs_ii["hubert"]["all_minus_ii_auc"])} 點，HuBERT）{all_ii_note}{mixed_note(vowels)}。
- **分不開的**：C1、C2 剩下的是房間、迴響、麥克風距離、招募來源和篩檢方式、聲音健康或抽菸（正常人現在抽菸 ESSUA {pts(smokes["ESSUA"])}%、HSA {pts(smokes["HSA"])}%），**不能說是房間**：地點方向淺層跟著底噪、深層跟著檔案長度（錄音流程），房間噪音只是一部分{channel_note()}{simulated_free_note()}。{simulated_text()}HSA 內低分原因分不開{svd_inside_note()}。

## 6. 判斷

{judgement}
"""


def main() -> None:
    OUTPUT.write_text(build(), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
