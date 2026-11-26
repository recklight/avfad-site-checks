"""S0 for SVD: labels, exclusions and the one-session-per-speaker set.

Usage: python -I s0_audit/svd_meta.py

Inputs: data/svd/voice_data.csv (the official session index),
s0_audit/out/svd_files.csv, svd_stats.csv and svd_eggcheck_*.csv.
Outputs: s0_audit/out/svd_sessions.csv (one row per session),
svd_file_status.csv (one row per speech file) and svd_describe_<material>.csv.

Exclusion rules follow D3; the label groups below are my reading of the 71
German labels and are listed in decisions.csv.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from describe import NUMERIC, table

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "out"

# Labels that are not voice disorders: fluency, articulation, resonance,
# syndromes and neck findings outside the larynx.
NON_VOICE = {
    "Balbuties",
    "Poltersyndrom",
    "Non-fluency-Syndrom",
    "Sigmatismus",
    "Rhinophonie aperta",
    "Rhinophonie clausa",
    "Rhinophonie mixta",
    "Orofaciale Dyspraxie",
    "Velopharyngoplastik",
    "Morbus Down",
    "Dish-Syndrom",
    "Mediale Halscyste",
}
# Partial removal of a vocal fold or the larynx: the voice is a surgical result.
POST_SURGERY = {"Chordektomie", "Frontolaterale Teilresektion"}
SINGER = {"Gesangsstimme", "Sängerstimme"}

# Notes that say the session was recorded after treatment of the voice problem.
# Thyroid surgery is excluded from the match: there it names the cause of a
# recurrent nerve palsy, not a treatment.
POST_TREATMENT = re.compile(
    r"Zustand (nach|danach)|nach Botox|Aufnahme nach (Botox|Injektion|OP|Therapie)|"
    r"Aufn\. nach Botox|nach (logop\w*\.? )?Therapie|postoperativ",
    re.IGNORECASE,
)
TREATMENT_IS_CAUSE = re.compile(r"Struma|Strumektomie|Schilddrüse", re.IGNORECASE)

# File problems listed in sbvoicedb's documentation and confirmed or
# extended by svd_eggcheck.py.
DOCUMENTED = {
    (139, "a_n"): "speech_egg_swapped",
    (139, "iau"): "speech_egg_swapped",
    (139, "phrase"): "speech_egg_swapped",
    (141, "a_n"): "speech_egg_swapped",
    (141, "iau"): "speech_egg_swapped",
    (141, "phrase"): "speech_egg_swapped",
    (980, "iau"): "speech_egg_swapped_partway",
    (980, "phrase"): "speech_egg_swapped",
    (1697, "iau"): "speech_egg_swapped",
    (1697, "phrase"): "speech_egg_swapped",
    (713, "iau"): "corrupted",
    (1573, "a_n"): "miscut",
}
# Swap screen: the speech file is at least 6 dB below its EGG partner in the
# F1/F0 band ratio while the EGG partner looks like speech (above -4 dB).
SCREEN_MARGIN_DB = -6.0
SCREEN_EGG_DB = -4.0


def split_labels(text: str) -> list[str]:
    return [t.strip() for t in str(text).split(",") if t.strip()] if isinstance(text, str) else []


def load_sessions() -> pd.DataFrame:
    meta = pd.read_csv(ROOT / "data" / "svd" / "voice_data.csv", dtype=str)
    s = pd.DataFrame(
        {
            "session": meta.AufnahmeID.astype(int),
            "speaker": meta.SprecherID.astype(int),
            "type": meta.AufnahmeTyp,
            "date": pd.to_datetime(meta.AufnahmeDatum),
            "birth": pd.to_datetime(meta.Geburtsdatum),
            "sex": meta.Geschlecht,
            "labels": meta.Pathologien.fillna(""),
            "note": meta.Diagnose.fillna(""),
        }
    )
    s["label"] = np.where(s.type == "n", "normal", "pathological")
    s["year"] = s.date.dt.year
    s["age"] = (s.date - s.birth).dt.days / 365.25
    s["n_labels"] = s.labels.map(lambda t: len(split_labels(t)))
    return s


def main() -> None:
    s = load_sessions()
    files = pd.read_csv(OUT / "svd_files.csv")
    stats = pd.read_csv(OUT / "svd_stats.csv")
    have = files.groupby("session").material.apply(set)
    extra = sorted(set(have.index) - set(s.session))
    s = pd.concat([s, pd.DataFrame({"session": extra})], ignore_index=True)
    s[["labels", "note"]] = s[["labels", "note"]].fillna("")
    for mat in ("a_n", "iau", "phrase"):
        s[f"has_{mat}"] = s.session.map(lambda x, m=mat: m in have.get(x, set()))
    s["source_zip"] = (
        s.session.map(files.groupby("session").source_zip.first())
        if "source_zip" in files
        else "data.zip"
    )

    labels = s.labels.map(split_labels)
    s["x_no_metadata"] = s.type.isna()
    s["x_no_audio"] = ~(s.has_a_n | s.has_iau | s.has_phrase)
    s["x_minor"] = s.age < 18
    s["x_singer"] = labels.map(lambda ls: any(label in SINGER for label in ls))
    s["x_non_voice"] = labels.map(lambda ls: any(label in NON_VOICE for label in ls))
    s["x_post_surgery_label"] = labels.map(lambda ls: any(label in POST_SURGERY for label in ls))
    s["x_post_treatment_note"] = s.note.map(
        lambda t: bool(POST_TREATMENT.search(t)) and not TREATMENT_IS_CAUSE.search(t)
    )
    session_flags = [
        "x_no_metadata",
        "x_no_audio",
        "x_minor",
        "x_singer",
        "x_non_voice",
        "x_post_surgery_label",
        "x_post_treatment_note",
    ]
    s["eligible"] = ~s[session_flags].any(axis=1)
    # One session per speaker: the earliest eligible one (ties by session id).
    elig = s[s.eligible].sort_values(["date", "session"])
    first = elig.groupby("speaker").session.first()
    s["primary"] = s.session.isin(first.values)
    s["speaker_sessions"] = s.speaker.map(s.groupby("speaker").size())
    s["speaker_has_both_labels"] = s.speaker.map(s.groupby("speaker").type.nunique() > 1)

    # File-level problems.
    egg = pd.concat(
        [pd.read_csv(p) for p in sorted(OUT.glob("svd_eggcheck_*.csv"))], ignore_index=True
    )
    egg = egg.drop_duplicates(["session", "material"])
    screen = egg[(egg.margin_db < SCREEN_MARGIN_DB) & (egg.egg_f1_f0_db > SCREEN_EGG_DB)]
    screened = {(int(a), b) for a, b in zip(screen.session, screen.material)}
    status = files[["session", "material", "wav", "duration_s"]].copy()
    if "reader_error" in files:
        status["reader_error"] = files.reader_error
    key = list(zip(status.session, status.material))
    status["file_problem"] = [DOCUMENTED.get(k, "") for k in key]
    status.loc[[k in screened and k not in DOCUMENTED for k in key], "file_problem"] = "swap_screen"
    status.loc[(status.duration_s.fillna(0) == 0) & (status.file_problem == ""), "file_problem"] = (
        "empty"
    )
    status = status.merge(egg[["session", "material", "margin_db", "egg_f1_f0_db"]], how="left")
    status = status.merge(
        s[["session", "speaker", "label", "eligible", "primary"]], how="left", on="session"
    )
    status["include"] = status.primary.fillna(False).astype(bool) & (status.file_problem == "")

    s.drop(columns=["birth"]).to_csv(OUT / "svd_sessions.csv", index=False)
    status.to_csv(OUT / "svd_file_status.csv", index=False)

    # Descriptive tables on the included files, by label.
    sess_cols = ["session", "age", "year", "sex"]
    for mat in ("a_n", "iau", "phrase"):
        inc = status[(status.material == mat) & status.include][["session", "label"]]
        df = inc.merge(s[sess_cols], on="session").merge(
            stats[stats.material == mat].drop(columns=["material"]), on="session"
        )
        t = table(df, "label", NUMERIC)
        t.to_csv(OUT / f"svd_describe_{mat}.csv", index=False)

    # Console summary.
    print("sessions in index:", int((~s.x_no_metadata).sum()), "speakers:", s.speaker.nunique())
    print("with audio:", int((~s.x_no_audio).sum()), "extra zip sessions:", extra)
    print(s[session_flags].sum().to_string())
    print("eligible:", int(s.eligible.sum()), "primary:", int(s.primary.sum()))
    print(s[s.primary].label.value_counts().to_string())
    print("file problems:", status.file_problem.value_counts().to_dict())
    print(status[status.include].groupby(["material", "label"]).size().unstack().to_string())


if __name__ == "__main__":
    main()
