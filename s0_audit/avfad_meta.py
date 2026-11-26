"""S0 for AVFAD: speakers, labels, recording places and the room-silence files.

Usage: python -I s0_audit/avfad_meta.py

Inputs: AVFAD_01_00_00.xlsx (sheet "AVFAD") in the README folder,
s0_audit/out/avfad_files.csv and s0_audit/out/avfad_stats.csv.
Outputs in s0_audit/out/:
  avfad_speakers.csv      one row per speaker, with the room-silence file matched to the visit
  avfad_silence.csv       one row per room-silence file
  avfad_file_status.csv   one row per speaker WAV, with format problems
  avfad_describe_*.csv    /a/ and silence statistics by label and by place

The label is CMVD-I dimension 1: code 0 is "Normal", every other code is a
voice disorder. No diagnosis in the sheet falls outside voice disorders, so
nothing is excluded by diagnosis; the candidates for exclusion are listed as
decisions.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from avfad_extract import OUT, RAW, require_access
from describe import NUMERIC, table

README = RAW / "AVFAD_01_00_00_1_README" / "AVFAD_01_00_00_1_README"
SHEET = README / "AVFAD_01_00_00.xlsx"

COLUMNS = {
    "File ID": "speaker",
    "Visit date": "visit_date",
    "Visit place": "place",
    "Age": "age",
    "Sex": "sex",
    "BMI": "bmi",
    "Surgery": "laryngeal_surgery",
    "Smoking": "smoking",
    "Singing": "singing",
    "Job (numeric system)": "job",
    "CMVD-I Dimension 1 (numeric system)": "cmvd_code",
    "CMVD-I Dimension 1 (word system)": "cmvd_name",
}

# Room-silence files: <year>_<month>_<day>_<place>_silence..., then optional
# tokens naming the room (room1, room2) or the speakers recorded in that
# session (ABC or ABCraw). A few names repeat ".wav".
SILENCE = re.compile(
    r"^(?P<y>\d{4})_(?P<m>\d{1,2})_(?P<d>\d{1,2})_(?P<place>[A-Za-z]+)_silence_?(?P<rest>.*?)"
    r"(?:\.wav)+$",
    re.IGNORECASE,
)
SPEAKER_TOKEN = re.compile(r"^(?P<code>[A-Z]{3})(?:raw)?$")
ROOM_TOKEN = re.compile(r"^room\d+$", re.IGNORECASE)
LENGTH_TOKENS = {"60", "60s", "seconds"}

# Speaker WAVs: <code><task>.wav, tasks 001-011 (README_01_00_00.txt).
SPEAKER_WAV = re.compile(r"^(?P<code>[A-Z]{3})(?P<task>\d{3})\.wav$", re.IGNORECASE)
TASKS = {
    "001": "i",
    "002": "a",
    "003": "u",
    "004": "cape_v_1",
    "005": "cape_v_2",
    "006": "cape_v_3",
    "007": "cape_v_4",
    "008": "cape_v_5",
    "009": "cape_v_6",
    "010": "reading",
    "011": "spontaneous",
}
MIN_DURATION_S = 0.5


def load_speakers() -> pd.DataFrame:
    raw = pd.read_excel(SHEET, sheet_name="AVFAD")
    raw.columns = [str(c).strip() for c in raw.columns]
    s = raw[list(COLUMNS)].rename(columns=COLUMNS)
    # Two cells carry a trailing space ("HSA ", "F ").
    for col in ("speaker", "place", "sex"):
        s[col] = s[col].astype(str).str.strip()
    # Missing BMI is either empty (7 rows) or "." (2 rows).
    s["bmi"] = pd.to_numeric(s.bmi.mask(s.bmi.astype(str) == "."))
    s["visit_date"] = pd.to_datetime(s.visit_date).dt.date
    s["year"] = pd.to_datetime(s.visit_date).dt.year
    s["label"] = np.where(s.cmvd_code == 0, "normal", "pathological")
    return s


def parse_silence(files: pd.DataFrame) -> pd.DataFrame:
    sil = files[files.path.str.contains("_silence", case=False) & (files.ext == ".wav")].copy()
    sil["name"] = sil.path.str.rsplit("/", n=1).str[-1]
    rows = []
    for name in sil.name:
        m = SILENCE.match(name)
        if m is None:
            rows.append({"name": name, "parsed": False})
            continue
        tokens = [t for t in m["rest"].split("_") if t and t.lower() not in LENGTH_TOKENS]
        speakers = [t["code"] for t in map(SPEAKER_TOKEN.match, tokens) if t]
        rooms = [t for t in tokens if ROOM_TOKEN.match(t)]
        other = [t for t in tokens if not SPEAKER_TOKEN.match(t) and not ROOM_TOKEN.match(t)]
        rows.append(
            {
                "name": name,
                "parsed": True,
                "visit_date": pd.Timestamp(int(m["y"]), int(m["m"]), int(m["d"])).date(),
                "place": m["place"].upper(),
                "room": " ".join(rooms),
                "listed_speakers": " ".join(speakers),
                "unparsed_tokens": " ".join(other),
            }
        )
    sil = sil.merge(pd.DataFrame(rows), on="name")
    # Byte-identical files carry the same group number.
    groups = sil.groupby("sha256").name.transform("size")
    sil["identical_files"] = groups
    sil["identical_group"] = np.where(groups > 1, sil.sha256.rank(method="dense"), np.nan)
    return sil


def match_silence(s: pd.DataFrame, sil: pd.DataFrame) -> pd.DataFrame:
    """Pick the room-silence file of each speaker's visit (date and place).

    A file that lists the speaker wins. Otherwise the visit's only file is
    taken, or, when the visit has several, the only one that lists nobody.
    Several candidates (room1 and room2, or one file per session) leave the
    speaker unmatched.
    """
    listed = {
        code: name
        for name, codes in zip(sil.name, sil.listed_speakers.fillna(""), strict=True)
        for code in codes.split()
    }
    by_visit = {k: g for k, g in sil[sil.parsed].groupby(["visit_date", "place"])}
    names, how = [], []
    for code, date, place in zip(s.speaker, s.visit_date, s.place, strict=True):
        g = by_visit.get((date, place))
        if code in listed:
            name, why = listed[code], "listed"
        elif g is None:
            name, why = "", "no_file_for_visit"
        elif len(g) == 1:
            name, why = g.name.iloc[0], "visit"
        else:
            unlisted = g[g.listed_speakers.fillna("") == ""]
            if len(unlisted) == 1:
                name, why = unlisted.name.iloc[0], "visit"
            else:
                name, why = "", "several_files"
        names.append(name)
        how.append(why)
    s = s.copy()
    s["silence_file"] = names
    s["silence_match"] = how
    return s


def file_status(files: pd.DataFrame) -> pd.DataFrame:
    w = files[(files.ext == ".wav") & ~files.path.str.contains("_silence", case=False)].copy()
    m = w.path.str.rsplit("/", n=1).str[-1].str.extract(SPEAKER_WAV)
    w["task"] = m.task
    w["material"] = m.task.map(TASKS)
    w["name_matches_folder"] = m.code.str.upper() == w.speaker
    problems = []
    for r in w.itertuples():
        p = []
        if r.n_frames == 0:
            p.append("empty")
        elif r.duration_s < MIN_DURATION_S:
            p.append("shorter_than_0.5s")
        if r.channels != 1:
            p.append(f"{int(r.channels)}_channels")
        if r.rate != 48000:
            p.append(f"rate_{int(r.rate)}")
        if r.ext_as_named != r.ext:
            p.append("upper_case_suffix")
        problems.append("; ".join(p))
    w["file_problem"] = problems
    cols = ["path", "folder", "speaker", "task", "material", "rate", "channels", "duration_s"]
    return w[[*cols, "name_matches_folder", "file_problem", "sha256"]]


def main() -> None:
    require_access()
    files = pd.read_csv(OUT / "avfad_files.csv")
    stats = pd.read_csv(OUT / "avfad_stats.csv")
    s = load_speakers()
    folders = files[files.ext == ".wav"].groupby("speaker").folder.first()
    s["folder"] = s.speaker.map(folders)

    # The inventory already carries rate, channels and duration; take the rest from stats.
    stats = stats.drop(columns=[c for c in stats.columns if c in files.columns and c != "path"])
    sil = parse_silence(files).merge(stats, on="path", how="left")
    s = match_silence(s, sil)
    status = file_status(files)
    a = status[status.task == "002"][["speaker", "path", "file_problem"]].rename(
        columns={"path": "a_path", "file_problem": "a_problem"}
    )
    s = s.merge(a, on="speaker", how="left")
    s["silence_identical_files"] = s.silence_file.map(sil.set_index("name").identical_files)

    s.to_csv(OUT / "avfad_speakers.csv", index=False)
    sil.drop(columns=["sha256"]).to_csv(OUT / "avfad_silence.csv", index=False)
    status.to_csv(OUT / "avfad_file_status.csv", index=False)

    # /a/ statistics by label, within HSA, and normal speakers by place.
    a_cols = files[["path", "duration_s"]].merge(stats, on="path")
    df = s.merge(a_cols.rename(columns={"path": "a_path"}), on="a_path", how="left")
    df["sex_f"] = (df.sex == "F").astype(float)
    cols = [*NUMERIC, "sex_f", "bmi", "laryngeal_surgery", "singing"]
    table(df, "label", cols).to_csv(OUT / "avfad_describe_a.csv", index=False)
    table(df[df.place == "HSA"], "label", cols).to_csv(
        OUT / "avfad_describe_a_hsa.csv", index=False
    )
    normal = df[(df.label == "normal") & df.place.isin(["ESSUA", "HSA"])]
    table(normal, "place", cols).to_csv(OUT / "avfad_describe_a_normal_by_place.csv", index=False)
    # Silence statistics: one row per speaker, so each visit counts by its speakers.
    room = s[s.silence_file != ""].merge(
        sil.rename(columns={"name": "silence_file"})[["silence_file", *NUMERIC[:13]]],
        on="silence_file",
    )
    table(room, "label", NUMERIC[:13]).to_csv(OUT / "avfad_describe_silence.csv", index=False)

    print("speakers:", len(s), s.label.value_counts().to_dict())
    print(pd.crosstab(s.place, s.label, margins=True).to_string())
    print("silence files:", len(sil), "parsed:", int(sil.parsed.sum()))
    print("silence by place:", sil.place.value_counts().to_dict())
    print(
        "unknown speaker codes in silence names:",
        sorted(set(" ".join(sil.listed_speakers.fillna("")).split()) - set(s.speaker)),
    )
    print("unparsed tokens:", [t for t in sil.unparsed_tokens.fillna("") if t])
    print(pd.crosstab(s.silence_match, s.label).to_string())
    print(pd.crosstab([s.place, s.silence_match], s.label).to_string())
    print("speaker files with problems:", status.file_problem.ne("").sum())
    print(status[status.file_problem != ""].groupby(["task", "file_problem"]).size().to_string())
    print("names match folder:", bool(status.name_matches_folder.all()))


if __name__ == "__main__":
    main()
