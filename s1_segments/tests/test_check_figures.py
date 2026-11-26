"""The acceptance figures. Run with the paper venv, which has matplotlib:

python -I -m pytest -q -p no:cacheprovider s1_segments/tests/test_check_figures.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_figures as cf


def fake_tables(tmp_path):
    s0 = tmp_path / "s0"
    s0.mkdir()
    n = 40
    voiced = [f"voice{i:03d}" for i in range(n)]
    pd.DataFrame(
        {"record": voiced, "label_header": ["healthy"] * 20 + ["pathological"] * 20}
    ).to_csv(s0 / "voiced_meta.csv", index=False)
    pd.DataFrame(
        {
            "session": [str(i) for i in range(n)] * 2,
            "material": ["a_n"] * n + ["iau"] * n,
            "label": (["normal"] * 20 + ["pathological"] * 20) * 2,
        }
    ).to_csv(s0 / "svd_file_status.csv", index=False)
    speakers = [f"S{i:02d}" for i in range(60)]
    pd.DataFrame(
        {
            "speaker": speakers,
            "place": ["ESSUA"] * 20 + ["HSA"] * 35 + ["ULSM"] * 5,
            "label": ["normal"] * 30 + ["pathological"] * 30,
        }
    ).to_csv(s0 / "avfad_speakers.csv", index=False)
    rows = [("voiced__" + v, "voiced", "a") for v in voiced]
    rows += [(f"svd__{i}-{m}", "svd", m) for i in range(n) for m in ("a_n", "iau")]
    rows += [("avfad__" + s, "avfad", "a") for s in speakers]
    rows += [("avfad_silence__0001", "avfad", "silence")]
    index = pd.DataFrame(rows, columns=["file_id", "database", "material"])
    index["wav"] = index.file_id + ".wav"
    index.to_csv(tmp_path / "index.csv", index=False)
    return s0, tmp_path / "index.csv"


def test_sample_is_stratified_blind_and_repeatable(tmp_path):
    s0, index = fake_tables(tmp_path)
    out = tmp_path / "out"
    sample = cf.sample_step(index, s0, out)
    key = pd.read_csv(out / "check_key.csv")
    assert key.groupby(key.stratum.str.split("_").str[0]).size().to_dict() == {
        "avfad": 30,
        "svd": 30,
        "voiced": 30,
    }
    counts = key.stratum.value_counts().to_dict()
    assert (
        counts["avfad_essua"]
        == counts["avfad_hsa_normal"]
        == counts["avfad_hsa_pathological"]
        == 10
    )
    assert counts["voiced_healthy"] == 15
    assert counts["svd_iau_normal"] in (7, 8)
    assert "avfad_silence__0001" not in set(key.file_id)
    assert not set(key.file_id) & {f"avfad__S{i}" for i in range(55, 60)}  # the ULSM speakers
    sheet = pd.read_csv(out / "check_sheet.csv")
    assert set(sheet.columns) == {"figure", "correct", "note"}
    assert len((out / "check_ids.txt").read_text().split()) == 90
    # Figure numbers do not run database by database.
    assert sample.database.iloc[:30].nunique() > 1
    again = cf.sample_step(index, s0, tmp_path / "again")
    assert again.file_id.tolist() == sample.file_id.tolist()


def test_figures_show_the_number_and_not_the_file(tmp_path, monkeypatch):
    monkeypatch.setattr(cf, "ROOT", tmp_path)
    rate = 8000
    x = (0.1 * np.sin(2 * np.pi * 200 * np.arange(rate * 2) / rate) * 32768).astype(np.int16)
    wavfile.write(tmp_path / "a.wav", rate, x)
    s1 = tmp_path / "out"
    (s1 / "tracks").mkdir(parents=True)
    times = np.arange(0, 2, 0.01)
    np.savez(
        s1 / "tracks" / "voiced__x.npz",
        times=times,
        level_db=np.full(times.size, -30.0),
        voiced=np.ones(times.size, dtype=bool),
        voiced_prob=np.ones(times.size),
    )
    pd.DataFrame(
        {
            "file_id": ["voiced__x", "voiced__y"],
            "status": ["ok", "empty"],
            "wav": ["a.wav", "b.wav"],
            "floor_db": [-60.0, np.nan],
            "threshold_db": [-52.0, np.nan],
        }
    ).to_csv(s1 / "files.csv", index=False)
    pd.DataFrame(
        {"file_id": ["voiced__x"], "kind": ["phonation"], "start_s": [0.1], "end_s": [1.9]}
    ).to_csv(s1 / "spans.csv", index=False)
    pd.DataFrame(
        {
            "figure": ["check_001", "check_002"],
            "file_id": ["voiced__x", "voiced__y"],
            "stratum": ["voiced_healthy", "voiced_healthy"],
        }
    ).to_csv(s1 / "check_key.csv", index=False)
    assert cf.draw_step(s1) == 1
    assert (s1 / "figures" / "check_001.png").stat().st_size > 1000
    assert not (s1 / "figures" / "check_002.png").exists()
