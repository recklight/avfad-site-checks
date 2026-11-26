import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import svd_extract
from test_readers import nsp_bytes


@pytest.fixture
def fake_svd(tmp_path, monkeypatch):
    root = tmp_path
    svd = root / "data" / "svd"
    svd.mkdir(parents=True)
    zpath = svd / "data.zip"
    rng = np.random.default_rng(1)
    expected = {}
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data/", b"")
        for session in ("7", "12"):
            for folder, stem in (("vowels", "a_n"), ("vowels", "iau"), ("sentences", "phrase")):
                x = rng.integers(-2000, 2000, 500).astype(np.int16)
                expected[(session, stem)] = x
                zf.writestr(f"data/{session}/{folder}/{session}-{stem}.nsp", nsp_bytes(x))
                zf.writestr(f"data/{session}/{folder}/{session}-{stem}-egg.egg", nsp_bytes(x // 2))
            zf.writestr(f"data/{session}/vowels/{session}-i_n.nsp", nsp_bytes(x))
        zf.writestr("data/overview.csv", "AufnahmeID;SprecherID\n7;1\n12;2\n")
    size = zpath.stat().st_size
    md5 = svd_extract.file_hash(zpath, "md5")
    monkeypatch.setattr(svd_extract, "ROOT", root)
    monkeypatch.setattr(svd_extract, "SVD", svd)
    monkeypatch.setattr(svd_extract, "ZIP", zpath)
    monkeypatch.setattr(svd_extract, "NSP_DIR", svd / "nsp")
    monkeypatch.setattr(svd_extract, "META_DIR", svd / "meta")
    monkeypatch.setattr(svd_extract, "WAV_DIR", root / "data" / "derived" / "svd")
    monkeypatch.setattr(svd_extract, "MANIFEST", svd / "manifest.json")
    monkeypatch.setattr(svd_extract, "OUT", root / "out")
    monkeypatch.setattr(svd_extract, "ZENODO", {**svd_extract.ZENODO, "size": size, "md5": md5})
    return root, expected


def test_extract_takes_only_the_three_speech_files(fake_svd):
    root, expected = fake_svd
    svd_extract.verify()
    svd_extract.extract()
    files = pd.read_csv(root / "out" / "svd_files.csv")
    assert sorted(files.material) == ["a_n", "a_n", "iau", "iau", "phrase", "phrase"]
    assert not list((root / "data" / "svd" / "nsp").rglob("*.egg"))
    assert not list((root / "data" / "svd" / "nsp").rglob("*i_n*"))
    for row in files.itertuples():
        rate, y = wavfile.read(root / row.wav)
        assert rate == 50000
        assert np.array_equal(y, expected[(str(row.session), row.material)])
        assert row.reader_problems != row.reader_problems or row.reader_problems == ""
    manifest = json.loads((root / "data" / "svd" / "manifest.json").read_text())
    assert manifest["zip_check"]["matches_record"]
    assert manifest["extraction"]["sessions"] == 2
    assert [m["member"] for m in manifest["extraction"]["meta_files"]] == ["data/overview.csv"]
    assert (root / "data" / "svd" / "meta" / "overview.csv").exists()


def test_extract_refuses_an_unverified_zip(fake_svd, monkeypatch):
    monkeypatch.setattr(svd_extract, "ZENODO", {**svd_extract.ZENODO, "md5": "0" * 32})
    with pytest.raises(SystemExit):
        svd_extract.verify()
    with pytest.raises(SystemExit):
        svd_extract.extract()
