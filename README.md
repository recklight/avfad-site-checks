# Analysis plan and code: site checks for voice pathology detection on AVFAD

English | [繁體中文](README.zh-TW.md)

Archived on Zenodo: [doi:10.5281/zenodo.24553193](https://doi.org/10.5281/zenodo.24553193) (all versions; version 1.0 is [doi:10.5281/zenodo.24553195](https://doi.org/10.5281/zenodo.24553195)).

This is the analysis plan and the code behind the manuscript "Site Checks for Evaluating Voice Pathology Detection on Multi-Site Case-Control Databases: A Pre-Specified Study of AVFAD" (Fu-Kai Chuang). The plan was frozen on 4 November 2026, before any detector was trained.

The code runs the whole study: the data audit, the segmentation, the feature extraction, the four confirmatory tests and the exploratory analyses, and the two scripts that write every number and the two data figures of the manuscript. The group-level results are in `experiments/results/`, as JSON, with an overview in Chinese in `summary.md`.

- `analysis_plan.md` is the plan in prose (Chinese), `analysis_plan.yaml` the same plan in the form the scripts read.
- `experiments/` holds the analysis scripts and their tests, the two freeze records and the results.
- `s0_audit/` reads the three databases and checks their metadata and files.
- `s1_segments/` finds the phonation and non-phonation spans; `out/params.json` holds its frozen settings.
- `features/` extracts the handcrafted, HuBERT and ComParE features.
- `paper/` has `make_numbers.py` and `figures.py`, their tests, and `numbers.txt`, which gives the source of each number.
- `experiments/pip_freeze.txt`, `s0_pip_freeze.txt` and `opensmile_env_freeze.txt` list the packages of the analysis, audit and openSMILE environments.

`MANIFEST.sha256` has the SHA-256 of every other file.

## The plan and the freeze records

`experiments/freeze.json` and `experiments/freeze_svd.json` are the freeze records the runs checked against, unchanged. The first holds the SHA-256 of the plan, the inputs, the feature tables and the scripts, with the package versions. The second was made before the SVD tests ran, for the C3 and C4 scripts, and includes `freeze.json`.

A few notes unrelated to the analysis were taken out of the two plan files here, so their SHA-256 differ from the ones in `freeze.json`. I can send the originals on request. The scripts import the statistics package under its published name, `voice_severity_bench` (<https://github.com/recklight/voice-severity-bench>, commit `1760203`), and some of their comments are worded differently; the code is the code that ran. `freeze.json` names the package under its former name at commit `35de45d`, and apart from the name its code is the same as `1760203`.

So `experiments/freeze.py verify` does not pass on this record. The plan files and five of the frozen scripts no longer match their hashes, the per-speaker feature tables and the S0 and S1 outputs cannot be shared, and it expects the package installed from commit `35de45d`, which is not a commit of the published repository. To check a rerun, rebuild those files from the raw data and compare their SHA-256 with `freeze.json`.

Every analysis added after the freeze has a dated entry in a deviation log, and most entries give the reading written down before the run. The scripts, `summary.md` and `numbers.txt` cite it as `experiments/deviations.md`. It is not part of this record; I can send it on request.

## What is not here

There are no recordings, features or per-speaker outputs. AVFAD is released under CC BY-NC-ND 4.0, so only group-level results can be shared, and I apply the same rule to the Saarbruecken Voice Database (SVD) and VOICED. The databases come from their own sources, and the scripts read them from these folders:

- AVFAD, from the University of Aveiro (described in doi:10.5772/intechopen.69643): the downloaded zip files go to `data/avfad/raw/`.
- SVD, from Zenodo (doi:10.5281/zenodo.16874898): `data.zip` and `healthy.zip` go to `data/svd/`, together with the session index `voice_data.csv` from <https://stimmdb.coli.uni-saarland.de/data/voice_data.csv>.
- VOICED, from PhysioNet (doi:10.13026/C25Q2N): the record files go to `data/voiced/`.

## Rerunning

1. Put HuBERT base (`facebook/hubert-base-ls960`, revision `dba3bb02fda4248b6e082697eee756de8fe8aa8a`) in `models/facebook__hubert-base-ls960/`, with that revision in a file named `REVISION`.
2. Make the analysis environment with Python 3.10.11: `python -m pip install -r experiments/pip_freeze.txt --extra-index-url https://download.pytorch.org/whl/cpu`. The PyTorch build is the CPU one, and the first line of `pip_freeze.txt` points at the published statistics package. The ComParE features need openSMILE in a separate environment (`opensmile_env_freeze.txt`). The segmentation ran with kudio 3.9.1 and librosa 0.11.0 on Python 3.9.13, as `s1_segments/out/params.json` records.
3. Run the audit (`s0_audit/`), the segmentation (`s1_segments/segment.py`) and the feature extraction (`features/`).
4. Run `experiments/run_first_batch.py` and `experiments/run_svd.py`, which hold the four confirmatory tests, before the exploratory scripts: many of those read the draws and predictions of the first batch. Each script's docstring gives its command line.
5. `paper/make_numbers.py` writes `paper/numbers.txt` from the result files. A few counts and timings in it come from my working notes, which are not part of this record; `numbers.txt` names the source of every value, so you can see which ones.

## Tests

```bash
python -m pytest experiments/tests paper/tests features/tests s0_audit/tests s1_segments/tests
```

With voice-severity-bench installed and the data rebuilt, 481 tests pass and one fails on purpose: `test_frozen_plan_files_match_their_recorded_hashes`, because the plan files here are not the frozen originals. `s1_segments/tests/test_segment.py` needs kudio and runs in the segmentation environment. On this record alone, without the data, the HuBERT files and the per-speaker results, 422 pass.

## Licences

The code (`*.py`, including the tests) is under the MIT License, see `LICENSE`. The plan, the result summaries, `numbers.txt`, the package lists and this README are under CC BY 4.0, see `LICENSE-CC-BY-4.0.txt`. The databases keep their own licences, and nothing from them is redistributed here.

## Citation

Author: Fu-Kai Chuang, ORCID [0009-0009-2123-8230](https://orcid.org/0009-0009-2123-8230).

Please cite the manuscript. I will add its reference to this record when it is published.

Generative AI (Claude, Anthropic) assisted with the analysis code and the documents in this record. The author reviewed the results and takes responsibility for the content.
