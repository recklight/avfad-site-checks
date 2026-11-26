# 分析計畫與程式：AVFAD 上嗓音病理偵測的地點檢查

[English](README.md) | 繁體中文

Zenodo 存檔：[doi:10.5281/zenodo.24553193](https://doi.org/10.5281/zenodo.24553193)（所有版本；1.0 版是 [doi:10.5281/zenodo.24553195](https://doi.org/10.5281/zenodo.24553195)）。

這裡是論文 "Site Checks for Evaluating Voice Pathology Detection on Multi-Site Case-Control Databases: A Pre-Specified Study of AVFAD"（Fu-Kai Chuang）背後的分析計畫和程式。計畫在 2026 年 11 月 4 日凍結，當時還沒有訓練任何偵測器。

程式涵蓋整個研究：資料稽核、切段、特徵擷取、四個確認性檢定和探索性分析，以及產生論文每個數字和兩張資料圖的兩支腳本。群組層級的結果放在 `experiments/results/`，格式是 JSON，總覽是中文的 `summary.md`。

- `analysis_plan.md` 是文字版的計畫（中文），`analysis_plan.yaml` 是同一份計畫、腳本讀的格式。
- `experiments/` 放分析腳本和測試、兩份凍結紀錄，以及結果。
- `s0_audit/` 讀三個資料庫，檢查詮釋資料和檔案。
- `s1_segments/` 找出發聲段和非發聲段；`out/params.json` 存凍結時的設定。
- `features/` 擷取手工特徵、HuBERT 和 ComParE 特徵。
- `paper/` 有 `make_numbers.py`、`figures.py` 和它們的測試，以及標出每個數字來源的 `numbers.txt`。
- `experiments/pip_freeze.txt`、`s0_pip_freeze.txt` 和 `opensmile_env_freeze.txt` 是分析、稽核和 openSMILE 三個環境的套件清單。

`MANIFEST.sha256` 列出其他每個檔的 SHA-256。

## 計畫和凍結紀錄

`experiments/freeze.json` 和 `experiments/freeze_svd.json` 是執行時拿來比對的凍結紀錄，沒有改動。第一份記了計畫、輸入、特徵表和腳本的 SHA-256，以及套件版本。第二份是跑 SVD 檢定之前為 C3、C4 的腳本做的，也包含 `freeze.json`。

這裡的兩個計畫檔拿掉了幾句和分析無關的備註，所以 SHA-256 和 `freeze.json` 記的不同，原檔可以向我索取。腳本用公開名稱 `voice_severity_bench`（<https://github.com/recklight/voice-severity-bench>，commit `1760203`）匯入統計套件，有些註解的措辭也不一樣；程式本身就是當時跑的那份。`freeze.json` 記的是套件以前的名稱和 commit `35de45d`，除了名稱，程式和 `1760203` 相同。

所以 `experiments/freeze.py verify` 在這裡不會通過：計畫檔和其中五支凍結腳本的雜湊對不上，逐人特徵表和 S0、S1 的輸出不能公開，而且它要從 commit `35de45d` 安裝套件，這不是公開 repo 裡的 commit。要檢查重跑的結果，請從原始資料重建那些檔案，再和 `freeze.json` 的 SHA-256 比對。

凍結之後加的每項分析，在偏離紀錄裡都有一筆附日期的條目，大部分條目寫了跑之前定下的判讀。腳本、`summary.md` 和 `numbers.txt` 引用它時寫作 `experiments/deviations.md`。它不在這裡，需要的話可以向我索取。

## 這裡沒有的東西

這裡沒有錄音、特徵或逐人輸出。AVFAD 以 CC BY-NC-ND 4.0 發布，只能分享群組層級的結果，Saarbruecken Voice Database（SVD）和 VOICED 我也用同樣的標準。資料庫請從各自的來源取得，腳本從下面這些資料夾讀取：

- AVFAD：向 University of Aveiro 取得（說明見 doi:10.5772/intechopen.69643），下載的 zip 檔放到 `data/avfad/raw/`。
- SVD：從 Zenodo 取得（doi:10.5281/zenodo.16874898），`data.zip` 和 `healthy.zip` 放到 `data/svd/`，連同場次索引 `voice_data.csv`（<https://stimmdb.coli.uni-saarland.de/data/voice_data.csv>）。
- VOICED：從 PhysioNet 取得（doi:10.13026/C25Q2N），紀錄檔放到 `data/voiced/`。

## 重跑

1. 把 HuBERT base（`facebook/hubert-base-ls960`，revision `dba3bb02fda4248b6e082697eee756de8fe8aa8a`）放到 `models/facebook__hubert-base-ls960/`，再把這個 revision 寫進一個叫 `REVISION` 的檔案。
2. 用 Python 3.10.11 建分析環境：`python -m pip install -r experiments/pip_freeze.txt --extra-index-url https://download.pytorch.org/whl/cpu`。PyTorch 是 CPU 版，`pip_freeze.txt` 第一行指向公開的統計套件。ComParE 特徵要在另一個環境裡用 openSMILE（`opensmile_env_freeze.txt`）。切段是用 kudio 3.9.1、librosa 0.11.0 在 Python 3.9.13 上跑的，`s1_segments/out/params.json` 有記錄。
3. 跑稽核（`s0_audit/`）、切段（`s1_segments/segment.py`）和特徵擷取（`features/`）。
4. 先跑 `experiments/run_first_batch.py` 和 `experiments/run_svd.py`，四個確認性檢定都在這兩支裡，再跑探索性腳本：很多探索性腳本會讀第一批的抽樣和預測。每支腳本的 docstring 寫了它的命令列用法。
5. `paper/make_numbers.py` 從結果檔產生 `paper/numbers.txt`。裡面有幾個計數和時間來自我的工作筆記，筆記不在這裡；`numbers.txt` 標了每個數字的來源，看得出是哪幾個。

## 測試

```bash
python -m pytest experiments/tests paper/tests features/tests s0_audit/tests s1_segments/tests
```

裝好 voice-severity-bench、重建資料之後，481 個測試通過，1 個照預期失敗：`test_frozen_plan_files_match_their_recorded_hashes`，因為這裡的計畫檔不是凍結時的原檔。`s1_segments/tests/test_segment.py` 需要 kudio，要在切段的環境裡跑。只有這裡的檔案、沒有資料、HuBERT 檔和逐人結果時，422 個通過。

## 授權

程式（`*.py`，含測試）採 MIT License，見 `LICENSE`。計畫、結果摘要、`numbers.txt`、套件清單和兩份 README 採 CC BY 4.0，見 `LICENSE-CC-BY-4.0.txt`。資料庫維持各自的授權，這裡沒有轉發任何資料庫的內容。

## 引用

作者：Fu-Kai Chuang，ORCID [0009-0009-2123-8230](https://orcid.org/0009-0009-2123-8230)。

請引用論文，論文發表後我會把出處補上。

本紀錄的分析程式和文件使用生成式 AI（Anthropic 的 Claude）協助完成；作者審閱了結果，並對內容負責。
