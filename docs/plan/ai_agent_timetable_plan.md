# AIエージェントによるタイテデータ作成（β版）実装計画

## 概要

ワークフロー ①プロジェクト設定 / ②画像登録 / ③画像切り取り / ④読み取り を、複数画像とメッセージを入力として **AIエージェントが自動実行** する機能を β版として実装する。

- 新規ページ「**AIエージェントによるタイテデータ作成（β版）**」を追加する。
- ①でプロジェクトを作成 → 本ページへ遷移して画像とメッセージを投入 → エージェントが ②③④ 相当を自動実行する。
- 最終的に得る状態は、通常ワークフローで ④ を一通り終えた後と同等（＝**⑥を開ける状態**）:
  - プロジェクト設定 / 画像種別 / 種別ごとの画像切り出し / 各ステージ元画像 / 各ステージの時間軸（未設定もあり得る）/ ステージ名 / ステージカラー / タイテデータ（通常 or 特典会併記）/ グループ名_採用 / コラボ判定。

### 設計の前提（ユーザー確定事項）

| 項目 | 決定 |
|------|------|
| エージェントLLM基盤 | **OpenAI Agents SDK**（既存OCR資産 `gpt_ocr.py` / 既存 `OPENAI_API_KEY` を流用） |
| データモデル | **「画像全体読み取り＋別途切り出し」へ刷新**（後方互換を保ちつつ拡張） |
| 人手介入（HITL） | **自動進行＋低信頼時のみ停止**して人に確認を求める |
| 周辺改善の同梱 | **5分毎グリッド線方式（ライムライト式）** と **グループ名 再判定フロー** を本計画に含める（エージェント非依存でも有効化） |
| 既存機能・UI | **保持**（②③④の手動フロー、⑤⑥⑦⑧はそのまま） |

---

## PoC検証結果（2026-06-21 実施）

計画着手前に、**完成済みプロジェクト「202606_【新】アイドル甲子園 FESTIVAL 2026」をテストケース**として、オーケストレータ（Claude）＋ワーカー6並列の構成で ②③④ 相当をエージェント的に実行し、生成データを完成済み正解データと自動照合した。既存データは無変更、全作業は `data_tmp/agent_sandbox/` に隔離（再現スクリプト: `score_full.py` / `materialize_and_score.py`、結果: `accuracy_report.txt` / `accuracy_full.txt`）。

### テストケースの特性
- 2イベント × {ライブ3ステージ / 特典会12ブース / 終演後3ステージ}。
- `特典会/raw.png` と `終演後/raw.png` が **md5一致＝同一画像の二重登録**（要件の「特典会画像内に終演後特典会」ケースを内包）。
- 終演後特典会は1バンドに複数ホール×複数グループが `21:30~22:50` 等でまとめ描画され、正解では **独立 `終演後/` 種別 ＋ 3グループを同一 `出番ID`／`コラボグループID` のコラボ**として保持。

### 定量結果（全362出番）

| 指標 | 結果 |
|------|------|
| 時刻一致 | **361 / 362 = 99.7%** |
| 名称一致（厳密） | 343 / 362 = 94.8% |
| 終演後特典会（最難ケース） | event_1/2 とも構造・時刻・名称まで完全再現（コラボ化含む） |
| 余剰読取（crop境界バグ由来） | 2件 |

名称不一致19件の内訳: **純粋な誤読は約8件**、残りは ①正解データ側の誤りで読み取りの方が正しいケース（`背越し`→`宵越し`、`しゅじゅ`→`じゅじゅ` 等 複数）、②`グループ名_採用` 補正で吸収される正規化差（ダイアクリティクス／括弧／アポストロフィ）。**人間が最終確定したデータと同等以上の精度**だった。誤読は「画数の多い漢字」「狭い列の小さな英字」に集中。

### 実装に反映すべき知見

1. **crop境界の排他化（実バグを発見）**: event_2 では `特典会` bbox 下端（1645）が `終演後` バンド（top=1489〜）と重なり、ブース読取に終演後の行が混入して余剰2件が発生した。→ **「特典会crop」と「終演後crop」を排他的領域として切り出す**ロジックを必須要件とする。これはリスク#1（終演後の帰属）の具体的裏付けでもあり、**案B（独立 `終演後/` 種別）が正解データと整合**することを確認した。
2. **「画像全体読み取り＋ハール/ブース振り分け」モデルの妥当性確認**: 終演後を含め切り出し精度に依存せず再現できた。データモデル刷新方針を支持。
3. **列crop→拡大フォールバックの有効性**: 狭い列・小文字の救済に、ワーカーが自発的に PIL で再拡大→再読取して精度を回復した実績あり。実装の `crop → upscale → 再読取` ツール（`read_timetable_perstage` フォールバック）に対応づける。
4. **HITL停止トリガの精緻化**: 微差（ダイアクリティクス/括弧/アポストロフィ）では**停止不要**。停止対象は **「画数の多い漢字の判読困難」「終演後バンドの境界判定」「新規グループ名」** に絞ると費用対効果が高い。

---

## 現状整理（実装の足場）

エージェントが呼び出す業務ロジックは Streamlit 非依存の **サービス層** に既に揃っており、これをエージェントの「ツール」としてラップする方針が取れる。

### 関連レイヤと主な関数

| レイヤ | ファイル | 主な再利用対象 |
|--------|---------|---------------|
| プロジェクト | [src/backend_functions/project_repository.py](../../src/backend_functions/project_repository.py) | `create_project_data` / `apply_project_setting` / `register_timetable_image` / `get_image_entry_by_dir_name` / `get_event_type_list` |
| ワークフロー | [src/workflow.py](../../src/workflow.py) | `ProjectWorkflow` / `ImageWorkflow`（`detect_stage_lines`/`split_evenly`/`save_stage_images`/`save_time_axis`）/ `OcrWorkflow` |
| 画像処理 | [src/backend_functions/image_processing.py](../../src/backend_functions/image_processing.py) | `detect_stageline` / `split_image_evenly` / `get_image_eachstage_byocr` / `save_stage_images` / `detect_timeline_onlyonestage` |
| OCR | [src/backend_functions/ocr_service.py](../../src/backend_functions/ocr_service.py) | `run_ocr_all_stages` / `detect_stage_names` / `correct_idol_names_all` / `autodetect_collab_all` |
| OCR下層 | [src/backend_functions/gpt_ocr.py](../../src/backend_functions/gpt_ocr.py) | `getocr_fes_timetable_structured` ほか（`GPT_MODEL_NAME="gpt-5.4"`、`client=OpenAI()`） |
| 時間軸 | [src/backend_functions/time_axis.py](../../src/backend_functions/time_axis.py) | `TimeAxisConverter` / `build_time_pixel_config` |
| グループ名 | [src/backend_functions/idolname.py](../../src/backend_functions/idolname.py) | `get_name_by_levenshtein_and_vector(_with_hint)` |
| UI | [src/app.py](../../src/app.py) | `st.radio("処理フェーズ", ...)`（L3784付近）＋ ディスパッチ（L3920付近） |

### 現状データモデルの要点（[docs/data_structure.md](../data_structure.md)）

- `data/projects/{pj}/project_info.json` … `event_detail[].timetables[]`（種別ごと）に `format` / `stage_num` / `stage_list[]` / `time_pixel`（**種別共通**）。
- 種別ごとフォルダ（`ライブ/` `特典会/` `ライブ特典会/`）に `raw.png` / `raw_cropped.png` / `stage_N.png` / `stage_N.json`。
- `stage_N.json` は **ステージ単位で切り出した画像から** OCR する前提。`ステージID` はトップレベル、特典会併記は `特典会[].ステージID` を子として保持。
- ⑥は `stage_N.json` 群＋`project_info.json` を読む。**エージェントの最終出力はこの形に materialize する必要がある。**

---

## データモデル刷新（「画像全体読み取り＋別途切り出し」）

### 課題（要件より）

1. 1枚に **ライブと特典会が別領域** で描かれるケース（従来は2枚に分けて登録）。
2. **終演後特典会**：複数ブースにまたがり複数グループがまとめて描画され、通常特典会とタイテ描画が異なる。トリミングするとブース名や時間帯が欠落する。
3. 稀に **ライブ側に終演後特典会**、**ステージが縦にも並ぶ** ケース。
4. 種別共通の時間軸では、終演後特典会など **種別横断で時間軸が揃わない** ケースに対応できない。

### 方針

**タイテ読み取りは「画像全体」を使い、ステージ領域の切り出しは読み取りとは独立に行う。**
切り出しは OCR 精度向上のためではなく **人間の照合用** に行う（LLM進化で全体一括読み取りでも精度が出るため）。うまくいかない場合のみエージェントがステージ単位読み取りへフォールバックする。

### 変更点（後方互換を維持）

1. **per-stage 時間軸（オプション）**
   - `project_info.json` の `stage_list[i]` に任意の `time_pixel` を追加できるようにする。
   - `TimeAxisConverter.from_project_info(pij, event_no, img_type, stage_no=None)` を拡張し、`stage_no` 指定時は **stage単位 time_pixel を優先**、無ければ種別共通へフォールバック。既存呼び出し（stage_no省略）は完全互換。
   - 生成・可視化（`generate_timetable_picture` / `compute_stage_layout`）の converter 取得箇所を stage単位対応に差し替え。

2. **画像全体OCR → per-stage JSON への materialize**
   - 新規プロンプト＋structured schema を追加（`src/prompt_system/fes_timetable_wholeimage*.txt` / `src/gpt_output_format/timetable_format.py`）。
   - 出力は「ステージ配列」を持つ構造：`{ "stages": [ { "ステージ名", "stage_color", "特典会フラグ", "タイムテーブル": [...] }, ... ] }`。
   - **materializer**（新規 `agent/materialize.py` 等）が、この出力を **既存の `stage_N.json` 群＋`project_info.json.stage_list`** へ書き出す。これにより ⑥以降は無改修で動く。
   - 各セル（出演要素）には bbox を持たせ、materializer が後段の照合用crop生成に使う。

3. **要素単位の種別分類（通常特典会 / 終演後特典会の分離）**
   > 📌 分離キーは **「縦位置（時刻）」ではなく「ブロックの意味・構造」**。終演後の時間帯はホール単位でバラバラで（PoC実測: ホール1000=21:10〜 / ホール500=21:30〜 / ホール300=21:00〜）、通常特典会の最終枠（〜21:50）と**同じ縦位置に同居**するため、画像を横一線で割る方法は原理的に破綻する。
   - 全体読み取り時、各セルを以下の構造的特徴で **通常 / 終演後 に分類**する（時刻は補助信号に留める）:
     | | 通常特典会 | 終演後特典会 |
     |---|---|---|
     | セル形状 | 1ブース幅の縦長カラム、1グループ/セル | 複数ブース幅にまたがる横長セル、複数グループまとめ |
     | ラベル | ブース名(A,B,C…) | **「終演後特典会」「終演後物販」** の明示ラベル |
     | 帰属 | ブース単位ステージ | ホール単位ステージ（同時刻の複数グループは**コラボ化**） |
   - **各視覚要素は1回だけ読み取り**、分類に従って `特典会/` か `終演後/` へ振り分ける。これにより**同一グループの二重カウントが構造的に発生しない**（PoCで観測した「列cropへの終演後行混入」は、先に幾何でカラムを切ってから独立読みした旧方式が原因。本モデルでは消失）。
   - 同一カラム内に通常と終演後が縦積みされる場合も、ラベル/横長化の検知でセル単位に正しく割れる。縦並びステージ・ライブ側への終演後混在も同じロジックで吸収。確信が持てない分類は HITL 停止トリガへ。

4. **照合用 crop は「分類の後」に要素 bbox から生成**
   - `stage_N.png` は引き続き生成するが、**読み取りの入力ではなく照合用**。bbox は `stage_list[i].bbox` に保持。
   - 終演後ステージのcropは横長バンド片、通常ブースは縦長カラム。両者が画面上 y で重なってもよい——**データはすでに要素単位で重複排除済み**なので、人間の確認画像が一部重複表示されるだけで実害なし。
   - ⇒ 「crop境界の排他化」は *pixel領域の排他* ではなく ***要素の排他割当て*** と定義する。
   - 切り出しに失敗／不要なケースでも、`stage_N.json`（全体読み取り由来）は成立する。

### 終演後特典会の帰属（PoCで決着）

独立 `終演後/` 種別（ホール単位ステージ）として materialize する。アイドル甲子園の正解データ（`終演後/stage_N.json`、3グループを同一 `出番ID`／`コラボグループID` のコラボとして保持）と整合を確認済み。各終演後ステージは **per-stage `time_pixel`**（上記1）で自前の時間軸を持ち、ホール間で時間帯が揃わなくても可視化・照合が成立する。

### 後方互換の担保

- 既存プロジェクトは `time_pixel`(種別) のみを持つ → そのまま動作（stage_no フォールバック）。
- materialize 後の `stage_N.json` スキーマは **既存と同一**（`ステージ名`/`ステージID`/`タイムテーブル[]`/`特典会[]`）。⑥編集・Excel・Stella 連携は無改修。
- 旧フロー（ステージ単位切り出し→読み取り）は ④ 画面で従来どおり利用可能。

---

## エージェントアーキテクチャ（OpenAI Agents SDK）

### 全体像

```
[新規ページ]  ── 入力(画像複数＋メッセージ) ──▶  AgentRunner(別スレッド)
     ▲                                              │
     │  進捗/HITL質問 (queue)                        ▼
     └──────────────  Streamlit再描画  ◀──  Orchestrator Agent
                                                 │ tools
              ┌──────────────────────────────────┼───────────────────────────┐
              ▼                ▼                  ▼               ▼            ▼
        ①project_tools   ②classify_tools   ③crop_tools     ④ocr_tools   review_tool
        (create/setting)  (種別/日付分類)   (切出/時間軸)   (全体読取/補正) (HITL)
```

- **Orchestrator Agent**：1体のエージェントが計画を立て、ツールを順に呼ぶ。必要に応じてステージ単位読み取りへフォールバック。
- **ツール**＝サービス層関数の薄いラッパ（`agent/tools.py`）。Streamlit非依存の `ocr_service` / `image_processing` / `project_repository` を呼ぶ。
- **実行ハーネス**（`agent/runner.py`）：Streamlit のメインスレッドをブロックしないよう **ワーカースレッド** で `Runner.run_*` を実行。進捗・HITL要求は `queue.Queue` 経由でページへ。
- **HITL**：低信頼時にエージェントが `request_human_review` ツールを呼ぶ → ワーカースレッドが `threading.Event` で待機 → ページが質問を描画 → ユーザー回答を queue に積む → 待機解除して続行。

### ディレクトリ構成（新規）

```
src/agent/
├── __init__.py
├── orchestrator.py   # Agent定義（instructions / tools / model設定）
├── tools.py          # サービス層を function tool 化
├── runner.py         # ワーカースレッド実行・進捗queue・HITL制御
├── materialize.py    # 全体読取結果 → stage_N.json / project_info.json
├── schemas.py        # 入出力 pydantic 型（信頼度フィールド含む）
└── prompts/          # orchestrator instructions ほか
```

### 依存追加

- `openai-agents`（OpenAI Agents SDK）を `requirements.txt` に追加。`OPENAI_API_KEY` は既存 `.env` を流用。
- モデルは Vision 対応の OpenAI モデル（既存 `GPT_MODEL_NAME` 系列）をオーケストレータ／読み取りに使用。

### ツール一覧（案）

| ツール | ラップ対象 | 役割 |
|--------|-----------|------|
| `create_project_and_setting` | `ProjectWorkflow.create_project` / `apply_project_setting` | ①イベント数・ライブ名・日付からプロジェクト確定 |
| `classify_images` | （新規）画像分類 LLM | ②添付画像を 日付×種別 に分類、ライブ/特典会別領域・ライムライト式を判定 |
| `register_image` | `ProjectWorkflow.register_image` | ②分類結果に基づき raw.png 登録（必要なら同画像を複数種別へ） |
| `crop_stage_regions` | `detect_stageline` / `split_image_evenly` / `get_image_eachstage_byocr` / `save_stage_images` | ③ステージ領域切り出し（照合用） |
| `set_time_axis` | `ImageWorkflow.save_time_axis`（stage拡張版） | ③時間軸設定（種別 or stage単位、未設定可） |
| `read_timetable_wholeimage` | （新規）全体読み取り＋`materialize` | ④画像全体から全ステージのタイテを読み、stage_N.json へ展開 |
| `read_timetable_perstage` | `run_ocr_all_stages` | ④フォールバック：ステージ単位読み取り |
| `detect_stage_names_colors` | `detect_stage_names` | ④ステージ名・カラー（特典会除く）読み取り |
| `correct_group_names` | `correct_idol_names_all`＋再判定 | ④グループ名補正（候補出し→再判定、後述） |
| `detect_collab` | `autodetect_collab_all` | ④コラボ判定 |
| `request_human_review` | （新規） | HITL：低信頼箇所を人に確認 |

---

## ②画像分類の詳細（要件の難所）

`classify_images` ツールで以下を判定する。

- **日付×種別への分割**：複数画像を `event_N`（日付）と種別（ライブ/特典会/ライブ特典会）に割り当てる。
- **時間表記の有無**：各枠に「HH:MM〜HH:MM」がある＝通常形式 / ない＝**ライムライト式**（別フロー：5分毎グリッド線方式へ）。
- **1枚にライブ＋特典会が別領域**：同一画像を「ライブ部分」「特典会部分」として **2エントリに登録**（従来踏襲）。あるいは刷新方針に従い、全体読み取りで両領域を一度に扱い materialize 時に振り分け（Phase 0 の未決事項と連動）。
- **終演後特典会**：特典会画像内に「終演後特典会」表記を検出した場合、トリミングせず **全体読み取り＋ブース振り分け** で処理（複数ブース・複数グループまとめ描画に対応）。
- **イレギュラー**：ライブ側に終演後特典会／ステージ縦並び を検出したら `request_human_review` で確認。

---

## ④読み取りの詳細

1. **全体読み取り**（`read_timetable_wholeimage`）：raw 画像全体を Vision で読み、`{stages:[...]}` を取得 → `materialize` で `stage_N.json` 群へ。
   - 信頼度が低い／件数が不自然な種別は `read_timetable_perstage` へフォールバック。
2. **ステージ名・カラー**：`detect_stage_names`（カラーは特典会除く、既存仕様）。
3. **グループ名 再判定フロー**（独立改善・後述）：読み取り → マスタ補正で候補出し → **画像＋候補を見て採用/新規決定**。
4. **時間軸**：ライムライト式は **5分毎グリッド線方式**（独立改善・後述）で `stage_N_addtime.png` 相当を生成してから読み取り。通常形式は枠内時刻をそのまま採用。
5. **コラボ判定**：`autodetect_collab_all`（終演後特典会の複数グループまとめもここで表現）。

---

## UI／ページ設計

[src/app.py](../../src/app.py) に以下を追加（既存ナビは保持し、末尾に追加）。

1. `st.radio("処理フェーズ", [... , "AIエージェント(β)"])` に選択肢追加。
2. ディスパッチに `elif page == "AIエージェント(β)": render_agent_page()`。
3. `render_agent_page()`：
   - **入力**：`st.file_uploader(accept_multiple_files=True)` ＋ 追加メッセージ `st.text_area` ＋ プロジェクト設定（イベント数／ライブ名／日付）。
   - **実行**：「エージェント実行」ボタンで `runner.start(...)`（ワーカースレッド起動）。
   - **進捗**：queue を `st.session_state` 経由でポーリングし、ステップログ／中間成果（分類結果・切り出し画像・読み取りJSON）を逐次表示。`st.autorefresh` 相当で再描画。
   - **HITL**：`request_human_review` 待機中は質問＋候補を描画し、回答を queue へ返す。
   - **完了**：「⑥へ進む」導線。失敗時はステップ単位の再試行。

> Streamlit はスクリプト再実行モデルのため、エージェント状態は **`st.session_state` に保持したワーカーハンドル＋queue** で橋渡しする。長時間実行・HITL のための再描画制御は Phase 1 で確立する。

---

## 独立改善（エージェント非依存でも有効化）

### A. 5分毎グリッド線方式（ライムライト式）

- 現状の横線検出（`detect_timeline_onlyonestage`、Canny+Hough）は **線検出精度に大きく依存**。
- 改善：時間軸（`time_pixel`）から **5分刻みの目盛線と時刻表記を一律付与** した `stage_N_addtime.png` を生成する関数を新設。
- 既存の `detect_timeline_onlyonestage` を置換／併存させ、④のライムライト式読み取り（`mode="notime"`）の前処理に組み込む。エージェント・手動フロー双方から利用。

### B. グループ名 再判定フロー

- 現状：`get_name_by_levenshtein_and_vector` でマスタ最近傍を即採用。新規グループに弱い。
- 改善：**「読み取り → マスタ補正で候補(top-k)出し → 画像＋候補をLLMに渡し、採用 or 候補に無い新規を決定」** の3段フロー。
- `ocr_service` に候補列挙関数（top-k）＋再判定 LLM 呼び出しを追加。**④手動フローにもオプションとして組み込む**（エージェント専用にしない）。

---

## フェーズ分割

### Phase 0：データモデル刷新＋独立改善（エージェント前提づくり）
- [ ] `stage_list[i].time_pixel`（任意）対応、`TimeAxisConverter.from_project_info` に `stage_no` 追加（後方互換）。終演後ステージは自前の軸を持つ。
- [ ] 全体読み取りプロンプト＋structured schema 追加（各セルに bbox ＋ 種別分類フィールド: 通常/終演後）。
- [ ] **要素単位の種別分類ロジック**（ラベル＋セル構造で 通常/終演後 を判定、時刻は補助。**幾何の横一線カットは採用しない**）。
- [ ] `agent/materialize.py`：全体読取結果 → 分類に従い `特典会/` `終演後/`（独立種別・コラボ化）へ振り分け → `stage_N.json`／`project_info.json` 書き出し。同一要素を二重カウントしないこと。
- [ ] 照合用 crop を**分類後に要素 bbox から生成**（y重複を許容）。
- [ ] 5分毎グリッド線方式（独立改善A）。
- [ ] グループ名 再判定フロー（独立改善B）。

### Phase 1：エージェント基盤
- [ ] `openai-agents` 依存追加。
- [ ] `agent/tools.py`（サービス層ラッパ）、`agent/orchestrator.py`（Agent定義）。
- [ ] `agent/runner.py`（ワーカースレッド・進捗queue・HITL `Event`）。
- [ ] 新規ページ骨格（入力・進捗表示・完了導線）。

### Phase 2：①②③④ 自動化フロー結線
- [ ] ① プロジェクト作成・設定。
- [ ] ② 画像分類（日付×種別、ライブ/特典会別領域、ライムライト式、終演後特典会）。
- [ ] ③ ステージ領域切り出し（照合用）＋時間軸（種別/stage）。
- [ ] ④ 全体読み取り→materialize→ステージ名/カラー→グループ名再判定→コラボ判定。
- [ ] フォールバック（全体→ステージ単位）。

### Phase 3：HITL（低信頼時の停止）
- [ ] `request_human_review` ツール＋待機・回答機構。
- [ ] 低信頼トリガ定義（分類の曖昧さ／件数異常／新規グループ多発／イレギュラー検出）。

### Phase 4：仕上げ・テスト
- [ ] ⑥へ遷移して整合確認（既存スキーマ無改修で開けること）。
- [ ] 代表画像セットでの結合テスト、エラー時の部分再実行。

---

## テスト方針

- **単体**：`materialize`（全体読取JSON → stage_N.json の正しさ）、`TimeAxisConverter` の stage_no フォールバック、5分グリッド生成、グループ名 再判定（候補→採用/新規）。既存 `tests/backend_functions/` 構成に追加。
- **結合**：代表画像（通常／ライムライト式／特典会併記／終演後特典会／縦並び）でエージェント一気通しを実行し、**⑥が無改修で開けること** をゴール条件に検証。
- **HITL**：低信頼トリガで停止→回答→続行のラウンドトリップ。

---

## リスク・未決事項

| # | 項目 | 内容・対応 |
|---|------|-----------|
| 1 | 終演後特典会の分離・帰属 | **方針決着**: 帰属＝独立 `終演後/` 種別（コラボ化、正解データと整合）。分離キー＝**要素の意味・構造分類**（ラベル＋セル形状）であり、**時刻の横一線カットではない**（終演後の時間帯はホール単位でバラつき、通常特典会と縦位置が同居するため幾何分割は破綻）。各要素を1回だけ読み1ステージへ排他割当て＝二重カウントなし。終演後ステージは per-stage 時間軸で自前の軸を持つ。データモデル節「要素単位の種別分類」を参照。 |
| 2 | Streamlit×長時間実行×HITL | スクリプト再実行モデル下でのワーカースレッド・queue・再描画制御。Phase 1 で PoC を先行。 |
| 3 | 全体読み取り精度 | ステージ多数／高密度画像で取りこぼし。フォールバック（ステージ単位）と件数チェックで担保。 |
| 4 | OpenAI Agents SDK のHITL | SDK のツール承認／中断機構の採用可否。最小実装は自前 `Event` 待機で回避可能。 |
| 5 | コスト・レイテンシ | 全体読取＋再判定でトークン増。種別単位の並列・キャッシュで緩和。 |
| 6 | 既存UI非回帰 | ②③④手動フロー・⑤⑥⑦⑧へ影響しないこと（time_axis 署名拡張は後方互換で担保）。 |

---

## 成果物（このβで「⑥を開ける状態」に必要なもの）

- `project_info.json`：イベント設定・種別・`stage_list`（名前/カラー/任意time_pixel/bbox）。
- 種別フォルダ：`raw.png`／（照合用）`stage_N.png`／`stage_N.json`（タイテ：通常 or 特典会併記、グループ名_採用、コラボ）。
- ⑥・Excel・Stella 連携は **無改修** で利用可能。
