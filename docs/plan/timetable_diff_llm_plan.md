# ⑤変更比較：LLMによるタイムテーブル変更案の提案・編集・反映

## Context（背景・目的）

現在の「⑤変更比較」（[src/app.py:1600](../../src/app.py#L1600) `render_comparison_section` / [src/frontend_functions/timetable_difference.py](../../src/frontend_functions/timetable_difference.py)）は、既存画像と新規画像のステージ別ピクセル差分を計算し、しきい値スライダーで「差分ありそうなステージ」を矩形＋ハッチングで可視化するところまで。

本タスクでは、差分ありと判定されたステージのうち**既にタイムテーブルデータ（stage_n.json）が存在するもの**を対象に、以下を実装する：
1. 既存/新規のステージ切り出し画像＋既存タイムテーブル＋既知グループ名一覧をOpenAI LLMに渡し、変更案（変更／削除／追加）を**構造化出力**で得る
2. 変更案を人間可読＋ID紐付きで表示
3. グループ名・時間・ブース、ならびにID再採番方針を人手で編集可能にする
4. 編集後の内容を実データ（stage_n.json、raw.png/切り出し画像、master_*.csv、可視化画像）に反映する

### 確定済みの方針（ユーザー回答）
- **LLM入力画像**: 各ステージの切り出し既存(`old_crop`)＋新規(`new_crop`)の2枚（差分判定で生成済み）
- **適用時の画像**: 新規画像でraw.pngと各ステージ切り出し画像も自動差し替え
- **ID再採番**: 出番ID・グループIDそれぞれ「保持／再採番(NULL化)」を選択可能にし、出番IDは手入力も可能。パターン別にデフォルトを決める（後述）
- **既知グループ名一覧**をLLM入力に付与し、表記揺れによる誤同定を防ぐ
- **構造化出力は既存実装を踏襲**：現行OCR本番経路（[ocr_service.py:52](../../src/backend_functions/ocr_service.py#L52) → [gpt_ocr.py:580](../../src/backend_functions/gpt_ocr.py#L580) `getocr_fes_*_structured` → [getocr_structured](../../src/backend_functions/gpt_ocr.py#L427)）が `response_format={"type":"json_schema", strict:True}` を使い実運用で動作している。L426の「うまく行かない」コメントは古い残骸で無視してよい。同じ流儀で変更案スキーマを定義する。
- **フロー（重要）**：**生成は個別ステージ可・反映は全体一括（原子的）**。LLMによる変更案「生成」は副作用が無いのでステージ個別ボタン＋全件ボタンの両方を提供。実データへの「反映」はraw.png差し替えがimg_type全体単位のため、全差分ステージをレビュー後に**一度だけ・まとめて**実行し状態の整合を保つ。
- **形式別の分岐（特典会併記型か否か）**：変更案のプロンプトとスキーマを、**特典会併記型（live_tokutenkai_heiki）と非併記型（ライブのみ／特典会のみ）で切り替える**。既存OCRが [response_format_live](../../src/backend_functions/gpt_ocr.py#L260) と [response_format_live_tokutenkai](../../src/backend_functions/gpt_ocr.py#L191)、用途別プロンプトファイルで分けているのと同じ思想。**共通化できる部分（基本指示・スキーマの共通プロパティ・操作の枠組み）は共通化し、特典会関連の差分のみ条件で付与する**。

---

## 全体構成（新規/変更ファイル）

| 種別 | パス | 役割 |
|---|---|---|
| 新規 | `src/prompt_system/fes_timetable_diff_propose_common.txt` | 変更案生成プロンプトの共通部（基本指示・出力規約） |
| 新規 | `src/prompt_system/fes_timetable_diff_propose_tokutenkai.txt` | 併記型のみ付与する特典会パート（共通部に連結） |
| 新規 | `src/backend_functions/timetable_diff_llm.py` | LLM入力整形・形式判定・提案生成・提案の実データ反映 |
| 変更 | [src/backend_functions/gpt_ocr.py](../../src/backend_functions/gpt_ocr.py) | `build_change_proposal_schema(with_tokutenkai)` 定義＋2枚画像を渡す `get_change_proposal` 追加 |
| 変更 | [src/workflow.py](../../src/workflow.py) `ImageWorkflow` | 提案生成（個別/全件）・全体一括反映のワークフロー追加 |
| 変更 | [src/app.py](../../src/app.py) `_render_diff_result` ほか | 個別/全件の生成ボタン・提案表示・2行編集UI・全体一括反映ボタン |

差分判定の純粋関数（timetable_difference）はそのまま再利用。LLM関連は新モジュールに分離し、既存のstructured-output／ID採番／画像再生成の関数を最大限再利用する。

---

## 1. バックエンド：LLM入力整形（`timetable_diff_llm.py`）

### `build_llm_timetable_input(stage_json: dict) -> dict`
`"タイムテーブル"` 配列のみを対象に軽量JSONへ整形：
- 各レコードに **`連番` = 配列インデックス** を付与
- グループ名の扱い：**`グループ名_採用` があればそれ（正式名）のみを入力し、`グループ名`（OCR生）は入力しない**。`グループ名_採用` が無い場合のみ `グループ名` を入力する（1レコードにつきグループ名は1つだけ渡す）
- 残すその他キー：`ライブステージ {from,to}`（あれば）、`特典会`（あれば）
- 特典会配列は各要素に `連番`（特典会内インデックス）を付与し、`from/to/ブース` のみ残す
- **除外するID系**：`出番ID` / `グループID` / `ステージID` / `対応出番ID` / `コラボステージID`
- フォーマット差（ライブのみ／特典会のみ／ライブ特典会併記）は「存在するキーだけ残す」方針で吸収

既知グループ名一覧 `known_groups` はプロンプト本文へ「既知グループ名一覧（表記の同定に使用）」として埋め込む。

---

## 2. バックエンド：OpenAI呼び出し（`gpt_ocr.py` に追加）

既存の structured-output 経路（[getocr_structured](../../src/backend_functions/gpt_ocr.py#L427)、[response_format_live_tokutenkai](../../src/backend_functions/gpt_ocr.py#L191)）を**そのまま踏襲**。違いは「画像が2枚・メモリ上のPIL」という点のみ。

- `encode_pil_image(pil_image) -> str`：PIL画像をPNG→base64化（既存 [encode_image](../../src/backend_functions/gpt_ocr.py#L334) のPIL版）
- `get_change_proposal(old_crop, new_crop, prompt_user, prompt_system, json_format) -> dict`：
  - `client.chat.completions.create(model=GPT_MODEL_NAME, response_format=json_format, ...)`（既存と同一）
  - `content` に「【既存】」ラベルtext＋old_crop画像、「【新規】」ラベルtext＋new_crop画像、最後に既存タイテJSON＋既知グループ名のtext
  - 返値は既存同様 `json.loads(response.choices[0].message.content)`
- `build_change_proposal_schema(with_tokutenkai: bool) -> dict`：既存スキーマと同じ `{"type":"json_schema","json_schema":{"name":..., "strict":True, "schema":{...}}}` 形式を返すビルダー。**共通部を1か所で定義し、`with_tokutenkai=True` の時だけ特典会関連プロパティ／種別を追加**する（live_tokutenkai専用とlive専用の2スキーマをコピペで持たない）。

### 変更案スキーマ（共通部＋特典会差分）
`操作` 配列。各要素の**共通部**：
- `種別`: `"変更"` | `"削除"` | `"追加"`
- `対象連番`: int（変更・削除時のライブレコードindex）
- 変更：`グループ名`（新・正式名／変更なしは空）、`ライブ時間{from,to}`
- 削除：`削除種別`（非併記では実質 `"全体"` のみ）
- 追加：`グループ名`、`ライブ時間{from,to}`
- `理由`: str（人手レビュー用の説明・表示専用）

**`with_tokutenkai=True`（併記型）でのみ追加**：
- 変更：`特典会変更`: [{`対象`: 連番 or "追加", `ブース`, `from`, `to`}]
- 削除：`削除種別` に `"特典会のみ"` を許可、`特典会連番`
- 追加：`特典会`: [{`ブース`,`from`,`to`}]

### プロンプト分岐
- 共通部 `fes_timetable_diff_propose_common.txt` を常に読み込み
- 併記型のときのみ `fes_timetable_diff_propose_tokutenkai.txt` を末尾連結（OCR側 [getocr_fes_timetable_notime_structured](../../src/backend_functions/gpt_ocr.py#L586) が live/tokutenkai でファイルを出し分けるのと同じ発想だが、本機能は共通部＋差分の連結で重複を最小化）

> strict:True 制約のため、未使用フィールドは空文字/空配列を返させる設計（required列挙＋additionalProperties:False）に揃える。

---

## 3. ワークフロー（`ImageWorkflow` に追加）

既存 `output_difference_image`（[workflow.py:403](../../src/workflow.py#L403)）の隣に追加。

### `propose_changes_for_stage(state, event_name, img_type, stage_no, diff_result) -> WorkflowResult`（生成・個別）
1. `stage_n.json` 存在チェック（`{pj_path}/{event_name}/{img_type}/stage_{stage_no}.json`）。**無ければ対象外**（success=False＋スキップ理由）
2. **形式判定** `with_tokutenkai`：`stage_list[].kind == "live_tokutenkai_heiki"`（[image_processing.py:462](../../src/backend_functions/image_processing.py#L462) 付近で付与）を第一に、無ければレコードに `特典会` キーが存在するかで判定。これでプロンプト・スキーマを出し分ける
3. `diff_result["stages"]` の該当 `stage_no` から `old_crop`/`new_crop` を取得
4. 既知グループ名 `known_groups` を解決：
   - [output_builder.load_existing_masters](../../src/backend_functions/output_builder.py#L124) で `master_idolname.csv` を読み `グループ名_採用` 一覧を取得
   - 無ければ当該イベント配下の全 `stage_*.json` の `グループ名_採用` 和集合をフォールバック
5. `build_llm_timetable_input` → `gpt_ocr.get_change_proposal`（`with_tokutenkai` で出し分けたプロンプト・スキーマを渡す）を呼び、`data` に「提案・整形済み既存タイテ・`with_tokutenkai`・連番→現ID対応表（出番ID/グループID/現名/現時間）」を返す

### `propose_changes_all_diff_stages(state, event_name, img_type, diff_stages, diff_result)`（生成・全件）
- stage_n.json有りの差分ステージをループ集約。既存OCRと同じ `ThreadPoolExecutor`（[ocr_service.py:112](../../src/backend_functions/ocr_service.py#L112)）で並列化

### `apply_all_change_proposals(state, event_name, img_type, new_image, edited_proposals_by_stage) -> WorkflowResult`（反映・全体一括）
全差分ステージの編集済み提案を **1トランザクションで** 反映（詳細は §5）。raw.png差し替えはここで一度だけ実行。

---

## 4. フロントエンド（`app.py`）

`_render_diff_result`（[src/app.py:1643](../../src/app.py#L1643)）の「差分のありそうなステージ」ループを拡張。

- **対象判定**：差分ステージのうち `stage_{n}.json` が存在するものだけ提案対象。無いものは「タイテ未作成のため対象外」とキャプション表示
- **生成ボタン（副作用なし）**：
  - 各ステージ見出し下に「このステージの変更案を生成」
  - 一覧上部に「差分ありの全ステージに変更案を生成」
- 提案は `st.session_state["_change_proposals"][stage_no]` に保持
- **反映は全体一括のみ**：全ステージのレビュー後、画面末尾に「全ステージの変更を反映」ボタン1つ（個別反映ボタンは設けない）。`uploaded_image_updated` が必須

### 提案の表示・編集UI（操作ごと・1レコード2行構成）
ID紐付け：提案の `連番` を現 `stage_n.json` レコードへ写像し、`出番ID`/`グループID`/現グループ名/現時間を解決して表示。
- 上段（読取専用）：元の `出番ID` / `グループID` / グループ名 / 時間（**併記型のみ**特典会のブース・時間も表示）
- 下段（編集欄）：グループ名・時間を編集（**併記型のみ**特典会のブース・時間の編集行を追加）（`st.text_input`/`st.data_editor`）＋ID方針コントロール
  - `出番ID`：`保持 / 再採番(NULL) / 手入力`（手入力時は数値入力欄）
  - `グループID`：`保持 / 再採番(NULL)`。**新グループ名がマスタに存在すれば該当IDを事前表示**（「既存マスタ: ID=NN」）し保持の既定値に充当

**ID方針デフォルト（パターン別）**：
| パターン | 出番ID 既定 | グループID 既定 |
|---|---|---|
| 同名・時刻のみ前後 | 保持 | 保持 |
| 同名・時刻変更だが**コラボ**（同一出番IDが複数）で部分変更 | 再採番(NULL) | 保持 |
| グループ名が変更 | 再採番(NULL) | 再採番(NULL)。新名がマスタ一致なら該当ID保持を提示 |
| 追加 | NULL | NULL（マスタ一致時は該当ID提示） |

- **グループ名変更ありでグループID保持を選んだ場合**：フラグを立て、反映時に `master_idolname.csv` の当該IDの `グループ名_採用` も更新（§5）

既存の `st.data_editor`＋`column_config` パターン（[src/app.py:2068](../../src/app.py#L2068)〜）を踏襲。

---

## 5. 実データへの反映（`apply_all_change_proposals` ＝全体一括・原子的）

実行順（img_type単位で1回）：

1. **新規画像のサイズ整合**：[replace_stage_images_from_new_raw](../../src/backend_functions/image_processing.py#L492) は新旧サイズ一致を要求（[image_processing.py:519](../../src/backend_functions/image_processing.py#L519)）。`analyze_difference_by_stage` 同様、保存前に**新規画像を既存raw.pngサイズへLANCZOSリサイズ**しbbox整合を確保
2. **各stage_n.jsonのJSON編集**（差分ありステージそれぞれ。`"タイムテーブル"` 配列を連番ベースで更新。削除でindexがずれないよう後方から処理 or 新配列構築）：
   - **変更**：`ライブステージ.from/to` 更新、`特典会[].from/to/ブース` 更新／特典会追加。`出番ID`/`グループID` はUI選択（保持＝現値／再採番＝`None`／手入力＝指定値）。併記で出番ID再採番時は子 `特典会[].出番ID`/`ステージID` も `None`
   - **削除**：`全体`＝レコード除去、`特典会のみ`＝該当 `特典会[連番]` 要素のみ除去
   - **追加**：`ライブステージ`＋（併記時）`特典会` を持つレコードを追加。ID系は `None`（採番は⑥）
   - **グループ名／採用名の付与（OCRパイプライン踏襲・ハイブリッド）**：diff LLM出力は **`グループ名`(raw)** に格納（名称変更時もrawを新規画像の読みで更新）。`グループ名_採用` は——マスタ完全一致なら採用名＋該当グループID／グループID保持の改名なら採用名＝新名（マスタも更新）／それ以外（新規・再採番）は**一旦空**にし、保存直後に `ocr_service.fill_empty_adopted_names`（確定リスト／チケット出演者／idolname埋め込み）で**補正生成**する
   - 保存は既存パターン（[ocr_service.py:76](../../src/backend_functions/ocr_service.py#L76) `json.dump(..., indent=4, ensure_ascii=False)`）に合わせる
3. **マスタ更新**：グループ名変更かつグループID保持のケースで `master_idolname.csv` の該当IDの `グループ名_採用` を更新（必要なら [_propagate_idolname_master_to_json](../../src/backend_functions/output_editor.py) で全json伝播）
4. **画像差し替え（raw.png/切り出し）**：[replace_stage_images_from_new_raw](../../src/workflow.py#L361) を**ここで一度だけ**実行（img_type全体のraw.png＋全ステージ切り出し＋raw_cropped）
5. **可視化画像の再生成**：当該ステージの `generate_timetable_picture`（ocr_service）＋イベント集約 [event_timetable_picture.regenerate_all_event_images](../../src/backend_functions/event_timetable_picture.py)。⑥のID採番はユーザーが⑥を開いた時 [OutputWorkflow.determine_id_master](../../src/workflow.py) でNULLのIDが採番される
6. `repo.update_timestamp` でプロジェクト更新時刻反映

> コラボ判定（UI既定値用）：同一 `出番ID` が複数レコードに存在するか（[timetabledata.py:617](../../src/backend_functions/timetabledata.py#L617) `autodetect_collab_groups` / [timetabledata.py:539](../../src/backend_functions/timetabledata.py#L539) `consolidate_collab_entries` を参考）で判定。

---

## 実装順序

0. 本計画書を `docs/plan/timetable_diff_llm_plan.md` に保存
1. プロンプト（`fes_timetable_diff_propose_common.txt`＋`_tokutenkai.txt`）＋ `build_change_proposal_schema(with_tokutenkai)`＋`get_change_proposal`/`encode_pil_image`（gpt_ocr.py）
2. `timetable_diff_llm.py`：`build_llm_timetable_input`・形式判定 → `get_change_proposal` 連携、併記/非併記両方で提案生成の単体動作確認
3. `ImageWorkflow.propose_changes_for_stage` / `_all_diff_stages`（known_groups解決含む）
4. `app.py`：生成ボタン（個別＋全件）＋提案表示・2行編集UI＋ID方針コントロール
5. `apply_all_change_proposals`（サイズ整合→各JSON編集→マスタ更新→raw一括差替え→可視化再生成）＋「全ステージの変更を反映」ボタン配線
6. テスト追加（tests/backend_functions, tests/frontend_functions）

---

## 検証（Verification）

- **単体**（`tests/backend_functions/`、API呼び出しはモック）：
  - `build_llm_timetable_input`：3フォーマット（ライブのみ／特典会のみ／併記）でID系除外・連番付与・グループ名_採用優先を検証
  - 反映ロジック：変更／削除（全体・特典会のみ）／追加、各ID方針（保持／NULL／手入力／コラボ部分変更）で出力JSONを検証（既存 [tests/fixtures](../../tests/fixtures) のstage json流用）
- **結合（手動）**：`.venv` でStreamlit起動（`streamlit run src/app.py`）→ ⑤で実在プロジェクトの新規画像をアップ→差分判定→個別/全件で「変更案を生成」→提案表示・編集→「全ステージの変更を反映」→ stage_n.json／raw.png／切り出し／可視化画像の更新を確認→⑥を開きNULLのIDが採番されることを確認
- OpenAI APIキーは既存 `.env` の `OPENAI_API_KEY` を使用（[gpt_ocr.py:9](../../src/backend_functions/gpt_ocr.py#L9)）
