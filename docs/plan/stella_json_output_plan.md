# Stella JSON出力対応 実装計画

## 概要

本アプリケーションで作成したタイムテーブルデータを、Stellaアプリ向けのJSON形式（`live{id}.json` / `liveList.json`）で出力し、GitHubリポジトリ（https://github.com/ys0512/timetableproj）へpush/PRを送るまでの一連の機能を実装する。

---

## 現状と目標の差分サマリ

> 📌 **2026-06 時点アップデート**: 本計画書策定以降に Phase 1 / Phase 2 / Phase 5 の大半が実装済となった。各 Phase 冒頭に最新の実装状況を記載。Phase 3 は部分実装、Phase 4 / 6 / 7-1 / 7-2 ⑥-D が残作業。

### 既にできていること
- ライブ + 特典会 + 縁日を横断統合したステージマスタ・アーティストマスタ・出番データの構築（⑥）
- Excel 出力
- ステージID / グループID / 出番ID の採番
- **コラボ公演対応**（Phase 1 完了）: `コラボグループID` / `コラボタイトル` の管理、④の自動検出 UI、⑥のグループ化処理 ([timetabledata.py:116-446](../../src/backend_functions/timetabledata.py)、[output_builder.py:425-713](../../src/backend_functions/output_builder.py)、[app.py:1337](../../src/app.py#L1337) ほか)
- **stageList 拡張カラム**（Phase 2 完了）: `master_stage.csv` の `ステージ名_短縮` / `カラー名` / `表示順` 追加、`color_preset.json` 整備、`make_stage_color_resolver()` 実装、`create_timetable_image()` の文字色対応、⑥のカラー編集 UI ([event_timetable_picture.py:120-154](../../src/backend_functions/event_timetable_picture.py)、[timetablepicture.py:209](../../src/frontend_functions/timetablepicture.py)、[app.py:2215-2326](../../src/app.py#L2215-L2326))
- **ステージカラー LLM 推定**: `gpt_ocr.getocr_fes_stagelist_with_color_structured` + `ocr_service.set_stage_name` で読み取り時に `stage_color` を自動付与（本計画書策定後に追加）
- **Stella JSON 変換関数**（Phase 5 完了）: `stella_export.py` (242行) に `build_stella_json` / `write_stella_json` / `update_live_list` / `increment_versions_on_push` 実装済 ([stella_export.py](../../src/backend_functions/stella_export.py))
- **⑥-A / ⑥-C UI**（Phase 7-2 部分完了）: `_render_stella_metadata_form()` (openTime/closeTime/notification) と `_render_stella_export_form()` (JSON 生成・ダウンロード) 実装済 ([app.py:2932-3025](../../src/app.py#L2932-L3025))

### 不足していること

| # | 項目 | 内容 |
|---|------|------|
| 3a | メタデータ拡張 | `_default_stella_metadata` に `date` / `dow` 追加、`stella_project_meta` (liveName/genre/release/pref) ブロック新設 |
| 4 | liveList.json 管理 | ライブ一覧（公開状態・都道府県・日付等）の管理、採番ロジック、bundleId 算出 |
| 6 | GitHub 連携 | リポ取得 → ID採番 → push/PR、`.env` 認証、Push 失敗ロールバック |
| 7-1 | ①画面 Stella 入力 + 採番ボタン | プロジェクト/イベント基本情報入力、Reserve-First 採番 UI |
| 7-2 ⑥-D | GitHub 連携 UI | Push (PR) / Push (直接) ボタン、接続状態表示 |

---

## Phase 1: コラボ公演対応

> 📌 **実装状況: 完了**（2026-06 時点）
> - `コラボグループID` / `コラボタイトル` の json_to_df / df_to_json: [timetabledata.py:116-353, 376-446](../../src/backend_functions/timetabledata.py)
> - ④の自動検出 UI: [app.py:1337](../../src/app.py#L1337) (一括検出チェックボックス) / [:1598-1604](../../src/app.py#L1598-L1604) (ステージ単位) / [:1627-1632](../../src/app.py#L1627-L1632) (全ステージ一括)
> - ⑥のコラボグループ → 同一出番ID 処理 / `コラボタイトル` 正規化: [output_builder.py:425-713](../../src/backend_functions/output_builder.py)、`_LIVE_OUTPUT_COLUMNS` ([:183-187](../../src/backend_functions/output_builder.py#L183-L187))
> - テスト: [tests/.../test_autodetect_collab_batch.py](../../tests/backend_functions/test_autodetect_collab_batch.py)
>
> 以下の設計記述は実装時の参照用として残す。

### 設計方針

コラボ公演は **コラボグループID** によって管理する。「コラボフラグ + 並び順による暗黙のグルーピング」は採用しない（並び順への依存を排除するため）。

- **ID確定前 (④段階)**: `コラボグループID` 列で同一グループを表す
- **ID確定後 (⑥段階)**: 同一の `出番ID` を持つ複数行が同一コラボ出番を表す（`出番ID` がコラボグループIDの役割を引き継ぐ）

### 1-1. ④読み取り後の表にカラム追加

OCR結果を表示・編集する④のDataFrameに以下のカラムを追加する:

| カラム名 | 型 | 説明 |
|---------|---|------|
| `コラボグループID` | int (nullable) | 同じ値を持つ行が1つのコラボ出番を構成。単独出番ならNull |
| `コラボタイトル` | str | コラボ公演の表示名。グループの全行で同値を保持（編集時の整合維持のため）。空ならStella側でアーティスト名連結表示 |

#### 動作イメージ

```
グループ名     | from  | to    | コラボグループID | コラボタイトル
---------------|-------|-------|-----------------|---------------
KOURiN         | 20:10 | 21:10 | 1               | 星のカケラ
SOMOSOMO       | 20:10 | 21:10 | 1               | 星のカケラ
虹色の飛行少女  | 20:10 | 21:10 | 1               | 星のカケラ
シングル出演者  | 21:30 | 22:00 | Null            |
```

上記の場合、`コラボグループID = 1` を持つ3行が1つのコラボ出番となる。

### 1-2. グルーピング自動検出ボタン

④のUIに「コラボ出番を自動検出」ボタンを配置する。

**ロジック**:
- **対象**: `コラボグループID` が **NULL（または列自体が存在しない）行のみ**。既にIDが入っている行（手動設定 / 前回検出済）は触らない。これにより複数回押しても冪等
- **採番スコープ**: **同一ステージ単位** でグルーピングする（ステージファイルごとに採番）。ステージをまたいだコラボはサポートしない
- 各ステージ内で、`コラボグループID = NULL` の行のみを抽出して `startTime` でグルーピング
- 同じ `startTime` の行が2件以上ある場合、それらの行に同一の `コラボグループID` を採番（**そのステージ内の既存IDの最大値+1から**）
- 単独行は `コラボグループID = Null` のまま
- ユーザーが手動でID追加/解除も可能

### 1-3. ⑥出力時のコラボ処理

⑥のDataFrame構築時、**同一ステージ内のコラボグループ単位** で **同一の出番IDを採番** する。

- 出番ID生成も **ステージ単位でグルーピング** する（コラボグループIDの採番方針と一貫）
- 同一 `コラボグループID` を持つ④の行群 → 同じ `出番ID` を持つ⑥の複数行として出力
- 単独出番は通常通り1行1出番IDで出力

⑥のDataFrame構造（追加カラム）:

| カラム名 | 型 | 説明 |
|---------|---|------|
| `コラボタイトル` | str | コラボ出番の表示名。同じ出番IDの全行で同値。非コラボはNull |

#### 値の正規化規則

④のDataFrame編集UIでは、同じ `コラボグループID` を持つ各行に異なる `コラボタイトル` / `from` / `to` が入力されてしまう可能性がある。⑥への変換時に以下の規則で正規化する:

- **採用ルール**: 同じグループの行の配列順に走査し、**最初に登場する `NULL` / 空文字以外の値** を採用する
- 対象列: `コラボタイトル`、`from`、`to`、その他コラボ内で共通であるべき列
- 全行が空/NULLなら、当該キーは空のまま（コラボタイトルなら Stella側でアーティスト名連結表示にフォールバック）

> 注: `コラボアーティストID` 列は不要（同一出番IDの行をgroupbyすればアーティストIDリストを構築できるため）。Stella JSON生成時(`build_stella_json`)に動的に組み立てる。

#### プレビュー表示時のタイトルフォールバック

⑥-Cやイベント画像など、本アプリ上のプレビューでコラボ出番を表示する際、`コラボタイトル` が空なら **Stellaクライアントと同じ規則でアーティスト名連結文字列を表示する** （例: `KOURiN・SOMOSOMO・虹色の飛行少女`）。Stella JSON本体には連結文字列は書き出さず、`title` フィールド自体を省略する（クライアント側の責務に委ねる）。

### 1-4. ④への反映タイミング

`コラボグループID` / `コラボタイトル` 列はOCR読み取り直後から④の各stage JSONに保存する（IDマスタ確定前から書ける）:

```json
{
    "グループ名": "KOURiN",
    "グループ名_採用": "KOURiN",
    "ライブステージ": { "from": "20:10", "to": "21:10" },
    "備考": "",
    "コラボグループID": 1,
    "コラボタイトル": "星のカケラ"
}
```

IDマスタ確定後は、⑥側で `コラボグループID → 出番ID` の対応関係を生成し、後続処理では出番IDのみを参照する（コラボグループIDは④側の編集用識別子として残る）。

#### 既存 stage_X.json の後方互換

既存ファイルには `コラボグループID` / `コラボタイトル` キーが無い。`timetabledata.json_to_df` 側で **欠損キーは `NULL` として扱う**（DataFrame上は `NaN` / `None`）。ユーザーが手動でコラボ設定するか、自動検出ボタンを押した時点で値が入る。書き戻し時 (`df_to_json`) はキーを必ず出力する（NULL の場合は `null` / `""` で出力）。

---

## Phase 2: stageList拡張フィールド

> 📌 **実装状況: 完了**（2026-06 時点）
> - `master_stage.csv` に `ステージ名_短縮` / `カラー名` / `表示順` 全て存在
> - `data/master/color_preset.json` 整備済（27色、`{name: [bg, fg]}` 形式）
> - `make_stage_color_resolver()`: [event_timetable_picture.py:120-154](../../src/backend_functions/event_timetable_picture.py#L120-L154)（計画書では「予約状態」と書かれていたが実装済）
> - `_default_color_resolver()`: [event_timetable_picture.py:96-98](../../src/backend_functions/event_timetable_picture.py#L96-L98)
> - `create_timetable_image(text_color_in_box: str = "black")`: [timetablepicture.py:209](../../src/frontend_functions/timetablepicture.py#L209)
> - ⑥のカラー編集 UI `_render_stage_color_editor()`: [app.py:2215-2326](../../src/app.py#L2215-L2326) — **プリセット selectbox + `st.color_picker` 方式で実装**（計画書のパレットグリッド案ではなく selectbox 方式が採用された）
> - スウォッチプレビュー: `_style_stage_color_column()` ([app.py:2196-2209](../../src/app.py#L2196-L2209))
> - `find_or_create_stage_id()` / `build_event_output()` での拡張カラム生成: [output_builder.py:63-122, 425-603](../../src/backend_functions/output_builder.py)
>
> **本計画書策定後に追加**: ステージカラー LLM 推定 (`gpt_ocr.getocr_fes_stagelist_with_color_structured` :783、`ocr_service.set_stage_name` :411-433) で OCR 読み取り時に `stage_color` を自動付与。
>
> **残作業**: なし（短縮名編集も ⑥編集モードの `data_editor` に組み込み済）。Phase 7-2 ⑥-B 参照。
>
> 以下の設計記述は実装時の参照用として残す。**特に 2-2-1 / 2-2-2 / 2-5-1 の「予約」「未実装」記述は実装と差分があるため、最新コードを正とすること**。

### 2-1. stageNameShort

ステージマスタに`stageNameShort`カラムを追加する。

- デフォルト: `stageName`と同値
- UIで個別に編集可能

### 2-2. colorName

各ステージに色を割り当てる。以下の2方式に対応する:

**方式A: プリセットカラー名**
Stellaで定義されている27色のプリセットから選択:

```
stage-red, stage-pink, stage-purple, stage-deep-purple, stage-indigo,
stage-blue, stage-blue2, stage-lightBlue, stage-cyan, stage-teal,
stage-green, stage-light-green, stage-light-green2, stage-lime,
stage-yellow, stage-amber, stage-orange, stage-deepOrange,
stage-brown, stage-brown2, stage-blueGrey, stage-grey, stage-black,
stage-white, stage-redGrey, stage-greenGrey, stage-yellowGrey
```

**方式B: カスタムカラーコード**
`#背景色-#文字色` の形式で直接指定（例: `#EA749E-#FFFFFF`）

**デフォルト割当ルール**:
- ライブステージ: プリセットカラー27色を **順番に循環** して割り当て（red, pink, purple, ... yellowGrey, red, pink, ...）。27ステージを超えると同色が複数ステージに割り当たるため、UIで重複を警告し手動調整を促す
- 特典会ステージ: `stage-grey`
- 縁日ステージ: `stage-blueGrey`

#### 2-2-1. UI: 実カラー表示を伴うカラー選択

プリセット名・カスタムコードのいずれも **実カラーを画面上にレンダリングして選ばせる** ことを必須とする。生の文字列 (`stage-red` 等) を選ばせる selectbox 単独UIは採用しない。

**ステージごとのカラー編集 UI 構成（⑥-B 内）**:

1. **モード切替**: `st.radio` で「プリセット」/「カスタム」を選択（horizontal）
2. **プリセットモード**:
   - **カラーパレットUI**: 27色を 9×3 程度のグリッドで実カラー表示し、クリックで選択
   - 実装方針:
     - 各セルを `st.button` で描画し、CSS で `background-color: {bg}; color: {fg}` を当てる
     - ボタンラベルは Stella のプリセット名を短く表現したテキスト（例: `red`）。視覚的に "色そのもの" で識別できるようにする
     - グローバルCSSは `st.markdown(..., unsafe_allow_html=True)` で1回だけ注入し、各ボタンには CSSセレクタが当たるよう `key` プレフィックスを揃える（例: `key=f"colorbtn_{stage_id}_{preset_name}"` + CSS は `button[kind="secondary"][data-testid*="colorbtn_"]` を狙う）
     - 現在選択中のプリセットには枠線 (outline) を付けて強調表示
   - **現状色のプレビュー**: パレットの上に「現在: ████ stage-red (#EA5A5A / #FFFFFF)」と実色スウォッチ付きで表示
3. **カスタムモード**:
   - 背景色用 `st.color_picker(label="背景色")` と 文字色用 `st.color_picker(label="文字色")` を2つ並べる（Streamlit標準のカラーピッカーで実カラー選択可能）
   - 入力結果から `#RRGGBB-#RRGGBB` を組み立て、`カラー名` 列に保存
   - プレビュー: 「████ #EA749E-#FFFFFF」を実色付きで表示
4. **一覧プレビュー**: ⑥-B のステージ一覧テーブルでも、各行に **実カラーのスウォッチ** （背景色 + 文字色サンプル "ABC"）を表示し、編集UIを開かずに現状確認できるようにする

#### 2-2-2. 一覧テーブルでの実カラー描画

ステージマスタを `st.dataframe` で表示する箇所では、`カラー名` 列の代わりに **HTML レンダリングのスウォッチ列** を追加する。

- 実装方針: `st.column_config` の `LinkColumn` / `ImageColumn` は色付きセル描画に不向きなため、`HtmlColumn`相当が必要 → 現状は `pandas.DataFrame.to_html(escape=False)` でHTML出力し、`st.markdown(unsafe_allow_html=True)` で描画する経路を採用する
- セル中身の例:
  ```html
  <span style="display:inline-block;padding:2px 8px;
               background:#EA5A5A;color:#FFFFFF;
               border-radius:3px;font-size:12px;">stage-red</span>
  ```
- 編集モードでは行ごとに「色を編集」ボタンを置き、押下で前述のパレット/カスタムUIを `st.popover` または `st.expander` で展開

#### 2-2-3. プリセット定義の参照元

[Phase 2-5-1](#2-5-1-ステージカラーの反映) で新設する `data/master/color_preset.json`（`stage-red → ["#EA5A5A", "#FFFFFF"]` の辞書）をUIからも参照する。実カラー値の単一情報源とすることで、画像出力（`event_timetable_picture`）とUIプレビューで色のズレを防ぐ。

### 2-3. stageOrder

**既存実装を流用する**（新規UI構築は不要）。

- `master_stage.csv` の `表示順` 列は [output_section_edit_plan Phase 1-2 で導入済](done/output_section_edit_plan.md) で、⑥編集モードに **D&D 並び替え UI** が既に組み込まれている ([app.py:1568-1615](../../src/app.py#L1568-L1615))
- 純関数ヘルパは [`stage_reorder.py`](../../src/frontend_functions/stage_reorder.py) にあり、`apply_stage_reorder()` が `表示順` を 0..N-1 で振り直す
- Stella JSON の `stageOrder` フィールドは、`master_stage.csv.表示順` の値をそのまま出力すればよい。本Phaseでの追加実装は不要

### 2-4. ステージマスタのDataFrame拡張

⑥のステージマスタDataFrame（現状は[output_builder.py:45](../../src/backend_functions/output_builder.py#L45)で構築、`stage_master[stage_id] = {"ステージ名": ..., "特典会フラグ": ...}` 形式）を以下に拡張:

| カラム名 | 型 | 説明 |
|---------|---|------|
| `ステージID` | int | (既存／index) |
| `ステージ名` | str | (既存) |
| `特典会フラグ` | bool | (既存) |
| `ステージ名_短縮` | str | stageNameShort |
| `カラー名` | str | colorName (プリセット名 or #hex-#hex) |
| `表示順` | int | stageOrder (既存) |

これに伴い、`find_or_create_stage_id()`([output_builder.py:42](../../src/backend_functions/output_builder.py#L42))で新規登録する辞書にも追加2フィールド (`ステージ名_短縮` / `カラー名`) のデフォルト値を入れる。`表示順` は [event_timetable_picture_plan の Phase 1-2 で導入済](done/event_timetable_picture_plan.md)。

#### 既存 `master_stage.csv` の後方互換

既存プロジェクトの `master_stage.csv` には `ステージ名_短縮` / `カラー名` 列が無い。読み込み時、欠損列は **デフォルト生成ロジックと同じ方法で埋める**:

- `ステージ名_短縮` が欠損 → `ステージ名` の値をそのままコピー
- `カラー名` が欠損 → デフォルト割当ルール（[Phase 2-2](#2-2-colorname)）に従って割当
  - ライブステージ: `表示順` を基準にプリセット27色を循環参照
  - 特典会ステージ: `stage-grey`
  - 縁日ステージ: `stage-blueGrey`

補完したファイルは次回保存時に新スキーマで書き戻される（明示的なマイグレーションスクリプトは不要）。

### 2-5. 集約タイテ画像 (`event_timetable_picture`) への反映

Phase 2-2 (`カラー名`) が `master_stage.csv` に入った時点で、
[event_timetable_picture.py](../../src/backend_functions/event_timetable_picture.py)
側にも以下の変更を加える。Phase 2-4 の DataFrame 拡張までは現状の
デフォルト挙動 (フォールバック色) で動く。

ヘッダラベルは引き続き `ステージ名` (短縮しない) をそのまま使う。
`ステージ名_短縮` は Stella JSON 出力 (`stageNameShort` フィールド) 向け
であり、集約画像のヘッダには影響させない。

#### 2-5-1. ステージカラーの反映

[event_timetable_picture_plan.md の Phase 3](done/event_timetable_picture_plan.md)
で予約していた `stage_color_resolver` を実装する。

`カラー名` プリセット (`stage-red` 等) およびカスタム指定 (`#bg-#fg`) は
いずれも **背景色 + 文字色の 2 色** を表現する。リゾルバは
`Callable[[int stage_id], tuple[str bg, str fg]]` 型で、内部実装も
現状の `_default_color_resolver()` が既に 2 色タプルを返す形になっている
ため、外側のシグネチャは変えずに済む。

1. `data/master/color_preset.json` (本プランで新設) から
   `stage-red → ("#EA5A5A", "#FFFFFF")` のように **(背景色, 文字色)** の
   タプル値を持つ dict を構築する。
2. `make_stage_color_resolver(pj_path, event_name)` を
   `event_timetable_picture.py` に追加:
   - `{event_name}/master_stage.csv` の `カラー名` カラムを読み、
     プリセット名なら `color_preset.json` から、`#bg-#fg` 形式なら直接
     `(bg, fg)` にパースして `Callable[[int], tuple[str, str]]` を返す。
   - 未設定 / マスタ無しなら現状の `_default_color_resolver()` にフォールバック
     (どちらも `(bg, fg)` の 2 色タプル)。
3. `save_event_type_images()` / `save_event_image()` /
   `regenerate_event_type_images()` / `regenerate_all_event_images()` の
   各保存ラッパー内部で、引数 `stage_color_resolver` が省略されたとき
   自動的に `make_stage_color_resolver()` を構築するように変更。
   呼び出し側 (workflow / app) の API は変更しない。
4. `_default_color_resolver()` は残しておく (モード B やマスタ無しのフォールバック用)。

#### 2-5-2. ボックスの文字色対応 (`create_timetable_image`)

リゾルバが返す 2 色のうち、現状で実際に画像に反映されているのは
**背景色のみ**。ヘッダ描画 (`_render_header()`) は既に `(bg, fg)` の両方を
受け取って使っているが、ボックス内のテキスト
([`create_timetable_image`](../../src/frontend_functions/timetablepicture.py))
は `box_color` (背景色) しか受け取らず、テキストは固定で黒
(`text_color = "black"`) になっている。プリセットの中には `stage-black`
のように背景が濃色のものがあり、黒文字では読めなくなるため、文字色も
引数化する。

- `box_color` を `box_fill_color` のエイリアスとし、新たに
  `text_color_in_box: str = "black"` 引数を追加する (デフォルト動作は不変)。
- `create_timetable_image()` 内のテキスト描画で `text_color` の代わりに
  `text_color_in_box` を使用。
- `event_timetable_picture._compose_columns()` から
  `(bg, fg) = color_fn(stage_id)` の `bg` を `box_color` (= `box_fill_color`)、
  `fg` を `text_color_in_box` に渡す。
- ヘッダ描画は既に `fg` を受け取っているので変更不要。

#### 2-5-3. 再生成タイミング

⑥編集モードで `カラー名` を変更したあと、
[OutputWorkflow.save_output_edits()](../../src/workflow.py) の末尾で既に
`regenerate_all_event_images()` を呼んでいるため、追加実装は不要
(`save_event_*_images()` 経由で新しいリゾルバが自動構築される)。
ステージカラー編集のみを画像に反映したい場合は、⑥閲覧モードの
「今すぐ再生成」ボタンで対応する。

---

## Phase 3: トップレベルメタデータ

> 📌 **実装状況: 部分実装**（2026-06 時点）
> - 実装済: `_default_stella_metadata()` / `get_stella_metadata()` / `set_stella_metadata()` / `ensure_stella_metadata()`: [project_repository.py:66-112](../../src/backend_functions/project_repository.py#L66-L112)
> - 実装済: 既存フィールド `openTime` / `closeTime` / `notification` / `notificationVersion` / `_last_pushed_notification`
>
> **残作業**:
> 1. `_default_stella_metadata()` に `date` (str) / `dow` (int|None) を追加
> 2. `project_info.json` トップに `stella_project_meta` ブロック (`liveName` / `genre` / `release` / `pref`) を新設するためのアクセサ関数追加
> 3. `create_project_data()`([project_repository.py:436-477](../../src/backend_functions/project_repository.py#L436-L477)) で `stella_project_meta` のデフォルト値を埋め込み
> 4. 既存 `ensure_stella_metadata()` を `stella_project_meta` にも適用する関数を追加（既存プロジェクト後方互換）

### 3-1. 管理対象フィールドと入力配置

Stella JSON のトップレベル + liveList エントリで使用するメタデータは、**入力タイミング・粒度で①画面と⑥画面に振り分ける**。

#### 3-1-A. ①画面で入力するフィールド（プロジェクト/イベント基本情報）

ライブ自体の属性（後から変えにくく、Stella 採番時に必要なもの）はすべて①画面で入力する。

**プロジェクト単位（全 event 共通）**:

| フィールド | 型 | デフォルト | UI |
|---|---|---|---|
| `liveName` | str | （空） | テキスト。`project_name` とは独立した「Stella上のライブ名」 |
| `genre` | int | `2` | selectbox: 1=ロック / 2=アイドル |
| `release` | int | `0` | selectbox: 0=非公開 / 1=公開(非アクティブ) / 2=公開 |
| `pref` | int | `13` (東京) | selectbox: 都道府県マスタ ([Phase 4-3](#4-3-都道府県マスタ)) |

→ `project_info_json["stella_project_meta"]` に保存（新設ブロック）。

**event 単位**:

| フィールド | 型 | デフォルト | UI |
|---|---|---|---|
| `date` | str (YYYYMMDD) | （空） | 日付ピッカー。複数 event の場合、1 つ目入力で他 event は翌日相当を自動入力（編集可） |
| `dow` | int (1-7) | （自動） | `date` から算出（月=1 〜 日=7）。表示のみ、編集不可 |

→ 各 `event_detail[i].stella_metadata` の `date` / `dow` フィールドに保存。

#### 3-1-B. ⑥画面で入力するフィールド（Stella JSON ヘッダ情報）

公演運営に紐づく情報（タイテ確定後に決まるもの）は⑥画面で入力する。

| フィールド | 型 | UI |
|---|---|---|
| `openTime` | str | 数値入力（時のみ、例: `"12"`） |
| `closeTime` | str | 数値入力（時のみ、例: `"23"`） |
| `notification` | str | テキスト入力（更新通知メッセージ） |

→ 各 `event_detail[i].stella_metadata` の同名フィールドに保存。

#### 3-1-C. システムが自動管理するフィールド

ユーザー入力ではなく、採番・Push 成功時にシステムが書き込む:

| フィールド | 型 | 書き込みタイミング |
|---|---|---|
| `liveId` | int | ①画面の **採番ボタン** で GitHub 側 `liveList.json` に Reserve-Push 成功後に書き込み ([Phase 4-4](#4-4-liveid採番ワークフローreserve-first方式)) |
| `bundleId` | str | 同上。event_num・既採番状況から算出 ([Phase 4-5](#4-5-bundleid-の確定タイミングと再計算)) |
| `jsonVersion` | int | Push 成功のたび `+1`。初回 Push で `1` |
| `notificationVersion` | str | 直近 Push 時の `notification` と現在値に **差分があれば +1**、無ければ据え置き。初回は `"1"` |
| `_last_pushed_notification` | str/null | notificationVersion 差分判定用の内部フィールド。Push 成功時に当時の `notification` 値で更新 |

#### 3-1-D. バージョンインクリメントとロールバックの順序

Push 失敗時にローカルメタデータが壊れないよう、**インクリメントは Push 成功確認後に行う**。詳細手順は [Phase 6-6](#6-6-push-失敗時のロールバック設計) を参照。

### 3-2. 保存場所

`project_info.json` の構造を以下のように拡張する:

```json
{
    "project_name": "...",
    "stella_project_meta": {
        "liveName": "歌舞伎町UP GATE↑↑",
        "genre": 2,
        "release": 0,
        "pref": 13
    },
    "event_num": 2,
    "event_detail": [
        {
            "event_no": 0,
            "event_name": "day1",
            "stella_metadata": {
                "date": "20260504",
                "dow": 1,
                "openTime": "12",
                "closeTime": "23",
                "notification": "",
                "notificationVersion": "1",
                "_last_pushed_notification": null
            },
            ...
        },
        ...
    ]
}
```

`liveId` / `bundleId` / `jsonVersion` はキー欠損（採番・Push 成功で初めてセット）。

#### 既存 `project_info.json` の後方互換

既存プロジェクトには `stella_project_meta` / `stella_metadata` ブロックが無い。読み込み時に既定値で補完する:

- `stella_project_meta` 補完値: `{"liveName": "", "genre": 2, "release": 0, "pref": 13}`
- `stella_metadata` 補完値: `{"date": "", "dow": null, "openTime": "", "closeTime": "", "notification": "", "notificationVersion": "1", "_last_pushed_notification": null}`
- 補完したファイルは次回保存時に書き戻される（明示的なマイグレーションスクリプトは不要）

①画面 / ⑥画面のフォーム表示時、空文字 / null のフィールドは未入力プレースホルダで表示する。

---

## Phase 4: liveList.json管理

### 4-1. liveListエントリの構造

```json
{
    "liveId": 547,
    "bundleId": "547",
    "liveName": "歌舞伎町UP GATE↑↑",
    "release": 2,
    "pref": 13,
    "date": "20260504",
    "dow": 1,
    "genre": 2,
    "serchKey": "0"
}
```

### 4-2. 各フィールドの入力方式

| フィールド | 入力方式 | 入力配置 |
|---|---|---|
| `liveId` | 自動採番（Reserve-First、[4-4](#4-4-liveid採番ワークフローreserve-first方式)） | - |
| `bundleId` | 自動算出（[4-5](#4-5-bundleid-の確定タイミングと再計算)） | - |
| `liveName` | テキスト | ①画面 (`stella_project_meta.liveName`) |
| `release` | selectbox: 0/1/2 | ①画面 (`stella_project_meta.release`) |
| `pref` | selectbox: 都道府県マスタ ([4-3](#4-3-都道府県マスタ)) | ①画面 (`stella_project_meta.pref`) |
| `date` | 日付ピッカー → YYYYMMDD | ①画面 (`event_detail[i].stella_metadata.date`) |
| `dow` | `date` から自動算出 (月=1 〜 日=7) | （自動） |
| `genre` | selectbox: 1/2 | ①画面 (`stella_project_meta.genre`) |
| `serchKey` | 固定 `"0"` | （Stella 側仕様確認後に編集UI追加検討、[後回し領域 C](#後回し領域の課題リスト-phase-4--6) 参照） |

> 採番ボタン押下前の **入力必須項目**: `liveName` + 全 event の `date`。他はデフォルト値で進められる。

### 4-3. 都道府県マスタ

`data/master/pref_master.json` として新設:

```json
[
    {"code": 1, "name": "北海道"},
    {"code": 2, "name": "青森県"},
    ...
    {"code": 13, "name": "東京都"},
    ...
    {"code": 47, "name": "沖縄県"},
    {"code": 48, "name": "タイ"},
    {"code": 49, "name": "台湾"}
]
```

**抽出元**: 現行の Stella JSON 生成ツール [`data_tmp/stella_json/JSON生成シート.xlsm`](../../data_tmp/stella_json/JSON生成シート.xlsm) の `prefList` シート右側列（47都道府県 + 48:タイ + 49:台湾、計49件）。マスタは初回手動でこのシートから書き起こす。

### 4-4. liveId採番ワークフロー（Reserve-First 方式）

**設計方針**: GitHub 側 `liveList.json` への push が成功してから手元の `stella_metadata.liveId` を確定する。手元の liveId は「GitHub に書き込めた liveId だけ」という不変条件を保つ。

#### 4-4-1. 採番ボタンの配置

①画面に「Stella liveId を採番」ボタンを設置する。⑥画面側には採番 UI を持たない（Push 系のみ）。

#### 4-4-2. ボタン押下時のフロー（新規採番）

```
[①画面 採番ボタン]
  ├─ バリデーション: liveName / 全 event の date が入力済か
  ├─ git pull (data/timetableproj/)
  ├─ max(liveList.liveId) を取得
  ├─ 未採番 event ごとに連番 (max+1, max+2, ...) を仮割当
  ├─ bundleId を確定 ([Phase 4-5](#4-5-bundleid-の確定タイミングと再計算))
  ├─ 各 event の liveList エントリを「フル構築」
  │    {liveId, bundleId, liveName, release, pref, date, dow, genre, serchKey:"0"}
  ├─ liveList.json に追記 (既採番 event の bundleId 更新があれば該当エントリも修正)
  ├─ commit → push
  ├─ 成功:
  │    各 event_detail[i].stella_metadata に liveId / bundleId / dow を書き込み
  │    project_info.json 保存
  └─ 失敗:
       ローカル一切無変更（内側リポは git reset --hard で巻き戻し、[Phase 6-6](#6-6-push-失敗時のロールバック設計)）
       UI: 「他の編集と衝突しました。再採番してください」
```

#### 4-4-3. 再採番ボタン（event 後追加 / 一部未採番）

①画面に **再採番ボタン**（採番ボタンと同位置、状態に応じてラベル切替）を配置し、以下の **案 B（既採番保持）** で動作する:

- **既採番 event の `liveId` は保持**（再採番しない）
- **未採番 event のみ** に `max(現liveList.liveId)+1` から連番採番
- **bundleId は再計算**（[Phase 4-5](#4-5-bundleid-の確定タイミングと再計算)）。event_num=1 → 増加で全 event の bundleId が変わる場合、既採番 event の `liveList.json` エントリも該当 push で更新（その event の `jsonVersion` も `+1`、次回 Push 時に live{id}.json も書き換え）

#### 4-4-4. 全リセットボタン（採番のやり直し）

採番済情報を丸ごと取り直したい場合のために、別途 **「採番をすべてクリアして取り直す」ボタン** を①画面に置く（確認ダイアログ付き）。挙動:

- 全 event の `stella_metadata.liveId` / `bundleId` / `jsonVersion` / `_last_pushed_notification` をクリア
- liveList.json 上の旧エントリは **削除しない**（過去 push 済のため）。ゴミとして残るが、Stella 側の release=0（非公開）なら影響しない方針
- クリア後に通常の採番ボタンを押下 → 新規 liveId で取り直し

#### 4-4-5. 認証 / リポ未取得時の挙動

- 認証情報が無い環境: 採番ボタンを無効化し、「`.env` に `GITHUB_TOKEN` を設定してください」と表示
- `data/timetableproj/` 未 clone: ボタン押下時に自動 clone を試行（[Phase 6-5](#6-5-ローカルリポジトリパス)）

### 4-5. bundleId の確定タイミングと再計算

**確定タイミング**: ①画面の採番ボタン押下時にのみ計算・書き込みする。⑥画面以降では bundleId に触れない。

**算出ロジック**:

```
if event_num == 1:
    bundleId = ""        # 空文字
else:
    # 既採番 event の bundleId を走査
    existing = [e.bundleId for e in events if e.bundleId と liveId が採番済]
    non_empty = [b for b in existing if b not in (None, "")]

    if non_empty:
        # 既に bundleId 確定済の event がある → その値を全 event に揃える
        bundleId = non_empty[0]
    else:
        # 全 event 未採番 or 全 event が bundleId="" (元 event_num=1 だった場合)
        # → これから採番する最初の event の liveId を bundleId とする
        bundleId = str(min(これから採番する liveId))
```

**event_num=1 → 増加した場合の挙動**:
- 元 event の `bundleId=""` を **新しい bundleId に上書きする**
- 該当 event の `liveList.json` エントリの bundleId も書き換え（採番 push と同一 commit 内で）
- 既 Push 済の `live{id}.json` は次回 Push 時に bundleId が更新される（jsonVersion +1）

**Push 時バリデーション**: `event_num > 1 かつ bundleId が "" / None / 未設定` の event を Push しようとした場合、Push を中断し「①画面で再採番してください」とアラート表示。

### 4-6. liveListPanel2.json 連携 (表示エリア・表示順制御) — ✅【実装済】

> 📌 **2026-06 実装済**。採番と同時に `liveListPanel2.json` を追記する。`liveListPanel.json` (flat v1) は対象外（運用上 Panel2 のみ参照のため）。

**ファイルの役割**: ライブを「どの年月エリアに / その中でどの順で / 複数日イベントをどうくくって」表示するかを制御する。

**構造** (運用リポ実物):
```json
[
  {"year":2026,
   "monthEnableList":[bool x12],            // index 0=1月..11=12月。表示がある月は true
   "monthList":[{"month":5,"liveIdlist":[[547,548],[550],...]}]}  // 各要素=1イベント(bundle)
]
```
- **bundle** = `liveIdlist` の各要素。複数日イベントは全日 liveId を 1 配列に束ねる (例 `[547,548]`)。本アプリの **同一プロジェクトの全 event の liveId 集合** (= 同一 `bundleId`) に対応。
- **配置**: イベントの **最初の日付** の年月エリア (月をまたいでも最初の日付の月に全 liveId)。
- **並び**: 月内は各 bundle の最初の日付の昇順、同一日付は追加順 (安定挿入)。

**採用ロジック**:
- 採番 (`reserve_live_ids`) で新規割当があるとき、プロジェクトの bundle を `liveListPanel2.json` に upsert し、**`liveList.json` と同一 commit/push** で反映 (原子性)。
- 既存 bundle (プロジェクトの既 liveId を含む) は一旦除去して再配置 → 再採番で日付/配置が変わるケースに対応。空になった月/年は除去し `monthEnableList` を再計算。
- 月内挿入位置の比較に使う既存 bundle の日付は、更新後の `liveList.json` から参照する (`{liveId: date}`)。
- 前提: 採番バリデーションで全 event の `date` 必須 (`liveId採番時に日付保持をマスト`)。
- push 失敗時は `liveList.json` と同じくロールバックで Panel2 も巻き戻る。

**採番後の date 変更への追従**: ①「Stella連携設定を保存」での `resync_live_list()` が、liveList 再反映に加えて **Panel2 の bundle を現在の date で再配置** する (`_resync_panel()`)。配置 (年月 / 月内順) が変わった場合のみ `liveListPanel2.json` も同一 commit に含めて push。`upsert_bundle` が旧位置から除去して再配置するため、月をまたぐ移動・空月の除去・`monthEnableList` 再計算も追従する。

**実装**: `stella_panel.py` (`read_panel` / `write_panel` / `upsert_bundle` 純関数) + `stella_reserve._update_panel()` (採番時) / `_resync_panel()` (date 変更時) / `project_bundle()` / `reserved_bundle()`。テスト: `test_stella_panel.py` / `test_stella_reserve.py`。

---

## Phase 5: Stella JSON出力関数

> 📌 **実装状況: 完了**（2026-06 時点）
> - [stella_export.py](../../src/backend_functions/stella_export.py) (242行) に以下が実装済:
>   - `build_stella_json()`: [:131-171](../../src/backend_functions/stella_export.py#L131-L171)
>   - `write_stella_json()`: [:174-191](../../src/backend_functions/stella_export.py#L174-L191)
>   - `update_live_list()`: [:194-217](../../src/backend_functions/stella_export.py#L194-L217)
>   - `increment_versions_on_push()`: [:220-241](../../src/backend_functions/stella_export.py#L220-L241)
>
> **Phase 4 / 6 着手時の確認事項**: `update_live_list()` の更新セマンティクスが [後回し領域 C](#後回し領域の課題リスト-phase-4--6--7-1--7-2-d) の決定事項と一致しているか、Reserve-First 採番フロー (Phase 4-4) と整合するか。
>
> 以下の設計記述は実装時の参照用として残す。

### 5-1. 変換ロジック

`src/backend_functions/stella_export.py`を新設し、以下の関数を実装:

```python
def build_stella_json(
    output_df: dict[str, pd.DataFrame],
    stella_metadata: dict,
) -> dict:
    """⑥のDataFrameからStella JSON形式のdictを構築する。

    output_df は `build_event_output()` の戻り値構造を想定:
      - "stage" / "idolname" / "live" (Stella JSON生成に使用)
      - "duration_distribution" / "group_count" / "overlap_alerts" /
        "group_appearances" (集計用。本関数では未使用)
    """
```

**変換処理**:

1. **artList構築**: idolname DataFrame → `[{"artId": id, "name": name}, ...]`
2. **stageList構築**: stage DataFrame → `[{"stageId": id, "stageName": ..., "stageNameShort": ..., "colorName": ..., "stageOrder": ...}, ...]`
   - `stageOrder`がNoneの場合は`stageId`をそのまま使用
3. **turnList構築**: live DataFrame を `出番ID` でgroupbyし、`[{"turnId": id, "startTime": ..., "min": ..., "artId": ..., "stageId": ..., ...}, ...]` を生成
   - **コラボ判定**: 同一 `出番ID` の行が複数ある場合 → `collabArtList: [artId1, artId2, ...]` を追加（先頭行の `artId` も含む）。単独行ならフィールド省略
   - **タイトル**: 当該グループの `コラボタイトル` が非空なら `title: "..."` を追加。空ならフィールド省略（Stella側でアーティスト名連結表示）
   - **代表 artId**: グループ先頭行（または最小 artId）を `artId` にセット
4. **トップレベル**: stella_metadataからliveId, jsonVersion等を設定

### 5-2. 出力形式と出力先

- 1行JSON（minify）として出力（live547.jsonと同様）
- **文字コード: UTF-8 BOM 付き** (`utf-8-sig`)。clone smoke test で運用リポを取得した結果、既存 `liveList.json` および全 `live{id}.json` が UTF-8 BOM 付き (`ef bb bf`) で配置されていることを確認 (2026-06)。`stella_export.write_stella_json()` / `update_live_list()` は `utf-8-sig` で読み書きする
- **一次出力先**: `data/projects/{pj_name}/event_{n}/live{liveId}.json`（本アプリ管理下）
- **GitHub反映時**: `data/timetableproj/` 配下へ JSON を書き出して commit/push

> 本アプリ側にもファイルを残すことで、GitHub Push前後の差分確認や、リポ未取得状態でも成果物が手元に残るメリットがある。
>
> 📌 **実装メモ (2026-06)**: `stella_push` の Push フローでは、内側リポ (`data/timetableproj/`) への書き出しは git add のため push 前に行うが、**本アプリ配下 (`data/projects/.../live{id}.json`) への確定コピーは push 成功後** (`_persist_local_copies()`) に保存する。これにより (1) push 失敗時に未 push の JSON を手元に残さない、(2) 手元コピーは常に「実際に push された内容・バージョン」と一致する。PR モードは push 後に内側リポを `reset --hard` で戻すため、手元に残る確定版は本アプリ配下のコピーが正となる。

#### 参考: 運用リポ取得時点の状況 (2026-06)

- `liveList.json` 件数: **603 件** / max `liveId`: **602**（→ 次回新規採番開始値の目安）
- `live{id}.json` 系のファイル数を含む全ファイル数: 578（マスタ系 `color.json` `delay_stage.json` 等を含む）
- トップレベルキー: `["liveList"]` のみ

### 5-3. liveList.json更新

```python
def update_live_list(
    live_list_path: str,
    new_entries: list[dict],
) -> None:
    """liveList.jsonに新規エントリを追加/更新する"""
```

---

## Phase 6: GitHub連携

### 6-1. 対象リポジトリ

- URL: https://github.com/ys0512/timetableproj
- 必要なファイル:
  - `liveList.json` — 全ライブ一覧
  - `live{id}.json` — 各ライブのタイムテーブルデータ

### 6-2. ワークフロー

```
[GitHub clone/pull] → [liveList読み込み] → [liveId採番]
    → [データ作成] → [JSON出力] → [commit] → [push (default branch / feature branch + PR)]
```

### 6-3. 実装する操作

| 操作 | 説明 | UI |
|------|------|----|
| Clone/Pull | リポジトリの最新状態を取得 | 編集開始時に自動 + 手動ボタン |
| liveId採番 | liveList.jsonの最大ID+1を自動取得 | 自動 |
| JSON出力 | live{id}.json + liveList.json更新を本アプリ`data/projects/...`に書き出し | ボタン |
| Push (PR) | 一時ブランチを作成 → commit → push → PR作成 | ボタン（**初期推奨**） |
| Push (直接) | デフォルトブランチに直接commit/push | ボタン（確認ダイアログ付き） |

### 6-4. 認証 (.env 方式 — 既存 AWS と同経路)

既存の AWS 認証が `.env` + `python-dotenv` で管理されている（[s3access.py](../../src/backend_functions/s3access.py)）ため、**GitHub PAT も同じ `.env` 方式に揃える**。Streamlit Secrets は使わない。

#### 6-4-1. トークン管理

`.env` に以下を追記:

```ini
GITHUB_TOKEN=github_pat_xxxxxxxx...
GITHUB_USER_NAME=timetable-detect-bot
GITHUB_USER_EMAIL=bot@example.com
```

読み込み優先順位:

1. プロセス環境変数 `GITHUB_TOKEN` (`.env` 経由を含む) — 主経路
2. 値が無い場合: UI 上で「GitHub 連携が利用できません」と表示し、JSON 出力までで停止（採番ボタン・Push ボタンは無効化）

> Streamlit Cloud 運用時は Streamlit の Secrets 機能ではなく、デプロイ時に環境変数として `GITHUB_TOKEN` 等を注入する。`.env` は `.gitignore` に既登録 (`/.env`) されておりリポにコミットされない。

#### 6-4-2. トークンスコープ

ys0512/timetableproj への push / PR 作成権限が必要:

- Classic PAT: `repo` スコープ
- Fine-grained PAT: `ys0512/timetableproj` を対象に `Contents: Read and write` + `Pull requests: Read and write`

> セキュリティ: PAT はリポジトリオーナーが発行・管理し、各環境の `.env`（または環境変数）にのみ設定する。コードや `project_info.json` にトークンを含めない。

#### 6-4-3. git 操作の実装方針

- 認証付きリモートURL: `https://x-access-token:{token}@github.com/ys0512/timetableproj.git` を使う方式に統一（HTTPSベース）
- PR作成は `PyGithub` ライブラリで REST API 経由（git CLI不要）
- clone/pull/push/commit は `GitPython` 経由（内部で git CLI を呼ぶため、実行環境に git バイナリが必要）

### 6-5. ローカルリポジトリパス

**`data/timetableproj/` 固定**。プロジェクトごとや設定ファイルでの切替は行わない（リポは 1 つしか無いため）。

#### 6-5-1. ディレクトリ配置とネスト git

```
timetable-detect/                     ← 外側 git (このアプリ本体)
├── .gitignore                        ← data/timetableproj/ を追加
├── data/
│   ├── master/                       ← 既存（マスタファイル）
│   ├── projects/                     ← 既存（プロジェクトデータ）
│   └── timetableproj/                ← 【新規】ここに git clone する
│       ├── .git/                     ← 内側の独立 git リポ
│       ├── liveList.json
│       └── live{id}.json
└── data_tmp/
    └── stella_json/                  ← 既存（Excel 試作品置き場、本機能では不使用）
```

- 外側 git は内側 `.git/` を見ないので **完全分離**。timetable-detect リポへの影響はゼロ
- `git submodule` は使わない（アプリが自由に reset/clean したいため、submodule 化するとむしろ管理が複雑になる）
- 内側 clone は実質「キャッシュ」扱い。壊れたら `data/timetableproj/` を削除して再 clone で復旧
- `data/` 直下の既存 `master/` / `projects/` は外側 git の管理対象 (一部 `data/timetable_sample` のみ ignore)。`timetableproj/` は新規 ignore 追加で同列に並べても干渉しない

#### 6-5-2. `.gitignore` 追記

```
data/timetableproj/
```

#### 6-5-3. clone / pull タイミング

- **初回**: アプリ起動後、最初に GitHub 連携機能を使う時点で `data/timetableproj/` が無ければ自動 clone
- **pull**: ①画面の採番ボタン押下時 / Push ボタン押下時に **毎回 pull → 最新化**してから処理開始（stale な liveList で衝突採番しないため）
- **手動更新**: ⑥-D に「リポ同期 (pull)」ボタンも残す（PR 進捗確認等のため）

#### 6-5-4. Streamlit Cloud 上の挙動

- Streamlit Cloud のファイルシステムはセッション跨ぎで揮発する可能性があるため、起動毎に clone される前提で動作する
- 同一インスタンス内では `@st.cache_resource` で clone 実行を 1 回に抑制
- ローカル開発でもクラウドでも **同じパス (`data/timetableproj/`)** を使うため、環境別パス設定は不要

### 6-6. Push 失敗時のロールバック設計

採番 push / 通常 push のいずれも、**ローカル `project_info.json` の更新は Push 成功後に行う**ことで失敗時の整合性を保つ。

#### 6-6-1. 共通スナップショット → 試行 → 確定/巻き戻しのパターン

```
[Push 開始]
  ├─ snapshot:
  │     before_hash = 内側リポ HEAD のコミット hash
  │     before_meta = 該当 event の stella_metadata の deepcopy
  ├─ 新メタデータを算出（メモリ上）:
  │     new_jsonVersion = before.jsonVersion + 1 (Push 系) / 未設定なら 1
  │     new_notificationVersion =
  │         before._last_pushed_notification != current.notification
  │             ? before.notificationVersion + 1
  │             : before.notificationVersion  ← 差分無ければ据え置き
  ├─ ローカルファイル書き込み:
  │     data/timetableproj/liveList.json (採番時のみ)
  │     data/timetableproj/live{id}.json (通常 Push 時)
  ├─ git add → git commit -m "..."（内側リポ）
  ├─ git push origin {branch}
  │
  ├─ 成功:
  │     project_info.json の stella_metadata を新メタデータで更新:
  │       - liveId / bundleId（採番時のみ）
  │       - jsonVersion = new_jsonVersion
  │       - notificationVersion = new_notificationVersion
  │       - _last_pushed_notification = current.notification
  │     project_info.json を保存
  │
  └─ 失敗:
        内側リポで `git reset --hard {before_hash}` → ローカルファイル巻き戻し
        project_info.json は触っていないので無変更
        UI でエラー表示（採番衝突 / 認証エラー / ネットワーク等の理由別）
```

#### 6-6-2. ポイント

- **インクリメントは Push 成功後**: jsonVersion / notificationVersion / _last_pushed_notification は失敗しても進まない → 次回リトライでも一意な番号
- **巻き戻しは内側リポで完結**: `git reset --hard {before_hash}` は内側のみに作用し、外側 timetable-detect リポには一切影響しない
- **採番時の特殊性**: liveId / bundleId も同じく Push 成功後に書き込み。失敗時はローカル `stella_metadata.liveId` 未設定のまま → リトライで pull → max+1 を取り直す
- **commit 後 push 前のローカルクラッシュ対応**: 次回起動時、内側リポに未 push の commit が残っていたら警告を出し、リセット or 強制 push のどちらかを選ばせる（運用上稀だが復旧フローを明示）

---

## Phase 7: UI統合

### 7-0. 現状の関数構造（2026-06 時点）

その後のリファクタと Stella 関連の先行実装で、⑥セクションは複数の関数に分離されている:

```
render_output_section()                     # app.py:2799 (44行) — タブ + 集計情報のみ
├── build_all_event_outputs() でデータ取得
├── event_tabs ループ
└── 「IDマスタを確定」ボタン

render_output_export_section()              # app.py:2842 — エクスポート系
├── 「クラウドにアップロード」ボタン
├── 「Excelデータを出力」ボタン
└── _render_stella_section(event_list)      # app.py:2868 — Stella 関連サブセクション
    ├── _render_stella_metadata_form()      # app.py:2932 — ⑥-A 相当 (openTime/closeTime/notification)
    └── _render_stella_export_form()        # app.py:2986 — ⑥-C 相当 (JSON 生成・ダウンロード)
```

ステージカラー編集 UI (⑥-B の色部分) は別関数 `_render_stage_color_editor()`([app.py:2215-2326](../../src/app.py#L2215-L2326)) として既に独立実装されている。

UI 描画は薄く保ち、データ組み立ては backend (`output_builder.py` / `stella_export.py`) に寄せる方針も同じく踏襲する。

### 7-1. ①画面に新設する Stella 入力サブセクション

> 📌 **実装状況: 未着手**。`render_project_setting()`([app.py:884](../../src/app.py#L884)) には Stella 関連入力は存在しない。

`render_project_setting()`([app.py:884](../../src/app.py#L884)) の末尾に以下を追加:

**①-Stella: ライブ基本情報 + 採番**

| ブロック | 内容 |
|---|---|
| プロジェクト単位入力 | `liveName` (テキスト) / `genre` (selectbox, デフォルト 2) / `release` (selectbox, デフォルト 0) / `pref` (selectbox, デフォルト 13:東京、[Phase 4-3](#4-3-都道府県マスタ) のマスタ参照) |
| event 単位入力 | 各 event の `date` (日付ピッカー、複数 event なら 1 つ目入力で他 event は翌日相当を自動補完・編集可)、`dow` (date から自動算出して表示のみ) |
| 採番状態表示 | 各 event の現状: 「未採番」 / 「採番済 liveId=N (bundleId=...)」を一覧で表示 |
| 採番ボタン | 「Stella liveId を採番」(未採番 event を対象に Reserve-First Push、[Phase 4-4-2](#4-4-2-ボタン押下時のフロー新規採番)) |
| 再採番ボタン | 採番済 event 保持 + 未採番 event 追番 + bundleId 再計算 ([Phase 4-4-3](#4-4-3-再採番ボタンevent-後追加--一部未採番)) — 採番ボタンと同位置でラベル切替 |
| 全リセットボタン | 「採番をすべてクリアして取り直す」(確認ダイアログ付き、[Phase 4-4-4](#4-4-4-全リセットボタン採番のやり直し)) |

- 入力値の保存先: `stella_project_meta` (プロジェクト単位) / `event_detail[i].stella_metadata` (event 単位)
- 認証情報 (`GITHUB_TOKEN`) が無い環境では採番系ボタンを無効化し、「`.env` を設定してください」のヘルプを表示

### 7-2. ⑥画面 Stella サブセクション

**⑥-A: Stella ヘッダ情報入力** — 📌 **実装済**: `_render_stella_metadata_form()`([app.py:2932-2983](../../src/app.py#L2932-L2983))
- `openTime` / `closeTime` / `notification`（[Phase 3-1-B](#3-1-b-⑥画面で入力するフィールドstella-json-ヘッダ情報)）
- 入力値は `event_detail[i].stella_metadata` に保持し、保存時に `project_info.json` に書き戻す
- **残作業（Phase 7-1 連動）**: `liveName` / `pref` / `genre` / `release` / `date` は①画面で入力する想定。①画面実装時に⑥-A から「表示のみ」セクションを追加して①画面へ誘導するリンクを置く

**⑥-B: ステージ詳細設定** — 📌 **実装済**
- `ステージ名_短縮`: ⑥編集モードのステージマスタ `st.data_editor` の column_config に組み込み済 ([app.py:2757-2773](../../src/app.py#L2757-L2773))。空入力時は `ステージ名` 同値で出力される旨をヘルプに記載
- `カラー名`: `_render_stage_color_editor()`([app.py:2397+](../../src/app.py#L2397)) で selectbox + `st.color_picker` カスタム + スウォッチプレビュー（計画書のパレットグリッド案ではなく selectbox 採用）
- 後方互換: 旧マスタに `ステージ名_短縮` / `カラー名` 列が無い場合のデフォルト埋めも実装済 ([app.py:2742-2747](../../src/app.py#L2742-L2747))
- stageOrder は [既存の D&D 並び替え UI](../../src/app.py#L1568-L1615) ([Phase 2-3](#2-3-stageorder)) を流用済

**⑥-C: Stella JSON 出力** — 📌 **実装済**: `_render_stella_export_form()`([app.py:2986-3025](../../src/app.py#L2986-L3025))
- 「Stella JSON を生成」ボタン → `stella_export.build_stella_json()` 呼び出し
- プレビュー表示・ダウンロードボタン実装済
- **残作業（Phase 4 連動）**: liveId 未採番の event を弾くバリデーション（「①画面で採番してください」のリンク表示）

**⑥-D: GitHub 連携 (Push)** — 📌 **実装済**
- ✅ 接続状態表示（`GITHUB_TOKEN` 検出済か / `data/timetableproj/` clone 済か）: `_render_stella_connection_status()`
- ✅ 「リポ同期 (pull)」ボタン（手動同期用、[Phase 6-5-3](#6-5-3-clone--pull-タイミング)）
- ✅ 現在の `liveId` / `jsonVersion` / `notificationVersion` 表示
- ✅ Push 時バリデーション結果表示（未採番 / event_num>1 で bundleId 未設定の場合は警告 + ①画面誘導、[Phase 4-5](#4-5-bundleid-の確定タイミングと再計算)）
- ✅ **Push (PR 作成)** ボタン — 初期推奨
- ✅ **Push (直接)** ボタン — 確認ダイアログ付き
- ✅ **プロジェクト全体一括 Push**（`_render_stella_bulk_push` / `stella_push.push_all_stella_json`）: 全イベントの `live{id}.json` を **1 コミット・1 PR** にまとめて push（pull は 1 回、全イベント検証後にまとめて push、1 件でも NG なら中断＝原子性）。event 数 1 のときは非表示
- ✅ 認証情報が無い環境では Push 系ボタンは無効化し、JSON 生成・ダウンロードのみ可能
- 関数構成: `_render_stella_section()` 内、接続状態 → 一括 Push → イベント別 expander（`_render_stella_push_form()`）の順

> **「リポ同期 (pull)」の必要タイミング**: 採番 (`reserve_live_ids`) / Push (`push_stella_json` / `push_all_stella_json`) は冒頭で毎回自動 pull するため、**通常操作では手動 pull は不要**。手動ボタンが要るのは (1) 初回 clone を先に済ませたい (2) 他者の GitHub 更新を採番前に手元で確認したい (3) 内側リポ破損時のリカバリ、の限定ケースのみ。

---

## 実装順序

### 完了済（2026-06 時点）

- ✅ Phase 1 (コラボ公演対応) — 全実装、テスト含む
- ✅ Phase 2 (stageList拡張カラム + カラーリゾルバ + LLM 色推定) — UI 設計は計画から変更され selectbox 採用
- ✅ Phase 5 (Stella JSON 変換関数 `stella_export.py`)
- ✅ Phase 7-2 ⑥-A (`_render_stella_metadata_form`) / ⑥-C (`_render_stella_export_form`)
- ✅ Phase 7-2 ⑥-B のカラー編集 (`_render_stage_color_editor`)

### 残作業（着手順）

```
[完了] Phase 3 残タスク (project_info.json スキーマ拡張)
     ├─ ✅ _default_stella_metadata に date/dow 追加
     ├─ ✅ stella_project_meta ブロック新設 (アクセサ追加)
     ├─ ✅ create_project_data でデフォルト埋め込み
     └─ ✅ read_project_data で後方互換補完 (ensure_stella_project_meta)
[完了] Phase 7-1 ①画面 Stella 入力ブロック (採番ボタン以外)
     ├─ ✅ liveName / genre / release / pref / event ごとの date 入力
     ├─ ✅ ProjectWorkflow.save_stella_input 経由で project_info.json 保存
     └─ ✅ 採番ボタンは disabled プレースホルダ (.env の GITHUB_TOKEN 設定で有効化予定)
[完了] Phase 4-3 都道府県マスタ
     └─ ✅ data/master/pref_master.json (49件)
[完了] Phase 6 環境整備
     ├─ ✅ requirements.txt に GitPython / PyGithub 追加
     ├─ ✅ .gitignore に data/timetableproj/ 追加
     ├─ ✅ .env に GITHUB_TOKEN / GITHUB_USER_NAME / GITHUB_USER_EMAIL プレースホルダ追加
     ├─ ✅ github_ops.py (clone_or_pull / commit_and_push / create_pull_request)
     └─ ✅ docs/stella_github_setup.md (PAT 受領後の手順書)
              ↓
[1] PAT 受領後の clone smoke test (docs/stella_github_setup.md §2-2 のワンライナー)
              ↓
[完了] Phase 4 stella_reserve.py 実装 (Reserve-First 採番、bundleId 再計算、ロールバック)
     ├─ ✅ reserve_live_ids() / re_reserve() / clear_all_reservations()
     ├─ ✅ compute_bundle_id() (Phase 4-5) / build_live_list_entry() (Phase 5-E)
     ├─ ✅ plan_reservations() (純粋ロジック) / validate_for_reservation()
     ├─ ✅ Push 失敗時のスナップショット/ロールバック (github_ops.reset_hard 追加)
     └─ ✅ tests/backend_functions/test_stella_reserve.py (27 ケース)
              ↓
[完了] Phase 7-1 採番ボタン有効化 (stella_reserve から呼び出し)
     ├─ ✅ ProjectWorkflow.reserve_stella_live_ids() / clear_stella_reservations()
     ├─ ✅ ①画面 _render_stella_reserve_block() — 採番状態表示 + 採番/再採番ボタン
     ├─ ✅ 全リセットボタン (確認ダイアログ付き、Phase 4-4-4)
     └─ ✅ GITHUB_TOKEN 未設定時はボタン無効化 + ヘルプ表示
              ↓
[完了] Phase 7-2 ⑥-D Push UI 実装
     ├─ ✅ stella_push.py (push_stella_json / validate_for_push、PR・直接 push、Phase 6-6 ロールバック)
     ├─ ✅ github_ops.commit_and_push に remote_branch 追加 (PR フィーチャブランチ用)
     ├─ ✅ ProjectWorkflow.sync_stella_repo() / push_stella_json()
     ├─ ✅ 接続状態表示 (_render_stella_connection_status) / リポ同期 (pull) ボタン
     ├─ ✅ Push (PR 作成) ボタン / Push (直接) ボタン (確認ダイアログ付き)
     ├─ ✅ Push 前バリデーション表示 (未採番 / bundleId 未設定で①画面誘導)
     └─ ✅ tests/backend_functions/test_stella_push.py (11 ケース)
```

[2]〜[4] は Reserve-First 採番 → Push 失敗ロールバック → bundleId 再計算 が密結合のため、分割せず順に着手する。設計の主要事項（Reserve-First、`.env` 認証、ネスト git の clone 配置、bundleId 再計算ルール、Push 失敗ロールバック）は本計画書で確定済。残課題は [後回し領域の課題リスト](#後回し領域の課題リスト-phase-4--6--7-1--7-2-d) を参照。

---

## ファイル変更一覧

### 完了済（参考）

Phase 1 / 2 / 5 / 7-2 ⑥-A/C の実装で以下が完了している:

| ファイル | 状態 | 内容 |
|---|---|---|
| `src/backend_functions/stella_export.py` | ✅ 新規実装済 | `build_stella_json` / `write_stella_json` / `update_live_list` / `increment_versions_on_push` |
| `src/backend_functions/output_builder.py` | ✅ 実装済 | コラボ処理 / stageList 拡張カラム対応 |
| `src/backend_functions/timetabledata.py` | ✅ 実装済 | `コラボグループID` / `コラボタイトル` の json_to_df / df_to_json |
| `src/backend_functions/project_repository.py` | ✅ 一部実装済 | `_default_stella_metadata` / `get_stella_metadata` / `set_stella_metadata` / `ensure_stella_metadata` |
| `src/backend_functions/event_timetable_picture.py` | ✅ 実装済 | `make_stage_color_resolver` / `_default_color_resolver` |
| `src/frontend_functions/timetablepicture.py` | ✅ 実装済 | `create_timetable_image(text_color_in_box)` |
| `src/app.py` | ✅ 部分実装済 | ④コラボ自動検出 UI、⑥-A `_render_stella_metadata_form`、⑥-B `_render_stage_color_editor`、⑥-C `_render_stella_export_form` |
| `data/master/color_preset.json` | ✅ 新規作成済 | ステージカラープリセット (27色、`{name: [bg, fg]}`) |

### 残作業

| ファイル | 変更種別 | 内容 | 対応 Phase |
|---|---|---|---|
| `src/backend_functions/project_repository.py` | 修正 | `_default_stella_metadata` に `date` / `dow` 追加。`stella_project_meta` 用の `_default_stella_project_meta()` / `get_stella_project_meta()` / `ensure_stella_project_meta()` を新設。`create_project_data` でデフォルト埋め込み | Phase 3 |
| `src/app.py` | ✅ 採番ボタン実装済 | `_render_stella_reserve_block()` で採番状態表示 + 採番/再採番/全リセット (確認付き)。`ProjectWorkflow.reserve_stella_live_ids` / `clear_stella_reservations` を workflow.py に追加 | Phase 7-1 |
| `src/app.py` | ✅ Push UI 実装済 | `_render_stella_section()` に接続状態表示 (`_render_stella_connection_status`)・リポ同期・⑥-D Push UI (`_render_stella_push_form`) を追加 | Phase 7-2 ⑥-D |
| `src/backend_functions/stella_push.py` | ✅ 新規実装済 | `push_stella_json()` (単一) / `push_all_stella_json()` (全イベント一括: 1 コミット・1 PR)、`validate_for_push()`、version インクリメント連携、Phase 6-6 スナップショット/ロールバック。`ProjectWorkflow.sync_stella_repo` / `push_stella_json` / `push_all_stella_json` を workflow.py に追加 | Phase 7-2 ⑥-D |
| `src/backend_functions/github_ops.py` | 修正 | `commit_and_push()` に `remote_branch` 引数追加 (PR フィーチャブランチ push 用) | Phase 7-2 ⑥-D |
| `tests/backend_functions/test_stella_push.py` | **新規** | Push バリデーション / version / PR・直接 / ロールバックのテスト (11 ケース) | Phase 7-2 ⑥-D |
| `src/backend_functions/github_ops.py` | **新規** | `.env` の `GITHUB_TOKEN` 経由 PAT 認証、`https://x-access-token:{token}@...` URL 方式、`GitPython` で clone/pull/commit/push、`PyGithub` で PR 作成 | Phase 6 |
| `src/backend_functions/stella_reserve.py` | ✅ 新規実装済 | Reserve-First 採番ロジック (`reserve_live_ids()` / `re_reserve()` / `clear_all_reservations()`)、bundleId 算出 (`compute_bundle_id`)、`build_live_list_entry` (Phase 5-E)、Push 失敗時のスナップショット/ロールバック | Phase 4 |
| `src/backend_functions/github_ops.py` | 修正 | ロールバック用 `reset_hard()` 公開ヘルパを追加 | Phase 4 |
| `tests/backend_functions/test_stella_reserve.py` | **新規** | 純粋ロジック + 採番成功/push 失敗ロールバックのテスト (27 ケース) | Phase 4 |
| `data/master/pref_master.json` | **新規** | 都道府県マスタ (47都道府県 + 48:タイ + 49:台湾、`JSON生成シート.xlsm` の prefList から抽出) | Phase 4 |
| `.env` | **更新** | `GITHUB_TOKEN` / `GITHUB_USER_NAME` / `GITHUB_USER_EMAIL` を追加 | Phase 6 |
| `.gitignore` | **更新** | `data/timetableproj/` を追加 | Phase 6 |
| `requirements.txt` | **更新** | `GitPython` / `PyGithub` を追加。`python-dotenv` は既存 | Phase 6 |

---

## 参考: Stella JSONの構造（完全版）

```json
{
    "liveId": 547,
    "jsonVersion": 13,
    "openTime": "12",
    "closeTime": "23",
    "notificationVersion": "2",
    "notification": "更新メッセージ",
    "artList": [
        {"artId": 0, "name": "-天-"},
        ...
    ],
    "stageList": [
        {
            "stageId": 0,
            "stageName": "Zepp Shinjuku (TOKYO)",
            "stageNameShort": "Zepp Shinjuku (TOKYO)",
            "colorName": "stage-red",
            "stageOrder": 0
        },
        ...
    ],
    "turnList": [
        {"turnId": 0, "startTime": "13:00", "min": 15, "artId": 96, "stageId": 0},
        {"turnId": 281, "startTime": "20:10", "min": 60, "artId": 26, "stageId": 44,
         "collabArtList": [26, 55, 138]},
        {"turnId": 199, "startTime": "13:30", "min": 30, "artId": 44, "stageId": 29,
         "title": "星のカケラ(あみ・あや・さあや・つむ・みいみ)"},
        ...
    ]
}
```

### turnListの特殊フィールド
- `collabArtList`: コラボ公演時のみ存在。参加する全アーティストIDのリスト
- `title`: コラボタイトルが設定されている場合のみ存在。未設定時はクライアント側でアーティスト名を自動連結して表示

### liveList.jsonの構造

```json
{
    "liveList": [
        {
            "liveId": 547,
            "bundleId": "547",
            "liveName": "歌舞伎町UP GATE↑↑",
            "release": 2,
            "pref": 13,
            "date": "20260504",
            "dow": 1,
            "genre": 2,
            "serchKey": "0"
        },
        ...
    ]
}
```

### colorNameプリセット一覧（27色）

```
stage-red, stage-pink, stage-purple, stage-deep-purple, stage-indigo,
stage-blue, stage-blue2, stage-lightBlue, stage-cyan, stage-teal,
stage-green, stage-light-green, stage-light-green2, stage-lime,
stage-yellow, stage-amber, stage-orange, stage-deepOrange,
stage-brown, stage-brown2, stage-blueGrey, stage-grey, stage-black,
stage-white, stage-redGrey, stage-greenGrey, stage-yellowGrey
```

カスタム指定: `#背景色-#文字色`（例: `#EA749E-#FFFFFF`）

---

## 後回し領域の課題リスト (Phase 4 / 6 / 7-1 / 7-2 ⑥-D)

主要設計（Reserve-First 採番、bundleId 再計算、`.env` 認証、ネスト git 配置、Push 失敗ロールバック）は本計画書本文で確定済。着手前に追加で詰める残課題のみ以下に残す。

### 着手前の環境前提（2026-06 確認時点）

以下はすべて **未整備**。Phase 4 / 6 着手の最初のステップで対応する:

- `data/timetableproj/` への clone 用ディレクトリは未作成、`.gitignore` 未追記
- `data/master/pref_master.json` 未作成
- `src/backend_functions/github_ops.py` / `stella_reserve.py` 未作成
- 依存ライブラリ (`GitPython` / `PyGithub`) は `requirements.txt` に未追加。`python-dotenv` は既存
- `.env` への `GITHUB_TOKEN` 等の追加未実施
- 既存 `stella_export.py` の `update_live_list()` / `increment_versions_on_push()` の挙動が、本計画書の Reserve-First 採番フロー (Phase 4-4) および Push 失敗ロールバック (Phase 6-6) と整合するか要確認（既実装が Reserve-First 想定でない場合は修正）

### C. liveList 運用詳細
- **`update_live_list()` の更新セマンティクス**: liveId 一致なら上書き / 不一致なら追加。削除は許可しない方針で良いか
- **liveList のソート規約**: liveId 昇順 / date 昇順 / 既存ファイルの並びを踏襲のいずれか
- **`serchKey` の意味と既定値**: Stella 側仕様確認が必要。現状の `"0"` の根拠と、編集 UI 提供の要否

#### C-1. ✅【実装済】採番後の liveList フィールド変更を GitHub に反映する

> 📌 **2026-06 実装済**（「採番=リンク」モデルで対応）。

**事象（当初）**: `liveName` / `release` / `pref` / `genre` / `date` は `liveList.json` のみが保持するフィールド（`live{id}.json` には含まれない）。採番後にこれらを①画面で変えても `liveList.json` に反映されなかった。

**採用した方針（「採番＝GitHubとリンク」モデル）**:
1. **リンク判定**: `stella_reserve.is_linked()` = 採番済イベントが 1 つでもある状態。リンク前はローカル保存のみ、リンク後は liveList も追従。
2. **①「Stella連携設定を保存」で自動反映**: リンク済み + `GITHUB_TOKEN` ありなら、ローカル保存後に `resync_live_list()` を自動実行し、採番済全イベントの liveList エントリを現在値で作り直して **default ブランチへ直接 push**。差分が無ければ no-op。
3. **失敗時の扱い**: liveList 反映が失敗しても **ローカル保存は成功扱い**。「要再同期」警告を出し、①画面の「liveList を再同期」ボタンで手動リトライ（`project_info.json` が真実の源、liveList はその派生ミラー）。
4. **⑥-D Push 時の release 変更オプション**: `push_stella_json` / `push_all_stella_json` に `release_override` を追加。⑦の Push 直前に公開状態を変えたいケースで、release 変更を **live{id}.json の Push と同一コミット/PR に同梱**（liveList.json も一緒に更新）。UI は Stella セクションの「Push 時の公開状態 (release)」セレクタで指定。
5. ⑥-A「メタデータ保存」(openTime/closeTime/notification) は `live{id}.json` 側のフィールドで liveList には無いため、liveList 反映の対象外（自動追従は①保存のみに紐づく）。

**実装**:
- `stella_reserve.is_linked()` / `build_current_live_list_entries(release_override=)` / `resync_live_list()`
- `stella_push.push_stella_json` / `push_all_stella_json` の `release_override` 引数 + `_stage_release_livelist()`
- `ProjectWorkflow.resync_stella_live_list()`、push 系に `release_override` 引数
- app: ①保存の自動 resync + 「liveList を再同期」リトライ + 「要再同期」警告、Stella セクションの release セレクタ
- tests: `test_stella_reserve.py`（resync 系）/ `test_stella_push.py`（release_override 系）

**注**: 全リセット→再採番の回避策は liveId 振り直し＋旧エントリのゴミ化のため引き続き非推奨。再同期対象は「採番済全イベント」、既存と差分が無いエントリは push スキップ。

### D. GitHub 運用規約
- **ブランチ命名**: `stella/live{id}` / `stella/live{id}-v{jsonVersion}` 等
- **commit message テンプレ**: 採番 push 時 / 通常 push 時 / 再採番 (bundleId 更新) 時の各テンプレ
- **PR タイトル / 本文テンプレ**: 自動生成文言、差分要約の含め方
- **既存 open 中の PR と重複したとき**: force-push / 別ブランチ作成 / エラー停止のいずれか
- **dry-run / プレビューモード** の必要性

### E. Phase 5 関数分割（Phase 4 着手時に追加）
- `build_live_list_entry()` 等、liveList エントリ専用の生成関数を追加（`build_stella_json` は live{id}.json 専用）

### F. 既存システムとの接続
- **既存「クラウドにアップロード」(S3)** と Stella リポ反映の順序・関係整理
- **`event_timetable_picture` の S3 同期**: `master_stage.csv` 拡張列の S3 同期挙動を検証

### G. テスト戦略
- 既存 [`test_output_builder.py`](../../tests/backend_functions/test_output_builder.py) / [`test_output_editor.py`](../../tests/backend_functions/test_output_editor.py) のコラボ列・ステージマスタ拡張対応
- `stella_export.py` の単体テスト（既存 `live547.json` 等を fixture として等価出力を検証）
- GitHub 連携のテスト方針（`GitPython` モック / `responses` で REST API スタブ / dry-run）
- 採番 Reserve-First のレース条件テスト（pull → push の間に別ユーザーが push したシナリオ）

### H. ドキュメント更新
- [docs/data_structure.md](../data_structure.md): `stella_project_meta` / `stella_metadata` / `カラー名` / `ステージ名_短縮` / `コラボグループID` を追記
- [docs/workflow.md](../workflow.md): Stella 出力フロー (①画面採番 → ⑥編集 → Push) を追加

### J. 入力検証 (Phase 7 補足)
- `openTime` / `closeTime` の範囲 (0-23) と前後関係 (`openTime <= closeTime`)
- `liveName` 必須チェック（①画面の採番ボタン押下前）
- `date` の妥当性チェック（過去日付の警告など）
- `pref` / `genre` / `release` の必須選択（デフォルト値があるので実質バリデーション不要だが、明示確認の必要性検討）
