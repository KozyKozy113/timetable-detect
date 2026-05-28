# Stella JSON出力対応 実装計画

## 概要

本アプリケーションで作成したタイムテーブルデータを、Stellaアプリ向けのJSON形式（`live{id}.json` / `liveList.json`）で出力し、GitHubリポジトリ（https://github.com/ys0512/timetableproj）へpush/PRを送るまでの一連の機能を実装する。

---

## 現状と目標の差分サマリ

### 既にできていること
- ライブ + 特典会 + 縁日を横断統合したステージマスタ・アーティストマスタ・出番データの構築（⑥）
- Excel出力
- ステージID / グループID / 出番ID の採番

### 不足していること

| # | 項目 | 内容 |
|---|------|------|
| 1 | コラボ公演対応 | collabArtList / コラボタイトル(title)の管理 |
| 2 | stageList拡張フィールド | stageNameShort, colorName, stageOrder |
| 3 | トップレベルメタデータ | liveId, jsonVersion, openTime, closeTime, notification等 |
| 4 | liveList.json管理 | ライブ一覧（公開状態・都道府県・日付等）の管理 |
| 5 | Stella JSON出力関数 | live{id}.json形式への変換・書き出し |
| 6 | GitHub連携 | リポ取得 → ID採番 → push/PR |

---

## Phase 1: コラボ公演対応

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

### 3-1. 管理対象フィールド

Stella JSONのトップレベルに必要なメタデータ:

| フィールド | 型 | 説明 | UI入力方法 |
|-----------|---|------|-----------|
| `liveId` | int | ライブID（GitHub上のliveListから採番） | **Phase 4で確定（本Phaseでは欠損キーとして保留）** |
| `jsonVersion` | int | データバージョン | **Push成功のたび +1**。初回Pushで `1`（Push実装は[Phase 6](#phase-6-github連携)で対応） |
| `openTime` | str | 開場時刻（時のみ、例: "12"） | 数値入力 |
| `closeTime` | str | 閉場時刻（時のみ、例: "23"） | 数値入力 |
| `notificationVersion` | str | 通知バージョン | **`notification` フィールドが変更されたPush時のみ +1**。初回は `"1"` |
| `notification` | str | 更新通知メッセージ | テキスト入力 |

#### バージョンのインクリメント実装

Push直前に以下を実行:

1. `jsonVersion` を `+1`
2. 直近Push時の `notification` 値と現在値を比較し、変わっていれば `notificationVersion` を `+1`
3. Push成功後に「直近Push時の notification 値」を `stella_metadata` の隠しフィールド（`_last_pushed_notification`）として保存しておく

### 3-2. 保存場所

`project_info.json`のevent_detail内に`stella_metadata`として保存。

本Phaseの時点では `liveId` / `bundleId` は **キーごと欠損** とし、Phase 4 で初めてセットする。`jsonVersion` も Push成功時に初めて `1` となるため、初期状態では欠損キーで構わない。

```json
{
    "event_detail": [
        {
            "event_no": 0,
            "event_name": "event_1",
            "stella_metadata": {
                "openTime": "12",
                "closeTime": "23",
                "notificationVersion": "1",
                "notification": "",
                "_last_pushed_notification": null
            },
            ...
        }
    ]
}
```

`_last_pushed_notification` は notificationVersion 自動管理用の内部フィールド。

#### 既存 `project_info.json` の後方互換

既存プロジェクトの `project_info.json` には `stella_metadata` ブロックが無い。読み込み時に **空のメタデータで補完する**:

- 補完値: `{"openTime": "", "closeTime": "", "notificationVersion": "1", "notification": "", "_last_pushed_notification": null}`
- 補完したファイルは次回保存時に書き戻される（明示的なマイグレーションスクリプトは不要）

⑥-A のフォーム表示時、空文字のフィールドは未入力プレースホルダで表示する。

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

| フィールド | 入力方式 | 選択肢 |
|-----------|---------|--------|
| `liveId` | 自動採番（既存liveListの最大ID + 1） | - |
| `bundleId` | 自動（複数日イベントの初日liveIdを文字列化） | - |
| `liveName` | テキスト入力 | - |
| `release` | selectbox | 0: 非公開 / 1: 公開(非アクティブ) / 2: 公開 |
| `pref` | selectbox | 都道府県マスタ（47都道府県 + タイ(48)） |
| `date` | 日付入力 → YYYYMMDD形式に変換 | - |
| `dow` | dateから自動算出（1=月〜7=日） | - |
| `genre` | selectbox | 1: ロック / 2: アイドル |
| `serchKey` | テキスト入力（デフォルト "0"） | - |

### 4-3. 都道府県マスタ

`data/master/pref_master.json`として保存:

```json
[
    {"code": 1, "name": "北海道"},
    {"code": 2, "name": "青森県"},
    ...
    {"code": 13, "name": "東京都"},
    ...
    {"code": 47, "name": "沖縄県"},
    {"code": 48, "name": "タイ"}
]
```

### 4-4. liveId採番ワークフロー

- **プロジェクト編集開始時**（イベント単位）: GitHub から `liveList.json` を `git pull` し、`max(liveId) + 1` を採番 → `stella_metadata.liveId` に書き込み、以降は不変
- **既存ライブの再編集**: 既に `stella_metadata.liveId` が設定されている場合は採番を行わず、その liveId を保持。Push時に `jsonVersion + 1` で同じ `live{id}.json` を上書き
- 同時編集による衝突を避けたい場合は、編集開始時にもう一度 `pull` してから最大値を確認する。実装上は「編集セッション開始ボタン」で明示的に pull → 採番を行う

### 4-5. bundleIdの自動設定

- **event_num == 1**: `bundleId = ""` （空文字）
- **event_num > 1**: 同一プロジェクト内の全イベントが同じ `bundleId` を共有。値はプロジェクト内 **最初に liveId を採番したイベントの liveId**（文字列化）とし、**一度確定したら不変** とする
- 「初日でなければならない」という制約は設けない。bundleId はあくまで複数イベントをまとめる識別子として機能すればよく、確定後は不変であることを優先する
- 採番ロジック: あるイベントで liveId 採番時、同プロジェクトの他イベントに既に `bundleId` が設定済なら、その値を採用。未設定なら自身の `liveId` を `bundleId` として全イベントへ書き込み

---

## Phase 5: Stella JSON出力関数

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
- 文字コード: UTF-8 (BOMなし)
- **一次出力先**: `data/projects/{pj_name}/event_{n}/live{liveId}.json`（本アプリ管理下）
- **GitHub反映時**: 一次出力先のファイルを `stella_repo_path` 配下にコピーしてから commit/push

> 本アプリ側にもファイルを残すことで、GitHub Push前後の差分確認や、リポ未取得状態でも成果物が手元に残るメリットがある。

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

### 6-4. 認証 (Streamlit Cloud 前提)

Streamlit Cloud 環境では git credential helper や ssh key の永続化が使えないため、**GitHub Personal Access Token (PAT) を Streamlit Secrets 経由で渡す方式** を採用する。

#### 6-4-1. トークン管理

優先順位:

1. `st.secrets["github"]["token"]` — Streamlit Cloud 本番環境での主経路
2. 環境変数 `GITHUB_TOKEN` — ローカル開発時のフォールバック
3. いずれも無い場合: UI上で「GitHub連携が利用できません」と表示し、JSON出力までで停止

`.streamlit/secrets.toml` のフォーマット:

```toml
[github]
token = "github_pat_xxxxxxxx..."
user_name = "timetable-detect-bot"      # commit author 表示名
user_email = "bot@example.com"          # commit author email
```

#### 6-4-2. トークンスコープ

ys0512/timetableproj への push / PR 作成権限が必要:

- Classic PAT: `repo` スコープ
- Fine-grained PAT: `ys0512/timetableproj` を対象に `Contents: Read and write` + `Pull requests: Read and write`

> セキュリティ: PAT はリポジトリオーナーが発行・管理し、Streamlit Cloud のSecrets画面でのみ登録する。コードや `project_info.json` にトークンを含めない。

#### 6-4-3. git 操作の実装方針

- 認証付きリモートURL: `https://x-access-token:{token}@github.com/ys0512/timetableproj.git` を使う方式に統一（HTTPSベース、Streamlit Cloud互換）
- PR作成は `PyGithub` ライブラリで REST API 経由（git CLI不要）
- clone/pull/push は `GitPython` または `subprocess` 経由の `git` CLI

### 6-5. ローカルリポジトリパス (アプリ全体設定で一元管理)

リポジトリは1つしかないため、**プロジェクトごとではなくアプリ全体の設定として一元管理** する。

#### 設定ファイル: `data/master/stella_config.json` (新設)

```json
{
    "stella_repo_url": "https://github.com/ys0512/timetableproj.git",
    "stella_repo_local_path": "/tmp/timetableproj",
    "default_branch": "main",
    "pr_base_branch": "main"
}
```

#### Streamlit Cloud 上の挙動

- Streamlit Cloud のファイルシステムはセッションを跨ぐと揮発する可能性があるため、`stella_repo_local_path` は `/tmp/timetableproj` 等の書き込み可能パスを使用
- `@st.cache_resource` を使い、アプリインスタンスごとに1回だけ clone を実行、以降は pull で更新
- ローカル開発時は `stella_config.json` の `stella_repo_local_path` を上書きして手元のパス（例: `C:/Users/.../timetableproj`）を指定できるようにする（`.streamlit/secrets.toml` 経由の上書きも可）

---

## Phase 7: UI統合

### 7-0. 現状の `render_output_section` 構造（前提）

[refactoring_render_output_section.md](refactoring_render_output_section.md) によるリファクタ後、`render_output_section()`([app.py:1284](../../src/app.py#L1284)) は約75行のUI描画関数になっており、データ組み立ては `_output.build_all_event_outputs()` に集約されている。現状の構造:

```
render_output_section()
├── build_all_event_outputs() でデータ取得
├── event_tabs ループ
│   ├── stage / idolname / live の3カラム表示
│   └── 集計情報（duration_distribution / group_count / overlap_alerts / group_appearances）
├── 「IDマスタを確定」ボタン
├── 「クラウドにアップロード」ボタン
└── 「Excelデータを出力」ボタン
```

Stella対応もこのリファクタ方針（UI描画は薄く保ち、データ組み立ては backend に寄せる）に合わせる。すなわち、新規追加するUIブロックも対応するデータ組み立て関数は `output_builder.py` / `stella_export.py` に置き、`render_output_section` 側はフォーム入力と関数呼び出しのみ持つ。

### 7-1. 新設するUIセクション

`render_output_section` の末尾（Excel出力ボタンの後）に以下のサブセクションを追加:

**⑥-A: Stellaメタデータ入力**
- openTime / closeTime
- liveName
- release / pref / genre
- notification
- 入力値は `app_state.output.stella_metadata`（Phase 3-2で追加）に保持し、保存時に `project_info.json` に書き戻す

**⑥-B: ステージ詳細設定**
- stageNameShort編集（テキスト入力）
- colorName選択（[Phase 2-2-1](#2-2-1-ui-実カラー表示を伴うカラー選択) のカラーパレット/カラーピッカーUI）
- stageOrder は [既存のD&D並び替えUI](../../src/app.py#L1568-L1615) ([Phase 2-3](#2-3-stageorder)) を流用するため、本セクションでは触らない
- 編集対象は `output_df[event_name]["stage"]` の拡張カラム（Phase 2-4）

**⑥-C: Stella JSON出力**
- 「Stella JSONを生成」ボタン → `stella_export.build_stella_json()` 呼び出し
- プレビュー表示
- ダウンロードボタン

**⑥-D: GitHub連携**
- 接続状態表示（PAT検出済か / clone済か）
- リポ同期ボタン (clone or pull)
- liveId確認（採番済 / 未採番、再編集時は既存ID表示）
- Push (PR作成) ボタン — **初期推奨**
- Push (直接) ボタン — 確認ダイアログ付き
- 認証情報が無い環境ではPush系ボタンは無効化し、JSON生成・ダウンロードのみ可能とする

なお、⑥セクションが長大化するため、必要に応じて `render_output_section` を細分割（例: `_render_stella_metadata_form()` 等のヘルパー関数化）することも検討する。

---

## 実装順序

### 先行実装（本計画書での着手対象）

```
Phase 1 (コラボ) ← 既存の④⑥に影響するため最初に着手
    ↓
Phase 2 (stageList拡張) ← ⑥のDF拡張
    ↓
Phase 3 (メタデータ) ← project_info.json拡張（liveId/bundleIdは欠損キーのまま）
    ↓
Phase 5 (JSON出力) ← liveId欠損のままでも live{id}.json 構造を組み立てる関数として実装可能
    ↓
Phase 7 (UI統合) — ⑥-A / ⑥-B / ⑥-C を中心に。⑥-D は後回し
```

### 後回し（GitHub連携を絡めて別途戦略検討）

```
Phase 4 (liveList) — liveId 採番・bundleId・liveList.json 更新
Phase 6 (GitHub連携) — clone/pull/push/PR
Phase 7 ⑥-D (GitHub連携UI)
```

Phase 4 / 6 は GitHub 連携の設計が絡むため、本計画書とは別タイミングで運用方針（commit/PR規約、認証、liveList更新セマンティクス等）を決めてから着手する。残課題は [後回し領域の課題リスト](#後回し領域の課題リスト-phase-4--6) を参照。

---

## ファイル変更一覧（予定）

| ファイル | 変更種別 | 内容 |
|---------|---------|------|
| `src/app_state.py` | 修正 | 現状の `OutputState` は `output_df` / `new_idolname` の2フィールド([app_state.py:55-59](../../src/app_state.py#L55-L59))。ここに `stella_metadata: dict` を追加 |
| `src/app.py` | 修正 | ④にコラボカラム追加、`render_output_section`([app.py:1284](../../src/app.py#L1284))にStella出力UI（⑥-A〜D）追加 |
| `src/workflow.py` | 修正 | `OutputWorkflow`([workflow.py:489](../../src/workflow.py#L489))にStellaエクスポート系メソッド追加 |
| `src/backend_functions/stella_export.py` | **新規** | Stella JSON変換・出力ロジック |
| `src/backend_functions/github_ops.py` | **新規** | GitHub連携操作 |
| `src/backend_functions/output_builder.py` | 修正 | `find_or_create_stage_id`([:42](../../src/backend_functions/output_builder.py#L42))/`build_event_output`([:321](../../src/backend_functions/output_builder.py#L321))にコラボ処理・stageList拡張カラム対応を追加 |
| `src/backend_functions/timetabledata.py` | 修正 | `コラボグループID` / `コラボタイトル` の `json_to_df` / `df_to_json` 対応 |
| `src/backend_functions/project_repository.py` | 修正 | `create_project_data`([:226](../../src/backend_functions/project_repository.py#L226))の `event_detail` スキーマに `stella_metadata` を追加 |
| `data/master/color_preset.json` | **新規** | ステージカラープリセット一覧 |

### 後回し領域のファイル変更（Phase 4 / 6 着手時）

| ファイル | 変更種別 | 内容 |
|---------|---------|------|
| `src/backend_functions/github_ops.py` | **新規** | clone/pull/push/PR作成。`st.secrets` / `GITHUB_TOKEN` 経由のPAT認証、URL埋め込み方式、PyGithub での PR作成 |
| `data/master/pref_master.json` | **新規** | 都道府県マスタ |
| `data/master/stella_config.json` | **新規** | `stella_repo_url` / `stella_repo_local_path` / `default_branch` / `pr_base_branch` |
| `.streamlit/secrets.toml` | **更新** | `[github] token / user_name / user_email` を追加（Streamlit Cloud側で設定、リポにはコミットしない） |

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

## 後回し領域の課題リスト (Phase 4 / 6)

Phase 4 (liveList) と Phase 6 (GitHub連携) は GitHub 連携運用と一体で設計するため、本計画書では着手しない。着手前に以下を別途決める。

### C. liveList / bundleId 運用詳細
- **`update_live_list()` の更新セマンティクス**: liveId一致なら上書き / 不一致なら追加。削除は許可するか?
- **liveList のソート規約**: liveId昇順 / date昇順 / 既存ファイルの並びを踏襲のいずれか
- **`serchKey` の意味と既定値**: Stella側仕様の確認が必要。現状の `"0"` の根拠
- **bundleId 確定タイミング**: event_num > 1 の場合、いつどのイベントが先に `liveId` を採番するかで bundleId 値が変わる。プロジェクト作成時に確定させるか、最初の Stella 出力時に確定させるか
- **同時編集衝突**: 複数ユーザーが同時に liveId を採番しに行った場合のレース対応

### D. GitHub 運用規約
- **ブランチ命名**: `stella/live{id}` / `stella/live{id}-v{jsonVersion}` 等
- **commit message テンプレ**: `[Stella] Add live547: {liveName} ({date})` 等
- **PR タイトル / 本文テンプレ**: 自動生成する文言、差分要約の含め方
- **既存open中のPR と重複したとき**: force-push / 別ブランチ作成 / エラー停止
- **Push失敗時のロールバック**: `jsonVersion` のインクリメントタイミングと巻き戻し
- **dry-run / プレビューモード** の必要性

### E. Phase 5 関数分割（Phase 4 着手時に追加）
- `build_live_list_entry()` 等、liveListエントリ専用の生成関数を追加（`build_stella_json` は live{id}.json 専用）

### F. 既存システムとの接続
- **既存「クラウドにアップロード」(S3)** と Stella リポ反映の順序・関係整理
- **`event_timetable_picture` の S3 同期**: `master_stage.csv` 拡張列の S3 同期挙動を検証

### G. テスト戦略
- 既存 [`test_output_builder.py`](../../tests/backend_functions/test_output_builder.py) / [`test_output_editor.py`](../../tests/backend_functions/test_output_editor.py) のコラボ列・ステージマスタ拡張対応
- `stella_export.py` の単体テスト（既存 `live547.json` 等を fixture として等価出力を検証）
- GitHub連携のテスト方針（モック / `responses` / dry-run）

### H. ドキュメント更新
- [docs/data_structure.md](../data_structure.md): `stella_metadata` / `カラー名` / `ステージ名_短縮` / `コラボグループID` を追記
- [docs/workflow.md](../workflow.md): Stella出力フローを追加

### J. 入力検証 (Phase 7 補足)
- `openTime` / `closeTime` の範囲 (0-23) と前後関係 (`openTime <= closeTime`)
- `liveName` 必須チェック
- `date` の妥当性チェック
- `pref` / `genre` / `release` の必須選択
