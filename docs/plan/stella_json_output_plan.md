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

### 1-1. ④読み取り後の表にカラム追加

OCR結果を表示・編集する④のDataFrameに以下のカラムを追加する:

| カラム名 | 型 | 説明 |
|---------|---|------|
| `コラボフラグ` | bool | True = 直上の出番とグルーピング（同一出番の一部） |
| `コラボタイトル` | str | コラボ公演の表示名（空なら自動でアーティスト名連結） |

#### 動作イメージ

```
出番  | グループ名     | from  | to    | コラボフラグ | コラボタイトル
------|---------------|-------|-------|------------|-------------
...   | KOURiN        | 20:10 | 21:10 | False      |
...   | SOMOSOMO      | 20:10 | 21:10 | True       |
...   | 虹色の飛行少女 | 20:10 | 21:10 | True       |
```

上記の場合、KOURiN + SOMOSOMO + 虹色の飛行少女が1つのコラボ出番となる。

### 1-2. グルーピング自動検出ボタン

④のUIに「コラボ出番を自動検出」ボタンを配置する。

**ロジック**:
- ステージごと（または全ステージ横断）に、同一`startTime`かつ同一ステージの出番を検出
- 2件目以降の出番の`コラボフラグ`をTrueに設定
- ユーザーが手動で修正も可能

### 1-3. ⑥出力時のコラボ処理

⑥のDataFrame構築時に、コラボフラグに基づいて処理する:

- コラボフラグ=Trueの行は独立した出番として出力 **しない**
- 代わりに、グループの先頭行（コラボフラグ=False）に以下を付与:
  - `コラボアーティストID`: コラボに参加する全アーティストのIDリスト（先頭含む）
  - `コラボタイトル`: ④で設定されたタイトル（未設定なら空）

⑥のDataFrame構造（追加カラム）:

| カラム名 | 型 | 説明 |
|---------|---|------|
| `コラボアーティストID` | str (JSON array) | `[26, 55, 138]`のようなリスト。非コラボならNull |
| `コラボタイトル` | str | コラボ出番の表示名。非コラボならNull |

### 1-4. IDマスタ確定後の④への反映

IDマスタ確定処理の後、④の各stage JSONにもコラボ情報を保存する:

```json
{
    "グループ名": "KOURiN",
    "グループ名_採用": "KOURiN",
    "ライブステージ": { "from": "20:10", "to": "21:10" },
    "備考": "",
    "コラボフラグ": false,
    "コラボアーティストID": [26, 55, 138],
    "コラボタイトル": ""
}
```

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
- ライブステージ: プリセットカラーを順番に割り当て（red, pink, purple, ...）
- 特典会ステージ: `stage-grey`
- 縁日ステージ: `stage-blueGrey`

UIではselectbox（プリセット選択）+ テキスト入力（カスタムカラーコード）の切り替えを想定。

### 2-3. stageOrder

ステージの表示順序。デフォルトはstageIdと同値。
ユーザーがUIで並び替え可能にする（ドラッグ or 数値入力）。

### 2-4. ステージマスタのDataFrame拡張

⑥のステージマスタDataFrame（現状は[output_builder.py:45](../../src/backend_functions/output_builder.py#L45)で構築、`stage_master[stage_id] = {"ステージ名": ..., "特典会フラグ": ...}` 形式）を以下に拡張:

| カラム名 | 型 | 説明 |
|---------|---|------|
| `ステージID` | int | (既存／index) |
| `ステージ名` | str | (既存) |
| `特典会フラグ` | bool | (既存) |
| `ステージ名_短縮` | str | stageNameShort |
| `カラー名` | str | colorName (プリセット名 or #hex-#hex) |
| `表示順` | int | stageOrder |

これに伴い、`find_or_create_stage_id()`([output_builder.py:42](../../src/backend_functions/output_builder.py#L42))で新規登録する辞書にも追加3フィールドのデフォルト値を入れる。

---

## Phase 3: トップレベルメタデータ

### 3-1. 管理対象フィールド

Stella JSONのトップレベルに必要なメタデータ:

| フィールド | 型 | 説明 | UI入力方法 |
|-----------|---|------|-----------|
| `liveId` | int | ライブID（GitHub上のliveListから採番） | 自動採番 or 手動入力 |
| `jsonVersion` | int | データバージョン（更新のたびにインクリメント） | 自動管理 |
| `openTime` | str | 開場時刻（時のみ、例: "12"） | 数値入力 |
| `closeTime` | str | 閉場時刻（時のみ、例: "23"） | 数値入力 |
| `notificationVersion` | str | 通知バージョン | 自動管理 |
| `notification` | str | 更新通知メッセージ | テキスト入力 |

### 3-2. 保存場所

`project_info.json`のevent_detail内に`stella_metadata`として保存:

```json
{
    "event_detail": [
        {
            "event_no": 0,
            "event_name": "event_1",
            "stella_metadata": {
                "liveId": 547,
                "jsonVersion": 1,
                "openTime": "12",
                "closeTime": "23",
                "notificationVersion": "1",
                "notification": ""
            },
            ...
        }
    ]
}
```

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

### 4-4. bundleIdの自動設定

- event_num > 1 のプロジェクトの場合、全イベントが同じbundleIdを共有
- bundleId = 初日イベントのliveId（文字列）
- event_num == 1 の場合、bundleId = "" (空文字)

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
3. **turnList構築**: live DataFrame → `[{"turnId": id, "startTime": ..., "min": ..., "artId": ..., "stageId": ..., ...}, ...]`
   - `コラボアーティストID`がある場合 → `"collabArtList": [...]`を追加
   - `コラボタイトル`がある場合 → `"title": "..."`を追加
4. **トップレベル**: stella_metadataからliveId, jsonVersion等を設定

### 5-2. 出力形式

- 1行JSON（minify）として出力（live547.jsonと同様）
- 文字コード: UTF-8 (BOMなし)
- 出力先: `data/projects/{pj_name}/event_{n}/live{liveId}.json`

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
    → [データ作成] → [JSON出力] → [commit] → [push / PR作成]
```

### 6-3. 実装する操作

| 操作 | 説明 | UI |
|------|------|----|
| Clone/Pull | リポジトリの最新状態を取得 | ボタン |
| liveId採番 | liveList.jsonの最大ID+1を自動取得 | 自動 |
| JSON出力 | live{id}.json + liveList.json更新をローカルに書き出し | ボタン |
| Push | 直接pushする場合 | ボタン（確認付き） |
| PR作成 | ブランチを作成しPRを送る場合 | ボタン |

### 6-4. 認証

- GitHubの認証情報はアプリ側で管理しない（git credentialに委譲）
- または環境変数`GITHUB_TOKEN`を使用

### 6-5. ローカルリポジトリパス

`project_info.json`にリポジトリパスを保存するか、アプリ設定で一元管理:

```json
{
    "stella_repo_path": "C:/Users/.../timetableproj"
}
```

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
- stageNameShort編集
- colorName選択（プリセット or カスタム）
- stageOrder設定
- 編集対象は `output_df[event_name]["stage"]` の拡張カラム（Phase 2-4）

**⑥-C: Stella JSON出力**
- 「Stella JSONを生成」ボタン → `stella_export.build_stella_json()` 呼び出し
- プレビュー表示
- ダウンロードボタン

**⑥-D: GitHub連携**
- リポ同期ボタン
- liveId確認
- Push / PR作成ボタン

なお、⑥セクションが長大化するため、必要に応じて `render_output_section` を細分割（例: `_render_stella_metadata_form()` 等のヘルパー関数化）することも検討する。

---

## 実装順序

```
Phase 1 (コラボ) ← 既存の④⑥に影響するため最初に着手
    ↓
Phase 2 (stageList拡張) ← ⑥のDF拡張
    ↓
Phase 3 (メタデータ) ← project_info.json拡張
    ↓
Phase 4 (liveList) ← 新規データ構造
    ↓
Phase 5 (JSON出力) ← Phase 1-4が揃って初めて完全な変換が可能
    ↓
Phase 6 (GitHub連携) ← Phase 5の出力をpushする
    ↓
Phase 7 (UI統合) ← 各Phaseと並行して段階的に実装
```

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
| `src/backend_functions/timetabledata.py` | 修正 | コラボフラグの `json_to_df` / `df_to_json` 対応 |
| `src/backend_functions/project_repository.py` | 修正 | `create_project_data`([:226](../../src/backend_functions/project_repository.py#L226))の `event_detail` スキーマに `stella_metadata` を追加 |
| `data/master/pref_master.json` | **新規** | 都道府県マスタ |
| `data/master/color_preset.json` | **新規** | ステージカラープリセット一覧 |

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
