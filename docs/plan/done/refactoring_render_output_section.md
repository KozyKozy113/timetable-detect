# `render_output_section` リファクタリング実装計画

## 概要

`app.py` の `render_output_section()`（L1099〜L1232, 134行）は、
**データ組み立てロジック**と**UI描画コード**が一体化しており、
テスト不可・可読性が低い状態にある。

本計画では、Streamlit非依存のデータ組み立て処理を `output_builder.py` に抽出し、
`render_output_section()` をUI描画のみの薄い関数にする。

### 対象ファイル

| ファイル | 変更内容 |
|---------|---------|
| `src/backend_functions/output_builder.py` | データ組み立て関数を追加 |
| `src/app.py` | `render_output_section()` をUI描画のみに書き換え |
| `src/workflow.py` | `OutputWorkflow` にデータ組み立てメソッドを追加 |

### 変更しないファイル

- `app_state.py` — `OutputState` の構造は変更不要
- `timetabledata.py` — `json_to_df`, `devide_df_live_tokutenkai` はそのまま利用
- `output_builder.py` の既存関数 — `determine_id_master`, `export_excel`, `listup_new_idolname` 等はそのまま

---

## 1. 現状の処理フロー分析

### 1.1 関数の全体構造

```
render_output_section()  # 134行, 最大4段ネスト
├── イベントリスト取得
├── output_df を空dictで初期化
├── イベントタブ ループ (event_tabs)
│   ├── [A] 既存マスタ読み込み (master_stage.csv, master_idolname.csv)
│   ├── [B] ステージマスタ構築 + タイムテーブル集約 ループ (event_type × stage)
│   │   ├── stage JSON読み込み → json_to_df
│   │   ├── 「特典会併記」分岐 → devide_df_live_tokutenkai
│   │   ├── 空行除去
│   │   └── ステージID割り当て (既存or新規)
│   ├── [C] 特典会ブース → ステージマスタへの統合
│   ├── [D] DataFrame化 (df_stage, df_stage_tokutenkai)
│   ├── [E] アーティストマスタ構築 (df_idolname)
│   ├── [F] 出番データ構築 (df_live) + 出番ID割り当て
│   └── [G] UI表示 (st.columns + st.dataframe) + state格納
├── 「IDマスタを確定」ボタン
├── 「S3アップロード」ボタン
└── 「Excelデータを出力」ボタン + ダウンロード
```

### 1.2 「IDマスタを確定」前後の挙動差

この関数の隠れた重要仕様として、`determine_id_master()` 実行前後で挙動が変わる。

| 状態 | master_stage.csv | master_idolname.csv | 出番ID列 | 挙動 |
|------|-----------------|--------------------|---------|----- |
| **確定前** | 存在しない | 存在しない | JSON内にない | stage/artist IDを0から採番。出番IDは連番インデックス |
| **確定後** | 存在する | 存在する | JSON内にある | 既存IDを引き継ぎ、新規分のみ追加採番。出番IDも既存維持 |

この差異は **CSVファイルの有無** と **JSONデータ内のID列の有無** で暗黙的に分岐している。
リファクタリング後もこの挙動を正確に再現する必要がある。

### 1.3 問題点の整理

| # | 問題 | 箇所 | 影響 |
|---|------|------|------|
| 1 | データ組み立てロジック(A〜F)がUI関数内に直書き | L1108〜L1214 | テスト不可、再利用不可 |
| 2 | 4段ネスト（event_tab → event_type → stage → for-else） | L1108〜L1164 | 可読性が極めて低い |
| 3 | ステージID検索が `for...else` パターンで2箇所重複 | L1149-1156, L1171-1178 | 同一ロジックの重複 |
| 4 | 「IDマスタ確定」前後の分岐が暗黙的 | L1112-1126 | CSVの有無で挙動が変わるが明示的でない |
| 5 | 変数スコープが広い | L1115-1128 | stage_master等がイベントループ全体にまたがる |

---

## 2. リファクタリング設計

### 2.1 抽出する関数の一覧

`output_builder.py` に以下の関数を追加する。

```
output_builder.py (追加分)
├── find_or_create_stage_id()      # ステージID検索の共通化
├── load_existing_masters()        # 既存マスタ読み込み
├── build_stage_master()           # ステージマスタ構築
├── build_artist_master()          # アーティストマスタ構築
├── build_turn_data()              # 出番データ構築
├── build_event_output()           # 1イベント分の全データ組み立て (上記を統合)
└── build_all_event_outputs()      # 全イベント一括処理
```

### 2.2 各関数の設計

#### `find_or_create_stage_id(stage_master, stage_name, is_tokutenkai, next_id) -> (id, next_id)`

L1149-1156 と L1171-1178 の重複 `for...else` パターンを統合。

```python
def find_or_create_stage_id(
    stage_master: dict,
    stage_name: str,
    is_tokutenkai: bool,
    next_id: int,
) -> tuple[int, int]:
    """既存マスタからステージIDを探索し、なければ新規採番する。

    Returns:
        (stage_id, next_id): 見つかった or 新規採番したID と、次に使うべきID
    """
    for k, v in stage_master.items():
        if v["ステージ名"] == stage_name:
            return k, next_id
    stage_master[next_id] = {"ステージ名": stage_name, "特典会フラグ": is_tokutenkai}
    return next_id, next_id + 1
```

#### `load_existing_masters(output_path) -> (stage_master, next_stage_id, idolname_master_df, next_artist_id)`

L1112〜L1126 を抽出。「IDマスタ確定」前後の差異がこの関数の戻り値で表現される。

```python
def load_existing_masters(
    output_path: str,
) -> tuple[dict, int, pd.DataFrame, int]:
    """既存のステージマスタ・アイドル名マスタを読み込む。

    master_stage.csv / master_idolname.csv が存在すれば読み込み、
    存在しなければ空の初期値を返す。
    これにより「IDマスタ確定」前後の挙動差を吸収する。

    Returns:
        (stage_master, next_stage_id, idolname_master_df, next_artist_id)
    """
```

#### `build_event_output(pj_path, event_name, event_no, project_info_json) -> dict | None`

L1108〜L1224（1イベント分の全ロジック）を集約するメイン関数。
内部で上記のヘルパー関数を使う。

```python
def build_event_output(
    pj_path: str,
    event_name: str,
    event_no: int,
    project_info_json: dict,
) -> dict[str, pd.DataFrame] | None:
    """1イベント分の output_df（stage / idolname / live）を組み立てて返す。

    データが存在しない場合は None を返す。

    処理フロー:
        1. load_existing_masters() で既存マスタ読み込み
        2. 全event_type × stageを走査し、ステージマスタとライブデータを構築
        3. 特典会データがあれば統合
        4. アーティストマスタを構築
        5. 出番データを構築（出番ID割り当て含む）

    Returns:
        {"stage": df_stage, "idolname": df_idolname, "live": df_live} or None
    """
```

#### `build_all_event_outputs(pj_path, project_info_json) -> dict`

全イベントを一括で処理する最上位関数。

```python
def build_all_event_outputs(
    pj_path: str,
    project_info_json: dict,
) -> dict[str, dict[str, pd.DataFrame]]:
    """全イベントの output_df を組み立てて返す。"""
```

### 2.3 リファクタリング後の `render_output_section()`

```python
def render_output_section():
    """⑥タイムテーブル情報の出力"""
    st.markdown("#### ⑥タイムテーブル情報の出力")

    event_list = get_event_name_list()
    app_state.output.output_df = _output.build_all_event_outputs(
        app_state.project.pj_path,
        app_state.project.project_info_json,
    )

    event_tabs = st.tabs(event_list)
    for event_name, event_tab in zip(event_list, event_tabs):
        if event_name not in app_state.output.output_df:
            continue
        with event_tab:
            data = app_state.output.output_df[event_name]
            output_cols = st.columns([1, 1, 3])
            with output_cols[0]:
                st.dataframe(data["stage"])
            with output_cols[1]:
                st.dataframe(data["idolname"])
            with output_cols[2]:
                st.dataframe(data["live"])

    st.button("IDマスタを確定", on_click=determine_id_master)
    st.button("プロジェクトデータをクラウドにアップロード ※通信料・保存料が発生するので留意",
              on_click=save_to_s3)
    if st.button("Excelデータを出力", on_click=output_data_for_stella):
        file_path = os.path.join(app_state.project.pj_path, "output.xlsx")
        with open(file_path, "rb") as file:
            excel_data = file.read()
        st.download_button(
            "ファイルをダウンロード",
            data=excel_data,
            file_name=f"{app_state.project.pj_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
```

**134行 → 約30行** に削減される。

### 2.4 `workflow.py` の変更

`OutputWorkflow` に `build_all_events` メソッドを追加。
ただし現在の `render_output_section` はコールバック経由ではなく描画時に直接データを組み立てているため、
ワークフロー層を経由するのではなく `_output.build_all_event_outputs()` を直接呼ぶ形でよい。

既存の `OutputWorkflow.determine_id_master()` 等のメソッドは変更不要。
これらは `app_state.output.output_df` を入力として使用しており、
`output_df` の生成方法が変わっても出力側のインターフェースは同じ。

---

## 3. 実装ステップ

### Step 1: `find_or_create_stage_id` の追加

**ファイル**: `src/backend_functions/output_builder.py`

重複している `for...else` パターン（L1149-1156, L1171-1178）を1つの関数に統合する。

**テスト観点**:
- 既存マスタにステージ名がある場合 → 既存IDを返す、next_idは変化しない
- 既存マスタにステージ名がない場合 → next_idを返す、マスタに追加される、next_idがインクリメント

### Step 2: `load_existing_masters` の追加

**ファイル**: `src/backend_functions/output_builder.py`

L1112〜L1126 のCSV読み込みロジックを抽出する。

**テスト観点**:
- CSVが存在する場合（IDマスタ確定後）→ 既存データとmax+1のIDを返す
- CSVが存在しない場合（IDマスタ確定前）→ 空dictと0を返す

### Step 3: `build_event_output` の実装

**ファイル**: `src/backend_functions/output_builder.py`

L1108〜L1224 の全ロジックを移植する。以下のサブ処理を含む:

1. `load_existing_masters()` で既存マスタ読み込み
2. event_type × stage ループ:
   - stage JSON読み込み → `timetabledata.json_to_df()`
   - 「特典会併記」形式の分離 → `timetabledata.devide_df_live_tokutenkai()`
   - 空行フィルタリング
   - `find_or_create_stage_id()` でステージID割り当て
3. 特典会ブース → ステージマスタ統合
4. ステージマスタ DataFrame化
5. アーティストマスタ構築（ユニーク名抽出 + 既存との結合）
6. 出番データ構築（アーティストIDマージ + 出番ID割り当て）
7. 表示用カラム選択

**注意点**:
- `project_repository` の関数を使ってイベント情報を取得する
- `timetabledata` の関数はそのまま利用する
- 「IDマスタ確定」前後の挙動差を `load_existing_masters()` の戻り値で吸収する

**テスト観点**:
- IDマスタ確定前: 全IDが0から連番で採番される
- IDマスタ確定後: 既存IDが維持され、新規分のみ追加採番される
- 特典会併記形式: ライブと特典会が正しく分離・統合される
- データなし: None が返る

### Step 4: `build_all_event_outputs` の実装

**ファイル**: `src/backend_functions/output_builder.py`

全イベントに対して `build_event_output()` をループで呼び出す薄いラッパー。

```python
def build_all_event_outputs(pj_path, project_info_json):
    event_list = repo.get_event_name_list(project_info_json)
    result = {}
    for event_name in event_list:
        event_no = repo.get_event_no_by_event_name(project_info_json, event_name)
        data = build_event_output(pj_path, event_name, event_no, project_info_json)
        if data is not None:
            result[event_name] = data
    return result
```

### Step 5: `render_output_section` の書き換え

**ファイル**: `src/app.py`

現在の134行のインライン実装を、Step 4の関数を呼び出す約30行のUI描画コードに置き換える。

**確認事項**:
- `app_state.output.output_df` の構造が従来と同一であること
  (`{event_name: {"stage": df, "idolname": df, "live": df}}`)
- `determine_id_master()` が `output_df` を正しく読み取れること
- `export_excel()` が `output_df` を正しく読み取れること
- `listup_new_idolname()` が `output_df` を正しく読み取れること

---

## 4. output_df のデータ構造（変更なし）

リファクタリング前後で `app_state.output.output_df` の構造は変わらない。
後続の `determine_id_master`, `export_excel`, `listup_new_idolname` がこの構造に依存しているため。

```python
output_df = {
    "イベント名A": {
        "stage": pd.DataFrame(
            columns=["ステージ名", "特典会フラグ"],
            index_name="ステージID"
        ),
        "idolname": pd.DataFrame(
            columns=["グループ名_採用"],
            index_name="グループID"
        ),
        "live": pd.DataFrame(
            columns=["ライブ_from", "ライブ_長さ(分)", "グループID",
                     "ステージID", "グループ名_raw", "グループ名",
                     "ステージ名", "備考"],
            index_name="出番ID"
        ),
    },
    "イベント名B": { ... },
}
```

---

## 5. リスクと対策

| リスク | 対策 |
|-------|------|
| `build_event_output` の出力が既存実装と微妙に異なる | 実データで既存実装の出力と差分比較する |
| `determine_id_master` が `output_df` の特定のカラム順に依存している | `determine_id_master` のコードを事前に確認済み。カラム名ベースのアクセスのため順序無関係 |
| 特典会併記形式の複雑な分岐を移植時に壊す | 特典会併記・通常形式それぞれのテストケースで検証 |
| `for...else` の `else` 節（Python独特の構文）を誤変換 | `find_or_create_stage_id` に統一することで `for...else` を排除 |

---

## 6. 効果見積もり

| 指標 | Before | After |
|------|--------|-------|
| `render_output_section` の行数 | 134行 | ~30行 |
| データロジックのテスト可能性 | 不可（Streamlit依存） | 可能（純粋関数） |
| ネストの深さ | 最大4段 | 最大2段 |
| `for...else` の重複 | 2箇所 | 0箇所（`find_or_create_stage_id` に統一） |
| `output_builder.py` の追加行数 | - | ~120行 |
