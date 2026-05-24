# ⑥フェーズ ステージ D&D 並び替え + DataFrame 編集 実装計画

## 目的・背景

⑥タイムテーブル情報の出力タブの **編集モード** におけるステージマスタ編集UI（[app.py:1469-1510](../../src/app.py#L1469-L1510)）について、以下の改善を行う:

1. ↑↓ ボタンによる隣接 swap だけでは、ステージが多い時に並び替えが手間
2. 行ごとの `text_input` / `checkbox` 列が縦長になり一覧性が低い

→ **D&D による並び替え** と **`st.data_editor` による表編集** を組み合わせる。

---

## スコープ

- ステージマスタ編集UI を「**D&D 並び替えエリア（左）+ `st.data_editor`（右）**」の2カラム構成に再編
- 既存の **↑↓ ボタン廃止**（D&D で代替）
- 各行の `text_input` (ステージ名) と `checkbox` (非活性化) は **`st.data_editor` に統合**
- D&D ラベルに **種別名** (`event_type` / dir_name) を併記表示

スコープ外:

- グループマスタ ([app.py:1513-1523](../../src/app.py#L1513-L1523)) / 出番マスタ ([app.py:1525-1568](../../src/app.py#L1525-L1568)) の編集UI（現状の `st.data_editor` 維持）
- バックエンド保存ロジック（[output_editor.py](../../src/backend_functions/output_editor.py) / [workflow.py:671-](../../src/workflow.py#L671) `save_output_edits`）。`表示順` 列の書き換えで完結する。
- ステージ ID の発番・削除（編集モードでは行追加・削除なし）
- **特典会フラグの編集**（live 行への再 join、heiki 整合性、`find_or_create_stage_id` の設計変更など波及が大きく、別計画として切り出し）

---

## 前提（変更なしのもの）

- `edits["stage"]` は `stage_id` を index とする `pd.DataFrame`。列は最低限以下を持つ:
  - `ステージ名` (str, 編集対象)
  - `表示順` (int, 並び順)
  - `特典会フラグ` (bool, 読み取り専用)
  - `非活性化フラグ` (bool, 編集対象)
- 編集モード開始は [workflow.py:623-664](../../src/workflow.py#L623-L664) `enter_output_edit_mode` で `output_df[event_name]["stage"]` をディープコピー
- 保存時の `表示順` は [output_editor.py:702-706](../../src/backend_functions/output_editor.py#L702-L706) で 0..N-1 に正規化済みとして再書き込みされる
- 種別 (`event_type` / dir_name) は stage_*.json を走査して得られる ([output_editor.py:_iter_stage_jsons:25-53](../../src/backend_functions/output_editor.py#L25-L53))

---

## 仕様

### 1. UI 構成

```
###### ステージマスタ (編集中)

┌──────────────────────┬─────────────────────────────────────────┐
│ ドラッグで並び替え      │ st.data_editor (編集中、行追加/削除なし)    │
│ ┌──────────────────┐ │ ┌────┬─────────┬──────┬─────────┐       │
│ │ [ライブ] ID:0 きら │ │ │ ID │ ステージ名│ 特典会│ 非活性化 │       │
│ │ [ライブ] ID:1 ぴか │ │ ├────┼─────────┼──────┼─────────┤       │
│ │ [特典会] ID:3 きら │ │ │ 0  │ きら     │ ☐    │ ☐       │       │
│ │ [縁日]   ID:5 縁日 │ │ │ 1  │ ぴか     │ ☐    │ ☐       │       │
│ │ ~~[ライブ] ID:2 削~~│ │ │ 3  │ きら     │ ☑    │ ☐       │       │
│ └──────────────────┘ │ │ 5  │ 縁日     │ ☐    │ ☐       │       │
│                       │ │ 2  │ 削除済    │ ☐    │ ☑       │       │
│                       │ └────┴─────────┴──────┴─────────┘       │
└──────────────────────┴─────────────────────────────────────────┘
```

- `st.columns([1, 2])` 等の2カラム構成（左：D&D / 右：data_editor）。expander は使わない
- 並び順は D&D の結果（`表示順` 列）に従い、`st.data_editor` も同順で描画
- `st.data_editor` は **`num_rows="fixed"`**（行追加削除不可）
- `表示順` 列は `st.data_editor` には出さない（D&D で間接編集）
- `ステージID` 列は data_editor 上で表示するが **編集不可**（→ §3 参照）

### 2. D&D 並び替えエリア

- `streamlit-sortables.sort_items(items, key=...)` を使用
- `items` は `["[ライブ] ID:0  きら", "[特典会] ID:3  きら", ...]` 形式の文字列リスト
- **ラベル仕様**:
  - フォーマット: `[{種別名}] ID:{stage_id}  {ステージ名}`
  - 種別名は stage_*.json の `event_type` (dir_name) から取得（§4.A）
  - 非活性化ステージは `~~ラベル~~` で打ち消し線（**並び順には含める**）
  - 種別マップに存在しないステージは `[不明]` を付与
- 戻り値（新しいラベル順）が前回と変化した場合のみ:
  - ラベルから `stage_id` を逆引きし、`edits["stage"]["表示順"]` を 0..N-1 で振り直す
  - `st.rerun()` で `st.data_editor` を新順序で再描画
- ステージ数 < 2 のとき D&D エリアは非表示 (`st.data_editor` のみ表示)

### 3. `st.data_editor` 編集仕様

`stage_id` を **index から外して通常列化**し、`disabled=True` にする。グループマスタは index ベース ([app.py:1513-1523](../../src/app.py#L1513-L1523)) だが、出番マスタ ([app.py:1532-1551](../../src/app.py#L1532-L1551)) は `ステージID` を編集不可な通常列として扱う先例があるため、それに合わせる。

| 列 | 表示 / column_config | 編集可否 | 備考 |
|---|---|---|---|
| `ステージID` | `NumberColumn("ステージID", disabled=True)` | 読み取り専用 | index から `reset_index` で通常列化 |
| `ステージ名` | `TextColumn("ステージ名", required=True)` | 編集可 | 既存の `text_input` 相当 |
| `特典会フラグ` | `CheckboxColumn("特典会", disabled=True)` | 読み取り専用 | 編集は別計画 |
| `非活性化フラグ` | `CheckboxColumn("非活性化")` | 編集可 | 既存の `checkbox` 相当 |
| `表示順` | 非表示 | — | D&D で間接編集 |

- 表示用 DataFrame は `sorted_stage.reset_index().rename(columns={"index": "ステージID"})` 等で作る（index 名がついていれば自動的に列名になる）
- 編集結果は `set_index("ステージID")` で index に戻し、`表示順` を元の値で補ってから `edits["stage"]` に書き戻す
- `key=f"stage_editor_{event_name}"` で session に紐付け

### 4. 種別 (`event_type`) 取得

#### A. stage_id → 種別名 マップの生成

`enter_output_edit_mode` 時に stage_*.json を走査して `dict[int, str]` を構築し、`edits["stage_kind_map"]` に保持する。

ロジック（[output_editor.py:_iter_stage_jsons:25-53](../../src/backend_functions/output_editor.py#L25-L53) を流用）:

```python
def build_stage_kind_map(
    pj_path: str, event_name: str, event_no: int, project_info_json: dict,
) -> dict[int, str]:
    """各 stage_id がどの event_type (dir_name) に属するかのマップ。

    heiki kind では トップレベル ステージID も 特典会[].ステージID も
    同じ event_type に紐付ける。
    """
    result: dict[int, str] = {}
    for _path, data, event_type, kind, _sn in _iter_stage_jsons(
        pj_path, event_name, event_no, project_info_json,
    ):
        top_sid = data.get("ステージID")
        if top_sid is not None:
            try:
                result[int(top_sid)] = event_type
            except (ValueError, TypeError):
                pass
        if kind == "live_tokutenkai_heiki":
            for turn in data.get("タイムテーブル", []) or []:
                for tk in turn.get("特典会", []) or []:
                    tk_sid = tk.get("ステージID")
                    if tk_sid is None:
                        continue
                    try:
                        result[int(tk_sid)] = event_type
                    except (ValueError, TypeError):
                        pass
    return result
```

- 配置場所: `src/backend_functions/output_editor.py`
- `enter_output_edit_mode` ([workflow.py:623-664](../../src/workflow.py#L623-L664)) から呼び出して `edits["stage_kind_map"]` に格納
- 不明（マップに無い stage_id）は UI 側で `"不明"` にフォールバック

#### B. キャンセル / 保存時の扱い

- `cancel_output_edit_mode` ([workflow.py:666-669](../../src/workflow.py#L666-L669)) で `edits` ごと破棄されるので追加対応不要
- `save_output_edits` ([workflow.py:671-](../../src/workflow.py#L671)) で `edits["stage_kind_map"]` は副次的なメタデータなので、保存対象から除外（既存処理は `edits["stage"] / ["idolname"] / ["live"]` のみ参照しているので影響なし）

---

## 実装方針

### A. 依存追加

`requirements.txt` に `streamlit-sortables` を追加（Streamlit 1.49.1 互換確認済み）。

### B. バックエンド追加（[output_editor.py](../../src/backend_functions/output_editor.py)）

- `build_stage_kind_map(pj_path, event_name, event_no, project_info_json) -> dict[int, str]` を新規追加（§4.A）

### C. workflow 修正（[workflow.py:623-664](../../src/workflow.py#L623-L664)）

`enter_output_edit_mode` 末尾近くで:

```python
edits["stage_kind_map"] = _output_editor.build_stage_kind_map(
    state.project.pj_path, event_name, event_no, state.project.project_info_json,
)
```

### D. ヘルパー関数追加（[app.py](../../src/app.py)）

```python
def _make_stage_dnd_label(stage_id, row, kind_map) -> str:
    """D&D アイテムのラベル文字列。stage_id をユニーク識別に含める。"""
    kind = kind_map.get(int(stage_id), "不明")
    base = f"[{kind}] ID:{stage_id}  {row['ステージ名']}"
    if bool(row.get("非活性化フラグ", False)):
        base = f"~~{base}~~"
    return base


def _apply_stage_reorder(event_name: str, new_order_stage_ids: list) -> None:
    """新しい stage_id 順に従って `表示順` を 0..N-1 で振り直す。"""
    edits = app_state.output.edits.get(event_name)
    if edits is None:
        return
    df = edits["stage"]
    for new_pos, sid in enumerate(new_order_stage_ids):
        df.at[sid, "表示順"] = int(new_pos)
```

### E. ステージマスタ編集UI の置換（[app.py:1469-1510](../../src/app.py#L1469-L1510)）

既存ループを以下に差し替え:

```python
st.markdown("###### ステージマスタ (編集中)")
sorted_stage = edits["stage"].sort_values("表示順")
kind_map = edits.get("stage_kind_map", {})

stage_cols = st.columns([1, 2])

# --- 左: D&D 並び替えエリア ---
with stage_cols[0]:
    if len(sorted_stage) >= 2:
        st.markdown("**ドラッグで並び替え**")
        from streamlit_sortables import sort_items
        labels = [
            _make_stage_dnd_label(sid, row, kind_map)
            for sid, row in sorted_stage.iterrows()
        ]
        id_by_label = dict(zip(labels, sorted_stage.index.tolist()))
        new_labels = sort_items(labels, key=f"sortable_stage_{event_name}")
        if new_labels != labels:
            new_order_ids = [id_by_label[lab] for lab in new_labels]
            _apply_stage_reorder(event_name, new_order_ids)
            st.rerun()

# --- 右: st.data_editor (ステージ名 / 非活性化 編集) ---
with stage_cols[1]:
    stage_display_df = (
        sorted_stage[["ステージ名", "特典会フラグ", "非活性化フラグ"]]
        .reset_index()
        .rename(columns={sorted_stage.index.name or "index": "ステージID"})
    )
    edited_stage = st.data_editor(
        stage_display_df,
        column_config={
            "ステージID": st.column_config.NumberColumn("ステージID", disabled=True),
            "ステージ名": st.column_config.TextColumn("ステージ名", required=True),
            "特典会フラグ": st.column_config.CheckboxColumn("特典会", disabled=True),
            "非活性化フラグ": st.column_config.CheckboxColumn("非活性化"),
        },
        num_rows="fixed",
        hide_index=True,
        key=f"stage_editor_{event_name}",
        use_container_width=True,
    )
    # ステージID を index に戻し、表示順を補って書き戻す
    edited_stage = edited_stage.set_index("ステージID")
    edited_stage["表示順"] = edits["stage"]["表示順"]
    edits["stage"] = edited_stage
```

### F. 削除する既存コード

- [app.py:1391-1420](../../src/app.py#L1391-L1420) `_move_stage_up` / `_move_stage_down`（不要化）
- [app.py:1473-1485](../../src/app.py#L1473-L1485) の ↑↓ ボタンレンダリング
- [app.py:1486-1510](../../src/app.py#L1486-L1510) の `text_input` / `checkbox` 行（`st.data_editor` に統合）

### G. session_state key cleanup（[app.py:1444-1451](../../src/app.py#L1444-L1451)）

`_on_save_edits` の cleanup 対象を更新:

- 削除対象から外す: `stage_name_*`, `stage_disabled_*`, `up_*`, `dn_*`
- 追加: `stage_editor_{event_name}`, `sortable_stage_{event_name}`

---

## 影響範囲

| ファイル | 変更内容 |
|---|---|
| [requirements.txt](../../requirements.txt) | `streamlit-sortables` 追加 |
| [src/backend_functions/output_editor.py](../../src/backend_functions/output_editor.py) | `build_stage_kind_map` 新規追加 |
| [src/workflow.py](../../src/workflow.py) | `enter_output_edit_mode` で `edits["stage_kind_map"]` をセット |
| [src/app.py](../../src/app.py) | `_move_stage_up/down` 削除 / `_make_stage_dnd_label` / `_apply_stage_reorder` 追加 / ステージマスタ編集UI を 2カラム D&D + `st.data_editor` に置換 / `_on_save_edits` の cleanup キー更新 |
| (新規) `tests/backend_functions/test_build_stage_kind_map.py` | `build_stage_kind_map` の単体テスト（通常 kind / heiki kind / 不明 stage_id） |
| (新規) `tests/frontend_functions/test_stage_reorder.py` | `_apply_stage_reorder` の単体テスト |

`save_output_edits` 等の既存保存パスは `表示順` 列の値だけ正しく振り直されていれば変更不要。

---

## 段階的実装案

1. **Phase 1**: `requirements.txt` に `streamlit-sortables` 追加 → ローカル動作確認（インポート / Streamlit 1.49.1 起動確認）
2. **Phase 2**: `build_stage_kind_map` 追加 + 単体テスト / `enter_output_edit_mode` で `stage_kind_map` をセット
3. **Phase 3**: `_make_stage_dnd_label` / `_apply_stage_reorder` 追加 + 単体テスト
4. **Phase 4**: ステージマスタ編集UI を 2カラム D&D + `st.data_editor` に置換、↑↓ ボタンと関連コード削除
5. **Phase 5**: `_on_save_edits` の session_state cleanup キーを更新、マニュアル E2E（並び替え→編集→保存→再ロード）

---

## 確定済み論点

1. **既存 ↑↓ ボタン**: 廃止（D&D で代替）
2. **ステージ名 / 非活性化編集**: `st.data_editor` に統合
3. **UI レイアウト**: 2カラム構成（左：D&D、右：`st.data_editor`）。expander は使わない
4. **D&D ラベル**: 種別名を併記 (`[ライブ] ID:0  きら` 形式)
5. **非活性化ステージ**: D&D 対象に含める（`~~strike~~` で打ち消し表示）
6. **`ステージID` 列**: index から外して通常列化、`NumberColumn(disabled=True)` で編集不可表示（ヘッダ名「ステージID」）
7. **`表示順` 列**: `st.data_editor` には出さず、D&D で間接編集
8. **行追加/削除**: `st.data_editor` は `num_rows="fixed"`
9. **特典会フラグ編集**: スコープ外（本計画では扱わない）

---

## 関連計画

- [done/timetable_order_plan.md](done/timetable_order_plan.md): ②画像登録セクションの並び順（配列順 + ↑↓ボタン + リセット）。本計画とは対象セクションが異なる。
