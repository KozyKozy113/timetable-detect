# ⑥出力タブ集計機能追加 実装計画

## 目的

⑥タイムテーブル情報の出力タブに、以下2つの集計機能を追加する。

1. **出演枠時間の頻度分布**: 各イベント内の出演枠を「ライブ枠の長さ(分)」でカウントし、ライブステージ／特典会ステージに分けて出現回数を表示。
2. **グループ別出演回数**: 各イベント内のグループごとの出演回数を、ライブステージ／特典会ステージに分けて表示。

いずれも既存の `event_tabs` (`st.tabs`) をそのまま利用し、イベントごとに表示する。

---

## スコープ

- [src/backend_functions/output_builder.py](../../src/backend_functions/output_builder.py): 集計用ピュア関数を追加（Streamlit非依存）
- [src/app.py](../../src/app.py) `render_output_section()`: 既存 `data["stage"]/["idolname"]/["live"]` の3カラム表示の **下** に集計セクションを追加
- 既存の `output_df` 構造には触らない（後段の `determine_id_master` / `export_excel` / `listup_new_idolname` への影響なし）

実装は本計画確定後に行う。

---

## 前提となるデータ構造

`build_event_output()` の戻り値（[output_builder.py:222-226](../../src/backend_functions/output_builder.py#L222-L226)）:

| キー | DataFrame の構造 |
|---|---|
| `"stage"` | index=`ステージID`, cols=`[ステージ名, 特典会フラグ(bool)]` |
| `"idolname"` | index=`グループID`, cols=`[グループ名_採用]` |
| `"live"` | index=`出番ID`, cols=`[ライブ_from, ライブ_長さ(分), グループID, ステージID, グループ名_raw, グループ名, ステージ名, 備考]` |

特記事項:
- `live["ライブ_長さ(分)"]` は `Int64` 型（[timetabledata.py:226](../../src/backend_functions/timetabledata.py#L226)）。NaN 行は既に `build_event_output()` 内で除去済み（[output_builder.py:145-148](../../src/backend_functions/output_builder.py#L145-L148)）。
- `live["ステージID"]` から `stage["特典会フラグ"]` を引いて 特典会ステージ／ライブステージを判別する。
- `live["グループID"]` は merge 済み。`live["グループ名"]` は `idolname` マスタの「グループ名_採用」と一致。

---

## 1. 集計関数の追加（output_builder.py）

### 1.1 `build_duration_distribution()`

```python
_DURATION_COL_LIVE = "ライブステージ"
_DURATION_COL_TOKUTENKAI = "特典会ステージ"


def build_duration_distribution(
    df_live: pd.DataFrame,
    df_stage: pd.DataFrame,
) -> pd.DataFrame:
    """出演枠の長さ(分)ごとの出現回数を、ライブ／特典会ステージ別にカウントする。

    Returns:
        index: 長さ(分) 昇順 Int64
        cols : [ライブステージ, 特典会ステージ]  ← 値は int (0埋め)
    """
```

実装の骨子:
1. `df_live` に `df_stage["特典会フラグ"]` を `ステージID` で left merge。
2. `groupby(["ライブ_長さ(分)", "特典会フラグ"]).size()` で集計。
3. `unstack("特典会フラグ", fill_value=0)` し、`True/False` を `特典会ステージ/ライブステージ` にリネーム。
4. 列が片方しか存在しない場合に備え、欠けた列は `0` で補う。
5. index を `ライブ_長さ(分)` 昇順にソートし、index 名は `長さ(分)`。

エッジケース:
- `df_live` が空 → 空 DataFrame (index 空、上記2列を持つ) を返す。
- 全ステージがライブ枠のみ／特典会枠のみでも、2列とも必ず存在させる。

### 1.2 `build_group_appearance_count()`

```python
_GROUP_COUNT_COL_LIVE = "ライブ出演回数"
_GROUP_COUNT_COL_TOKUTENKAI = "特典会出演回数"
_GROUP_COUNT_COL_TOTAL = "合計"


def build_group_appearance_count(
    df_live: pd.DataFrame,
    df_stage: pd.DataFrame,
) -> pd.DataFrame:
    """グループごとの出演回数を、ライブ／特典会ステージ別にカウントする。

    Returns:
        index: グループID
        cols : [グループ名, ライブ出演回数, 特典会出演回数, 合計]
    """
```

実装の骨子:
1. `df_live` に `df_stage["特典会フラグ"]` を `ステージID` で left merge。
2. `groupby(["グループID", "グループ名", "特典会フラグ"]).size().unstack(fill_value=0)`。
3. `True/False` 列を `特典会出演回数/ライブ出演回数` にリネーム、欠けた列は 0 で補う。
4. `合計 = ライブ出演回数 + 特典会出演回数` を追加。
5. **並び替えは UI 側で行う** ため、ここではデフォルトで `グループID` 昇順を返す。

エッジケース:
- グループ名はマスタの「グループ名_採用」を採用（`df_live["グループ名"]`）。`グループ名_raw` は使わない。
- 同一グループが両方のステージに出演 → 2列とも > 0。
- 片方しか出演していないグループも行として残す（もう片方は 0）。

### 1.3 `build_event_output()` の戻り値拡張

```python
return {
    "stage": df_stage,
    "idolname": df_idolname,
    "live": df_live[_LIVE_OUTPUT_COLUMNS],
    "duration_distribution": build_duration_distribution(df_live_full, df_stage),
    "group_count": build_group_appearance_count(df_live_full, df_stage),
}
```

注意点:
- 集計には `_LIVE_OUTPUT_COLUMNS` で絞る **前** の `df_live`（`ステージID`/`グループID` を含む完全版）を使う。本来 [output_builder.py:225](../../src/backend_functions/output_builder.py#L225) では絞り後の `df_live[_LIVE_OUTPUT_COLUMNS]` を返しているが、集計関数も同じ列を要求するので順番上問題ない（`ステージID`, `グループID`, `グループ名`, `ライブ_長さ(分)` は `_LIVE_OUTPUT_COLUMNS` に含まれる）。→ **絞った後の `df_live` をそのまま渡せばよい**。
- 既存の `determine_id_master` / `export_excel` / `listup_new_idolname` は `output_df[event_name]["stage"/"idolname"/"live"]` のみを参照しているため、キーが増えても無害（[output_builder.py:263-266](../../src/backend_functions/output_builder.py#L263-L266), [310](../../src/backend_functions/output_builder.py#L310), [351](../../src/backend_functions/output_builder.py#L351)）。

---

## 2. UI 追加（app.py `render_output_section()`）

[src/app.py:1110-1122](../../src/app.py#L1110-L1122) の `for event_name, event_tab in zip(event_list, event_tabs):` ブロックを、既存の3カラム表示はそのままに、その **下** に集計セクションを追加する。

```python
with event_tab:
    output_cols = st.columns([1, 1, 3])
    with output_cols[0]:
        st.dataframe(data["stage"])
    with output_cols[1]:
        st.dataframe(data["idolname"])
    with output_cols[2]:
        st.dataframe(data["live"])

    st.divider()
    st.markdown("##### 集計情報")
    aggr_cols = st.columns(2)
    with aggr_cols[0]:
        st.markdown("**出演枠時間の頻度分布**")
        st.dataframe(data["duration_distribution"])
    with aggr_cols[1]:
        st.markdown("**グループ別出演回数**")
        sort_key = f"group_count_sort_{event_name}"
        sort_mode = st.radio(
            "並び順",
            ["合計回数(降順)", "グループID(昇順)"],
            key=sort_key,
            horizontal=True,
        )
        df_group = data["group_count"]
        if sort_mode == "合計回数(降順)":
            df_group = df_group.sort_values(
                ["合計", "ライブ出演回数"], ascending=[False, False],
            )
        st.dataframe(df_group)
```

設計判断:
- **並び替えは UI 側**で行う（`output_builder` 側ではソートしない）。これにより `st.radio` の state を `output_df` 再生成と切り離せる。
- `st.radio` の `key` は **イベント名を含める**（`f"group_count_sort_{event_name}"`）。複数イベントタブで独立した state を持つため。
- 「合計回数(降順)」を default にする（推奨）。同点時の安定ソートとして第2キーに `ライブ出演回数` を入れる。
- 区切りは `st.divider()` + `st.markdown("##### 集計情報")` で視覚的に分離。expander は使わない（毎回開閉する手間を避ける）。

---

## 3. 影響範囲まとめ

| ファイル | 変更内容 | 行数目安 |
|---|---|---|
| [src/backend_functions/output_builder.py](../../src/backend_functions/output_builder.py) | `build_duration_distribution()` / `build_group_appearance_count()` 追加、`build_event_output()` の返却 dict に2キー追加 | +60行 |
| [src/app.py](../../src/app.py) `render_output_section()` (L1115-1122) | 既存3カラム表示の下に集計セクション追加 | +20行 |

`tests/` 配下の output_builder のテスト（既に存在する場合）には、新関数2つのユニットテストを追加する。

---

## 4. 検討した代替案と却下理由

| 案 | 却下理由 |
|---|---|
| 頻度分布を 2 DataFrame 横並び | 比較しにくいとユーザーが判断 → 1表で `ライブステージ/特典会ステージ` 列分け |
| 頻度分布を棒グラフ | 数値の正確な確認が必要なため DataFrame で表示 |
| グループ別出演回数を「ライブ専用列」「特典会専用列」+「両方列」の3分割 | 仕様は「別個に計算」のみ → シンプルに2列で十分 |
| 集計を `expander` に格納 | 毎回開閉する手間が発生。`divider` で視覚分離するだけで十分 |
| 集計関数を `timetabledata.py` に置く | `output_builder` は既に `stage/idolname/live` の集約 DataFrame を扱う層。同じ層の派生集計はここに置くのが自然 |
| 並び替えを output_builder 側で行う | `st.radio` を切り替えるたびに `build_all_event_outputs` を再実行する必要が出てしまう。並び順はビュー側責務 |

---

## 5. 確定済み論点

1. **Q: 頻度分布の表示形式** → 1 表で列分け（行=長さ(分)、列=ライブ/特典会ステージ）
2. **Q: グループ別出演回数の並び順** → UI で「合計回数(降順) / グループID(昇順)」を切替可能に
3. **Q: 新機能の配置** → 既存3カラム表示の下に「集計情報」セクションとして追加
4. **Q: 「特典会フラグ」の判定根拠** → `stage["特典会フラグ"]` 列（既存ロジックの結果）をそのまま使用。[image_kind_and_stage_kind_plan.md](image_kind_and_stage_kind_plan.md) の kind ベース再設計が入っても出力 DataFrame の `特典会フラグ` 列は据え置きのため、本機能はその変更とは独立に実装可能。

---

## 6. 実装ステップ

1. **Step 1**: `output_builder.py` に `build_duration_distribution()` を追加 + ユニットテスト
2. **Step 2**: `output_builder.py` に `build_group_appearance_count()` を追加 + ユニットテスト
3. **Step 3**: `build_event_output()` の戻り値に新キー2つを追加
4. **Step 4**: `app.py` `render_output_section()` の event_tab ブロックに集計セクション追加
5. **Step 5**: 既存プロジェクトで動作確認（特典会のみのイベント、ライブのみのイベント、両方混在のイベント の3パターン）

各 Step は独立しており、Step 1-3 まで完了した時点で `output_df` の中身が増えるだけで UI には影響しない（後方互換）。Step 4 で初めて UI に現れる。
