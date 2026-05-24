# ⑥出力確認・編集 編集モード 実装計画

## 実装状況 (2026-05-25 時点)

| Phase | 内容 | ステータス |
|---|---|---|
| **Phase 0** | データモデル変更 (ステージIDトップレベル化 + マイグレーション) | ✅ 完了 |
| **Phase 1** | ステージマスタに 表示順 / 非活性化フラグ カラム追加 + `output_editor.py` 保存基盤 | ✅ 完了 |
| **Phase 2** | ステージマスタ編集UI (↑↓・ステージ名・非活性化) + ページ遷移警告 + バリデーション | ✅ 完了 |
| **Phase 3** | グループマスタ編集UI (`グループ名_採用` 編集) | ✅ 完了 |
| **Phase 4** | 出番マスタ編集UI + 特典会併記 `対応出番ID` 採番 + JSON書き戻し | ✅ 完了 |
| **Phase 5** | 出番マスタのステージID変更 (stage_*.json 間エントリ移動) | ✅ 完了 |

### Phase 0+1+2 で実装されたもの

- **データモデル**
  - `stage_*.json` のトップレベルに `ステージID` を保持（特典会併記形式では子=`特典会[].ステージID` も維持）
  - `project_info.stage_list[i].stage_id` を追加
  - `find_or_create_stage_id` に `existing_stage_id` 引き当て対応（ステージ名変更時の二重採番を防止）
  - 旧形式（出番粒度に `ステージID`）からのマイグレーション（[src/backend_functions/project_migration.py](../../src/backend_functions/project_migration.py) `migrate_stage_id_to_toplevel`、冪等、プロジェクトロード時に自動実行）
- **`master_stage.csv` スキーマ拡張**
  - `表示順` (int、0始まり連番): ⑥出力 / Excel の並び順を司る
  - `非活性化フラグ` (bool): True なら ⑥出力・集計・Excel から除外（実体は保持）
  - `load_existing_masters` で後方互換補完
- **編集UI**（[src/app.py](../../src/app.py) `render_output_section` / `_render_event_output_editor`）
  - IDマスタ確定後のイベントタブにのみ「編集モード」トグル表示
  - ステージマスタ: カードUI で ↑↓ 並び替え + ステージ名 `st.text_input` 編集 + 非活性化 `st.checkbox`
  - グループマスタ / 出番マスタは Phase 3/4 までは閲覧専用
  - 保存時バリデーション（空名 / 同イベント内重複の検知）
  - 保存処理: `master_stage.csv` 書き出し → `stage_*.json` トップレベル `ステージ名` 同期 → `project_info.stage_list[i].stage_name` 同期
- **ページ遷移制御**
  - サイドバーの「処理フェーズ」radio 切り替え時、編集内容に実際の差分があれば「保存して移動 / 破棄して移動 / キャンセル」を提示（[src/app.py](../../src/app.py) `_has_unsaved_edits` で `DataFrame.equals` 比較）
  - 差分なし or 確認後の遷移時には、編集モードのトグルやウィジェットキーを掃除し、戻ってきた時に必ず通常画面から始まるようにする

### Phase 0-2 で発生した既知の課題と対処

1. **トグルON だけで「未保存の編集あり」と誤検知**: `bool(app_state.output.edits)` では実変更を判定できないため、`output_df` ベースラインと `edits` を `DataFrame.equals` で値比較する方式に変更
2. **`st.session_state.exist_pj_name cannot be modified after the widget ...`**: 警告ボタン内で `_sync_to_session(app_state)` を呼ぶと、すでに instantiate 済みの selectbox キーを書き換えてエラー。当該呼び出しを除去
3. **保存後に編集モードトグルが ON のまま残る**: callback で `st.session_state[toggle_key] = False` を明示セット（pop だと一部ケースでウィジェット側の状態が残るため）

### Phase 3+4 で実装されたもの (2026-05-25)

- **`対応出番ID` 配線**
  - `id_apply_to_json` ([src/backend_functions/timetabledata.py](../../src/backend_functions/timetabledata.py)) で `live_tokutenkai_heiki` 形式の `特典会[].対応出番ID` に親 `出番ID` をコピー
  - 既存プロジェクト向けマイグレーション `backfill_tokutenkai_corresponding_turn_id` ([src/backend_functions/project_migration.py](../../src/backend_functions/project_migration.py)) を追加。プロジェクトロード時に冪等実行
- **グループマスタ編集 (Phase 3)**
  - 編集中作業コピー `edits[event_name]["idolname"]` を追加 ([src/workflow.py](../../src/workflow.py) `enter_output_edit_mode`)
  - `st.data_editor` で `グループ名_採用` のみ編集可（行追加・削除不可）
  - 保存処理: `master_idolname.csv` 書き出し + `_propagate_idolname_master_to_json` で全 `stage_*.json` の `タイムテーブル[].グループ名_採用` を `グループID` 引き当てで同期
  - バリデーション: 空名 / 重複なし
- **出番マスタ編集 (Phase 4)**
  - 編集中作業コピー `edits[event_name]["live"]` を追加。`特典会フラグ` を `ステージID` で stage マスタから join 付与、`対応出番ID` を `build_corresponding_turn_id_map` で stage_*.json から逆引きして付与 (差分検知用 baseline は `edits["_live_baseline"]` に保管)
  - `st.data_editor` で編集可: `ライブ_from` / `ライブ_長さ(分)` / `グループID` (SelectboxColumn) / `対応出番ID` (NumberColumn; 特典会行のみ意味) / `備考`
  - 読取専用: `出番ID` (index) / `ステージID` (Phase 5 まで) / `ステージ名` / `グループ名_raw` / `グループ名` (採用) / `特典会フラグ`
  - 保存処理: `turn_id_data.csv` 書き出し + `_propagate_live_edits_to_json` で `stage_*.json` に書き戻し
    - 全 stage_*.json をメモリにロード後、`parent_index` (親出番ID→位置) と `booth_index` (booth出番ID→位置) を構築
    - **対応出番ID 変更行は先に「移動」処理**: 旧親.`特典会[]` から要素を取り除き、新親.`特典会[]` (別 stage_*.json 可) に同要素を追加。booth 自身の `出番ID` / `ステージID` は保持、`対応出番ID` のみ更新
    - その後、ライブ行・特典会行とも値更新を反映 (`ライブステージ.from/to`、`グループID`、`グループ名_採用`、`備考` / `特典会[].from/to`)
    - 変更があった stage_*.json のみ書き戻し
  - グループID 変更時は idolname マスタを引いて `グループ名_採用` (df_live の "グループ名" 列) を再導出
  - バリデーション: HH:MM 形式 / 正の整数 / グループID 存在 / **対応出番ID は同一イベントのライブ行 出番ID を指す** / ステージID 不変 (Phase 5 まで)
- **未保存編集の検知 / キー掃除拡張**
  - `_has_unsaved_edits` ([src/app.py](../../src/app.py)) で idolname の `グループ名_採用` と live の編集4カラムを比較
  - `_clear_edit_mode_widgets` / `_on_save_edits` で `idolname_editor_*` / `live_editor_*` キーも掃除

### Phase 5 で実装されたもの (2026-05-25)

- **出番マスタの `ステージID` 編集解禁**
  - `st.data_editor` の `ステージID` 列を `SelectboxColumn` に変更 ([src/app.py](../../src/app.py) `_render_event_output_editor`)。選択肢は ID + ステージ名 (特典会フラグ立ちは `[特典会]` サフィックス)
- **バリデーション拡張** ([src/backend_functions/output_editor.py](../../src/backend_functions/output_editor.py) `validate_live_master_edits`)
  - `df_live_original` ベースの「ステージID 不変チェック」を削除
  - 代わりに `df_stage` を受け取り、新しい `ステージID` がマスタに存在すること / 新ステージの `特典会フラグ` が行の `特典会フラグ` と一致することを検証
- **`_propagate_live_edits_to_json` の Phase 5 対応**
  - 親エントリ (`parent_index` 該当: 非heiki ライブ / 非heiki 特典会 / heiki ライブ親) で top-level `ステージID` ≠ 新値なら `live_moves` に分類
  - 移動先 stage_*.json は `_build_file_by_top_stage_id` (top-level `ステージID` → (file_idx, kind)) で引き当て。kind 不一致 (例: live ⇔ heiki) は移動拒否
  - 移動処理: `loaded[old].タイムテーブル.pop(turn_idx)` → `loaded[new].タイムテーブル.append(entry)`。`特典会[]` は親エントリ配下なので同行する
  - 親移動後はインデックスを完全再構築 (`_build_live_indices(loaded)`)、その後で Phase 4 の `tk_moves` (対応出番ID) と値更新を実施
  - heiki 特典会行で `ステージID` を変更した場合は `_update_tk_fields` 内で `特典会[j].ステージID` を in-place で更新 (ファイル移動なし)
- **`save_event_edits` でのステージ名再導出**: `ステージID` 変更行の `ステージ名` 列を `master_stage` から再導出して `turn_id_data.csv` に書き出す (グループ名と同じ流儀)
- **テスト追加** ([tests/backend_functions/test_output_editor.py](../../tests/backend_functions/test_output_editor.py))
  - `test_validate_live_accepts_stage_id_change_phase5` / `_rejects_unknown_stage_id` / `_rejects_stage_id_tk_flag_mismatch`
  - `test_save_event_edits_phase5_moves_live_entry_between_files`
  - `test_save_event_edits_phase5_move_preserves_other_field_edits` (移動と同時の from/長さ 編集)
  - `test_save_event_edits_phase5_heiki_booth_stage_id_update_in_place` (booth別ID は in-place)

---

## 目的・背景

⑥出力確認・編集タブ ([app.py:1284-1356](../../src/app.py#L1284-L1356)) では、`build_all_event_outputs` ([output_builder.py:469](../../src/backend_functions/output_builder.py#L469)) が組み立てたステージマスタ / グループマスタ / 出番マスタ (live) を `st.dataframe` で表示しているが、いずれも読み取り専用。

実運用では以下の手修正が頻繁に必要だが、現在はプロジェクトのソースデータ（`stage_*.json` や project_info）まで戻らないと修正できない:

- **ステージ並び順の変更**: 現状は `stage_list[]` の登録順に依存。Excel 出力時の見やすさのため並べ替えたい。
- **ステージ名の修正**: OCR 由来のブース名が崩れていた場合の手直し。
- **グループ名の修正**: マスタに無いグループや表記揺れの是正。
- **出番マスタの修正**: 誤読された出演時間、誤って割り当てられたグループID／ステージIDの差し替え。

これらを ⑥出力確認・編集タブ内で完結させる。

---

## スコープ

- ⑥出力確認・編集タブに **「編集モード」トグル** を追加（**IDマスタ確定後のみ表示**）
- 編集モード中に以下を編集可能にする:
  - **ステージマスタ**: 並び順（↑↓ボタン）+ ステージ名（テキスト入力）
  - **グループマスタ**: グループ名_採用（テキスト入力）
  - **出番マスタ (live)**: グループID / ライブ_from / ライブ_長さ(分) / 備考
    - ステージID 変更は Phase 5 で対応（それまで disabled）
- **「保存」ボタン**で一括永続化、**「キャンセル」ボタン**で破棄
- 編集中に他ページへ遷移しようとした場合は警告表示
- 編集はイベント単位（タブ単位）で独立

スコープ外:
- 行の追加・削除（既存行の編集のみ）
- 集計表（duration_distribution / group_count / overlap_alerts / group_appearances）の編集
- 特典会の `ブース` 編集（特典会併記形式の内部構造変更を伴うため）
- マスタ確定前（`master_*.csv` 未生成）の編集 — 編集モードトグル自体を非表示にする

---

## 前提

- [output_builder.py:469](../../src/backend_functions/output_builder.py#L469) `build_all_event_outputs` は毎回ソース (`stage_*.json` / `project_info.json` / `master_*.csv`) から再構築する。
- **マスタ確定後**（`event_N/master_stage.csv` 等が存在する状態、[output_builder.py:49-80](../../src/backend_functions/output_builder.py#L49-L80) `load_existing_masters` がそれらを起点に動く）が編集の前提。
- ステージマスタの「ステージID」と出番マスタの「ステージID」は外部キー、グループも同様。
- 現状: `stage_*.json` 内では「ステージ名」が `find_or_create_stage_id` の結合キーとして使われており、**ステージ名変更で二重採番される構造的問題**がある（§0 で根本対応）。

---

## 仕様

### 0. データモデル変更（前提となる構造変更）

ステージ名変更による二重採番（[output_builder.py:42-46](../../src/backend_functions/output_builder.py#L42-L46) `find_or_create_stage_id` がステージ名一致で引き当てているため、ステージ名を変えるとマスタに同じ物理ステージが二重登録される）を根本解決するため、**ステージIDの保持場所を変える**。

#### 0-1. `stage_*.json` の構造変更

ステージIDを **出番粒度 → ステージ粒度（JSON ファイル単位）** に移動。

**Before**:
```json
{
    "ステージ名": "MORI MORI STAGE",
    "タイムテーブル": [
        {"グループ名": "...", "出番ID": 0, "グループID": 51, "ステージID": 0},
        {"グループ名": "...", "出番ID": 1, "グループID": 71, "ステージID": 0},
        ...
    ]
}
```

**After**:
```json
{
    "ステージ名": "MORI MORI STAGE",
    "ステージID": 0,
    "タイムテーブル": [
        {"グループ名": "...", "出番ID": 0, "グループID": 51},
        {"グループ名": "...", "出番ID": 1, "グループID": 71},
        ...
    ]
}
```

#### 0-2. 特典会併記形式 (`live_tokutenkai_heiki`) の二層構成

特典会併記形式では 1 ファイル内で「親=ライブステージ」と「子=ブース別ステージ」が混在するため、二層で `ステージID` を保持:

```json
{
    "ステージ名": "TOKYO STAGE",
    "ステージID": 0,                              // ← 親: ライブステージのID
    "タイムテーブル": [
        {
            "グループ名": "アイドルA",
            "出番ID": 1,
            "グループID": 12,
            // "ステージID": 0  ← 撤去（親トップレベルから参照）
            "ライブステージ": {"from": "10:00", "to": "10:20"},
            "特典会": [
                {
                    "from": "10:30", "to": "11:30",
                    "ブース": "ブースA",
                    "ステージID": 10,             // ← 子: ブース別ステージID（既存維持）
                    "対応出番ID": 1
                }
            ]
        }
    ]
}
```

- 親ステージIDは JSON トップレベルに 1 つ
- 子ステージID（ブース別）は `特典会[].ステージID` に存続（ブースは OCR で動的に発見されるため、ブース粒度の集約場所がない）

#### 0-3. `project_info.json` の `stage_list[i]` への `stage_id` 追加

```json
{
    "stage_list": [
        {
            "stage_no": 0,
            "stage_name": "MORI MORI STAGE",
            "stage_id": 0,                       // ← 新規追加
            "bbox": {...},
            "kind": "live"
        },
        ...
    ]
}
```

- `stage_id` は IDマスタ確定 (`determine_id_master`) のタイミングで採番・書き戻し
- 未確定時は `null` または欠落
- `live_tokutenkai_heiki` 形式でも `stage_list[i].stage_id` は **親ライブステージのID** を保持。ブース別IDは project_info には持たない（動的発見のため）

#### 0-4. `find_or_create_stage_id` のロジック再編

引き当ての優先順位:

1. **既存ID指定がある**（`stage_*.json` トップレベル `ステージID` または `特典会[].ステージID` が埋まっている）→ そのIDを使用し、ステージ名がマスタと異なれば**マスタのステージ名を更新**（編集モードの変更を反映）
2. **既存ID指定がない**（マスタ未確定時の初回採番）→ ステージ名一致でマスタを検索、無ければ新規採番

```python
def find_or_create_stage_id(
    stage_master: dict,
    stage_name: str,
    is_tokutenkai: bool,
    next_id: int,
    existing_stage_id: int | None = None,
) -> tuple[int, int]:
    if existing_stage_id is not None and existing_stage_id in stage_master:
        if stage_master[existing_stage_id]["ステージ名"] != stage_name:
            stage_master[existing_stage_id]["ステージ名"] = stage_name
        return existing_stage_id, next_id
    for k, v in stage_master.items():
        if v["ステージ名"] == stage_name:
            return k, next_id
    stage_master[next_id] = {"ステージ名": stage_name, "特典会フラグ": is_tokutenkai}
    return next_id, next_id + 1
```

これにより:
- マスタ確定後にステージ名を編集しても、トップレベル `ステージID` でマスタを引き当てられるので二重採番が起きない
- マスタの `ステージ名` も自動追随する（編集モードでステージ名を変えた後、`build_event_output` を再実行するとマスタ側のステージ名も更新される）

#### 0-5. マイグレーション

既存プロジェクトを開いた時、[project_migration.py](../../src/backend_functions/project_migration.py) で以下を実行:

1. **`stage_*.json` のステージID トップレベル化**:
   - `タイムテーブル[0].ステージID` が存在する場合、その値をトップレベル `ステージID` に昇格
   - 全出番から `ステージID` カラムを削除
   - `live_tokutenkai_heiki` 形式の場合、`特典会[].ステージID` はそのまま維持
2. **`project_info.json` の `stage_list[i].stage_id` 補完**:
   - 対応する `stage_*.json` のトップレベル `ステージID` をコピー
   - `stage_*.json` が無い・ステージID未確定の場合は `null`

`determine_id_master` 側の責務追加（[output_builder.py:492-526](../../src/backend_functions/output_builder.py#L492-L526)）:
- 出番ID採番後、各 `stage_*.json` のトップレベルに `ステージID` を書き込む
- `project_info.json` の `stage_list[i].stage_id` も同期更新
- `特典会[].対応出番ID` を採番（[本計画 §6-b](#6-b-特典会併記形式-live_tokutenkai_heiki-の対応)）

#### 0-6. 関連箇所の修正

- [output_builder.py:321-466](../../src/backend_functions/output_builder.py#L321-L466) `build_event_output`:
  - `stage_*.json` 読み込み時にトップレベル `ステージID` を取得して `find_or_create_stage_id` に渡す
  - `df_edit_live` 構築時、各行の `ステージID` カラムは親JSONの値で埋める
  - 特典会併記形式の `特典会[]` 由来行は `特典会[].ステージID` を使用
- [timetabledata.py:316-341](../../src/backend_functions/timetabledata.py#L316-L341) `id_apply_to_json`:
  - 出番粒度の `ステージID` 書き込みを削除
  - 代わりにトップレベル `ステージID` を書き込む処理を追加
- [timetabledata.py json_to_df/df_to_json](../../src/backend_functions/timetabledata.py#L48):
  - 出番粒度の `ステージID` カラムの受け渡しを廃止
  - トップレベル `ステージID` は呼び出し側で別途引き渡す

---

### 1. 編集モード切り替え

⑥出力確認・編集タブの先頭に `st.toggle("編集モード", key="output_edit_mode_{event_name}")` をイベントタブごとに配置。

**起動条件**: そのイベントの `master_stage.csv` / `master_idolname.csv` / `turn_id_data.csv` がすべて存在する（= IDマスタ確定済み）場合のみトグルを表示する。未確定時はトグル位置に「編集するには先に『IDマスタを確定』してください」とヒントを表示。

- **OFF**: 現状の `st.dataframe` 表示（読み取り専用）
- **ON**: ステージ／グループ／出番の各セクションを編集 UI に切り替え。下部に「保存」「キャンセル」ボタンを表示
- 集計表は編集モードでも参考表示として残す（出番マスタの編集中はリアルタイム反映せず、保存後の再計算結果を表示）

編集中の状態は `app_state.output.edits[event_name]` に保持（編集対象の作業コピー dict）。タブ切り替えやページ遷移で破棄されないよう注意。

### 1-b. 編集中の他ページ遷移警告

サイドバーの `st.radio("処理フェーズ", ...)` で別ページに遷移しようとした際、`app_state.output.edits` が空でなければ警告を出す。

実装方針:
- ページ遷移を検知するため、現在ページ (`page`) と前回ページ (`app_state.ui.last_page` を新設) を比較
- 編集中（`edits` が非空）かつページが変わった場合、`st.warning` で「未保存の編集があります。保存しますか？破棄しますか？」と表示
- 「保存して移動」「破棄して移動」「キャンセル（前ページに留まる）」の 3 ボタンを提示
- ユーザーが選択するまで遷移先の本処理（`render_*` 呼び出し）は実行しない

タブ切り替え（同一ページ内のイベントタブ）は警告対象外（編集状態はイベント単位で独立しているため、他タブを見ても作業中タブの edits は失われない）。

### 2. ステージマスタの編集

「表」ではなく**カード／リスト型 UI** で表示する。並び替えは ↑↓ ボタンで行う。

```
###### ステージマスタ
┌──────────────────────────────────────────────────────────┐
│ [↑][↓]  ID:0  [ステージ名: STAGE-A    ] 特典会:☐  表示順:0  非活性化:☐ │
│ [↑][↓]  ID:1  [ステージ名: STAGE-B    ] 特典会:☐  表示順:1  非活性化:☐ │
│ [↑][↓]  ID:2  [ステージ名: 旧物販ブース] 特典会:☑  表示順:2  非活性化:☑  ← グレーアウト │
└──────────────────────────────────────────────────────────┘
```

- **並び順は新規追加カラム `表示順` (stageOrder) に保存**。`stage_list[]`（②画像登録で使われる入力側）の順序とは独立
- `表示順` カラムは [stella_json_output_plan.md §2-3](stella_json_output_plan.md) の Phase 2-3 で計画されている `stageOrder` と同一フィールド。**本計画はその先行実装**として扱う（stella 計画側はステージマスタ拡張のうちの 1 カラムとして扱われていたが、本計画で先に導入）
- `↑`: 表示順を 1 つ前と入れ替え（先頭なら無効）
- `↓`: 表示順を 1 つ後と入れ替え（末尾なら無効）
- `表示順` の値は 0 始まりの連番で再採番して保存（欠番なし）
- 各カードの `st.text_input` でステージ名を編集
- 特典会フラグは表示のみ（変更不可。仕様変更扱いのため別タスク）
- ステージID は保存後も維持（並び替えても ID は変えない）
- **非活性化フラグ** は `st.checkbox` で編集可能。後述 §2-b を参照

⑥出力確認・編集の `df_stage` 表示順、Excel 出力、Stella JSON 出力（将来）は `表示順` 昇順で並べ替えて出す。

### 2-b. ステージ非活性化フラグ

ステージマスタに **`非活性化フラグ`** カラム（bool、デフォルト False）を追加し、編集モードで切り替えられるようにする。物理削除ではなくソフトデリート方式とし、データは残しつつ表示上は除外する。

#### 振る舞い

| 場面 | 非活性化ステージの扱い |
|---|---|
| ⑥出力確認・編集の閲覧モード（`st.dataframe` 表示） | **除外** |
| 編集モードのステージマスタ UI | **表示**（グレーアウトで視覚区別）。再活性化のため |
| 集計表（duration_distribution / group_count / overlap_alerts / group_appearances） | **除外** |
| Excel 出力 | **除外** |
| Stella JSON 出力（将来） | **除外** |
| 出番マスタ表示・集計（非活性化ステージに紐づく出番行） | **連動して除外** |
| `master_stage.csv` / `turn_id_data.csv` / `stage_*.json` のデータ実体 | **保持**（フラグだけ立てる） |
| ②画像登録の `stage_list[]` | **影響なし**（OCR入力管理側は触らない） |

#### 編集 UI

§2 のカードに `st.checkbox("非活性化", ...)` を追加。非活性化済みのカードはコンテナのスタイルでグレーアウト表示（Streamlit 標準の範囲では完全グレーアウトは難しいので、ステージ名やラベルに `~~` や色付きマーカーで視覚区別）。

並び替え対象には含める（`表示順` は保持し、再活性化時に元の位置で復活）。

#### 保存処理への影響

- `master_stage.csv` に `非活性化フラグ` カラムが含まれる
- `build_event_output` の戻り値 `df_stage` / `df_live` / 集計系 DataFrame は **非活性化ステージとそれに紐づく出番を除外して生成**
- ただし `master_stage.csv` / `turn_id_data.csv` への保存は除外せず実データを保持
- 後方互換: `load_existing_masters` で `非活性化フラグ` カラムが無い既存マスタは `False` で補完

#### Phase 配置

Phase 2（ステージマスタ編集UI 導入）と同時に実装する。`master_stage.csv` のスキーマ拡張 + UI チェックボックス + `build_event_output` の除外フィルタ。

### 3. グループマスタの編集

`st.data_editor` を使用。

```
###### グループマスタ
| グループID | グループ名_採用     |
|------------|---------------------|
| 0          | [アイドルA       ]  |
| 1          | [アイドルB       ]  |
```

- `グループID` は読み取り専用
- `グループ名_採用` のみ編集可
- 行追加・削除は無効化（`num_rows="fixed"`）

### 4. 出番マスタの編集

`st.data_editor` を使用。

```
###### 出番マスタ
| 出番ID | ライブ_from | ライブ_長さ(分) | グループID | ステージID | グループ名_raw | グループ名 | ステージ名 | 備考 |
|--------|-------------|-----------------|------------|------------|----------------|------------|------------|------|
| 0      | [10:00]     | [20]            | [0 ▼]      | (read)     | (read)         | (read)     | (read)     | [  ] |
```

- **編集可能**: `ライブ_from` / `ライブ_長さ(分)` / `グループID` / `備考`
- **読み取り専用**: `出番ID` / `ステージID` / `グループ名_raw` / `グループ名` / `ステージ名`
- `グループID` は `SelectboxColumn` で選択肢制約（ID + 名前の併記表示）
- `ステージID` は Phase 5 で編集可能化（それまで disabled）
- `ライブ_from` は `HH:MM` 形式の文字列バリデーション
- `ライブ_長さ(分)` は正の整数バリデーション
- 保存時に `グループ名` カラムは グループID から再導出して上書き

### 5. 保存

「保存」ボタン押下時の処理順序:

1. ステージマスタの作業コピーから `master_stage.csv` を上書き（並び順を反映、ステージ名更新）
2. ステージ名変更を `stage_*.json` のトップレベル `ステージ名` および `project_info.json` の `stage_list[i].stage_name` に伝播
   - 対応関係は §0 で導入したトップレベル `ステージID` ⇔ `stage_list[i].stage_id` で引き当てる
3. グループマスタの作業コピーから `master_idolname.csv` を上書き
4. 出番マスタの作業コピーから `turn_id_data.csv` を上書き
5. **`stage_*.json` への書き戻し**: 出番マスタの修正（時刻 / グループ名 / ステージ振り分け）を元 JSON に反映 → §実装方針 D
6. `build_all_event_outputs` を再実行し、表示用 dataframe をリフレッシュ
7. 編集モードを OFF に戻す

### 6. キャンセル

作業コピー (`app_state.output.edits[event_name]`) を破棄し、編集モードを OFF に戻すのみ。

### 6-b. 特典会併記形式（`live_tokutenkai_heiki`）の対応

**現状の課題**:
- 1 つの JSON エントリが `ライブステージ` と `特典会[]` を両方持つ（[data_structure.md タイムテーブルJSONフォーマット§特典会併記形式](../data_structure.md#特典会併記形式)）
- ⑥出力確認・編集では `devide_df_live_tokutenkai` ([output_builder.py:371](../../src/backend_functions/output_builder.py#L371)) でライブ行と特典会行に分解される
- 編集後の出番マスタ行を元 JSON に書き戻す際、特典会行とライブ行を **同じ JSON エントリに紐付けて戻す手段がない**

**対応案: フォーマット拡張**

`特典会[]` の各要素に `対応出番ID` (=ライブ側の出番ID) を追加する:

```json
{
    "グループ名": "アイドルグループA",
    "グループ名_採用": "アイドルグループA",
    "ライブステージ": { "from": "10:00", "to": "10:20" },
    "特典会": [
        {
            "from": "10:30",
            "to": "11:30",
            "ブース": "A",
            "対応出番ID": 1
        }
    ],
    "出番ID": 1
}
```

これにより:
- 編集後の特典会行（出番マスタ上の独立行）を元 JSON に書き戻す際、`対応出番ID` から正しい親エントリを特定できる
- 同一アーティストが複数の特典会枠を持つ場合（`特典会[]` が 2 要素以上）にも対応可能

**採番タイミング**:
- `対応出番ID` は `determine_id_master` ([output_builder.py:492](../../src/backend_functions/output_builder.py#L492)) で `出番ID` を採番するのと同じタイミングで、親エントリの `出番ID` を `特典会[]` 各要素にコピーする形で埋める
- 編集モードは IDマスタ確定後のみ起動するため、編集時点で `対応出番ID` は必ず存在することが保証される
- 既存プロジェクト（マスタ確定済み）には `determine_id_master` を再実行することで `対応出番ID` を埋められる。または初回 ⑥出力アクセス時に補完処理を呼ぶ

**書き戻し側の変更**:
- `_propagate_live_edits_to_json` で、出番マスタ行の `特典会フラグ` を見て:
  - ライブ行（特典会フラグ=False）→ 親エントリの `ライブステージ.from/to` 等を更新
  - 特典会行（特典会フラグ=True）→ `対応出番ID` に該当する親エントリの `特典会[]` のうち、ブース名一致または順序一致で要素を特定して更新

これらの実装は Phase 4-5 で対応（Phase 1-3 では特典会併記イベントのみ出番マスタ編集を無効化）。

**ステージIDの二層構成との関係**:
- §0-2 で導入したとおり、`live_tokutenkai_heiki` 形式では JSON トップレベル `ステージID` がライブステージID、`特典会[].ステージID` がブース別ID。
- 編集モードでステージマスタの並び替えやステージ名変更を行った場合:
  - ライブステージのID変更があれば JSON トップレベルを更新
  - ブース由来のステージID変更があれば該当する `特典会[].ステージID` を更新
- ブースは ⑥出力上ではステージマスタの 1 行として表示されるため、編集 UI 上は両者を区別せず扱える（特典会フラグで内部的に振り分け）

### 7. バリデーション

保存前に以下をチェック。エラー時は `st.error` で警告して保存を中断:

- ステージ名 / グループ名_採用 が空でない
- ステージ名 / グループ名_採用 がイベント内で重複していない
- `ライブ_from` が `HH:MM` 形式
- `ライブ_長さ(分)` が正の整数
- 選択された `グループID` / `ステージID` がマスタに存在する

---

## 実装方針

### A. 編集モード状態管理

[app_state.py](../../src/app_state.py) の `OutputState` に以下を追加:

```python
@dataclass
class OutputState:
    output_df: dict[str, dict[str, pd.DataFrame]] | None = None
    new_idolname: pd.DataFrame | None = None
    # 追加: イベント名 -> 編集中の作業コピー
    edits: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)
```

編集モード ON 切替時に `output_df[event_name]` のディープコピーを `edits[event_name]` に格納。

### B. ステージマスタの拡張カラム + 並び替え UI

#### B-1. `表示順` (stageOrder) カラム追加

`master_stage.csv` のスキーマを拡張:

| カラム | 型 | 説明 |
|--------|-----|------|
| ステージID | int | (既存／index) |
| ステージ名 | str | (既存) |
| 特典会フラグ | bool | (既存) |
| **表示順** | int | **新規。0始まりの連番** |
| **非活性化フラグ** | bool | **新規。デフォルト False。True のステージは ⑥出力・Excel・Stella JSON から除外（§2-b）** |

`find_or_create_stage_id` ([output_builder.py:42](../../src/backend_functions/output_builder.py#L42)) で新規ステージを追加する際、`表示順 = 既存の最大値 + 1` を初期値として設定する。

`build_event_output` の `df_stage` / `df_live` 生成後、`df_stage` を `表示順` 昇順でソートして返す。既存マスタの読み込み ([output_builder.py:49-80](../../src/backend_functions/output_builder.py#L49-L80)) で `表示順` カラムが無い場合は `ステージID` をそのまま `表示順` として補完（後方互換）。

#### B-2. ↑↓ ボタンによる並び替え UI

`streamlit-sortables` などの追加依存は使わず、Streamlit 標準で実装:

```python
for display_pos, (stage_id, row) in enumerate(
    edits["stage"].sort_values("表示順").iterrows()
):
    cols = st.columns([1, 1, 1, 6, 2])
    with cols[0]:
        st.button("↑", key=f"up_{event_name}_{stage_id}",
                  disabled=(display_pos == 0),
                  on_click=_move_stage_up, args=(event_name, stage_id))
    with cols[1]:
        st.button("↓", key=f"dn_{event_name}_{stage_id}",
                  disabled=(display_pos == n_stages - 1),
                  on_click=_move_stage_down, args=(event_name, stage_id))
    with cols[2]:
        st.markdown(f"ID:{stage_id}")
    with cols[3]:
        new_name = st.text_input("ステージ名", value=row["ステージ名"],
                                 key=f"stage_name_{event_name}_{stage_id}",
                                 label_visibility="collapsed")
        edits["stage"].at[stage_id, "ステージ名"] = new_name
    with cols[4]:
        st.markdown(f"特典会:{'☑' if row['特典会フラグ'] else '☐'}")
```

`_move_stage_up/down` は `edits["stage"]` の `表示順` を 2 行スワップして 0 始まりで再採番する。

#### B-3. stella_json_output_plan.md との関係

[stella_json_output_plan.md §2-3](stella_json_output_plan.md) では `stageOrder` を `stageNameShort` / `colorName` と並ぶステージマスタ拡張カラムとして計画。本計画では `stageOrder`（= `表示順`）のみを先行導入する。stella 計画側で `stageNameShort` / `colorName` を追加する際は、本計画で導入した `表示順` カラムをそのまま利用できる。

### C. 編集 UI（グループ / 出番）

`st.data_editor` の `column_config` でカラムごとの編集可否・型・選択肢を制御。

```python
edited_idolname = st.data_editor(
    edits["idolname"],
    column_config={
        "グループID": st.column_config.NumberColumn(disabled=True),
        "グループ名_採用": st.column_config.TextColumn(required=True),
    },
    num_rows="fixed",
    key=f"idolname_editor_{event_name}",
)
```

出番マスタの `SelectboxColumn` 用に「ID: 名前」形式の選択肢リストを動的生成。

### D. 保存処理: マスタ CSV + stage_*.json への書き戻し

新規モジュール [src/backend_functions/output_editor.py](../../src/backend_functions/output_editor.py) を追加。

```python
def save_event_edits(
    pj_path: str,
    event_name: str,
    event_no: int,
    project_info_json: dict,
    edits: dict[str, pd.DataFrame],
) -> None:
    """編集結果を master_*.csv と stage_*.json に書き戻す。"""
    output_path = os.path.join(pj_path, event_name)

    # 1. master CSV を保存
    edits["stage"].to_csv(os.path.join(output_path, "master_stage.csv"))
    edits["idolname"].to_csv(os.path.join(output_path, "master_idolname.csv"))
    edits["live"].to_csv(os.path.join(output_path, "turn_id_data.csv"))

    # 2. 出番マスタの修正を stage_*.json に反映
    _propagate_live_edits_to_json(
        pj_path, event_name, event_no, project_info_json, edits["live"], edits["stage"],
    )

    # 3. project_info の stage_list の並び順をマスタの並び順に追従させるか?
    #    → §残オープン論点 §2
```

`_propagate_live_edits_to_json` は以下を実施:

- 出番ID をキーに `stage_*.json` の各エントリを引き当て
- `ライブステージ.from` / `to` / `グループ名_採用` を更新
- `ステージID` 変更があった場合は、対応する別 `stage_*.json` へ移動（削除 + 追加）

ステージ移動は副作用が大きいため、最初の Phase ではグループ名 / 時刻の修正のみ反映し、ステージID 変更は「未対応」として警告表示する案も検討する → §残オープン論点 §3

### E. UI 統合 (`app.py`)

[app.py:1284-1356](../../src/app.py#L1284-L1356) `render_output_section` をリファクタ:

- ヘルパー関数 `_render_event_output_view(event_name, data)` と `_render_event_output_editor(event_name, data)` に分離
- トグル状態に応じてどちらかを呼び出す
- 編集モード時は集計セクションを下に折りたたみ表示（編集中の負荷を減らす）

### F. workflow 追加

[src/workflow.py](../../src/workflow.py) に編集系メソッドを追加:

```python
class ProjectWorkflow:
    def enter_output_edit_mode(self, event_name: str): ...
    def cancel_output_edit_mode(self, event_name: str): ...
    def save_output_edits(self, event_name: str): ...
```

`save_output_edits` 内で `output_editor.save_event_edits` を呼び、その後 `build_all_event_outputs` を再実行。

---

## 影響範囲

**Phase 0（データモデル変更）の影響**:

| ファイル | 変更内容 |
|---|---|
| [src/backend_functions/output_builder.py](../../src/backend_functions/output_builder.py) | `find_or_create_stage_id` を `existing_stage_id` パラメータ対応に再編（§0-4）。`build_event_output` で `stage_*.json` トップレベル `ステージID` を取得して引き渡し。`determine_id_master` で `stage_*.json` トップレベルおよび `project_info.stage_list[i].stage_id` への `ステージID` 書き戻しを追加 |
| [src/backend_functions/timetabledata.py](../../src/backend_functions/timetabledata.py) | `json_to_df` / `df_to_json` で出番粒度の `ステージID` カラム受け渡しを廃止。`id_apply_to_json` でトップレベル `ステージID` を書き込む処理に変更 |
| [src/backend_functions/project_migration.py](../../src/backend_functions/project_migration.py) | 既存 `stage_*.json` のステージIDトップレベル化、`project_info.stage_list[i].stage_id` 補完のマイグレーション追加 |
| [src/backend_functions/project_repository.py](../../src/backend_functions/project_repository.py) | `stage_list` 関連アクセサに `stage_id` 取得/設定メソッドを追加。`create_project_data` のスキーマに `stage_id` を含める |
| [src/gpt_output_format/timetable_format.py](../../src/gpt_output_format/timetable_format.py) | GPT 出力スキーマから出番粒度の `ステージID` を削除（OCR は出番粒度ステージIDを生成しない） |
| [docs/data_structure.md](../data_structure.md) | `stage_*.json` のステージID位置変更、`stage_list[i].stage_id`、`特典会[].対応出番ID` を反映 |

**Phase 1-5（編集モード本体）の影響**:

| ファイル | 変更内容 |
|---|---|
| [src/app.py:1284-1356](../../src/app.py#L1284-L1356) | `render_output_section` をビュー／エディタに分割、編集モードトグル追加、ページ遷移警告 |
| [src/app_state.py](../../src/app_state.py) | `OutputState.edits` フィールド追加、`UIState.last_page` 追加（遷移警告用） |
| [src/workflow.py](../../src/workflow.py) | 編集モード制御メソッド追加 |
| (新規) [src/backend_functions/output_editor.py](../../src/backend_functions/output_editor.py) | 編集結果を CSV / JSON / project_info に書き戻すロジック |
| [src/backend_functions/output_builder.py](../../src/backend_functions/output_builder.py) | `find_or_create_stage_id` で `表示順` を採番、`build_event_output` で `df_stage` を `表示順` 昇順にソート、`load_existing_masters` で `表示順` カラムを後方互換補完。`determine_id_master` で `特典会[].対応出番ID` を埋める処理を追加 |
| [src/gpt_output_format/timetable_format.py](../../src/gpt_output_format/timetable_format.py) | GPT 出力スキーマに `特典会[].対応出番ID` を追加（OCR 段階では None、ID確定時に埋める） |
| [src/backend_functions/timetabledata.py](../../src/backend_functions/timetabledata.py) | `json_to_df` / `df_to_json` で `対応出番ID` を受け渡し |
| (新規) `tests/backend_functions/test_output_editor.py` | 書き戻しロジックの単体テスト |
| (新規) `tests/backend_functions/test_stage_id_migration.py` | Phase 0 マイグレーションの単体テスト |

---

## 段階的実装案

0. **Phase 0**: データモデル変更（§0）。
   - `stage_*.json` トップレベルへの `ステージID` 昇格、出番粒度からの撤去
   - `project_info.stage_list[i].stage_id` 追加
   - `find_or_create_stage_id` の `existing_stage_id` 対応
   - `determine_id_master` で新しい書き込み先に対応
   - 既存プロジェクトのマイグレーション + 単体テスト
   - 後続 Phase の前提となるため最初に実施。Phase 0 完了時点でも既存機能は動作する（編集UIはまだ無い）

1. **Phase 1**: ステージマスタに `表示順` カラム追加。`find_or_create_stage_id` / `build_event_output` / `load_existing_masters` を `表示順` 対応に。`output_editor.save_event_edits` を実装（マスタ CSV 書き戻しのみ）+ 単体テスト。
2. **Phase 2**: ⑥出力確認・編集タブに編集モードトグル（IDマスタ確定後のみ表示）+ ステージ並べ替え UI（↑↓ボタン）+ ステージ名編集 + 保存／キャンセル + ページ遷移警告。ステージ名変更は §0 のロジックにより `master_stage.csv` 更新 → 次回 `build_event_output` 時に各 stage_*.json 由来の名前もマスタ側で追随更新される。
3. **Phase 3**: グループマスタ編集 UI 追加。グループ名_採用 の更新を `stage_*.json` の `グループ名_採用` フィールドに伝播。
4. **Phase 4**: 出番マスタ編集 UI 追加（時刻・備考・グループID）。`stage_*.json` への書き戻し。特典会併記形式の `対応出番ID` 採番ロジックを `determine_id_master` に追加（Phase 0 でスキーマは入っている）。
5. **Phase 5**: 出番マスタの `ステージID` 変更を `stage_*.json` 間でのエントリ移動として実装。リスク高のため最後。

---

## 確定済み論点

1. **マスタ確定前の編集可否**: 編集モードトグルは IDマスタ確定済み（`master_*.csv` 全揃い）のときのみ表示。
2. **ステージ並び順 vs `stage_list`**: project_info の `stage_list[]` の配列順は触らない。ステージマスタに `表示順` カラムを追加し、これを並び替え対象とする。`stage_list[]` は ②画像登録の入力順を保持し、`表示順` は ⑥出力確認・編集 / Excel / Stella JSON の表示順を司る。
3. **ステージID 変更の扱い**: Phase 5 で実装済。`ステージID` 編集時、親エントリ (ライブ / 非heiki 特典会 / heiki ライブ親) は `タイムテーブル[i]` ごと別 stage_*.json に移動。heiki booth は `特典会[j].ステージID` を in-place 更新。kind 跨ぎ (live ⇔ heiki) は不許可。
4. **特典会併記形式の書き戻し**: `live_tokutenkai_heiki` 形式の `特典会[]` 各要素に `対応出番ID` を追加するフォーマット拡張で対応（§6-b）。`対応出番ID` の採番は `determine_id_master` で `出番ID` 採番と同時に行う。編集モードはマスタ確定後のみ起動するため、編集時点で `対応出番ID` は必ず埋まっている。
5. **編集中の他ページ遷移**: 警告ダイアログ（保存して移動／破棄して移動／キャンセル）を表示。
6. **並び替え UI**: ↑↓ ボタン方式（`streamlit-sortables` 等の追加依存は不要）。
7. **ステージ名変更時の二重採番問題**: §0 のデータモデル変更（ステージIDのトップレベル化 + `find_or_create_stage_id` の `existing_stage_id` 引き当て対応）で根本解決。
8. **ステージID保持場所**: `stage_*.json` トップレベル（親=ライブステージID）+ `特典会[].ステージID`（子=ブース別ID）の二層構成。`project_info.stage_list[i].stage_id` にも親IDを保持。
9. **ステージ削除のニーズ**: ソフトデリート方式の `非活性化フラグ` で対応（§2-b）。物理削除はせず、出力系から除外するのみ。再活性化可能。
10. **イベント単位でのID独立性**: 現状仕様通り（`load_existing_masters` がイベントディレクトリ単位）。ブース別ステージIDも `find_or_create_stage_id` でイベント単位独立採番。
11. **マイグレーション方式**: 旧形式（出番粒度ステージID）と新形式（トップレベル）の判別は **フィールド有無**（トップレベル `ステージID` の存在）で行う。マイグレーションは冪等。

---

## 残オープン論点

現時点で残課題なし。実装着手後に発見された論点をここに追記する。

---

## 関連計画

- [timetable_order_plan.md](timetable_order_plan.md): タイムテーブル（画像 / kind）の並び順。本計画とは対象データが異なる（あちらは `timetables[]`、こちらは `master_stage`）が、「↑↓ UI」「同一イベント内のみ並び替え可能」の設計思想を共有する。
- [stella_json_output_plan.md](stella_json_output_plan.md): Stella JSON 出力計画。Phase 2-3 で計画されている `stageOrder` カラムを本計画で先行導入する。`stageNameShort` / `colorName` は stella 計画側で追加し、本計画の `表示順` と共存させる。
- [done/refactoring_render_output_section.md](done/refactoring_render_output_section.md): `render_output_section` のリファクタ実績。本計画でさらに編集モード分割を加える。
