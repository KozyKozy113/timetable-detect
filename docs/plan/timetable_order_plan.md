# タイムテーブル並び順 実装計画

## 目的・背景

`event_detail.timetables[]` の表示順について、現状は [project_repository.py:123-139](../../src/backend_functions/project_repository.py#L123-L139) の `get_sorted_image_no_list` が「kind 順 + image_no 昇順」で都度ソートしている。これでは以下ができない。

- ユーザーが UI 上で並び順を手動で入れ替える
- 同 kind 内で意味のある順序（例: ステージA → ステージB → ステージC）を保持する

[done/abandoned_stage_kind_and_image_restructure_plan.md](done/abandoned_stage_kind_and_image_restructure_plan.md) では `stage_kinds[]` / `images[]` の 3 層モデルと `sort_order` フィールドで対応する案を立てたが、要件に対して過剰と判断。**本計画ではよりシンプルに「`timetables[]` の配列順 = 表示順」**として実装する。

スキーマ拡張は行わず、既存の `timetables[]` 配列順序を真実の値として扱う。

---

## スコープ

- `timetables[]` の配列順を表示順として採用するためのソート関数の差し替え
- 画像登録時の **挿入位置ルール**（kind ベースのデフォルト配置）
- ② 画像登録セクションへの **手動並び替え UI（↑↓）** の追加
- 並び順リセット機能

スコープ外:
- `stage_kinds[]` / `images[]` 3 層分解（廃案）
- 1 stage_kind に複数画像、画像なし stage_kind（廃案）
- raw 画像の `raw_images/` 配下移動（現状維持）
- `stage_list[i].image_no` 追加・手動ステージ追加 UI（別タスク）

---

## 前提（変更なしのもの）

- `event_detail.timetables[]` は migration 後 list 形式 ([project_migration.py:63-100](../../src/backend_functions/project_migration.py#L63-L100))。
- 各エントリは `image_no` を ID として保持（[project_migration.py:71](../../src/backend_functions/project_migration.py#L71)）。`image_no` はファイル参照には使われず、エントリ識別子として `dir_name` と並んで参照される。
- 画像とステージ種別は **1 : 1**。
- raw 画像配置は **現状維持** (`event_N/{dir_name}/raw.png`)。
- `stage_list[i]` は `stage_no` / `stage_name` のみ。`image_no` フィールドは持たない。

---

## 仕様

### 1. 並び順の定義

**`timetables[]` の配列順 = 表示順**。`sort_order` フィールドは追加しない。

UI / OCR / output_builder などで「ソート済み一覧」が必要な場面では、配列をそのまま走査する。

### 2. デフォルト並び順ルール

新規プロジェクト・新規イベントで複数 timetable が並ぶ際の初期順序、および「並び順リセット」時の順序は以下:

| 優先度 | バケット |
|---|---|
| 0 | `dir_name == "ライブ"` |
| 1 | `kind == "live"`（`dir_name` が "ライブ" 以外） |
| 2 | `dir_name == "特典会"` |
| 3 | `kind == "tokutenkai"` / `live_tokutenkai_heiki`（`dir_name` が "特典会" 以外） |
| 4 | それ以外 |

`dir_name` 完全一致を `kind` 一致より優先する。同バケット内は登録順（既存の配列順）を維持する。

### 3. 新規登録時の挿入位置ルール

`register_timetable_image` ([project_repository.py:307-340](../../src/backend_functions/project_repository.py#L307-L340) 付近) で新エントリを追加するときの挿入位置:

挿入位置は §2 のバケット順に従い、「新エントリと同じバケットの末尾の次」に挿入する。

| 条件 | 挿入位置 |
|---|---|
| 既存 `dir_name` と一致（上書き登録） | **その場所を維持**（image_no も使い回し、現状の挙動を踏襲） |
| 新規 + `dir_name == "ライブ"` (バケット 0) | **配列先頭** (index 0) |
| 新規 + `kind == "live"` (バケット 1) | バケット 0 の末尾の直後（無ければ先頭） |
| 新規 + `dir_name == "特典会"` (バケット 2) | バケット 0/1 の末尾の直後（無ければ先頭） |
| 新規 + `kind == "tokutenkai"` / `live_tokutenkai_heiki` (バケット 3) | バケット 0/1/2 の末尾の直後 |
| 新規 + その他 (バケット 4) | **配列末尾** |

`image_no` は引き続き `next_image_no()` で採番（欠番許容 / 配列インデックスとは独立）。

### 4. 手動並び替え UI

② 画像登録セクション ([app.py:700-728](../../src/app.py#L700-L728)) の「登録済みタイムテーブル画像一覧」に ↑↓ ボタンを追加する。

```
###### 登録済みタイムテーブル画像一覧 [並び順をリセット]
- 画像数：3
┌─────────────┬─────────────┬─────────────┐
│ event_1/ライブ│ event_1/特典会│ event_2/縁日 │
│ [↑][↓][削除] │ [↑][↓][削除] │ [↑][↓][削除] │
│ <preview>    │ <preview>    │ <preview>    │
└─────────────┴─────────────┴─────────────┘
```

- **↑**: 配列内で 1 つ前のエントリと swap（先頭なら無効）
- **↓**: 配列内で 1 つ後のエントリと swap（末尾なら無効）
- **並び順リセット**: イベントごと、上記デフォルトルールで再ソート
- 並び替えはイベント横断ではなく、**同一 event 内のみ**

イベントを跨ぐ並び替えは行わない（既存のイベント単位列レイアウトを維持）。

---

## 実装方針

### A. ソート関数の差し替え

[project_repository.py:123-139](../../src/backend_functions/project_repository.py#L123-L139) `get_sorted_image_no_list` を「配列順そのまま」に変更:

```python
def get_sorted_image_no_list(project_info_json, event_no):
    """timetables[] の配列順で image_no リストを返す。"""
    entries = get_image_entry_list(project_info_json, event_no)
    return [e["image_no"] for e in entries]
```

旧仕様（kind 順ソート）が必要な箇所は「デフォルト並び順計算ヘルパー」(`_default_sort_key`) として分離し、新規登録の挿入位置決定と reset 関数のみで使用する。

### B. 新規追加: 挿入・並び替えロジック

```python
# 新規画像の挿入位置を返す
def compute_insert_index(project_info_json, event_no, dir_name, kind) -> int

# 同一イベント内で隣接 swap
def move_timetable_up(project_info_json, event_no, image_no) -> None
def move_timetable_down(project_info_json, event_no, image_no) -> None

# デフォルトルールで再ソート
def reset_timetable_order(project_info_json, event_no) -> None
```

挿入位置決定は `(kind バケット, 既存配列上の最後の同バケット位置 + 1)` で計算。

### C. `register_timetable_image` の修正

新エントリの append を、`compute_insert_index` の結果での `list.insert(idx, entry)` に置き換える。既存 dir_name 上書きパスは現状通り（位置維持）。

### D. UI 追加 (`app.py`)

[app.py:700-728](../../src/app.py#L700-L728) の登録済み一覧ループに、各カラム内へ ↑↓ ボタンと、見出し横に「並び順をリセット」ボタンを追加。

- ↑↓ は `workflow.ProjectWorkflow.move_timetable_up/down` 経由
- リセットは `workflow.ProjectWorkflow.reset_timetable_order` 経由
- 各ハンドラは並び替え後に `save_project_info()` を呼ぶ

### E. workflow.py 追加メソッド

```python
class ProjectWorkflow:
    def move_timetable_up(self, event_no, image_no): ...
    def move_timetable_down(self, event_no, image_no): ...
    def reset_timetable_order(self, event_no): ...
```

---

## 影響範囲

| ファイル | 変更内容 |
|---|---|
| [src/backend_functions/project_repository.py](../../src/backend_functions/project_repository.py) | `get_sorted_image_no_list` を配列順返却に変更 / `compute_insert_index` / `move_timetable_up/down` / `reset_timetable_order` 追加 / `register_timetable_image` で挿入位置使用 |
| [src/workflow.py](../../src/workflow.py) | `move_timetable_up/down` / `reset_timetable_order` メソッド追加 |
| [src/app.py:700-728](../../src/app.py#L700-L728) | 登録済み一覧に ↑↓ と「並び順をリセット」ボタン追加 |
| [tests/backend_functions/test_project_repository_accessors.py](../../tests/backend_functions/test_project_repository_accessors.py) | `get_sorted_image_no_list` のテストを「配列順」基準に書き換え |
| (新規) `tests/backend_functions/test_timetable_order.py` | `compute_insert_index` / `move_*` / `reset_*` の単体テスト |

OCR / output_builder 等の `get_sorted_image_no_list` 利用箇所は **挙動が「kind 順」から「配列順」に変わる**。配列順が常にデフォルトルールに従う初期状態を保てるよう、register 経路で挿入位置を制御するのが要。

---

## 段階的実装案

1. **Phase 1**: `compute_insert_index` / `reset_timetable_order` / `move_timetable_up/down` を実装 + 単体テスト。`get_sorted_image_no_list` を配列順返却に変更。
2. **Phase 2**: `register_timetable_image` を `compute_insert_index` 経由に変更。既存プロジェクトを開いて初期表示が崩れないことを確認（既存配列順が kind ベースで並んでいなければ「並び順リセット」で揃える運用）。
3. **Phase 3**: ② UI に ↑↓ ボタンと「並び順をリセット」ボタンを追加。workflow メソッド経由でハンドリング。
4. **Phase 4**: 既存プロジェクトの一括マイグレーション（任意）。初回ロード時に `reset_timetable_order` を呼んで配列順をデフォルトルールに揃えるかどうかを決定。揃えないなら現状の配列順をそのまま尊重。

---

## 確定済み論点

1. **配列順 = 表示順**: `sort_order` フィールドは追加しない。
2. **画像 : ステージ種別 = 1 : 1** を維持。
3. **デフォルト順**: ライブ → 特典会 → それ以外。
4. **挿入ルール**: ライブ=先頭、特典会=ライブ直後、その他=末尾。
5. **手動並び替え UI**: ② に ↑↓ + リセット。同一イベント内のみ。
6. **raw 画像配置**: 現状維持 (`event_N/{dir_name}/raw.png`)。
7. **`stage_list[i].image_no` 追加・手動ステージ追加**: 別タスク。

---

## 残オープン論点

1. **初回ロード時の自動リセット**: 既存プロジェクトを開いたとき、配列順をデフォルトルールで強制的に並べ直すか、現状の配列順を尊重するか。デフォルトは「尊重」とし、ユーザーが「並び順をリセット」を押した時のみ揃える方針を推奨。
2. **イベント跨ぎの並び替え**: 本計画ではスコープ外。必要になれば別タスクで対応。

---

## 関連計画

- [done/abandoned_stage_kind_and_image_restructure_plan.md](done/abandoned_stage_kind_and_image_restructure_plan.md): 3 層モデル案。**本計画の採用により廃案**。同ファイル内の「kind 順デフォルト」「ライブ・特典会優先」の考え方のみ本計画に継承。
