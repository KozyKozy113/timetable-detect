# ステージ種別 × 画像 概念分離 実装計画

## 目的・背景

現状の `project_info.json` は「登録した 1 枚の画像」「その画像から派生するステージ群」「ステージ種別（live / tokutenkai / live_tokutenkai_heiki / 縁日 等）」がすべて `event_detail.timetables[]` の 1 エントリに混在している。これは以下のケースで破綻する。

- 同一の「ステージ種別」に複数枚の画像をアップロードしたい（例: 縁日の領域が画像内に分散しており別撮影で 2 枚登録したい）
- ステージ種別を画像なしで登録したい（例: 縁日のステージを手動入力だけで作りたい）
- 個別ステージを画像由来でなく手動で追加したい
- ステージ種別の並び順と画像の並び順を独立に扱いたい場合がある

本計画では `event_detail` を **3 層モデル** に再構成する。

- **stage_kinds[]**: ステージ種別。ライブ / 特典会 / ライブ特典会 / 縁日 など。**旧 `dir_name` が相当**。0..N 枚の画像を持つ。stage_list を所有する。
- **images[]**: 登録された raw 画像。必ず 1 つの stage_kind に所属する。
- **stage_kinds[].stage_list[]**: 個別ステージ。任意で image_no を参照する（画像由来 / 手動）。

これに伴い raw 画像のディスク配置も `event_N/raw_images/image_{image_no}.png` に分離する。raw_cropped・stage_*.png・stage_*.json は引き続き `event_N/{dir_name}/` 配下（= stage_kind 配下）に置く。

過去計画との関係は末尾「廃案・関連計画」を参照。

---

## スコープ

設計のみ。実装は別タスク。

「ステージ種別を画像登録なしで単独追加する UI」「同一 stage_kind に複数画像を割り当てる UI」は本計画ではスキーマで先回りするのみで、UI 実装は未スコープ。

---

## 概念モデル

```
event_detail
├── stage_kinds[]                       (種別: ライブ / 特典会 / 縁日 ...)
│   ├── stage_kind_no
│   ├── dir_name          ← 旧 dir_name
│   ├── display_name
│   ├── kind              ← live / tokutenkai / live_tokutenkai_heiki
│   ├── sort_order        ← 種別の並び順
│   └── stage_list[]
│       ├── stage_no
│       ├── stage_name
│       ├── image_no      ← null 可。画像由来時のみ値が入る
│       └── bbox          ← image_no がある時のみ
└── images[]                            (登録した raw 画像)
    ├── image_no
    ├── stage_kind_no     ← どの種別の素材か
    ├── format            ← 通常 / ライムライト式 (heiki なら省略)
    ├── sort_order        ← 画像の並び順
    ├── raw_crop_box      ← クロップ枠（画像単位）
    └── time_pixel        ← 時間軸対応（画像単位）
```

### 関係性まとめ

| 関係 | カーディナリティ | 備考 |
|---|---|---|
| stage_kind : image | 1 : 0..N | 画像 0 枚の stage_kind を許容 |
| stage_kind : stage_list | 1 : 1 | stage_list は stage_kind に所有される |
| stage : image | N : 0..1 | 各ステージは任意で 1 枚の画像に紐付く |
| image : stage_kind | N : 1 | 画像は必ず 1 つの種別に属する |

---

## 新スキーマ詳細

### event_detail 全体

```json
{
    "event_no": 0,
    "event_name": "event_1",
    "ticket_urls": [],
    "stage_kinds": [
        {
            "stage_kind_no": 0,
            "dir_name": "ライブ",
            "display_name": "ライブ",
            "kind": "live",
            "sort_order": 0,
            "stage_num": 5,
            "stage_list": [
                {"stage_no": 0, "stage_name": "STAGE-A", "image_no": 0, "bbox": [...]},
                {"stage_no": 1, "stage_name": "STAGE-B", "image_no": 0, "bbox": [...]}
            ]
        },
        {
            "stage_kind_no": 1,
            "dir_name": "特典会",
            "display_name": "特典会",
            "kind": "tokutenkai",
            "sort_order": 1,
            "stage_num": 3,
            "stage_list": [
                {"stage_no": 0, "stage_name": "BOOTH-1", "image_no": 1, "bbox": [...]},
                {"stage_no": 1, "stage_name": "BOOTH-2", "image_no": null, "bbox": null}
            ]
        },
        {
            "stage_kind_no": 2,
            "dir_name": "縁日",
            "display_name": "縁日",
            "kind": "tokutenkai",
            "sort_order": 2,
            "stage_num": 0,
            "stage_list": []
        }
    ],
    "images": [
        {
            "image_no": 0,
            "stage_kind_no": 0,
            "format": "通常",
            "sort_order": 0,
            "raw_crop_box": {...},
            "time_pixel": {...}
        },
        {
            "image_no": 1,
            "stage_kind_no": 1,
            "format": "通常",
            "sort_order": 1,
            "raw_crop_box": {...}
        }
    ]
}
```

### stage_kinds[] フィールド

| フィールド | 型 | 説明 |
|---|---|---|
| `stage_kind_no` | int | イベント内一意の整数連番。欠番許容。 |
| `dir_name` | str | ステージ派生物の保存先サブフォルダ名。同イベント内で一意。`"raw_images"` は予約名で登録不可。 |
| `display_name` | str | UI 表示名。原則 `dir_name` と同値。 |
| `kind` | str | `"live"` / `"tokutenkai"` / `"live_tokutenkai_heiki"`。並び順デフォルト計算と特典会フラグ導出に使う。 |
| `sort_order` | int | 種別の表示順。0..N-1 の純列。 |
| `stage_num` | int | `len(stage_list)` を反映。 |
| `stage_list` | list[dict] | 個別ステージのリスト。 |

### stage_list[] フィールド

| フィールド | 型 | 説明 |
|---|---|---|
| `stage_no` | int | 該当 stage_kind 内一意の整数連番。 |
| `stage_name` | str | ステージ名。 |
| `image_no` | int \| null | 画像由来のとき該当 image_no、手動追加なら null。 |
| `bbox` | list \| null | 親画像内での bbox。`image_no` が null なら null。 |
| `kind` | str | 旧 image_kind_plan で導入された stage 個別 kind。基本は親 stage_kind.kind を継承。 |

### images[] フィールド

| フィールド | 型 | 説明 |
|---|---|---|
| `image_no` | int | イベント内一意の整数連番。欠番許容。 |
| `stage_kind_no` | int | 所属 stage_kind の `stage_kind_no`。必須 (非 null)。 |
| `format` | str | `"通常"` / `"ライムライト式"`。stage_kind.kind=heiki ならフィールド省略。 |
| `sort_order` | int | 画像の表示順。0..N-1 の純列。 |
| `raw_crop_box` | dict | クロップ枠。 |
| `time_pixel` | dict | 時間軸ピクセル対応。 |

---

## ファイル配置

### 基本

```
event_1/
  raw_images/
    image_0.png         ← image_no=0 の raw
    image_1.png         ← image_no=1 の raw
  ライブ/               ← stage_kind_no=0 (dir_name="ライブ")
    raw_cropped.png
    stage_0.png, stage_0.json, ...
  特典会/               ← stage_kind_no=1
    raw_cropped.png
    stage_0.png, stage_0.json, ...
  縁日/                 ← stage_kind_no=2 (画像なし)
    (空、または手動追加された stage_*.json のみ)
```

### 多画像 stage_kind の扱い（将来）

1 つの stage_kind に画像が 2 枚以上ある場合、`raw_cropped.png` 等の画像スコープなアセットが衝突する。本計画では以下のルールで対応する。

- **0 or 1 枚の場合（現状）**: 従来通り `{dir_name}/raw_cropped.png`。
- **2 枚以上の場合**: 以下のいずれか（実装フェーズで詳細化）。
  - 案A: `{dir_name}/image_{image_no}/raw_cropped.png` のように画像ごとにサブフォルダを切る。
  - 案B: `{dir_name}/raw_cropped_{image_no}.png` のように接尾辞で識別。

`stage_*.png` / `stage_*.json` は **stage_kind スコープ**（`{dir_name}/stage_{stage_no}.png`）とする。複数画像から派生したステージでも、stage_no は stage_kind 全体での連番なので衝突しない。生成時はそれぞれの `image_no` + `bbox` を見て対応する raw からクロップする。

現状実装は **0 or 1 枚前提** で進め、多画像対応は別タスクで詰める（残オープン論点）。

---

## 並び順の同期ルール

### 基本方針

- `stage_kinds[].sort_order` と `images[].sort_order` は別々に値を持つ。
- ユーザーがどちらか片方を並び替えた瞬間、**もう片方をベストエフォートで連動更新する**（同期方向は「触った側 → 触らなかった側」の一方向）。
- 完全同期できない場合（画像なし stage_kind の存在、1 stage_kind に複数画像）は、部分的にだけ同期する。
- 状態フラグ（linked/unlinked）は持たない。常に連動を試みる方針で、ユーザーが「連動させたくない」場合は触らなかった側を再度手動調整する。

### A. stage_kinds を並び替えた場合

```python
def reorder_stage_kinds(new_order: list[stage_kind_no]):
    # 1. stage_kinds[].sort_order を new_order 通りに再割当て
    for i, sk_no in enumerate(new_order):
        stage_kinds[sk_no].sort_order = i

    # 2. images[].sort_order を再計算
    #    stage_kinds の新しい順に各種別の images を current 相対順で並べ直す
    new_image_order = []
    for sk_no in new_order:
        imgs = sorted(
            [img for img in images if img.stage_kind_no == sk_no],
            key=lambda img: img.sort_order,
        )
        new_image_order.extend(imgs)
    for i, img in enumerate(new_image_order):
        img.sort_order = i
```

### B. images を並び替えた場合

```python
def reorder_images(new_order: list[image_no]):
    # 1. images[].sort_order を new_order 通りに再割当て
    for i, img_no in enumerate(new_order):
        images[img_no].sort_order = i

    # 2. stage_kinds[].sort_order を再計算
    #    images の新しい順を走査し、初出 stage_kind_no を順に並べる
    new_sk_order = []
    seen = set()
    for img_no in new_order:
        sk_no = images[img_no].stage_kind_no
        if sk_no not in seen:
            new_sk_order.append(sk_no)
            seen.add(sk_no)

    # 3. 画像のない stage_kind は元の sort_order 相対順を維持し、
    #    new_sk_order に登場しない順序で末尾寄りに差し込む
    imageless = [sk for sk in stage_kinds if sk.stage_kind_no not in seen]
    imageless.sort(key=lambda sk: sk.sort_order)  # 元の相対順を保つ
    for sk in imageless:
        # 元の相対位置（current sort_order）を尊重し最寄り位置に差し込む
        _insert_imageless_sk(new_sk_order, sk)

    for i, sk_no in enumerate(new_sk_order):
        stage_kinds[sk_no].sort_order = i
```

「画像のない stage_kind の差し込み位置」は厳密には不定なので、シンプルに **元の sort_order 値を基準に二分探索で挿入** する程度の実装で十分（端のケースは末尾扱い）。

### C. 新規 image 登録

新 image の `sort_order` を計算するときは、所属 stage_kind の sort_order と、同 stage_kind 内の既存画像の sort_order を見て、`(所属 stage_kind のブロック内末尾 + 1)` に入れる。挿入後は両側の sort_order を 0..N-1 純列に振り直す。

### D. 新規 stage_kind 登録

`sort_order` は [image_kind_and_stage_kind_plan.md](image_kind_and_stage_kind_plan.md) の kind 順 + dir_name 完全一致優先のバケットルール（[image_display_order_plan.md](image_display_order_plan.md) で定義済み）でデフォルト位置を決める。

### E. リセット

`reset_sort_order(event_no)`: stage_kinds を `(bucket, stage_kind_no)` で並べ直し、images の sort_order を A の手順で再計算。

---

## 画像登録 UI の刷新

### ② 画像登録セクション [src/app.py:738-762](../../src/app.py#L738-L762)

種別選択を「既存 stage_kind を選択 or 新規作成」UI に変更。

```
[イベント]   event_1
[ステージ種別]  ●既存 ○新規作成
   ●既存の場合: selectbox(既存 stage_kind 一覧 by sort_order)
   ○新規作成の場合: text_input(dir_name) + radio(kind: ライブ/特典会/ライブ特典会併記/その他...)
[形式]       通常 / ライムライト式  (stage_kind.kind == heiki なら disabled)
[ファイル]   <uploaded_image>
[登録]
```

「新規作成」を選んだとき、`dir_name` の重複は警告。`kind` 選択肢は radio で 4 種（ライブ・特典会・ライブ特典会併記・その他）。「その他」は内部的にどの enum 値になるかをさらに選択するか、別途検討（実装フェーズ）。

### 登録済み画像一覧（②内）

stage_kind 単位でグルーピング表示する。

```
■ ライブ          [↑][↓][並び順リセット]
   ├ image_0.png   [↑][↓][削除]
   ├ image_3.png   [↑][↓][削除]
■ 特典会          [↑][↓]
   ├ image_1.png   [↑][↓][削除]
■ 縁日 (画像なし)  [↑][↓]
   (画像なし)
```

- 外側 ↑↓: stage_kind の並び替え → `reorder_stage_kinds()`
- 内側 ↑↓: 画像の並び替え → `reorder_images()`
- 「並び順リセット」: イベント単位でデフォルトルールに戻す

「画像なし stage_kind」は (画像なし) と表示。stage_kind 単独追加 UI は未実装（残オープン論点）。

---

## ロジック設計

### project_repository.py — アクセサ層

旧 `timetables` ベースの関数群を全廃し、新スキーマアクセサに置き換える。

#### 新規追加

```python
# stage_kind 系
def get_stage_kind_list(pij, event_no) -> list[dict]
def get_stage_kind_by_no(pij, event_no, sk_no) -> dict | None
def get_stage_kind_by_dir_name(pij, event_no, dir_name) -> dict | None
def get_sorted_stage_kind_no_list(pij, event_no) -> list[int]
def next_stage_kind_no(pij, event_no) -> int
def find_stage_kind_dir_name_conflict(pij, event_no, dir_name) -> int | None

def register_stage_kind(pij, event_no, dir_name, kind, sort_order=None) -> int
def delete_stage_kind(pij, event_no, sk_no) -> None    # 紐づく images も一括削除

# image 系
def get_image_list(pij, event_no) -> list[dict]
def get_image_by_no(pij, event_no, image_no) -> dict | None
def get_images_for_stage_kind(pij, event_no, sk_no) -> list[dict]   # sort_order 順
def get_sorted_image_no_list(pij, event_no) -> list[int]
def next_image_no(pij, event_no) -> int

def register_image_entry(pij, event_no, sk_no, format, raw_crop_box=None) -> int
def delete_image_entry(pij, event_no, image_no) -> None    # stage_list 内 image_no=null 化

# stage 系
def get_stage_list(pij, event_no, sk_no) -> list[dict]
def get_stage_name(pij, event_no, sk_no, stage_no) -> str
def get_stages_for_image(pij, event_no, image_no) -> list[dict]   # 親 sk を跨ぐ可能性は無いが念のため

# 並び順
def reorder_stage_kinds(pij, event_no, new_order: list[int]) -> None
def reorder_images(pij, event_no, new_order: list[int]) -> None
def reset_sort_order(pij, event_no) -> None

# パス
def raw_image_path(pj_path, event_name, image_no) -> str
def stage_kind_dir(pj_path, event_name, dir_name) -> str
```

#### 削除/書き換え

| 旧関数 | 新挙動 |
|---|---|
| `get_event_type_list` | `[sk.dir_name for sk in get_sorted_stage_kind_no_list()]` を返す互換シム（呼び出し側を順次置換） |
| `get_stage_name_list` | `dir_name` → stage_kind_no を解決し `stage_list` から名前リスト |
| `get_image_entry_*` | `images[]` / `stage_kinds[]` 用に分割 |
| `register_timetable_image` | 内部で `register_stage_kind` (新規 or 既存) + `register_image_entry` + raw 保存 |
| `delete_timetable_image` | 「画像のみ削除」と「stage_kind ごと削除」を呼び分け可能に |
| `cleanup_image_artifacts` | 旧 raw.png 削除を `raw_image_path()` に切替 |

### project_migration.py — 旧 → 新スキーマ移行

`migrate_project_info(pij, pj_path=None)` を 4 ステップ構成に拡張。

```python
def migrate_project_info(pij, pj_path=None):
    for event_detail in pij.get("event_detail", []):
        _migrate_legacy_dict_to_list(event_detail)            # Step 1 (既存)
        _migrate_one_layer_to_three_layer(event_detail)       # Step 2 (新)
        _ensure_sort_orders(event_detail)                     # Step 3 (新)
        if pj_path:
            _migrate_raw_image_files(event_detail, pj_path)   # Step 4 (新)
    return pij
```

#### Step 2: 1 層 → 3 層分解

旧スキーマ（image と stage_list が一体）から stage_kinds / images / stage_list[].image_no を組み立てる。

```python
def _migrate_one_layer_to_three_layer(event_detail):
    if _is_already_three_layer(event_detail):
        return
    old_timetables = event_detail.get("timetables", [])

    stage_kinds, images = [], []
    for entry in old_timetables:
        sk_no = entry["image_no"]          # 旧 image_no を stage_kind_no として継承
        image_no = entry["image_no"]       # image_no も同値を維持
        # stage_kind を作る
        stage_kinds.append({
            "stage_kind_no": sk_no,
            "dir_name": entry["dir_name"],
            "display_name": entry.get("display_name", entry["dir_name"]),
            "kind": entry["kind"],
            "sort_order": None,           # Step 3 で補完
            "stage_num": entry.get("stage_num", 0),
            "stage_list": [
                {**stg, "image_no": image_no} for stg in entry.get("stage_list", [])
            ],
        })
        # image を作る
        img = {
            "image_no": image_no,
            "stage_kind_no": sk_no,
            "sort_order": None,
            "raw_crop_box": entry.get("raw_crop_box"),
            "time_pixel": entry.get("time_pixel"),
        }
        if "format" in entry:
            img["format"] = entry["format"]
        images.append(img)

    event_detail["stage_kinds"] = stage_kinds
    event_detail["images"] = images
    event_detail.pop("timetables", None)
```

- 旧 `image_no` は **stage_kind_no と image_no の両方として同値を継承**。これにより既存ファイルパス（image_no 連動）が保てる。
- 既存プロジェクトには「画像なし stage_kind」も「stage_list[].image_no = null」も存在しないので Step 2 で 1:1 構造になる。

#### Step 3: sort_order 補完

```python
def _ensure_sort_orders(event_detail):
    stage_kinds = event_detail.get("stage_kinds", [])
    if any(sk.get("sort_order") is None for sk in stage_kinds):
        # デフォルトバケットルールで初期化
        ordered = sorted(stage_kinds, key=lambda sk: (_default_bucket(sk), sk["stage_kind_no"]))
        for i, sk in enumerate(ordered):
            sk["sort_order"] = i

    # images の sort_order は stage_kind の sort_order に追従させる
    images = event_detail.get("images", [])
    sk_by_no = {sk["stage_kind_no"]: sk for sk in stage_kinds}
    images.sort(key=lambda img: (sk_by_no[img["stage_kind_no"]]["sort_order"], img["image_no"]))
    for i, img in enumerate(images):
        img["sort_order"] = i
```

#### Step 4: raw.png 物理移動

旧 `event_N/{dir_name}/raw.png` → `event_N/raw_images/image_{image_no}.png` に移動。冪等。

### workflow.py — ワークフロー層

| メソッド | 変更 |
|---|---|
| `ProjectWorkflow.register_image` | repo の新シグネチャに合わせる: stage_kind を「既存選択 or 新規作成」で受け取る |
| `ProjectWorkflow.delete_image` | image のみ削除（stage_kind は残す）。stage_list 内の image_no を null 化 |
| `ProjectWorkflow.delete_stage_kind` | **新規**: stage_kind ごと削除（紐づく画像とディスク資産も） |
| `ProjectWorkflow.reorder_stage_kinds` | **新規** |
| `ProjectWorkflow.reorder_images` | **新規** |
| `ProjectWorkflow.reset_sort_order` | **新規** |
| `ImageWorkflow.output_difference_image` | raw パスを `raw_image_path()` 経由に |

### output_builder.py — 出力ビルド

`event_type_list` ベースのループ箇所は `get_sorted_stage_kind_no_list()` + dir_name 取得に書き換える。特典会フラグ導出は stage_list[].kind ベース ([image_kind_plan の §5](image_kind_and_stage_kind_plan.md#5-出力ビルド時の特典会フラグ導出後方互換性なし置き換え)) のまま。

### ocr_service.py — OCR

`event_type_list` ベースの参照を `get_sorted_stage_kind_no_list()` 経由に。stage_list の参照経路は変わるが、画像ファイルパス（raw_cropped.png, stage_*.png）は stage_kind の `dir_name` で組み立てるので大きな変更はない。

### image_processing.py

`raw.png` 参照箇所 ([L515-525](../../src/backend_functions/image_processing.py#L515-L525)) を `raw_image_path()` 経由に置換。`raw_old` のリネーム先も raw_images/ 配下とする。

---

## 影響範囲

| ファイル | 変更内容 |
|---|---|
| [src/backend_functions/project_repository.py](../../src/backend_functions/project_repository.py) | アクセサ層を 3 層モデル用に全面書き換え |
| [src/backend_functions/project_migration.py](../../src/backend_functions/project_migration.py) | Step 2-4 追加 |
| [src/backend_functions/image_processing.py:515-525](../../src/backend_functions/image_processing.py#L515-L525) | raw パス置換 |
| [src/backend_functions/ocr_service.py:69,180,213-245,385](../../src/backend_functions/ocr_service.py#L69) | アクセサ呼び出し置換 |
| [src/backend_functions/output_builder.py:36-56,346,509](../../src/backend_functions/output_builder.py#L36-L56) | 同上 |
| [src/workflow.py:118-182,296](../../src/workflow.py#L118-L182) | register/delete のシグネチャ刷新、reorder メソッド追加 |
| [src/app.py:648-782](../../src/app.py#L648-L782) | ② 画像登録 UI 刷新（stage_kind 選択／新規作成 + 一覧の grouping + ↑↓） |
| [src/app.py:718,810,1009](../../src/app.py#L718) | raw パス置換 |
| [src/app.py:1100-1190](../../src/app.py#L1100-L1190) `render_output_section` | stage_kind 経由でループ |
| [tests/backend_functions/test_project_repository_accessors.py](../../tests/backend_functions/test_project_repository_accessors.py) | 新スキーマ用に書き直し |
| (新規) `tests/backend_functions/test_project_migration_three_layer.py` | Step 2-4 のマイグレーション単体テスト |

---

## 動作例

### 例1: 通常 2 種別 2 画像（現状の典型）

```
stage_kinds:
  [{sk_no:0, dir_name:"ライブ",   kind:"live",       sort_order:0, stage_list:[...]},
   {sk_no:1, dir_name:"特典会",   kind:"tokutenkai", sort_order:1, stage_list:[...]}]
images:
  [{image_no:0, stage_kind_no:0, format:"通常", sort_order:0, raw_crop_box:...},
   {image_no:1, stage_kind_no:1, format:"通常", sort_order:1, raw_crop_box:...}]
```

### 例2: 同一 stage_kind に画像 2 枚（将来）

```
stage_kinds: [{sk_no:0, dir_name:"縁日", kind:"tokutenkai", sort_order:0, stage_list:[
    {stage_no:0, image_no:0, bbox:[..]},
    {stage_no:1, image_no:0, bbox:[..]},
    {stage_no:2, image_no:1, bbox:[..]},
]}]
images:
  [{image_no:0, stage_kind_no:0, sort_order:0, ...},
   {image_no:1, stage_kind_no:0, sort_order:1, ...}]
```

### 例3: 画像なし stage_kind（将来）

```
stage_kinds:
  [{sk_no:0, dir_name:"ライブ", kind:"live",       sort_order:0, stage_list:[...]},
   {sk_no:1, dir_name:"縁日",   kind:"tokutenkai", sort_order:1, stage_list:[
       {stage_no:0, stage_name:"射的", image_no:null, bbox:null}
   ]}]
images:
  [{image_no:0, stage_kind_no:0, sort_order:0, ...}]
```

### 例4: 画像並び替えで stage_kind 順も連動

```
初期:
  stage_kinds: [ライブ(0), 特典会(1)]
  images: [image_0→ライブ(0), image_1→特典会(1), image_2→ライブ(2)]

reorder_images([2, 1, 0]) を実行:
  → images: image_2(0), image_1(1), image_0(2)
  → stage_kind sort_order 再計算:
      image_2(ライブ), image_1(特典会), image_0(ライブ)
      初出順: ライブ, 特典会 → stage_kinds: [ライブ(0), 特典会(1)]
      (この例では結果同じ)

別例: reorder_images([1, 2, 0]) を実行:
  → 初出順: 特典会, ライブ → stage_kinds: [特典会(0), ライブ(1)]
```

---

## 段階的実装案

1. **Phase 1**: マイグレータ拡張（Step 2-4）+ 単体テスト。既存プロジェクト読込で 3 層モデルに自動変換、raw.png が raw_images/ に移動することを検証。
2. **Phase 2**: アクセサ層の新関数群追加。既存関数は内部で新スキーマを読むよう書き換え（旧呼び出し口を温存して段階移行）。テスト追加。
3. **Phase 3**: `register_image_entry` / `register_stage_kind` / `delete_*` を新スキーマで実装。raw 保存先を raw_images/ に切替。
4. **Phase 4**: `reorder_stage_kinds` / `reorder_images` / `reset_sort_order` 実装 + 同期ロジックの単体テスト。
5. **Phase 5**: ② 画像登録 UI 刷新（既存/新規 stage_kind selector + grouped 一覧 + ↑↓）。
6. **Phase 6**: 各セクション（③④⑤⑥）で raw.png パスとアクセサ呼び出しを置換。
7. **Phase 7**: 旧 timetables 互換シム（`get_event_type_list` 等）の削除と最終クリーンアップ。
8. **Phase 8（未スコープ）**: stage_kind 単独追加 UI、同一 stage_kind 多画像 UI、画像なしステージ手動追加 UI。

各 phase は単独でマージ可能になるよう、旧呼び出し口は phase 内で内部書き換えして互換性を保つ。

---

## 確定済み論点

1. **3 層モデル採用**: stage_kinds[] / images[] / stage_list[]（stage_kind 配下）の 3 層。
2. **images : stage_kinds = N : 1**: 1 stage_kind に 0..N 画像、画像は必ず 1 stage_kind に属する。
3. **stage_list の所属**: stage_kind 配下。各 stage は任意で image_no 参照（null 可）。
4. **並び順の同期挙動**: フラグなし、常にベストエフォートで連動。触った側 → 触らなかった側へ一方向再計算。
5. **画像登録 UI**: 既存 stage_kind 選択 or 新規作成の二択。
6. **raw 画像パス**: `event_N/raw_images/image_{image_no}.png`。
7. **stage_kind スコープアセット**: 0 or 1 画像時は `{dir_name}/raw_cropped.png` 据え置き。多画像時のレイアウトは別タスク。
8. **`raw_images`**: dir_name の予約名。UI バリデーションで弾く。

---

## 残オープン論点

1. **同一 stage_kind に複数画像があるときのアセット配置**: 案A（`{dir_name}/image_{N}/...`）と案B（`{dir_name}/raw_cropped_{N}.png`）のどちらにするか。多画像ユースケースが具体化する段階で決める。
2. **stage_kind 単独追加 UI / 画像なしステージ手動追加 UI**: 本計画ではスキーマだけ先回り。UI は別タスク。
3. **`reorder_images()` 時の画像なし stage_kind の差し込みルール**: 「元の sort_order を基準に二分探索挿入」で簡易対応する想定だが、より自然なロジックが欲しい場合は仕様詰めが必要。
4. **新規 stage_kind 作成時の `kind` 入力 UI**: radio 4 択（ライブ / 特典会 / ライブ特典会併記 / その他）の「その他」をさらにどう細分化するか。実装フェーズで詰める。
5. **`stage_list[].image_no = null` 状態での OCR / クロップ等の挙動**: 「画像由来でないステージ」は OCR 対象外、クロップ不可、として扱う想定だが、UI でどう disable するかは実装フェーズで。

---

## 関連計画

- [image_kind_and_stage_kind_plan.md](image_kind_and_stage_kind_plan.md) の kind 設計と「`format == '特典会併記'` 廃止」「stage_list[i].kind 付与」「`get_project_json` 内マイグレータ呼び出し」等は **本計画の前提として継承**。

## 設計検討の経緯（参考）

本計画に至るまで、以下の中間案を検討して廃案にしている（ドキュメントは削除済み）。

- **image_order 案**: `event_detail.image_order: list[int]` で並び順を管理する案。本計画の `stage_kinds[].sort_order` / `images[].sort_order` に置き換え。
- **images / timetables 二分化案**: 元画像と stage_list を 2 層に分ける案。stage_kind を独立エンティティとして扱う必要があり、本計画の 3 層モデルに発展。
- **旧 `event_type` 文字列リテラル一致**: 特典会フラグを `event_type == "特典会"` で判定していた仕組み。`stage_list[i].kind` ベースに置換済み（image_kind_and_stage_kind_plan で確定）。
