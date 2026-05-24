# 読み取り結果画像への特典会併記表示 実装計画

## 目的・背景

④読み取り結果タブでは、`stage_{i}_timetable.png` を生成して元タイテ画像 (`stage_{i}.png`) と並べて表示している ([app.py:1144-1167](../../src/app.py#L1144-L1167))。

しかし [create_timetable_image](../../src/frontend_functions/timetablepicture.py#L78) は **`ライブステージ` 枠のみ**を描画しており、`live_tokutenkai_heiki`（特典会併記形式）の JSON に含まれる `特典会[]` は無視されている。結果、特典会併記形式では生成画像に特典会のブース・時間が現れず、元画像との比較に必要な情報が欠落している。

各グループのライブ枠の右側に、特典会のブース名と時刻を **ライブ列と全く同じ描画ロジック** で描画する。

---

## 基本方針

> **「ライブ列の描画ロジックを、グループ名→ブース名に置き換えるだけ」で特典会列を生成する。** 二列描画は別画像として独立に作成し、最後に横方向に合体する。

これにより:

- 既存の `create_timetable_image` の **動的幅決定 / 1〜3 行モード切替 / フォントサイズ縦逆算 / 重複インデント / `_wrap_by_pixel` 折り返し** のロジックがすべてそのまま流用できる
- 特典会列のために新しいレイアウト分岐を内部に持ち込む必要がない
- ブース名が長いケースもライブ用のロジック（折り返し・省略）でそのまま吸収される

---

## スコープ

- 特典会列描画用に **元 JSON → 擬似ライブ JSON** に変換する関数を追加
- `create_timetable_image` に **時刻ラベル列の表示／非表示切り替え** 引数を追加
- `create_timetable_image` の `MAX_GEN_WIDTH` 縮小処理を **呼び出し側で制御可能** にする（合体後に判定するため）
- [generate_timetable_picture](../../src/backend_functions/ocr_service.py#L314) で `kind == "live_tokutenkai_heiki"` を判定し、ライブ列画像と特典会列画像を別々に生成して合体

スコープ外:
- 元画像側の crop 範囲・列分割 UI 変更
- ⑤以降（Excel・Stella JSON 等）での特典会描画
- `tokutenkai` 単独フォルダの描画ロジック変更
- 特典会のブース別ステージIDの描画

---

## 前提（現状実装の確認）

### 識別キー

特典会併記形式は `project_info_json` の画像エントリの `kind == "live_tokutenkai_heiki"` で識別する ([project_repository.py:420](../../src/backend_functions/project_repository.py#L420))。フォルダ名 `img_type` はカスタム名になり得るため、フォルダ名判定は不可 ([app.py:144-148](../../src/app.py#L144-L148))。

`run_batch_ocr` 内では既に同じ判定ロジックが使われている ([ocr_service.py:445](../../src/backend_functions/ocr_service.py#L445))。

### JSON 構造

[data_structure.md §特典会併記形式](../data_structure.md#特典会併記形式) の通り、各タイムテーブル要素は `ライブステージ` と `特典会[]`（0..N 要素）を持つ。`特典会[].ブース` / `from` / `to` の 3 項目が描画対象。

### 現状のレイアウト決定ロジック

[timetablepicture.py](../../src/frontend_functions/timetablepicture.py) の `create_timetable_image` は:

- 縦軸（`start_margin` / `time_line_spacing` / `image_height`）は引数で受領
- 元画像の box 横幅 `source_box_width` を受け取り、`box_height_min / source_box_width` のアスペクト比から 1 行 / 2 行モードを切替
- フォントサイズは `box_height_min` から逆算（縦駆動）
- 横幅は各イベントの `required_inner_width` から動的決定
- ステップ H で `image.width > MAX_GEN_WIDTH = 2000` なら画像全体を LANCZOS 縮小

`generate_timetable_picture` は `factor = max(1.0, TARGET_PPM / source_ppm)` で縦のみクランプし、`source_box_width = source_width * factor` を渡す。

---

## 仕様

### 1. 描画レイアウト（合体イメージ）

```
┌──────────────────────┬─────────────────────┐
│  ライブ列画像         │  特典会列画像        │
│ (時刻ラベル列あり)    │ (時刻ラベル列なし)   │
│                       │                      │
│ 10:00 ┌─box─┐         │       ┌─box─┐        │
│       │ライブ│         │       │ブース│       │
│ 10:30 │ 名  │         │       │ 名  │        │
│       └────┘          │       └────┘         │
│ ...                   │ ...                  │
└──────────────────────┴─────────────────────┘
       ↑                       ↑
   現状の描画               同じロジックで描画
```

- ライブ列と特典会列は **別画像として独立に生成**し、横方向に paste で合体
- 縦パラメータ（`start_margin` / `time_line_spacing` / `image_height`）は両画像で完全に揃える（=同じY位置で並ぶ）
- 特典会列画像は **時刻ラベル列を省略**（中央に時刻ラベルが重複しないため）
- 合体後に `MAX_GEN_WIDTH` 超過なら全体を LANCZOS 縮小

### 2. 特典会用擬似 JSON の生成

複数特典会のグループは、**ライブ枠を時間軸で N 等分**して各特典会を独立した擬似ライブ枠として配置する。これにより `create_timetable_image` のロジック側は「ライブ枠が縦に細かく並んでいる」としか見えず、自動的にフォントが縦逆算で小さくなる。

```python
def _build_tokutenkai_view_json(json_data: dict) -> dict:
    """特典会併記JSON → 特典会列描画用の擬似ライブJSON。

    各 live の 特典会[] の各要素を擬似ライブ枠に変換。
    複数特典会のときライブ枠を時間軸で N 等分し、末尾は live to に合わせる。
    時刻文字列上はブース名と「live の from～to (分)」ではなく、
    特典会自身の実時刻表記を表示するため、グループ名にブース名を入れた上で
    ライブステージの from/to を「特典会自身の実時刻」にする。
    """
    time_format = "%H:%M"
    new_timetable = []

    for live in json_data.get("タイムテーブル", []):
        tklist = live.get("特典会", []) or []
        if not tklist:
            continue  # 特典会なし → 右側空白

        try:
            live_from = datetime.strptime(live["ライブステージ"]["from"], time_format)
            live_to   = datetime.strptime(live["ライブステージ"]["to"],   time_format)
        except (KeyError, ValueError):
            continue

        total_min = int((live_to - live_from).total_seconds() / 60)
        n = len(tklist)
        # 分単位の境界（末尾は live_to と一致）
        boundary = [round(total_min * i / n) for i in range(n + 1)]
        boundary[-1] = total_min

        for i, tk in enumerate(tklist):
            sub_from = live_from + timedelta(minutes=boundary[i])
            sub_to   = live_from + timedelta(minutes=boundary[i + 1])
            # 退化（sub_from == sub_to）を防ぐため最低 1 分
            if sub_to <= sub_from:
                sub_to = sub_from + timedelta(minutes=1)

            booth = (tk.get("ブース") or "").strip() or "(ブース未定)"

            # 注: 擬似 live の ライブステージ.from/to は描画の Y 位置のための仮想時刻。
            # time_text 表示用には _display_* フィールドを別途持たせる。
            new_timetable.append({
                "グループ名":      booth,
                "グループ名_採用": booth,
                "ライブステージ": {
                    "from": sub_from.strftime(time_format),
                    "to":   sub_to.strftime(time_format),
                },
                "_display_time_from": tk.get("from", ""),
                "_display_time_to":   tk.get("to", ""),
            })

    return {
        "ステージ名": json_data.get("ステージ名", ""),
        "タイムテーブル": new_timetable,
    }
```

#### 重要な性質

- `_display_time_from` / `_display_time_to` は描画時に time_text 構築で参照される（§3）
- ライブ枠を N 等分するとき、各擬似ライブ枠の縦サイズ = `live_box_height / N`。これにより縦逆算フォントが自動で 1/N に小さくなる
- 退化（sub_from == sub_to）回避: ライブ枠が極端に短くて分単位で N 等分できないケース（例: 5 分ライブ × 特典会 6 件）でも sub_to が sub_from を超えるよう最低 1 分確保

### 3. `create_timetable_image` の最小限の変更

3 点だけ追加:

#### 3-1. `time_text` の参照フィールド切り替え

現状 ([timetablepicture.py:194](../../src/frontend_functions/timetablepicture.py#L194)):

```python
time_text = f"{live['ライブステージ']['from']} ～ {live['ライブステージ']['to']} ({minutes})"
```

これを次のように修正:

```python
disp_from = live.get("_display_time_from") or live["ライブステージ"]["from"]
disp_to   = live.get("_display_time_to")   or live["ライブステージ"]["to"]
# minutes は disp_from/to から再計算
disp_minutes = _diff_minutes(disp_from, disp_to)
time_text = f"{disp_from} ～ {disp_to} ({disp_minutes})"
```

これで時刻表記フォーマット自体はライブ側と完全に同じ（半角チルダ + スペース + 半角括弧）に統一される。

#### 3-2. 時刻ラベル列の省略

引数 `show_timeline_labels: bool = True` を追加。`False` のとき:

- 時刻ラベル（"10:00", "10:30" 等）を描画しない
- `timeline_text_margin` を 0 にする（横方向の余白を詰める）
- 30 分グリッドの横線も描画しない（合体時にライブ列のグリッドで十分）

#### 3-3. `MAX_GEN_WIDTH` 縮小の制御

引数 `apply_max_width_clamp: bool = True` を追加。`False` のときはステップ H の LANCZOS 縮小をスキップ。

合体時はライブ列・特典会列とも `apply_max_width_clamp=False` で生成し、合体後に呼び出し側で改めて判定する。

### 4. `generate_timetable_picture` での合体

```python
def generate_timetable_picture(
    stage_no, pj_path, event_name, img_type, time_match,
    time_axis_converter=None,
    project_info_json=None,                          # 新規
    event_no=None,                                   # 新規
) -> Optional[str]:
    ...
    # kind 判定
    is_heiki = False
    if project_info_json is not None:
        if event_no is None:
            event_no = repo.get_event_no_by_event_name(project_info_json, event_name)
        entry = repo.get_image_entry_by_dir_name(project_info_json, event_no, img_type)
        is_heiki = entry is not None and entry.get("kind") == "live_tokutenkai_heiki"
    ...

    if time_match and time_axis_converter is not None and os.path.exists(stage_img_path):
        # 現状の factor / image_height / source_box_width 算出は不変
        ...
        live_source_box_width = source_box_width / 2 if is_heiki else source_box_width

        # ライブ列画像
        live_img = timetablepicture.create_timetable_image(
            json_data,
            start_margin=start_margin,
            time_line_spacing=time_line_spacing,
            image_height=image_height,
            source_box_width=live_source_box_width,
            apply_max_width_clamp=not is_heiki,  # 併記なら合体後に判定
        )

        if is_heiki:
            tk_json = timetablepicture._build_tokutenkai_view_json(json_data)
            tk_img = timetablepicture.create_timetable_image(
                tk_json,
                start_margin=start_margin,
                time_line_spacing=time_line_spacing,
                image_height=image_height,
                source_box_width=live_source_box_width,   # 同じ値
                box_color="lightblue",
                show_timeline_labels=False,
                apply_max_width_clamp=False,
            )
            timetable_image = _hstack_images(live_img, tk_img)  # 横並び合体
            if timetable_image.width > timetablepicture.MAX_GEN_WIDTH:
                scale = timetablepicture.MAX_GEN_WIDTH / timetable_image.width
                new_h = max(1, int(round(timetable_image.height * scale)))
                timetable_image = timetable_image.resize(
                    (timetablepicture.MAX_GEN_WIDTH, new_h), _PILImage.LANCZOS,
                )
        else:
            timetable_image = live_img
    else:
        # time_match=False パス（フォールバック）も同様に実装
        ...
```

#### 4-1. 合体ヘルパー

```python
def _hstack_images(left: Image.Image, right: Image.Image) -> Image.Image:
    """左右に画像を横並びで合体。高さが違う場合は左の高さに合わせる。"""
    h = left.height
    if right.height != h:
        new_w = round(right.width * h / right.height)
        right = right.resize((new_w, h), Image.LANCZOS)
    combined = Image.new("RGB", (left.width + right.width, h), "white")
    combined.paste(left, (0, 0))
    combined.paste(right, (left.width, 0))
    return combined
```

`timetablepicture.py` 内に置く（`create_timetable_image` と関連が深いため）。

### 5. `time_match=False` フォールバックパス

[ocr_service.py:372](../../src/backend_functions/ocr_service.py#L372) の `else` 節（`source_box_width` なし）も併記対応:

```python
else:
    live_img = timetablepicture.create_timetable_image(
        json_data,
        apply_max_width_clamp=not is_heiki,
    )
    if is_heiki:
        tk_json = timetablepicture._build_tokutenkai_view_json(json_data)
        tk_img = timetablepicture.create_timetable_image(
            tk_json,
            box_color="lightblue",
            show_timeline_labels=False,
            apply_max_width_clamp=False,
        )
        timetable_image = _hstack_images(live_img, tk_img)
        # MAX_GEN_WIDTH 縮小
        ...
    else:
        timetable_image = live_img
```

このパスでは縦軸パラメータが各画像で個別に決まるため、ライブと特典会で `image_height` がズレる可能性がある。`_hstack_images` で右画像の高さをライブに合わせてリサイズする（縦軸の整合は保たれる）。

### 6. 呼び出し側への引数伝播

| 場所 | 変更 |
|---|---|
| [workflow.py:494](../../src/workflow.py#L494) `ProjectWorkflow.generate_timetable_picture` | `state.project.project_info_json` を `_ocr.generate_timetable_picture` に渡す |
| [app.py:579](../../src/app.py#L579), [app.py:603](../../src/app.py#L603) | `event_no` も渡す（既に取得済み or `get_event_no_by_event_name` で取得） |

`project_info_json` が None のときは旧挙動（特典会列なし）にフォールバック。

### 7. テスト

新規 [tests/frontend_functions/test_timetablepicture_tokutenkai.py](../../tests/frontend_functions/test_timetablepicture_tokutenkai.py):

- `_build_tokutenkai_view_json` の単体テスト:
  - 単一特典会のグループが正しく擬似ライブ 1 件に変換される
  - 複数特典会のグループが N 等分され、末尾の `to` がライブの `to` と一致する
  - `特典会[]` 空のグループはスキップされる
  - `_display_time_from` / `_display_time_to` に元の実時刻が保持される
  - 退化ケース（極端に短いライブ × 多数特典会）でも sub_to > sub_from が保たれる
- `create_timetable_image(show_timeline_labels=False)` で時刻ラベルと縦軸グリッドが描かれないこと
- `create_timetable_image(apply_max_width_clamp=False)` で `MAX_GEN_WIDTH` 超過時にも縮小されないこと
- `generate_timetable_picture` の統合テスト:
  - `kind == "live_tokutenkai_heiki"` のとき特典会列が合体される（`lightblue` ピクセル検出）
  - それ以外では従来通りライブ列のみ
  - 合体後の幅が `MAX_GEN_WIDTH` を超えるとき LANCZOS 縮小される

---

## 影響範囲

| ファイル | 変更内容 | 行数目安 |
|---------|---------|---------|
| [src/frontend_functions/timetablepicture.py](../../src/frontend_functions/timetablepicture.py) | `_build_tokutenkai_view_json` 追加、`create_timetable_image` に `show_timeline_labels` / `apply_max_width_clamp` 追加、`_display_time_*` 参照、`_hstack_images` ヘルパー | 100 行前後 |
| [src/backend_functions/ocr_service.py](../../src/backend_functions/ocr_service.py) | `generate_timetable_picture` に `project_info_json` / `event_no` 追加、kind 判定、ライブ列＋特典会列の合体ロジック | 50 行前後 |
| [src/workflow.py](../../src/workflow.py) | `ProjectWorkflow.generate_timetable_picture` の引数伝播 | 数行 |
| [src/app.py](../../src/app.py) | `_ocr_wf.generate_timetable_picture` 呼び出し箇所の引数追加（2 箇所） | 数行 |
| (新規) `tests/frontend_functions/test_timetablepicture_tokutenkai.py` | 単体・統合テスト | 150 行前後 |

---

## 段階的実装案

1. **Phase 1**:
   - `_build_tokutenkai_view_json` 実装 + 単体テスト
   - `create_timetable_image` に `show_timeline_labels` / `apply_max_width_clamp` / `_display_time_*` 対応追加
   - `_hstack_images` 追加
   - `generate_timetable_picture` の `time_match=True` パス対応
2. **Phase 2**:
   - `time_match=False` フォールバックパス対応
   - `workflow.py` / `app.py` の引数伝播
   - 実プロジェクト動作確認（`data/projects/2026_05_ガルガルMORIMORI/event_1/ライブ特典会/` 等）
3. **Phase 3**（必要なら）:
   - フォント／色／余白の最終調整

---

## 確定済み論点

1. **特典会枠の Y 位置**: ライブと同じ Y 位置。複数特典会はライブ枠を時間軸で N 等分し、末尾はライブ to に合わせる（分単位で丸め）
2. **複数特典会の表示**: N 等分された各枠に「ブース名 / 時刻 (分)」を表示。フォントは縦逆算で自動的に 1/N サイズに縮む
3. **特典会なし出番**: 擬似 JSON でスキップ → 右側空白
4. **枠内表示**: ブース名（= 擬似グループ名）+ 時刻表記 `"from ～ to (分)"`（**ライブ列と完全同じフォーマット**）
5. **時刻表記**: ライブと同じ `f"{from} ～ {to} ({minutes})"`（半角チルダ + スペース + 半角括弧）
6. **判定方法**: `project_info_json` の `kind == "live_tokutenkai_heiki"`
7. **適用範囲**: `live_tokutenkai_heiki` のみ
8. **枠色**: ライブ = 黄、特典会 = 水色（`box_color` 引数を `"lightblue"` で渡す。`create_timetable_image` 側に専用引数は追加しない）
9. **時刻ラベル列**: 特典会列では非表示（新引数 `show_timeline_labels=False`）
10. **`source_box_width` の解釈**: 併記モードでは `source_box_width / 2` を両画像に渡す（若干ライブ寄りだが妥当）
11. **`MAX_GEN_WIDTH` 縮小**: 合体後に呼び出し側で判定。各画像生成時は `apply_max_width_clamp=False`
12. **ライブ列幅の event 間揃え**: 不要（独立画像方式により発生しない）
13. **ブース名が長い場合の折り返し**: ライブ用の `_wrap_by_pixel` / `_truncate_lines` がそのまま適用される（特別な対応は不要）
14. **極端に短いライブ × 多数特典会の縦溢れ**: 「5 分ライブ × 特典会 4 件」のようなケースは `MIN_FONT_SIZE = 12` に張り付いてはみ出す可能性があるが、**超レアケースとして無視**。特別な省略・縮小対応は行わない。
15. **特典会列のグリッド線**: 描画しない。`show_timeline_labels=False` 時は時刻ラベルだけでなく **30 分ごとの横線も描かない**（合体時にライブ列のグリッドで十分）。

---

## 残オープン論点

1. **`source_box_width / 2` の妥当性**: 元画像が「ライブ列 + 特典会列」を等幅で含む前提。実プロジェクトでズレが大きければ画像登録スキーマで列比を持つ拡張を検討（過剰スコープ）。実装着手後に動作確認で見直し可能。

---

## 関連計画

- [timetable_picture_layout_plan.md](timetable_picture_layout_plan.md): 本計画の前提となるレイアウト改善計画。当該計画から実装段階で複数の変更が入っており（`source_box_width` ベースの動的幅決定、1〜3 行モード自動切替、LANCZOS 縮小）、本計画は現行実装に整合している。
- [output_section_edit_plan.md §0-2](output_section_edit_plan.md): 特典会併記形式のステージID二層構成（親=ライブ、子=ブース別）。本計画では描画のみ扱い、ステージID 自体には触れない。
