# ⑥出力確認・編集 — 全ステージ統合タイテ画像 実装計画

## 目的・背景

⑥タブには既にステージマスタ・グループマスタ・出番マスタ・集計情報が並んでいるが、「視覚的に出番全体を見渡したい」ニーズが満たせていない。
[④ 読み取り結果確認](../../src/app.py#L1129) では `_timetable.png`（[ocr_service.generate_timetable_picture](../../src/backend_functions/ocr_service.py#L314) → [timetablepicture.create_timetable_image](../../src/frontend_functions/timetablepicture.py#L161)）でステージ単位の画像を生成しているが、これらはステージごと・種別ごとにバラバラ。

⑥では以下2種類の俯瞰画像を生成する:

1. **種別単位画像**: 1イベント × 1種別の全ステージを横並びで1枚に
2. **種別横断画像**: 1イベントの全種別 × 全ステージをまとめて1枚に

並び順は `master_stage.csv` の `表示順`（[output_section_edit_plan Phase 1-2 で導入済](done/output_section_edit_plan.md)）を参照する。

ステージカラーは [stella_json_output_plan.md Phase 2-2](stella_json_output_plan.md) で導入予定の `master_stage.csv.カラー名` を反映するが、未実装の段階ではデフォルト色で動かす。

---

## 現状把握

| 項目 | 既存実装 | 流用方針 |
|---|---|---|
| 1ステージ画像生成 | [`create_timetable_image()`](../../src/frontend_functions/timetablepicture.py#L161) | 各ステージ列の描画にそのまま使う（既存パラメータで `source_box_width` / `start_margin` / `time_line_spacing` / `image_height` / `box_color` を制御可） |
| 縦軸スケール決定 | [`generate_timetable_picture()`](../../src/backend_functions/ocr_service.py#L394-L425) で `time_axis_converter` から `source_ppm` を求め `factor = max(1.0, TARGET_PPM / source_ppm)` で拡大 | 同じロジックを抽出して再利用 |
| 特典会併記の列分割 | [`_build_tokutenkai_view_json()`](../../src/frontend_functions/timetablepicture.py#L88) で擬似ライブJSONに変換 | ライブ列 / 特典会列の2画像生成にそのまま使う |
| 横並び結合 | [`_hstack_images()`](../../src/frontend_functions/timetablepicture.py#L149) | N枚版に拡張 |
| ステージ表示順 | [`build_event_output()`](../../src/backend_functions/output_builder.py#L497-L512) で `df_stage` を `表示順` でソート済 | そのまま使う |
| ステージカラー | **未実装**（[stella_json_output_plan.md Phase 2-2](stella_json_output_plan.md)） | デフォルト色で動くようにし、`カラー名` カラム導入後に解決マップを追加 |

---

## スコープ

- **対象**: 1イベント単位の俯瞰画像。`event_num > 1` のプロジェクトでもイベントタブごとに独立した画像を生成する（プロジェクト全体を1枚にまとめる機能は対象外）。
- **永続化方針**: 生成画像は PNG としてプロジェクト配下に保存する（S3 同期対象）。④の個別ステージ画像と同様、ソースデータ更新時に再生成する。
- **挿入位置**: 閲覧モードのみ。編集モード中は再生成負荷とキャッシュ不整合を避けるため非表示。

### 保存パスとファイル名

| 種類 | 保存パス |
|---|---|
| 種別単位 (ライブ列) | `{pj_path}/{event_name}/{img_type}/all_stages_live.png` |
| 種別単位 (特典会列) | `{pj_path}/{event_name}/{img_type}/all_stages_tokutenkai.png` |
| 種別横断 | `{pj_path}/{event_name}/all_stages.png` |

- `live_tokutenkai_heiki` 種別は `all_stages_live.png` と `all_stages_tokutenkai.png` の2枚を生成
- `kind == "live"` 種別は `all_stages_live.png` のみ
- `kind == "tokutenkai"` 種別は `all_stages_tokutenkai.png` のみ

### 再生成タイミング

| トリガ | 呼び出し箇所 | 再生成対象 |
|---|---|---|
| 個別ステージの `_timetable.png` 生成後 | [app.py:`output_timetable_picture_onlyonestage()`](../../src/app.py#L601) | 当該 (event_name, img_type) の種別単位画像 + 当該 event の種別横断画像 |
| 全ステージ一括の `_timetable.png` 生成後 | [app.py:`output_timetable_picture_eachstage()`](../../src/app.py#L609) | 同上 |
| バッチOCR後の一括生成後 | [app.py:`_run_batch_ocr_together()` の生成ループ後](../../src/app.py#L578-L582) | 触れた全 (event_name, img_type) の種別単位画像 + 各 event の種別横断画像 |
| ⑥編集の保存時 | [workflow.py:`OutputWorkflow.save_output_edits()`](../../src/workflow.py#L626) | 当該 event の全種別単位画像 + 種別横断画像 |
| IDマスタ確定時 | [workflow.py:`OutputWorkflow.determine_id_master()`](../../src/workflow.py#L530) | 全イベントの全画像（ステージID確定でステージマスタ参照が初めて成立するため） |

`generate_timetable_picture()` 自体には**集約画像生成を組み込まない**（per-stage 呼び出しが多重になり N×N 回生成されてしまうため）。呼び出し側のループ末尾で 1 回だけ集約版を生成する。

---

## Phase 1: 種別単位の統合タイテ画像生成

### 1-1. 新規モジュール

[src/backend_functions/event_timetable_picture.py](../../src/backend_functions/event_timetable_picture.py) を新設し、純粋関数として以下を提供する（Streamlit非依存・フロント状態非依存）:

```python
def build_event_type_image(
    pj_path: str,
    event_name: str,
    img_type: str,
    project_info_json: dict,
    *,
    variant: str = "live",   # "live" | "tokutenkai"
    stage_color_resolver: Callable[[int], tuple[str, str]] | None = None,
) -> Image.Image | None:
    """1イベント・1種別の全ステージを横並びにした統合タイテ画像を返す。

    内部で {pj_path}/{event_name}/master_stage.csv の有無を判定し、
      - 存在する → 表示順でソート（モード A）
      - 存在しない → project_info.stage_list[] 順（モード B）
    を自動切替する。
    入力データは全て永続化済ファイル（stage_*.json / master_stage.csv /
    project_info.json）から読み込み、フロントの DataFrame には依存しない。

    - kind == "live_tokutenkai_heiki" のとき、variant="live"/"tokutenkai" で出し分け
    - kind == "live" のとき variant="tokutenkai" は None
    - kind == "tokutenkai" のとき variant="live" は None
    """
```

### 1-2. 縦軸スケールの統一

種別内の各ステージは同じ `time_pixel` を持つが、生成画像の縦サイズが微妙に異なるケースに備え、種別単位で**共通の (start_margin, time_line_spacing, image_height)** を計算する:

1. 当該種別の `image entry` から `time_axis.build_converter` で `TimeAxisConverter` を取得（[time_axis.py:43](../../src/backend_functions/time_axis.py#L43)）
2. 種別配下の `raw_cropped.png` で `source_width / source_height` を確認
3. [ocr_service.py:412-423](../../src/backend_functions/ocr_service.py#L412-L423) と同じロジックで `factor` を算出
4. `image_height = round(source_height * factor)`、`start_margin`、`time_line_spacing` を確定

抽出した縦軸決定ロジックは `ocr_service.py` から `event_timetable_picture._compute_vertical_layout()` に切り出して両者で共有する（リファクタ）。

### 1-3. ステージ幅の統一

「各ステージの幅は共通にする」要件への対応:

- 種別の `raw_cropped.png` 全体幅から `source_box_width_common = source_width * factor / stage_num_visible` を算出
- `live_tokutenkai_heiki` の場合は更に `/ 2`（既存 [ocr_service.py:425](../../src/backend_functions/ocr_service.py#L425) と同じ考え方）
- 各ステージで `create_timetable_image(..., source_box_width=source_box_width_common, apply_max_width_clamp=False)` を呼ぶ
- 戻り画像の実描画幅は内容に応じて多少異なるため、`max(widths)` に揃えてから `Image.new` で背景白パディングして列幅を統一する

### 1-4. ステージ並び順

`master_stage.csv` の有無を関数内で判定し、以下2モードを自動切替する:

**モード A: IDマスタ確定済（`{event_name}/master_stage.csv` が存在する）**
- `master_stage.csv` を読み込み（[`load_existing_masters()`](../../src/backend_functions/output_builder.py#L92) を共用）、`表示順` 昇順でソート
- `非活性化フラグ == True` のステージは除外
- ステージID → (event_type, stage_no) の対応は [project_info.json の stage_list[i].stage_id](done/output_section_edit_plan.md) で解決
- 当該種別 (`img_type`) に属するステージのみフィルタ。`variant="tokutenkai"` の場合は `特典会フラグ == True` のステージかつ当該種別配下のブースに限定

**モード B: IDマスタ未確定（`master_stage.csv` が無い）**
- `project_info.json` の `stage_list[]` 登録順（= 元の入力順）でステージを並べる
- `variant="live"` は `tokutenkai` 種別以外を、`variant="tokutenkai"` は `tokutenkai` / `live_tokutenkai_heiki` の特典会ブース要素を出力
- 特典会併記種別の特典会ブースは `_build_tokutenkai_view_json()` 適用後の出現順
- 非活性化フラグの判定はスキップ（マスタが未確定のため）

### 1-5. ヘッダ（ステージ名）描画

各ステージ列の上部に高さ `_HEADER_H = 32px`（フォント `MIN_FONT_SIZE` 相当）の見出しエリアを設け、`ステージ名_短縮` があれば優先、無ければ `ステージ名` を描画する。背景色は Phase 3 のステージカラーを反映、未実装段階ではグレー (`#DDDDDD`) を使用する。

### 1-6. 横並び結合

`_hstack_images()` を N 枚版に一般化した `_hstack_many(images, gap=4, bg="white")` を `timetablepicture.py` に追加（既存関数は内部から呼ぶ）。

### 1-7. 特典会併記時の2画像生成

- `kind == "live_tokutenkai_heiki"` のとき、対象ステージJSONを2系統に変換:
  - **live**: 元 JSON をそのまま `create_timetable_image(json, ...)`
  - **tokutenkai**: `_build_tokutenkai_view_json(json)` を渡し、ブース別の擬似ライブJSONに変換
- 特典会画像のステージ並び順は `df_stage[特典会フラグ] & event_type==img_type` を `表示順` でソート
- いずれも `source_box_width_common` を共通幅とする

戻り値仕様:

```python
# kind == "live_tokutenkai_heiki" のとき
{"live": Image, "tokutenkai": Image}
# それ以外
{"live": Image} or {"tokutenkai": Image}
```

---

## Phase 2: 種別横断の統合タイテ画像生成

### 2-1. 関数

```python
def build_event_image(
    pj_path: str,
    event_name: str,
    project_info_json: dict,
    *,
    stage_color_resolver: Callable[[int], tuple[str, str]] | None = None,
) -> Image.Image | None:
    """1イベントの全種別×全ステージをまとめた統合タイテ画像を返す。
    モード A/B の自動切替は build_event_type_image と同様。
    """
```

### 2-2. 縦軸の全種別統一

種別ごとに `time_pixel` が異なるため、種別横断時は以下のルールで縦軸を統一する:

- 種別ごとの `factor` を全て計算
- 全種別の `min(start_time)` / `max(end_time)` を取得（30分刻みで丸め）
- `gen_ppm = max(各種別の source_ppm * factor)` を採用 ＝ 最も縦長な種別を基準にスケール
- 共通の `image_height = total_minutes * gen_ppm + margin*2 + 見出し`、`start_margin = (start_time - 最早:00時刻).minutes * gen_ppm + 見出し`
- 各種別の `create_timetable_image` には統一した `(start_margin, time_line_spacing, image_height)` を渡す

これにより全種別の時刻軸（左端）が一致し、列同士の出番が同じ Y 座標で並ぶ。

### 2-3. レイアウト

```
┌──────────┬──────────┬──────────┬──────────┬──────────┐
│時間軸     │ ライブ列 │ ライブ列 │ 特典会列 │ 縁日列    │
│           │ stage_A  │ stage_B  │ booth_C  │ booth_D  │
│ 12:00 ─── │          │          │          │          │
│ 12:30 ─── │  ...     │          │          │          │
└──────────┴──────────┴──────────┴──────────┴──────────┘
```

- 時間軸は最左の1列のみ（種別単位画像と同様、`show_timeline_labels=True` は最初の `create_timetable_image` 呼び出しのみ、以降 `False`）
- 種別境界に縦の太線（例: `width=3, fill="gray"`）を入れる
- 種別ヘッダ（"ライブ" / "特典会" 等）を最上部に幅でまとめて描画

### 2-4. 順序

- 種別の並び: `repo.get_event_type_list()` 既存順（live → tokutenkai → live_tokutenkai_heiki）
- 各種別内のステージ並び: `表示順` 昇順
- `live_tokutenkai_heiki` 種別は1ブロック内でライブ列群・特典会列群の順に並べる

---

## Phase 3: ステージカラー反映（後付け）

[stella_json_output_plan.md Phase 2-2](stella_json_output_plan.md) で `master_stage.csv` に `カラー名` カラムが追加されたあとに対応。

### 3-1. カラー解決マップ

`data/master/color_preset.json`（[stella plan で新規](stella_json_output_plan.md)）から `stage-red → ("#EA5A5A", "#FFFFFF")` のような (背景色, 文字色) マップを作る。

### 3-2. リゾルバ

```python
def make_stage_color_resolver(
    pj_path: str, event_name: str,
) -> Callable[[int], tuple[str, str]]:
    """ステージID → (背景色, 文字色) を返す解決関数を構築する。
    {event_name}/master_stage.csv の `カラー名` 列を読み、
    プリセット名なら color_preset.json から、"#bg-#fg" 形式なら直接パース。
    未設定 / マスタ無しなら ("#FFFF99" ライブ / "#DDDDDD" 特典会, "#000000")。
    """
```

### 3-3. `create_timetable_image` 拡張

現状 `box_color` は単色文字列のみ。背景色＋文字色を渡せるよう以下を追加:

- `box_fill_color: str = "yellow"`（既存 `box_color` のエイリアス）
- `text_color_in_box: str = "black"`

ヘッダ描画にも同じ色を流用する。

---

## Phase 4: 永続化と再生成タイミング

### 4-1. 保存関数

`event_timetable_picture.py` に保存ラッパーを追加:

```python
def save_event_type_images(
    pj_path, event_name, img_type, project_info_json,
) -> dict[str, str]:
    """build_event_type_image() を全 variant について実行し PNG 保存。
    返り値: {"live": path, "tokutenkai": path} (該当する variant のみ)
    既存ファイルは上書き。データ無しなら旧ファイルを削除（取り残し防止）。
    ステージカラーリゾルバは内部で make_stage_color_resolver() を構築。
    """

def save_event_image(
    pj_path, event_name, project_info_json,
) -> str | None:
    """build_event_image() を実行し PNG 保存。返り値: 保存パス or None"""
```

これらはプロジェクトディレクトリと `project_info_json` のみを受け取り、フロント状態 (`AppState` / `output_df` / `df_stage`) には一切依存しない。

### 4-2. 上位ラッパー

`OutputWorkflow` に統合エントリを追加:

```python
def regenerate_event_type_images(
    self, state, event_name, img_type,
) -> WorkflowResult:
    """1種別の集約画像（種別単位 + 種別横断）を再生成。"""

def regenerate_all_event_images(
    self, state, event_name,
) -> WorkflowResult:
    """1イベントの全集約画像を再生成。"""
```

内部では永続化ファイル (`master_stage.csv` / `stage_*.json` / `project_info.json`) のみを参照し、`state.output.output_df` は使わない。`master_stage.csv` の有無で モード A/B が自動切替される。

### 4-3. 呼び出し側の修正

| 修正箇所 | 修正内容 |
|---|---|
| [`output_timetable_picture_onlyonestage()`](../../src/app.py#L601) | 末尾で `regenerate_event_type_images(event, img_type)` 呼び出し |
| [`output_timetable_picture_eachstage()`](../../src/app.py#L609) | ループ後に `regenerate_event_type_images(event, img_type)` を1回呼び出し |
| [`_run_batch_ocr_together()` のループ](../../src/app.py#L568-L582) | 触れた (event, img_type) ペアを集めて、ループ後に `regenerate_event_type_images` を一括実行 |
| [`OutputWorkflow.save_output_edits()`](../../src/workflow.py#L626) | `output_df` 再構築後に `regenerate_all_event_images(event)` を呼び出し |
| [`OutputWorkflow.determine_id_master()`](../../src/workflow.py#L530) | `project_info` 永続化後、全イベントに対し `regenerate_all_event_images` を呼び出し |

### 4-4. UI表示

[app.py:1284](../../src/app.py#L1284) の `render_output_section` 内、`_render_event_output_view()`（閲覧モード）の末尾、`_render_event_aggregations()` の後ろに新セクションを追加。**編集モード (`_render_event_output_editor`) には追加しない**。

```
##### 全ステージ統合タイテ画像
[種別: ○ ライブ ○ 特典会 ○ ガッツリMORIMORI(併記)  ○ 種別横断]
[ライブ列 / 特典会列]   ← live_tokutenkai_heiki 選択時のみ表示
[画像表示]  ← st.image(保存済PNGパス)
[ダウンロードボタン]
```

- 既に永続化されているため、UI 側はファイル存在チェックして `st.image()` で表示するだけ
- ファイルが存在しない場合は「今すぐ再生成」ボタンの案内を表示
- 「今すぐ再生成」ボタンも置く（強制再生成用、Phase 3 のステージカラー変更後などに使う）

### 4-5. S3同期

[`project_repository._DERIVATIVE_FILE_PATTERNS`](../../src/backend_functions/project_repository.py#L472) に `all_stages*.png` を追加し、派生ファイル扱いとする。
S3 アップロード対象に含めるか除外するかは既存の `_timetable.png` の扱いに合わせる（要確認）。

---

## Phase 5: テスト

### 5-1. 単体テスト（既存テストインフラがあれば）

| ケース | 期待動作 |
|---|---|
| 単一種別・単一ステージ | 既存 `_timetable.png` とほぼ同等の画像 |
| 単一種別・複数ステージ | 列幅が全ステージで揃う / 表示順通り |
| `live_tokutenkai_heiki` 種別 | live/tokutenkai が別画像 / 特典会画像は ブース順 |
| 種別横断 | 時間軸が左1本に統一 / 各種別の出番が同じ Y 座標 |
| 出番ゼロのステージ | 空白列として表示（潰さない） |
| 非活性化ステージ | 出力対象外 |

### 5-2. 手動確認

[data/projects/2026_05_ガルガルMORIMORI/](../../data/projects/) など `live_tokutenkai_heiki` 入り実プロジェクトで目視確認。

---

## ファイル変更一覧（予定）

| ファイル | 変更種別 | 内容 |
|---|---|---|
| `src/backend_functions/event_timetable_picture.py` | **新規** | `build_event_type_image()` / `build_event_image()` / `save_event_type_images()` / `save_event_image()` / `_compute_vertical_layout()` |
| `src/backend_functions/ocr_service.py` | 修正 | 縦軸決定ロジックを `event_timetable_picture._compute_vertical_layout()` に外出し（[L412-L425](../../src/backend_functions/ocr_service.py#L412-L425)） |
| `src/frontend_functions/timetablepicture.py` | 修正 | `_hstack_many()` 追加、Phase 3 で `text_color_in_box` 追加 |
| `src/workflow.py` | 修正 | `OutputWorkflow.regenerate_event_type_images()` / `regenerate_all_event_images()` 追加、`save_output_edits` / `determine_id_master` に再生成フック追加 |
| `src/app.py` | 修正 | `_render_event_output_view` に Phase 4-4 のUIブロック追加、`output_timetable_picture_onlyonestage` / `output_timetable_picture_eachstage` / `_run_batch_ocr_together` に再生成フック追加 |
| `src/backend_functions/project_repository.py` | 修正 | `_DERIVATIVE_FILE_PATTERNS` に `all_stages*.png` 追加 |
| `data/master/color_preset.json` | **新規** | Phase 3 で追加（stella_plan と共通） |

---

## 実装順序

```
Phase 1 (種別単位画像)
    ↓
Phase 2 (種別横断画像)
    ↓
Phase 4 (UI統合)  ← Phase 1/2 が動いた時点で並行着手可
    ↓
Phase 3 (ステージカラー)  ← stella_plan Phase 2-2 完了後に実施
    ↓
Phase 5 (テスト)
```

---

## 設計上の留意点

1. **`raw_cropped.png` が存在しないケース**: 4で `raw_cropped` を作る前のフローもあり得る。その場合は `factor=1.0`, `source_box_width=None` フォールバック（`create_timetable_image` のデフォルト挙動に委ねる）。
2. **`time_pixel` 未設定ステージ**: `TimeAxisConverter` が None を返したら `time_match=False` ルートと同等の素朴生成にフォールバック（[ocr_service.py:442-447](../../src/backend_functions/ocr_service.py#L442-L447) 参照）。
3. **MAX_GEN_WIDTH (2000px)**: 種別横断は容易にこれを超える。`apply_max_width_clamp=False` で生成し、最終合体後に全体縮小する（Phase 2-2 の image_height 計算と合わせ縦横比保持）。`MAX_GEN_WIDTH` を超えていい上限は別途 `MAX_COMBINED_WIDTH = 6000` 程度を新設。
4. **生成コスト**: 1イベント10ステージ × 種別横断で2-3秒オーダー想定。UI ではボタン押下でのみ生成 + キャッシュで再描画を抑える。
5. **非活性化ステージ**: 既存 [`build_event_output()`](../../src/backend_functions/output_builder.py#L559-L567) で `df_stage_visible` から既に除外済 → そのまま渡せばよい。
6. **複数イベント**: `event_num > 1` プロジェクトでもイベントタブごとに独立した画像を生成する。プロジェクト全体を1枚にまとめる機能は本計画のスコープ外。
7. **IDマスタ未確定時の再生成**: ステージマスタが無くても集約画像は生成する（Phase 1-4 のモード B、`project_info.stage_list[]` 登録順）。IDマスタ確定時にモード A で再生成され、`表示順` による並びに切り替わる。
8. **取り残しファイル**: 種別の `kind` が後から変わる（例: `live` → `live_tokutenkai_heiki`）と、不要な variant の PNG が残る可能性がある。`save_event_type_images()` 内で「生成対象外の variant ファイルは削除する」処理を入れる。
9. **既存プロジェクト**: 既に④を完了済みのプロジェクトを開いた際は、自動再生成しない（ファイルが無いままUI表示のメッセージで「再生成」ボタンを促す）。マイグレーションは行わない。
