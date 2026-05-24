# タイムテーブル生成画像レイアウト 改善計画

## 目的・背景

OCR セクション ([app.py:1149-1164](../../src/app.py#L1149-L1164)) では、読み取り結果から生成したタイムテーブル画像 (`stage_{i}_timetable.png`) を元画像 (`stage_{i}.png`) と横並びで表示している。

現状の [create_timetable_image](../../src/frontend_functions/timetablepicture.py#L12) は次の課題を持つ:

1. **横幅が固定**（`image_width = 400` または `text_font_size * 20/30`）。元画像の幅と無関係に決まるため、並べたときに片方が極端に細い／太いことがある。
2. **折り返し幅が「あ」基準**（[timetablepicture.py:127](../../src/frontend_functions/timetablepicture.py#L127)）。`font.getbbox("あ")[2]` を 1 字幅として全角想定で割り算しているため、英数字混在のアーティスト名で改行精度が崩れる。
3. **縦スケールが元画像由来**。`time_match` モードでは [ocr_service.py:343-345](../../src/backend_functions/ocr_service.py#L343-L345) で元画像の `time_pixel` から `time_line_spacing` を直接コピーするので、元画像が低解像度（短い時間軸 / 小さい crop）だと生成画像も低解像度になり、テキストが潰れる。
4. **`time_match=False` パスの単体可読性も保証されていない**。デフォルト 30 分=90px は妥当だが、フォントサイズ・横幅・折り返し精度については (1)(2) と同じ問題を抱える。

並べたときの整合は **合成時の縦リサイズで吸収する** 設計に切り替え、生成画像自体は「単体としての可読性」を最優先で組み立て直す。

---

## スコープ

- [create_timetable_image](../../src/frontend_functions/timetablepicture.py#L12) のレイアウト決定ロジックを「外部から幅・高さを受け取り、その中に最適なフォントサイズで描画する」構造に変更
- [generate_timetable_picture](../../src/backend_functions/ocr_service.py#L313) で **元画像の `time_pixel` + 解像度** から生成画像のパラメータを算出
- `time_pixel` が無いフォールバックパスでも同じ可読性ロジックを共有
- [app.py:1149-1164](../../src/app.py#L1149-L1164) の合成ステップに **生成画像の縦リサイズ** を追加し、元画像と縦軸が一致した状態で並べる

スコープ外:
- 元画像側の `time_pixel` 取得 UI ([app.py:959-998](../../src/app.py#L959-L998)) の変更
- 差分画像 ([timetable_difference.py](../../src/frontend_functions/timetable_difference.py)) の変更
- 特典会併記形式の表示（既存ロジック踏襲）
- 重複時インデント (`duplicate_margin`) の挙動変更

---

## 前提

- 元画像 `stage_{i}.png` はステージ単位の縦長クロップ（1 ステージ列分）
- 元画像側の時刻↔ピクセル対応は [time_axis.py:18-24](../../src/backend_functions/time_axis.py#L18-L24) `TimePixelConfig` に永続化されている
  - `start_pix` : 基準時刻 (`time_start`) の y 座標
  - `total_pix` / `total_duration` : 時間軸全体のピクセル高さと分数
  - **これらの座標系は per-stage の `stage_{i}.png` と同じ y 座標系**
- 合成は [app.py:1154-1158](../../src/app.py#L1154-L1158) で行われており、現在は **リサイズせず** 単純に paste している

---

## 設計（核となる数式）

### 想定する最悪ケース

| 項目 | 値 |
|------|----|
| アーティスト名最長 | 日本語 20 文字 |
| 許容行数 | 最大 2 行（= 1 行 10 文字） |
| 最短イベント長 | 15 分 |
| 最長イベント長 | 90 分（縦に余裕がある側、制約にはならない） |

### 可読性パラメータの導出

快適な読み取り `font_size` を **20px** とする:

- 行高 ≈ `font_size * 1.2 = 24px`
- 15 分枠に必要な縦テキスト量 = (アーティスト名 2 行 + 時刻 1 行) × 24 = **72px**
- 1 分あたりピクセル `PPM ≥ 72 / 15 = 4.8 px/min`
- 1 行 10 文字日本語の横幅 ≈ `10 × 20 = 200px`
- 列の余白（時刻ラベル 60 + box_margin*2 + text_margin*2 + 外周 margin*2）= 140px
  - 内訳: `margin*2`(20) + `timeline_text_margin`(60) + `box_margin*2`(40) + `text_margin*2`(20)
- → ステージ列の最小幅 = 200 + 140 = **340px**

定数として以下を導入する（[timetablepicture.py](../../src/frontend_functions/timetablepicture.py) 先頭）:

```python
# 可読性目標
TARGET_FONT_SIZE = 20            # 目標フォントサイズ (px)
MIN_FONT_SIZE    = 12            # 縮小許容下限
LINE_HEIGHT_RATIO = 1.2          # 行高 / font_size
MAX_NAME_LINES    = 2            # アーティスト名の許容行数
WORST_NAME_LEN_JP = 20           # 最悪想定の日本語文字数
SHORTEST_EVENT_MIN = 15          # 最短イベント長 (分)

# 上記から導出される目標 PPM とサイズ閾値
TARGET_PPM      = 5.0            # ≈ TARGET_FONT_SIZE * LINE_HEIGHT_RATIO * (MAX_NAME_LINES + 1) / SHORTEST_EVENT_MIN
MIN_BOX_WIDTH   = 200            # WORST_NAME_LEN_JP / MAX_NAME_LINES * TARGET_FONT_SIZE
MIN_GEN_WIDTH   = 340            # MIN_BOX_WIDTH + 余白合計 140

# 生成画像の絶対サイズ上限（メモリ・描画コスト保護）
MAX_GEN_HEIGHT  = 4000           # 12 時間 × TARGET_PPM = 3600 + マージン余裕
MAX_GEN_WIDTH   = 1500           # 元画像が極端に細い場合の factor 暴走対策
```

`TARGET_PPM = 5` を内部的な「目標」とし、実 PPM は元画像との整合性から逆算する。

最大値については **倍率（factor）ではなく絶対 px** で表現する。理由:
- `factor` は元画像サイズ依存の派生量なので「何が問題か」が伝わりにくい
- メモリ・描画コストは `gen.width * gen.height` の絶対値で決まる
- 上限を超える場合は `gen_ppm` を犠牲にして `MAX_GEN_*` を守る（= 単体可読性が下がる代わりにファイルサイズを抑える）

### A. `time_pixel` あり（`time_match=True`）のパス

元画像 1 ステージ列の `source.width`, `source.height` と `TimePixelConfig` (`source_ppm = total_pix / total_duration`, `source_start_pix`) から:

```python
# 目標を満たす factor を最初に算出
factor = max(
    1.0,                                   # 縮小はしない（合成時の LANCZOS upscale を避ける）
    TARGET_PPM / source_ppm,               # 単体可読性を確保
    MIN_GEN_WIDTH / source.width,          # ステージ列が極端に細い場合の保険
)

# 絶対サイズ上限で頭打ち（メモリ保護）
factor = min(
    factor,
    MAX_GEN_HEIGHT / source.height,
    MAX_GEN_WIDTH  / source.width,
)

# factor < 1 になり得るのは MAX_GEN_* > source.* の前提が崩れた場合のみ
factor = max(factor, 1.0)                  # 念のため再クランプ

gen_ppm        = source_ppm * factor
image_width    = round(source.width  * factor)
image_height   = round(source.height * factor)
start_margin   = round(source_start_pix * factor)
time_line_spacing = gen_ppm * 30
```

`factor` は内部の中間変数。仕様上の上限はあくまで `MAX_GEN_HEIGHT` / `MAX_GEN_WIDTH`（絶対 px）。

**上限到達時の挙動**: 元画像が極端に低解像度な場合に `MAX_GEN_HEIGHT` に当たり、`gen_ppm < TARGET_PPM` となって単体可読性が下がる。これは妥協として明示する（コメントで残す）。

→ 合成時に `image_output.resize((source.width, source.height))` をかけると:
- 縦：30 分線が元画像と完全一致
- 横：元画像と 1:1
- 上端：最初の時刻線位置が一致
- 高解像度レンダリング → LANCZOS 縮小でアンチエイリアスが効き、テキストが綺麗

### B. `time_pixel` なし（`time_match=False`）のフォールバックパス

元画像参照無しでも **同じ可読性原則** で組み立てる:

```python
# 入力 JSON から時間軸の必要情報を取得
total_minutes = (latest_end - earliest_start) を 30 分丸めで拡張
min_event_min = min(event 長)

# 基準値: 単体可読性のために TARGET_* をそのまま採用
gen_ppm        = TARGET_PPM
image_height   = round(start_margin_default + gen_ppm * total_minutes + bottom_margin)
image_width    = round(MIN_GEN_WIDTH)   # = MIN_BOX_WIDTH + 余白
start_margin   = start_margin_default   # 固定値（例: 30px）
time_line_spacing = gen_ppm * 30
```

ステージ列としての横幅の絶対基準が無いので `MIN_GEN_WIDTH` を採用する。`min_event_min < 15` の極端なケースに備えて、必要に応じて `gen_ppm` を引き上げる:

```python
required_ppm_for_short = TARGET_FONT_SIZE * LINE_HEIGHT_RATIO * (MAX_NAME_LINES + 1) / min_event_min
gen_ppm = max(TARGET_PPM, required_ppm_for_short)
```

### C. フォントサイズの確定（A/B 共通）

外側で決まった `image_width` と `time_line_spacing` を受け取った後、`create_timetable_image` 内部で `text_font_size` を逆算する:

```python
box_width  = image_width - timeline_text_margin - 2*box_margin - 2*text_margin - 2*margin
box_height_shortest = time_line_spacing * min_event_min / 30

# 横幅由来の上限
size_from_width  = box_width / max(WORST_NAME_LEN_JP / MAX_NAME_LINES, 1)
# 縦由来の上限（最短イベントが (MAX_NAME_LINES + 1) 行を収容できる）
size_from_height = box_height_shortest / ((MAX_NAME_LINES + 1) * LINE_HEIGHT_RATIO)

text_font_size = clamp(min(size_from_width, size_from_height, TARGET_FONT_SIZE), MIN_FONT_SIZE, ...)
```

これで「箱に対して最大限大きく、ただし TARGET_FONT_SIZE を超えない」サイズに収束する。

### D. 折り返しの実測化

[timetablepicture.py:127](../../src/frontend_functions/timetablepicture.py#L127) の `textwrap.fill` を、`font.getlength` ベースの greedy 改行関数に置換:

```python
def _wrap_by_pixel(text: str, font: ImageFont.FreeTypeFont, max_px: float) -> list[str]:
    """1 文字ずつ実測しながら greedy に折り返す。日英混在に対応。"""
    lines: list[str] = []
    cur = ""
    for ch in text:
        if font.getlength(cur + ch) <= max_px:
            cur += ch
        else:
            if cur:
                lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines
```

`MAX_NAME_LINES` を超える行は末尾を `…` で省略するオプションも検討（後述）。

### E. 行高の正規化

[timetablepicture.py:129, 134](../../src/frontend_functions/timetablepicture.py#L129) の `font.getbbox(line)[3]` は文字列依存（ディセンダ有無で変動）。代わりに `font_size * LINE_HEIGHT_RATIO` を固定行高として使い、行間が揃うようにする。

---

## 実装ステップ

### Phase 1: `create_timetable_image` のシグネチャ拡張と内部リファクタ

ファイル: [src/frontend_functions/timetablepicture.py](../../src/frontend_functions/timetablepicture.py)

変更内容:
- 新引数 `image_width`, `image_height` を追加（既存の `start_margin`, `time_line_spacing`, `box_color` の後に）
- 旧来の「`time_line_spacing` から `text_font_size` と `image_width` を一気に決める」分岐 ([L37-L47](../../src/frontend_functions/timetablepicture.py#L37-L47)) を撤去
- 新しい順序:
  1. JSON から `total_minutes`, `min_event_min`, `start_time`/`end_time` を計算
  2. 引数で渡された `image_width`, `image_height`, `start_margin`, `time_line_spacing` を採用
  3. それらから `text_font_size` を § C の式で逆算
  4. `_wrap_by_pixel` で折り返し、固定行高で描画
- モジュール先頭に § C の定数を追加
- `_wrap_by_pixel` ヘルパー関数を追加（モジュール private）

互換性メモ:
- 既存呼び出し（[ocr_service.py:341](../../src/backend_functions/ocr_service.py#L341), [L347](../../src/backend_functions/ocr_service.py#L347)）は `image_width`/`image_height` を渡さない呼び方が残るため、これらを `None` 既定にし、`None` の場合は § B のフォールバック式で内部計算する。
- これにより `create_timetable_image(json_data)` 単独呼び出しも従来どおり動く。

### Phase 2: `generate_timetable_picture` でパラメータ算出

ファイル: [src/backend_functions/ocr_service.py](../../src/backend_functions/ocr_service.py)

変更内容 ([L313-L351](../../src/backend_functions/ocr_service.py#L313-L351)):
- `time_match=True` かつ `time_axis_converter is not None` の分岐内で:
  - `stage_img_path` の `Image.open` でサイズ取得（既に開いている箇所があれば共有）
  - `TimePixelConfig` から `source_ppm`, `source_start_pix` を取り出す
  - § A の式で `factor`, `image_width`, `image_height`, `start_margin`, `time_line_spacing` を算出
  - これらを `create_timetable_image` に渡す
- `time_match=False` または `time_axis_converter is None` のパス:
  - 引数なしで `create_timetable_image(json_data)` を呼ぶ（Phase 1 で内部フォールバックが対応する）

### Phase 3: 合成ステップで縦リサイズ

ファイル: [src/app.py](../../src/app.py)

変更内容 ([L1154-L1158](../../src/app.py#L1154-L1158)):

```python
if image_output.height != image.height:
    new_w = round(image_output.width * image.height / image_output.height)
    image_output = image_output.resize((new_w, image.height), Image.LANCZOS)
new_width  = image.width + image_output.width
new_height = image.height
new_image  = Image.new("RGB", (new_width, new_height), "white")
new_image.paste(image, (0, 0))
new_image.paste(image_output, (image.width, 0))
```

`Image.LANCZOS` 縮小により Phase 1/2 の高解像度レンダリングが活きる。

### Phase 4: 動作確認

確認観点:
1. **時刻線の一致**: 元画像と生成画像で 10:00, 10:30, ... の横線位置が並べたときに揃う
2. **横幅の一致**: 元画像と生成画像の幅が合成後ほぼ同じ
3. **テキストの可読性**:
   - 15 分枠に 20 文字日本語名が 2 行で収まる
   - 半角・全角混在名で改行位置が文字幅に従う（例: "ZOC SPECIAL ガールズユニット"）
   - 90 分枠が異常に大きく見えない（font は TARGET_FONT_SIZE 上限で頭打ち）
4. **`time_pixel` 未設定プロジェクト**: フォールバックパスでも単体で読みやすい画像が出る
5. **回帰**: 重複時インデント (`duplicate_margin`)、特典会併記、極端に短い／長いタイムテーブル

確認は既存プロジェクトの代表例（`data/projects/2026_05_ガルガルMORIMORI` 等）で実施。

---

## 影響範囲

| ファイル | 変更内容 | 行数目安 |
|---------|---------|---------|
| [src/frontend_functions/timetablepicture.py](../../src/frontend_functions/timetablepicture.py) | 全面リファクタ（シグネチャ拡張・定数追加・折り返し実装差し替え・行高正規化） | 100 行前後 |
| [src/backend_functions/ocr_service.py](../../src/backend_functions/ocr_service.py) | `generate_timetable_picture` のパラメータ算出ロジック差し替え | 20 行前後 |
| [src/app.py](../../src/app.py) | OCR セクションの合成箇所に縦リサイズを追加 | 数行 |

外部に新規ファイルは作らない。

---

## リスクと対応

| リスク | 影響 | 対応 |
|--------|------|------|
| 元画像が極端に低解像度で生成画像が巨大化する | メモリ・描画時間 | `MAX_GEN_HEIGHT = 4000` / `MAX_GEN_WIDTH = 1500` で絶対 px 上限を設ける。上限到達時は `gen_ppm < TARGET_PPM` となり単体可読性は下がるが、ファイルサイズは保証される |
| `MIN_GEN_WIDTH` が大きくて横方向に余白が増える | 並べた時のバランス | `MIN_GEN_WIDTH` は最終調整。実データで確認しながら値を詰める |
| `_wrap_by_pixel` が単語境界を無視して英単語を途中で切る | 可読性 | Phase 1 では許容。気になる場合は「半角英数連続中は単語境界を優先」フォローを Phase 5 として切り出す |
| 行が `MAX_NAME_LINES` を超えるアーティスト名がある | 箱からあふれる | 末尾省略 `…` をオプション化（デフォルト ON）。実装は Phase 1 内に含める |
| 既存の `box_color`, `duplicate_margin` 等の動作変更 | 回帰 | 既存挙動を維持。新引数は末尾に追加し、既定値で従来動作を再現可能にする |

---

## 後続検討（本計画外）

- アーティスト名の左寄せ・中央寄せ切り替え
- 重複時インデントの自動最適化（複数同時イベントを横に並べる、など）
- 特典会併記行のスタイル区別（薄い枠色など）
- 縦方向の時刻ラベル間隔を 30 分以外（15 分 / 60 分）に切り替え可能化
