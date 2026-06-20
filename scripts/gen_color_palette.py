"""④ステージカラーをLLMに読み取らせる機能で使うカラーパレット画像の生成スクリプト。

data/master/color_preset.json のデフォルトカラー(全プリセット)を1枚の
カラーパレット画像に集約再現する。各スウォッチは実タイムテーブル同様に
「背景色(bg) の上に前景色(fg) でカラー名 + bg HEX」を描画する。
カラー定義は backend_functions.stage_color（color_preset.json）に一元化済み。

color_preset.json を更新したら本スクリプトを再実行してパレット画像を作り直す。

実行: python scripts/gen_color_palette.py
出力: data/master/color_palette.png (stage_color.COLOR_PALETTE_PATH)
"""
import os
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from backend_functions import stage_color  # noqa: E402

FONT_PATH = os.path.join(ROOT, "src", "frontend_functions", "Fonts", "NotoSansJP-Regular.otf")
OUT_PATH = stage_color.COLOR_PALETTE_PATH

# --- レイアウト設定 ---
COLS = 3                 # 列数
SWATCH_W = 360           # スウォッチ幅
SWATCH_H = 110           # スウォッチ高さ
GAP = 16                 # スウォッチ間の余白
MARGIN = 32              # 画像外周の余白
BG_CANVAS = "#FFFFFF"    # キャンバス背景
NAME_FONT_SIZE = 30
HEX_FONT_SIZE = 22


def main() -> None:
    preset = stage_color.load_color_preset()
    items = list(preset.items())
    rows = (len(items) + COLS - 1) // COLS

    canvas_w = MARGIN * 2 + COLS * SWATCH_W + (COLS - 1) * GAP
    canvas_h = MARGIN * 2 + rows * SWATCH_H + (rows - 1) * GAP

    image = Image.new("RGB", (canvas_w, canvas_h), BG_CANVAS)
    draw = ImageDraw.Draw(image)

    try:
        name_font = ImageFont.truetype(FONT_PATH, NAME_FONT_SIZE)
        hex_font = ImageFont.truetype(FONT_PATH, HEX_FONT_SIZE)
    except OSError:
        print("フォント読み込み失敗、デフォルトフォントを使用:", FONT_PATH)
        name_font = ImageFont.load_default()
        hex_font = ImageFont.load_default()

    for idx, (name, (bg, fg)) in enumerate(items):
        col = idx % COLS
        row = idx // COLS
        x0 = MARGIN + col * (SWATCH_W + GAP)
        y0 = MARGIN + row * (SWATCH_H + GAP)
        x1 = x0 + SWATCH_W
        y1 = y0 + SWATCH_H

        # 白系は境界が分かるよう薄い枠線を付ける
        outline = "#CCCCCC"
        draw.rectangle([x0, y0, x1, y1], fill=bg, outline=outline, width=1)

        # カラー名 (fg 色)
        draw.text((x0 + 18, y0 + 22), name, fill=fg, font=name_font)
        # bg / fg の HEX (fg 色)
        draw.text((x0 + 18, y0 + 62), f"bg {bg} / fg {fg}", fill=fg, font=hex_font)

    image.save(OUT_PATH)
    print(f"保存しました: {OUT_PATH}  ({canvas_w}x{canvas_h}, {len(items)}色)")


if __name__ == "__main__":
    main()
