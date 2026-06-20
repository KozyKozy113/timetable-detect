"""ステージカラー設定の単一窓口。

カラーの源泉は `data/master/color_preset.json`（{カラー名: [bg, fg]} を順序付きで保持）。
カラー名リスト・(bg, fg) 解決・カラーパレット画像パスを、推定 / ビルド / 描画 /
パレット生成の各ロジックで共通利用するためにここへ集約する。
"""
from __future__ import annotations

import json
import os

_MASTER_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "master")
)
COLOR_PRESET_PATH = os.path.join(_MASTER_DIR, "color_preset.json")
COLOR_PALETTE_PATH = os.path.join(_MASTER_DIR, "color_palette.png")


def load_color_preset() -> dict[str, tuple[str, str]]:
    """color_preset.json を読み込み {カラー名: (bg, fg)} を返す。読込失敗時は空 dict。"""
    try:
        with open(COLOR_PRESET_PATH, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return {}
    result: dict[str, tuple[str, str]] = {}
    for k, v in raw.items():
        if isinstance(v, list) and len(v) >= 2:
            result[k] = (str(v[0]), str(v[1]))
    return result


def get_preset_color_names() -> list[str]:
    """プリセットのカラー名を JSON の定義順で返す（デフォルトカラー循環の基準順）。"""
    return list(load_color_preset().keys())
