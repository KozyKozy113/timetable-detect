"""per-stage タイムテーブル画像のレイアウト計算。

元画像の時間軸 (time_pixel) と JSON のイベント範囲から、
- 描画用パラメータ (start_margin / time_line_spacing / image_height など)
- UI 表示時の整列に必要な拡張オフセット (top_extension_px / bottom_extension_px)
を一括で算出する。

生成側 (ocr_service.generate_timetable_picture) と UI 表示側 (app.py) で
共通のソースから値を導出するため、レイアウト計算ロジックの単一情報源となる。
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

from backend_functions.time_axis import TimeAxisConverter
from frontend_functions import timetablepicture


@dataclass
class StageLayout:
    """per-stage タイムテーブル画像のレイアウト計算結果。

    座標系: 描画用 (Y シフト適用後)。`start_margin` / `image_height` を
    `create_timetable_image` にそのまま渡せば、`source-aligned` 領域は
    Y = [top_extension_px, top_extension_px + source_aligned_height_px] に
    必ず配置される。
    """
    factor: float
    gen_ppm: float
    time_line_spacing: float
    source_aligned_height_px: int       # = round(source_height * factor)
    top_extension_px: int               # 上方向の拡張ピクセル (>= 0)
    bottom_extension_px: int            # 下方向の拡張ピクセル (>= 0)
    start_margin: int                   # Y シフト後 (>= 0)
    image_height: int                   # top_ext + src_aligned + bot_ext
    source_box_width: float             # = source_width * factor
    ext_start: _dt.datetime
    ext_end: _dt.datetime


def compute_stage_layout(
    json_data: dict,
    source_size: tuple[int, int],
    converter: TimeAxisConverter,
) -> StageLayout | None:
    """1ステージ分のレイアウトを計算して返す。

    `json_data["タイムテーブル"]` が空 / 全てパース失敗の場合は None。

    Y シフトを適用するため、`source_start_pix < 0` (extended start が
    元画像の time_start より前) のときも描画が image 外に出ない。

    `image_height` が MAX_GEN_HEIGHT を超えそうな場合は factor をさらに
    抑制して全体を MAX_GEN_HEIGHT 内に収める (異常系の保険)。
    """
    time_format = "%H:%M"
    timetable = json_data.get("タイムテーブル", []) or []
    starts: list[_dt.datetime] = []
    ends: list[_dt.datetime] = []
    for live in timetable:
        try:
            s = _dt.datetime.strptime(live["ライブステージ"]["from"], time_format)
            e = _dt.datetime.strptime(live["ライブステージ"]["to"], time_format)
        except (KeyError, ValueError, TypeError):
            continue
        starts.append(s)
        ends.append(e)
    if not starts or not ends:
        return None

    raw_start = min(starts)
    raw_end = max(ends)
    ext_start = raw_start.replace(minute=0)
    ext_end = raw_end.replace(minute=0) + _dt.timedelta(hours=1)

    source_width, source_height = source_size
    if source_height <= 0:
        return None

    config = converter.config
    if config.total_duration <= 0 or config.total_pix <= 0:
        return None
    source_ppm = config.total_pix / config.total_duration

    # --- factor 算出 (event_timetable_picture._compute_vertical_layout と整合) ---
    factor = max(1.0, timetablepicture.TARGET_PPM / source_ppm)
    factor = min(factor, timetablepicture.MAX_GEN_HEIGHT / source_height)
    factor = max(factor, 1.0)

    # --- 上下拡張の算出 ---
    source_start_pix = converter.time_to_pix(ext_start.time())
    source_end_pix = converter.time_to_pix(ext_end.time())
    total_min = int((ext_end - ext_start).total_seconds() / 60)

    def _build(f: float) -> tuple[int, int, int, int, int]:
        """factor を受けて (top_ext, src_aligned, bot_ext, start_margin, image_height) を返す。

        下方向拡張がある場合は末尾ラベル切れ防止用の `_MARGIN` (=10) を加算する。
        既存 `ocr_service` 旧ロジック (`image_height = max(source*factor, data_bottom_y + margin)`)
        と等価な振る舞いを保ち、拡張なしケースでは saved file 高さが従来と一致する。
        """
        start_margin_unshifted = round(source_start_pix * f)
        top_ext = max(0, -start_margin_unshifted)
        start_margin_shifted = start_margin_unshifted + top_ext
        src_aligned = int(round(source_height * f))
        gen_ppm_local = source_ppm * f
        data_bottom_unshifted = start_margin_unshifted + int(round(total_min * gen_ppm_local))
        if data_bottom_unshifted > src_aligned:
            bot_ext = data_bottom_unshifted + timetablepicture._MARGIN - src_aligned
        else:
            bot_ext = 0
        img_h = top_ext + src_aligned + bot_ext
        return top_ext, src_aligned, bot_ext, start_margin_shifted, img_h

    top_extension_px, source_aligned_height_px, bottom_extension_px, start_margin, image_height = _build(factor)

    # --- MAX_GEN_HEIGHT クランプ (image_height 基準) ---
    if image_height > timetablepicture.MAX_GEN_HEIGHT:
        factor_cap = factor * timetablepicture.MAX_GEN_HEIGHT / image_height
        factor = max(1.0, factor_cap)
        top_extension_px, source_aligned_height_px, bottom_extension_px, start_margin, image_height = _build(factor)

    gen_ppm = source_ppm * factor
    time_line_spacing = gen_ppm * 30
    source_box_width = source_width * factor

    return StageLayout(
        factor=factor,
        gen_ppm=gen_ppm,
        time_line_spacing=time_line_spacing,
        source_aligned_height_px=source_aligned_height_px,
        top_extension_px=top_extension_px,
        bottom_extension_px=bottom_extension_px,
        start_margin=start_margin,
        image_height=image_height,
        source_box_width=source_box_width,
        ext_start=ext_start,
        ext_end=ext_end,
    )
