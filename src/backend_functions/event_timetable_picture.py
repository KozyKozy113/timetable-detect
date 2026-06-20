"""イベント単位の統合タイムテーブル画像生成。

⑥出力確認・編集タブで表示する「1イベントの全ステージを1枚に俯瞰する画像」を生成する。
2種類の画像を提供する:
  - 種別単位画像 (build_event_type_image): 1種別の全ステージを横並び
  - 種別横断画像 (build_event_image): 全種別×全ステージを横並び

入力は全て永続化済ファイル (stage_*.json / master_stage.csv / project_info.json /
raw_cropped.png) から読み込み、フロント状態 (AppState / DataFrame) には依存しない。

レイアウト方針:
  - 全ステージ列の幅・フォントサイズを統一する (内容に応じた伸縮はしない)
  - 最終画像は (列幅合計) : (共通縦幅) の自然なアスペクト比のまま保存する
  - ステージ名見出しの縦幅は、列幅とフォントサイズから必要行数を計算して決定する
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from backend_functions import project_repository as repo
from backend_functions import stage_color as _stage_color
from backend_functions import time_axis as _time_axis
from frontend_functions import timetablepicture


_TYPE_SEPARATOR_W = 3
_DEFAULT_LIVE_BG = "#FFFF99"
_DEFAULT_TOKUTENKAI_BG = "#DDDDDD"
_DEFAULT_TEXT = "#000000"

# 1ステージ列の幅の上限と下限 (px)。
_COLUMN_WIDTH_MIN = 220
_COLUMN_WIDTH_MAX = 480
# 集約画像のフォントサイズの下限・上限
_AGGREGATED_FONT_SIZE_MIN = 18
_AGGREGATED_FONT_SIZE_MAX = 32
# 全体画像の横幅・縦幅の目標上限 (px)。ブラウザでスムーズに開けるサイズに抑える。
# 越えそうな場合は列幅 / gen_ppm を適応的に縮める。
_TARGET_MAX_TOTAL_WIDTH = 14000
_TARGET_MAX_TOTAL_HEIGHT = 3200
# ステージ名見出しの最大行数 (これ以上は末尾省略)
_HEADER_MAX_LINES = 3
# 見出し内の上下パディング (px)
_HEADER_VPAD = 4


# ---------------------------------------------------------------------------
# 共通: 縦軸レイアウト計算
# ---------------------------------------------------------------------------

def _compute_vertical_layout(
    converter: _time_axis.TimeAxisConverter,
    source_width: int,
    source_height: int,
) -> dict:
    """raw_cropped.png サイズ + TimeAxisConverter から factor 系を算出する。

    ocr_service.generate_timetable_picture と同じロジックを共有。
    """
    config = converter.config
    source_ppm = config.total_pix / config.total_duration

    factor = max(1.0, timetablepicture.TARGET_PPM / source_ppm)
    factor = min(factor, timetablepicture.MAX_GEN_HEIGHT / source_height)
    factor = max(factor, 1.0)

    gen_ppm = source_ppm * factor
    image_height = round(source_height * factor)
    time_line_spacing = gen_ppm * 30
    source_box_width = source_width * factor

    return {
        "factor": factor,
        "gen_ppm": gen_ppm,
        "source_ppm": source_ppm,
        "image_height": image_height,
        "time_line_spacing": time_line_spacing,
        "source_width": source_width,
        "source_height": source_height,
        "source_box_width": source_box_width,
    }


# ---------------------------------------------------------------------------
# 共通: ステージカラー
# ---------------------------------------------------------------------------

def _default_color_resolver(is_tokutenkai: bool) -> tuple[str, str]:
    bg = _DEFAULT_TOKUTENKAI_BG if is_tokutenkai else _DEFAULT_LIVE_BG
    return bg, _DEFAULT_TEXT


# ---------------------------------------------------------------------------
# Phase 2-5-1: カラープリセット読み込み + ステージカラーリゾルバ
# カラー設定は stage_color モジュール（color_preset.json）へ集約済み。
# ---------------------------------------------------------------------------


def _parse_custom_color(value: str) -> tuple[str, str] | None:
    """`#bg-#fg` 形式をパースし (bg, fg) を返す。失敗時 None。"""
    if not isinstance(value, str):
        return None
    parts = value.split("-")
    if len(parts) != 2:
        return None
    bg, fg = parts[0].strip(), parts[1].strip()
    if not bg.startswith("#") or not fg.startswith("#"):
        return None
    return bg, fg


def make_stage_color_resolver(
    pj_path: str, event_name: str,
) -> Optional[Callable[[int], tuple[str, str]]]:
    """`master_stage.csv` の `カラー名` を解決する Callable を返す。

    マスタが無ければ None を返し、呼び出し側で `_default_color_resolver` に
    フォールバックさせる。
    """
    master_stage = _load_master_stage(pj_path, event_name)
    if master_stage is None or "カラー名" not in master_stage.columns:
        return None
    preset = _stage_color.load_color_preset()
    # ステージID -> (bg, fg) の dict を事前構築
    color_map: dict[int, tuple[str, str]] = {}
    for sid, row in master_stage.iterrows():
        raw_name = row.get("カラー名")
        is_tk = bool(row.get("特典会フラグ", False))
        if not isinstance(raw_name, str) or raw_name == "":
            color_map[int(sid)] = _default_color_resolver(is_tk)
            continue
        if raw_name in preset:
            color_map[int(sid)] = preset[raw_name]
            continue
        custom = _parse_custom_color(raw_name)
        if custom is not None:
            color_map[int(sid)] = custom
            continue
        color_map[int(sid)] = _default_color_resolver(is_tk)

    def resolver(stage_id: int) -> tuple[str, str]:
        try:
            return color_map[int(stage_id)]
        except (KeyError, ValueError, TypeError):
            return _default_color_resolver(False)
    return resolver


# ---------------------------------------------------------------------------
# 共通: master_stage.csv 読み込み / ステージ並びの構築
# ---------------------------------------------------------------------------

def _load_master_stage(pj_path: str, event_name: str) -> Optional[pd.DataFrame]:
    csv_path = os.path.join(pj_path, event_name, "master_stage.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path, index_col=0)
    if "表示順" not in df.columns:
        df["表示順"] = range(len(df))
    if "非活性化フラグ" not in df.columns:
        df["非活性化フラグ"] = False
    df["非活性化フラグ"] = df["非活性化フラグ"].fillna(False).astype(bool)
    df["表示順"] = df["表示順"].astype(int)
    return df


def _build_heiki_booth_entries(stage_data: list[dict]) -> list[dict]:
    """特典会併記タイプの全ステージ統合用に、ブース単位で列を再編成する。

    各ステージ個別画像では「ステージ列 + そのステージの特典会ブース列」を
    並べるが、全ステージ統合ではブースが物理的な場所として全ステージで共有
    されるため、ブースを1列 (=ステージ扱い) とし、各枠の表示名を
    特典会ではなく出演グループ名とする必要がある。

    各ステージのライブ枠に紐づく 特典会[] を ブース ごとに集約し、枠の
    縦位置・表示時刻には特典会自身の from/to を用いる。stage_id には特典会の
    ステージID を採用し、master_stage による並び替え・カラー解決に委ねる。
    """
    time_format = "%H:%M"
    booths: dict[str, dict] = {}
    order: list[str] = []
    for s in stage_data:
        jd = s.get("json_data") or {}
        for live in jd.get("タイムテーブル", []):
            group_name = (live.get("グループ名_採用") or live.get("グループ名") or "").strip()
            for tk in live.get("特典会", []) or []:
                booth = (tk.get("ブース") or "").strip()
                tk_from = (tk.get("from") or "").strip()
                tk_to = (tk.get("to") or "").strip()
                if not booth or not tk_from or not tk_to:
                    continue
                try:
                    _dt.datetime.strptime(tk_from, time_format)
                    _dt.datetime.strptime(tk_to, time_format)
                except ValueError:
                    continue
                if booth not in booths:
                    booths[booth] = {"stage_id": tk.get("ステージID"), "timetable": []}
                    order.append(booth)
                booths[booth]["timetable"].append({
                    "グループ名": group_name,
                    "グループ名_採用": group_name,
                    "ライブステージ": {"from": tk_from, "to": tk_to},
                })

    entries: list[dict] = []
    for booth in order:
        info = booths[booth]
        if not info["timetable"]:
            continue
        entries.append({
            "stage_id": info["stage_id"],
            "stage_no": None,
            "label": booth,
            "label_short": booth,
            "json_data": None,
            "tk_json_data": {"ステージ名": booth, "タイムテーブル": info["timetable"]},
            "is_tokutenkai": True,
        })
    return entries


def _build_type_stage_entries(
    pj_path: str, event_name: str, img_type: str,
    project_info_json: dict, event_no: int,
    master_stage: Optional[pd.DataFrame],
    variant: str,
) -> list[dict]:
    """1種別配下のステージ列描画情報を組み立てる。

    モードA (master_stage がある) → 表示順でソート、非活性化除外。
    モードB → project_info.stage_list[] の登録順。
    """
    entry = repo.get_image_entry_by_dir_name(project_info_json, event_no, img_type)
    if entry is None:
        return []
    kind = entry.get("kind")
    is_heiki = kind == "live_tokutenkai_heiki"
    is_tokutenkai_type = kind == "tokutenkai"

    if variant == "live":
        if is_tokutenkai_type:
            return []
    elif variant == "tokutenkai":
        if kind == "live":
            return []
    else:
        return []

    stage_list = entry.get("stage_list", [])
    stage_data: list[dict] = []
    for stage_info in stage_list:
        stage_no = stage_info["stage_no"]
        stage_id = stage_info.get("stage_id")
        json_path = os.path.join(
            pj_path, event_name, img_type, f"stage_{stage_no}.json",
        )
        if not os.path.exists(json_path):
            continue
        try:
            with open(json_path, encoding="utf-8") as f:
                jd = json.load(f)
        except (OSError, ValueError):
            continue
        stage_data.append({
            "stage_no": stage_no,
            "stage_id": stage_id,
            "stage_name": stage_info.get("stage_name", f"stage_{stage_no}"),
            "json_data": jd,
        })

    entries: list[dict] = []

    if variant == "live":
        for s in stage_data:
            entries.append({
                "stage_id": s["stage_id"],
                "stage_no": s["stage_no"],
                "label": s["stage_name"],
                "label_short": s["stage_name"],
                "json_data": s["json_data"],
                "tk_json_data": None,
                "is_tokutenkai": False,
            })
    else:  # tokutenkai
        if is_heiki:
            # 全ステージ統合ではブースをステージ扱いとし、表示名はグループ名にする。
            entries = _build_heiki_booth_entries(stage_data)
        elif is_tokutenkai_type:
            for s in stage_data:
                entries.append({
                    "stage_id": s["stage_id"],
                    "stage_no": s["stage_no"],
                    "label": s["stage_name"],
                    "label_short": s["stage_name"],
                    "json_data": s["json_data"],
                    "tk_json_data": None,
                    "is_tokutenkai": True,
                })

    if master_stage is not None:
        def _key(e):
            sid = e.get("stage_id")
            if sid is None or sid not in master_stage.index:
                return (1, 0)
            return (0, int(master_stage.loc[sid, "表示順"]))

        filtered = []
        for e in entries:
            sid = e.get("stage_id")
            if sid is None:
                filtered.append(e)
                continue
            if sid not in master_stage.index:
                continue
            if bool(master_stage.loc[sid, "非活性化フラグ"]):
                continue
            filtered.append(e)
        filtered.sort(key=_key)
        return filtered

    return entries


# ---------------------------------------------------------------------------
# 共通: 統一フォントサイズ・列幅・ヘッダ高さ
# ---------------------------------------------------------------------------

def _font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(timetablepicture.font_path, max(timetablepicture.MIN_FONT_SIZE, int(size)))
    except (OSError, IOError):
        return ImageFont.load_default()


def _collect_min_event_min(stage_entries: list[dict]) -> int:
    """全ステージのイベントから最短イベント分数を返す。"""
    time_format = "%H:%M"
    mins: list[int] = []
    for e in stage_entries:
        for jd in (e.get("json_data"), e.get("tk_json_data")):
            if not jd:
                continue
            for live in jd.get("タイムテーブル", []):
                try:
                    s = _dt.datetime.strptime(live["ライブステージ"]["from"], time_format)
                    t = _dt.datetime.strptime(live["ライブステージ"]["to"], time_format)
                    m = int((t - s).total_seconds() / 60)
                    if m > 0:
                        mins.append(m)
                except (KeyError, ValueError, TypeError):
                    continue
    return min(mins) if mins else 15


def _compute_unified_text_layout(
    stage_entries: list[dict],
    gen_ppm: float,
    timeline_column_width: int = 80,
) -> tuple[int, int]:
    """全ステージ共通の (font_size, column_width) を返す。

    フォントサイズは _AGGREGATED_FONT_SIZE_MIN〜_MAX にクランプ。
    列幅は font_size から必要テキスト幅 + マージンを計算した上で、
    _COLUMN_WIDTH_MIN〜_MAX にクランプする。
    さらに、ステージ数が多い場合は (列幅×N + 時間軸) が
    _TARGET_MAX_TOTAL_WIDTH を超えないよう適応的に列幅を縮める。
    """
    min_event_min = _collect_min_event_min(stage_entries)
    box_height_min = max(1.0, gen_ppm * min_event_min)
    # 2 行表示が成り立つフォントサイズ (box_height_min 基準)
    avail_h = max(1.0, box_height_min - 2 * timetablepicture._BOX_VPAD)
    font_size_from_h = int(avail_h / (2 * timetablepicture.LINE_HEIGHT_RATIO))
    font_size = max(_AGGREGATED_FONT_SIZE_MIN,
                    min(_AGGREGATED_FONT_SIZE_MAX, font_size_from_h))

    # 列幅: 時刻テキスト "00:00 ～ 00:00 (00)" が確実に収まる最低幅を計算
    font = _font(font_size)
    sample_w = font.getlength("00:00 ～ 00:00 (00)")
    col_for_text = int(round(
        sample_w
        + 2 * timetablepicture._TEXT_MARGIN
        + 2 * timetablepicture._BOX_MARGIN
        + 2 * timetablepicture._MARGIN
        + timetablepicture._WIDTH_SAFETY_PAD
    ))
    # 列幅: box_height_min / TARGET_BOX_ASPECT_2LINE を基準とする幅
    col_from_aspect = int(round(
        box_height_min / timetablepicture.TARGET_BOX_ASPECT_2LINE
        + 2 * timetablepicture._TEXT_MARGIN
        + 2 * timetablepicture._BOX_MARGIN
        + 2 * timetablepicture._MARGIN
    ))
    natural_col = max(col_for_text, col_from_aspect)
    natural_col = max(_COLUMN_WIDTH_MIN, min(_COLUMN_WIDTH_MAX, natural_col))

    # 適応縮小: 全体幅が目標上限を超えるなら列幅を縮める (ただし MIN は維持)
    num_stages = max(1, len(stage_entries))
    budget = (_TARGET_MAX_TOTAL_WIDTH - timeline_column_width) / num_stages
    if budget < natural_col:
        column_width = max(_COLUMN_WIDTH_MIN, int(budget))
    else:
        column_width = natural_col
    return font_size, column_width


def _cap_gen_ppm_for_aggregated(
    gen_ppm: float, total_minutes: int,
) -> float:
    """gen_ppm を、image_height が _TARGET_MAX_TOTAL_HEIGHT を超えないようにクランプ。"""
    if total_minutes <= 0 or gen_ppm <= 0:
        return gen_ppm
    margin = timetablepicture._MARGIN
    max_spacing_total = max(1, _TARGET_MAX_TOTAL_HEIGHT - 4 * margin)
    max_gen_ppm = max_spacing_total / total_minutes
    return min(gen_ppm, max_gen_ppm)


def _measure_header_height(
    labels: list[str], column_width: int, font_size: int,
) -> tuple[int, int]:
    """ステージ名一覧から見出し領域の (height, line_height) を返す。

    label 1 件ごとに column_width 内での折り返し行数を測り、
    最大行数 (最大 _HEADER_MAX_LINES) を採用して全列共通の高さとする。
    """
    font = _font(font_size)
    try:
        ascent, descent = font.getmetrics()
        line_h = max(1, ascent + descent)
    except Exception:
        line_h = max(1, int(round(font_size * timetablepicture.LINE_HEIGHT_RATIO)))
    inner_w = max(1, column_width - 2 * _HEADER_VPAD)
    max_lines = 1
    for label in labels:
        if not label:
            continue
        lines = timetablepicture._wrap_by_pixel(label, font, inner_w)
        n = min(_HEADER_MAX_LINES, max(1, len(lines)))
        if n > max_lines:
            max_lines = n
    height = max_lines * line_h + 2 * _HEADER_VPAD
    return height, line_h


def _render_header(
    label: str, width: int, height: int, font_size: int, line_height: int,
    bg: str, fg: str,
) -> Image.Image:
    """ステージ名ヘッダ画像を生成 (折り返し対応)。"""
    img = Image.new("RGB", (max(1, width), max(1, height)), bg)
    draw = ImageDraw.Draw(img)
    # 下辺の区切り線
    draw.line([(0, height - 1), (width - 1, height - 1)], fill="#888888", width=1)
    if not label:
        return img
    font = _font(font_size)
    inner_w = max(1, width - 2 * _HEADER_VPAD)
    lines = timetablepicture._wrap_by_pixel(label, font, inner_w)
    lines = timetablepicture._truncate_lines(lines, font, inner_w, _HEADER_MAX_LINES)
    total_text_h = line_height * len(lines)
    y = max(_HEADER_VPAD, int((height - total_text_h) / 2))
    for line in lines:
        text_w = font.getlength(line)
        x = max(_HEADER_VPAD, int((width - text_w) / 2))
        draw.text((x, y), line, fill=fg, font=font)
        y += line_height
    return img


def _stack_header_over(body: Image.Image, header: Image.Image) -> Image.Image:
    w = max(body.width, header.width)
    h = body.height + header.height
    img = Image.new("RGB", (w, h), "white")
    img.paste(header, (0, 0))
    img.paste(body, (0, header.height))
    return img


# ---------------------------------------------------------------------------
# Phase 1: 種別単位画像
# ---------------------------------------------------------------------------

def build_event_type_image(
    pj_path: str,
    event_name: str,
    img_type: str,
    project_info_json: dict,
    *,
    variant: str = "live",
    stage_color_resolver: Optional[Callable[[int], tuple[str, str]]] = None,
) -> Optional[Image.Image]:
    """1イベント・1種別の全ステージを横並びにした統合タイテ画像を返す。

    全ステージ列の幅・フォントサイズを統一する。
    最終画像は (列幅合計 + 時間軸列幅) × 共通縦幅 のままリサイズ無しで返す。
    """
    event_no = repo.get_event_no_by_event_name(project_info_json, event_name)
    if event_no is None:
        return None
    entry = repo.get_image_entry_by_dir_name(project_info_json, event_no, img_type)
    if entry is None:
        return None

    master_stage = _load_master_stage(pj_path, event_name)
    stage_entries = _build_type_stage_entries(
        pj_path, event_name, img_type,
        project_info_json, event_no, master_stage, variant,
    )
    if not stage_entries:
        return None

    converter = _time_axis.TimeAxisConverter.from_project_info(
        project_info_json, event_no, img_type,
    )
    raw_path = os.path.join(pj_path, event_name, img_type, "raw_cropped.png")
    if converter is None or not os.path.exists(raw_path):
        return None
    with Image.open(raw_path) as _src:
        sw, sh = _src.size
    vlayout = _compute_vertical_layout(converter, sw, sh)

    unified_start, unified_end = _collect_time_range(stage_entries)
    if unified_start is None or unified_end is None:
        return None

    # 縦サイズの上限: 必要 source-pixel 空間高さ × factor ≤ _TARGET_MAX_TOTAL_HEIGHT
    # データ範囲 (unified_end) が元画像の縦範囲を超える場合に備え、必要な高さを算出。
    src_h = vlayout["source_height"]
    src_ppm = vlayout["source_ppm"]
    factor = vlayout["factor"]
    src_data_end_pix = converter.time_to_pix(unified_end.time())
    required_src_h = max(src_h, src_data_end_pix)
    factor_cap = _TARGET_MAX_TOTAL_HEIGHT / max(1, required_src_h)
    factor = max(1.0, min(factor, factor_cap))
    gen_ppm = src_ppm * factor if src_ppm > 0 else vlayout["gen_ppm"]
    time_line_spacing = gen_ppm * 30
    margin = timetablepicture._MARGIN
    start_margin = int(round(converter.time_to_pix(unified_start.time()) * factor))
    total_minutes = int((unified_end - unified_start).total_seconds() / 60)
    # データ末尾の y + 余白を確保 (元画像基準とデータ基準のうち大きい方)
    data_bottom_y = start_margin + int(round(total_minutes * gen_ppm))
    image_height = max(int(round(src_h * factor)), data_bottom_y + margin)

    # 共通フォントサイズと列幅
    font_size, column_width = _compute_unified_text_layout(stage_entries, gen_ppm)
    # ヘッダ高さ
    labels = [e.get("label_short") or e.get("label", "") for e in stage_entries]
    header_height, header_line_height = _measure_header_height(
        labels, column_width, font_size,
    )

    return _compose_columns(
        stage_entries=stage_entries,
        converter=converter,
        unified_start=unified_start,
        unified_end=unified_end,
        start_margin=start_margin,
        time_line_spacing=time_line_spacing,
        image_height=image_height,
        column_width=column_width,
        font_size=font_size,
        header_height=header_height,
        header_line_height=header_line_height,
        stage_color_resolver=stage_color_resolver,
        separator_block_change=None,
    )


def _compose_columns(
    *,
    stage_entries: list[dict],
    converter: _time_axis.TimeAxisConverter,
    unified_start: _dt.datetime,
    unified_end: _dt.datetime,
    start_margin: int,
    time_line_spacing: float,
    image_height: int,
    column_width: int,
    font_size: int,
    header_height: int,
    header_line_height: int,
    stage_color_resolver: Optional[Callable[[int], tuple[str, str]]],
    separator_block_change: Optional[list[int]],
) -> Optional[Image.Image]:
    """列群を生成・合成する共通処理。"""
    color_fn = stage_color_resolver

    # 時間軸列を最左に配置
    timeline_img = timetablepicture.create_timeline_image(
        start_time=unified_start, end_time=unified_end,
        start_margin=start_margin, time_line_spacing=time_line_spacing,
        image_height=image_height, font_size=font_size,
        line_extent_width=0,
    )
    # 時間軸列のヘッダはラベル無しの空白 (高さだけ揃える)
    timeline_header = _render_header(
        "", timeline_img.width, header_height, font_size, header_line_height,
        bg="white", fg=_DEFAULT_TEXT,
    )
    timeline_column = _stack_header_over(timeline_img, timeline_header)

    stage_columns: list[Image.Image] = []
    for e in stage_entries:
        is_tokutenkai = bool(e["is_tokutenkai"])
        bg_color, fg_color = (
            color_fn(e["stage_id"]) if (color_fn and e.get("stage_id") is not None)
            else _default_color_resolver(is_tokutenkai)
        )
        jd = e["json_data"] if e["json_data"] is not None else e["tk_json_data"]
        if jd is None:
            continue
        body = timetablepicture.create_timetable_image(
            jd,
            start_margin=start_margin,
            time_line_spacing=time_line_spacing,
            image_height=image_height,
            box_color=bg_color,
            show_timeline_labels=False,
            apply_max_width_clamp=False,
            start_time_override=unified_start,
            end_time_override=unified_end,
            force_image_width=column_width,
            force_font_size=font_size,
            text_color_in_box=fg_color,
        )
        if body is None:
            # データ無しでも空の列を確保 (列幅・順序の整合性を保つ)
            body = Image.new("RGB", (column_width, image_height), "white")
        header = _render_header(
            e.get("label_short") or e.get("label", ""),
            column_width, header_height, font_size, header_line_height,
            bg=bg_color, fg=fg_color,
        )
        stage_columns.append(_stack_header_over(body, header))

    if not stage_columns:
        return None

    # 種別境界の区切り線 (Phase 2 のみ使用)
    separators = separator_block_change or []
    # 列の前に時間軸列 1 つ挿入するので、separator indices は +1 補正
    separators_adj = [i + 1 for i in separators]

    all_columns = [timeline_column] + stage_columns
    combined = timetablepicture._hstack_many(
        all_columns,
        separator_width=_TYPE_SEPARATOR_W if separators_adj else 0,
        separator_color="#666666",
        separator_indices=separators_adj,
    )
    return combined


def _collect_time_range(
    stage_entries: list[dict],
) -> tuple[Optional[_dt.datetime], Optional[_dt.datetime]]:
    time_format = "%H:%M"
    starts: list[_dt.datetime] = []
    ends: list[_dt.datetime] = []
    for e in stage_entries:
        for jd in (e.get("json_data"), e.get("tk_json_data")):
            if not jd:
                continue
            for live in jd.get("タイムテーブル", []):
                try:
                    starts.append(_dt.datetime.strptime(
                        live["ライブステージ"]["from"], time_format,
                    ))
                    ends.append(_dt.datetime.strptime(
                        live["ライブステージ"]["to"], time_format,
                    ))
                except (KeyError, ValueError, TypeError):
                    continue
    if not starts or not ends:
        return None, None
    s = min(starts).replace(minute=0)
    e = max(ends).replace(minute=0) + _dt.timedelta(hours=1)
    return s, e


# ---------------------------------------------------------------------------
# Phase 2: 種別横断画像 (全体)
# ---------------------------------------------------------------------------

def build_event_image(
    pj_path: str,
    event_name: str,
    project_info_json: dict,
    *,
    stage_color_resolver: Optional[Callable[[int], tuple[str, str]]] = None,
) -> Optional[Image.Image]:
    """1イベントの全種別×全ステージをまとめた統合タイテ画像を返す。

    縦軸を全種別で統一し、全ステージ列の幅・フォントサイズを共通化する。
    """
    event_no = repo.get_event_no_by_event_name(project_info_json, event_name)
    if event_no is None:
        return None
    event_type_list = repo.get_event_type_list(project_info_json, event_no)
    if not event_type_list:
        return None

    master_stage = _load_master_stage(pj_path, event_name)

    # 全ステージ統合エントリ + 各 (種別, variant) ブロック境界
    all_entries: list[dict] = []
    block_changes: list[int] = []  # all_entries 上の境界 (このインデックスの直後に区切り線)
    last_block_key: Optional[tuple[str, str]] = None

    max_gen_ppm = 0.0
    all_starts: list[_dt.datetime] = []
    all_ends: list[_dt.datetime] = []

    for img_type in event_type_list:
        entry = repo.get_image_entry_by_dir_name(project_info_json, event_no, img_type)
        if entry is None:
            continue
        kind = entry.get("kind")
        if kind == "live":
            variants_for_type = ["live"]
        elif kind == "tokutenkai":
            variants_for_type = ["tokutenkai"]
        elif kind == "live_tokutenkai_heiki":
            variants_for_type = ["live", "tokutenkai"]
        else:
            variants_for_type = ["live"]

        converter = _time_axis.TimeAxisConverter.from_project_info(
            project_info_json, event_no, img_type,
        )
        raw_path = os.path.join(pj_path, event_name, img_type, "raw_cropped.png")
        if converter is None or not os.path.exists(raw_path):
            continue
        with Image.open(raw_path) as _src:
            sw, sh = _src.size
        vlayout = _compute_vertical_layout(converter, sw, sh)
        max_gen_ppm = max(max_gen_ppm, vlayout["gen_ppm"])

        for v in variants_for_type:
            entries = _build_type_stage_entries(
                pj_path, event_name, img_type,
                project_info_json, event_no, master_stage, v,
            )
            if not entries:
                continue
            ts, te = _collect_time_range(entries)
            if ts is not None and te is not None:
                all_starts.append(ts)
                all_ends.append(te)
            block_key = (img_type, v)
            if last_block_key is not None and block_key != last_block_key:
                block_changes.append(len(all_entries) - 1)
            for e in entries:
                all_entries.append(e)
            last_block_key = block_key

    if not all_entries or not all_starts or not all_ends or max_gen_ppm <= 0:
        return None

    unified_start = min(all_starts).replace(minute=0)
    unified_end = max(all_ends).replace(minute=0) + _dt.timedelta(hours=1)
    total_minutes = int((unified_end - unified_start).total_seconds() / 60)

    gen_ppm = _cap_gen_ppm_for_aggregated(max_gen_ppm, total_minutes)
    time_line_spacing = gen_ppm * 30
    margin = timetablepicture._MARGIN
    start_margin = margin
    image_height = int(start_margin + margin + (total_minutes // 30) * time_line_spacing)

    # 共通フォントサイズ・列幅 (全種別の全ステージから決定)
    font_size, column_width = _compute_unified_text_layout(all_entries, gen_ppm)
    labels = [e.get("label_short") or e.get("label", "") for e in all_entries]
    header_height, header_line_height = _measure_header_height(
        labels, column_width, font_size,
    )

    return _compose_columns(
        stage_entries=all_entries,
        converter=None,  # 種別横断では converter を使わない (start_margin は明示的に渡す)
        unified_start=unified_start,
        unified_end=unified_end,
        start_margin=start_margin,
        time_line_spacing=time_line_spacing,
        image_height=image_height,
        column_width=column_width,
        font_size=font_size,
        header_height=header_height,
        header_line_height=header_line_height,
        stage_color_resolver=stage_color_resolver,
        separator_block_change=block_changes,
    )


# ---------------------------------------------------------------------------
# 保存ラッパー
# ---------------------------------------------------------------------------

def _event_type_image_path(
    pj_path: str, event_name: str, img_type: str, variant: str,
) -> str:
    fname = "all_stages_live.png" if variant == "live" else "all_stages_tokutenkai.png"
    return os.path.join(pj_path, event_name, img_type, fname)


def _event_image_path(pj_path: str, event_name: str) -> str:
    return os.path.join(pj_path, event_name, "all_stages.png")


def _expected_variants(kind: Optional[str]) -> list[str]:
    if kind == "live":
        return ["live"]
    if kind == "tokutenkai":
        return ["tokutenkai"]
    if kind == "live_tokutenkai_heiki":
        return ["live", "tokutenkai"]
    return ["live"]


def _save_event_type_variant(
    pj_path: str,
    event_name: str,
    img_type: str,
    variant: str,
    project_info_json: dict,
    expected: set,
    stage_color_resolver: Optional[Callable[[int], tuple[str, str]]],
) -> Optional[tuple[str, str]]:
    """1 (img_type, variant) の集約画像を生成・保存。
    生成された場合は (variant, out_path) を返す。生成不要 / 失敗時は None。
    """
    out_path = _event_type_image_path(pj_path, event_name, img_type, variant)
    if variant not in expected:
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
        return None
    img = build_event_type_image(
        pj_path, event_name, img_type, project_info_json,
        variant=variant,
        stage_color_resolver=stage_color_resolver,
    )
    if img is None:
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
        return None
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)
    return (variant, out_path)


def save_event_type_images(
    pj_path: str,
    event_name: str,
    img_type: str,
    project_info_json: dict,
    *,
    stage_color_resolver: Optional[Callable[[int], tuple[str, str]]] = None,
) -> dict[str, str]:
    """build_event_type_image() を全 variant について実行し PNG 保存。
    variant 間は並列実行する。
    """
    event_no = repo.get_event_no_by_event_name(project_info_json, event_name)
    if event_no is None:
        return {}
    entry = repo.get_image_entry_by_dir_name(project_info_json, event_no, img_type)
    if entry is None:
        return {}
    if stage_color_resolver is None:
        stage_color_resolver = make_stage_color_resolver(pj_path, event_name)
    expected = set(_expected_variants(entry.get("kind")))

    variants = ("live", "tokutenkai")
    result: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(variants)) as executor:
        futures = [
            executor.submit(
                _save_event_type_variant,
                pj_path, event_name, img_type, variant,
                project_info_json, expected, stage_color_resolver,
            )
            for variant in variants
        ]
        for future in futures:
            res = future.result()
            if res is not None:
                variant, out_path = res
                result[variant] = out_path
    return result


def save_event_image(
    pj_path: str,
    event_name: str,
    project_info_json: dict,
    *,
    stage_color_resolver: Optional[Callable[[int], tuple[str, str]]] = None,
) -> Optional[str]:
    """build_event_image() を実行し PNG 保存。"""
    out_path = _event_image_path(pj_path, event_name)
    if stage_color_resolver is None:
        stage_color_resolver = make_stage_color_resolver(pj_path, event_name)
    img = build_event_image(
        pj_path, event_name, project_info_json,
        stage_color_resolver=stage_color_resolver,
    )
    if img is None:
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
        return None
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)
    return out_path


def regenerate_event_type_images(
    pj_path: str,
    event_name: str,
    img_type: str,
    project_info_json: dict,
    *,
    include_cross_type: bool = True,
    stage_color_resolver: Optional[Callable[[int], tuple[str, str]]] = None,
) -> None:
    """1種別の集約画像を再生成する。include_cross_type=True なら全体画像も。
    include_cross_type=True の場合、種別単位と種別横断を並列実行する。
    """
    if stage_color_resolver is None:
        stage_color_resolver = make_stage_color_resolver(pj_path, event_name)
    if not include_cross_type:
        save_event_type_images(
            pj_path, event_name, img_type, project_info_json,
            stage_color_resolver=stage_color_resolver,
        )
        return
    with ThreadPoolExecutor(max_workers=2) as executor:
        type_future = executor.submit(
            save_event_type_images,
            pj_path, event_name, img_type, project_info_json,
            stage_color_resolver=stage_color_resolver,
        )
        cross_future = executor.submit(
            save_event_image,
            pj_path, event_name, project_info_json,
            stage_color_resolver=stage_color_resolver,
        )
        type_future.result()
        cross_future.result()


def regenerate_all_event_images(
    pj_path: str,
    event_name: str,
    project_info_json: dict,
    *,
    stage_color_resolver: Optional[Callable[[int], tuple[str, str]]] = None,
) -> None:
    """1イベントの全集約画像を再生成する。
    全 (種別 × variant) と種別横断画像を単一プールで並列実行する。
    """
    event_no = repo.get_event_no_by_event_name(project_info_json, event_name)
    if event_no is None:
        return
    if stage_color_resolver is None:
        stage_color_resolver = make_stage_color_resolver(pj_path, event_name)

    # 種別×variant のタスクを構築
    type_variant_tasks: list[tuple[str, str, set]] = []
    for img_type in repo.get_event_type_list(project_info_json, event_no):
        entry = repo.get_image_entry_by_dir_name(project_info_json, event_no, img_type)
        if entry is None:
            continue
        expected = set(_expected_variants(entry.get("kind")))
        for variant in ("live", "tokutenkai"):
            type_variant_tasks.append((img_type, variant, expected))

    # 種別横断画像も同じプール内で並列実行 (+1)
    max_workers = max(1, len(type_variant_tasks) + 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _save_event_type_variant,
                pj_path, event_name, img_type, variant,
                project_info_json, expected, stage_color_resolver,
            )
            for img_type, variant, expected in type_variant_tasks
        ]
        cross_future = executor.submit(
            save_event_image,
            pj_path, event_name, project_info_json,
            stage_color_resolver=stage_color_resolver,
        )
        for future in futures:
            future.result()
        cross_future.result()
