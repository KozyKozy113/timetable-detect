"""
画像処理ロジック（ステージ分割、タイムライン検出）。

Streamlitに依存しない純粋な画像処理関数群。
UIレンダリング用のデータは戻り値として返す。
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from operator import itemgetter
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from backend_functions import gpt_ocr


# ---------------------------------------------------------------------------
# パラメータ / 結果 データクラス
# ---------------------------------------------------------------------------

@dataclass
class StageLineDetectParams:
    """detect_stagelineのパラメータ群"""
    x_minlength_rate: float
    x_edge_threshold_1: int
    x_edge_threshold_2: int
    x_hough_threshold: int
    x_hough_gap: int
    x_identify_interval: int


@dataclass
class StageLineResult:
    """detect_stagelineの結果"""
    stage_line_list: pd.DataFrame
    images_eachstage: list[Image.Image]
    images_eachstage_bbox: list[dict]
    annotated_image: Image.Image


@dataclass
class TimelineDetectParams:
    """detect_timeline_onlyonestageのパラメータ群"""
    y_minlength_rate: float
    y_edge_threshold_1: int
    y_edge_threshold_2: int
    y_hough_threshold: int
    y_hough_gap: int
    y_identify_interval: int
    y_ignoretime_threshold: float


@dataclass
class TimelineDetectResult:
    """detect_timeline_onlyonestageの結果"""
    timeline_df: pd.DataFrame
    addtime_image_path: str


# ---------------------------------------------------------------------------
# 画像キャッシュ
# ---------------------------------------------------------------------------

_image_cache: dict[str, Image.Image] = {}


def get_image(img_path: str) -> Image.Image:
    """画像をキャッシュ付きで読み込む。同じパスは一度だけ読み込む。"""
    if img_path not in _image_cache:
        _image_cache[img_path] = Image.open(img_path)
    return _image_cache[img_path]


# ---------------------------------------------------------------------------
# 純粋関数
# ---------------------------------------------------------------------------

def get_x_freq(image: np.ndarray, stage_num: int) -> pd.Series:
    """画像の矩形X位置の出現頻度を返す。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        rectangles.append((x, y, w, h))

    df_rectangle = pd.DataFrame(rectangles, columns=["横位置", "縦位置", "横幅", "縦幅"])
    min_width = image.shape[1] / stage_num / 3
    x_appear = pd.Series(df_rectangle[df_rectangle["横幅"] > min_width]["横位置"])
    return x_appear.value_counts()


def get_xpoint(image: Image.Image, stage_num: int) -> list[int]:
    """GPT-OCRによりステージ分割点のX座標リストを取得する。"""
    import json
    x_freq = get_x_freq(np.array(image), stage_num)
    response = gpt_ocr.get_xpoint(x_freq, stage_num)
    return json.loads(response.choices[0].message.content)["xpoint"]


def get_image_eachstage_byocr(
    image: Image.Image,
    stage_num: int,
) -> tuple[list[Image.Image], list[dict]]:
    """GPT-OCRベースでステージ画像を分割する。

    Returns:
        (images_eachstage, bboxes)
    """
    if stage_num == 1:
        return [image], [{"left": 0, "top": 0, "right": image.width, "bottom": image.height}]
    for _ in range(3):
        try:
            xpoints = get_xpoint(image, stage_num)
            break
        except ValueError:
            continue
    else:
        raise ValueError("ステージ分割点の取得に失敗しました")

    images_eachstage = []
    bboxes = []
    for i in range(stage_num):
        x_from = xpoints[i]
        if i < stage_num - 1:
            x_to = xpoints[i + 1]
        else:
            x_to = image.width
        width = x_to - x_from
        if i > 0:
            x_from = x_from - width / 10
        else:
            x_from = 0
        rect_img = image.crop((x_from, 0, x_to, image.height))
        images_eachstage.append(rect_img)
        bboxes.append({"left": int(x_from), "top": 0, "right": int(x_to), "bottom": image.height})
    return images_eachstage, bboxes


def split_image_evenly(
    image: Image.Image,
    stage_num: int,
    crop_box: dict,
) -> tuple[list[Image.Image], list[dict]]:
    """画像を均等に分割する（5%のオーバーラップ付き）。

    Args:
        image: 分割対象画像
        stage_num: 分割数
        crop_box: 元画像に対するクロップ領域 {"left", "top", "width", "height"}

    Returns:
        (images_eachstage, bboxes)
    """
    images_eachstage = []
    bboxes = []
    crop_offset_x = crop_box["left"]
    crop_offset_y = crop_box["top"]
    crop_bottom = crop_offset_y + crop_box["height"]

    if stage_num > 0:
        width, height = image.size
        segment_width = width / stage_num
        for j in range(stage_num):
            left = round(max(0, (j - 0.05) * segment_width))
            right = round(min((j + 1.05) * segment_width, width))
            segment = image.crop((left, 0, right, height))
            images_eachstage.append(segment)
            bboxes.append({
                "left": left + crop_offset_x,
                "top": crop_offset_y,
                "right": right + crop_offset_x,
                "bottom": crop_bottom,
            })
    return images_eachstage, bboxes


# ---------------------------------------------------------------------------
# ステージ線検出
# ---------------------------------------------------------------------------

def detect_stageline(
    image: Image.Image,
    params: StageLineDetectParams,
    crop_box: dict,
) -> StageLineResult:
    """ステージ領域を特定する縦線を検出し、分割結果を返す。

    Args:
        image: クロップ済み画像
        params: エッジ検出・ハフ変換パラメータ
        crop_box: 元画像に対するクロップ領域 {"left", "top", "width", "height"}

    Returns:
        StageLineResult（UI表示用の annotated_image を含む）
    """
    bgr_img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    height, width = gray_img.shape
    minlength = height * params.x_minlength_rate

    im_edges = cv2.Canny(
        gray_img, params.x_edge_threshold_1, params.x_edge_threshold_2, L2gradient=True
    )
    lines = cv2.HoughLinesP(
        im_edges, rho=1, theta=np.pi / 360,
        threshold=params.x_hough_threshold,
        minLineLength=minlength, maxLineGap=params.x_hough_gap,
    )

    line_list = []
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 == x2 or abs((y1 - y2) / (x1 - x2)) > np.tan(np.pi / 180 * 85):
                line_list.append([x1, y1, x2, y2, abs(y1 - y2)])
                draw.line((x1, y1, x2, y2), fill="red", width=20)

    line_list.sort(key=itemgetter(0, 1, 2, 3))
    line_x_list = pd.DataFrame(
        line_list, columns=["x1", "y1", "x2", "y2", "length"]
    ).groupby("x1").sum()[["length"]]

    x_before = 0
    new_line_x_list = []
    for x, row in line_x_list.iterrows():
        if x - x_before < params.x_identify_interval:
            if len(new_line_x_list) > 0:
                new_line_x_list[-1][0].append(x)
                new_line_x_list[-1][1] += row["length"]
            else:
                new_line_x_list.append([[0, x], row["length"]])
        else:
            if len(new_line_x_list) > 0:
                new_line_x_list[-1][0] = np.mean(new_line_x_list[-1][0])
            new_line_x_list.append([[x], row["length"]])
        x_before = x
    if len(new_line_x_list) > 0:
        new_line_x_list[-1][0] = np.mean(new_line_x_list[-1][0])

    stage_line_list = pd.DataFrame(new_line_x_list, columns=["x", "length"])

    # ステージ画像に分割
    crop_offset_x = crop_box["left"]
    crop_offset_y = crop_box["top"]
    crop_bottom = crop_offset_y + crop_box["height"]

    left = 0
    num = len(stage_line_list) + 1
    images_eachstage = []
    images_eachstage_bbox = []
    for i in range(num):
        if i < num - 1:
            right = stage_line_list.iat[i, 0]
        else:
            right = width
        images_eachstage.append(image.crop((left, 0, right, height)))
        images_eachstage_bbox.append({
            "left": int(left) + crop_offset_x,
            "top": crop_offset_y,
            "right": int(right) + crop_offset_x,
            "bottom": crop_bottom,
        })
        left = right

    return StageLineResult(
        stage_line_list=stage_line_list,
        images_eachstage=images_eachstage,
        images_eachstage_bbox=images_eachstage_bbox,
        annotated_image=image_copy,
    )


# ---------------------------------------------------------------------------
# タイムライン検出
# ---------------------------------------------------------------------------

def detect_timeline_onlyonestage(
    img_path: str,
    params: TimelineDetectParams,
    pix_to_time_fn: Callable[[float], object],
    time_length_to_pix_fn: Callable[[float, bool], float],
) -> Optional[TimelineDetectResult]:
    """1ステージのタイムライン横線を検出し、時刻付き画像を生成して保存する。

    Args:
        img_path: ステージ画像のパス
        params: 検出パラメータ
        pix_to_time_fn: ピクセル→時刻変換関数（TimeAxisConverter.pix_to_time）
        time_length_to_pix_fn: 分→ピクセル変換関数（TimeAxisConverter.time_length_to_pix）

    Returns:
        TimelineDetectResult or None（線が検出されなかった場合）
    """
    if not os.path.exists(img_path):
        raise ValueError(f"画像が見つかりません: {img_path}")

    image = Image.open(img_path)
    bgr_img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    height, width = gray_img.shape
    minlength = height * params.y_minlength_rate

    im_edges = cv2.Canny(
        gray_img, params.y_edge_threshold_1, params.y_edge_threshold_2, L2gradient=True
    )
    lines = cv2.HoughLinesP(
        im_edges, rho=1, theta=np.pi / 360,
        threshold=params.y_hough_threshold,
        minLineLength=minlength, maxLineGap=params.y_hough_gap,
    )
    if lines is None or len(lines) == 0:
        return None

    line_list = []
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 != x2 and abs((y1 - y2) / (x1 - x2)) < np.tan(np.pi / 180 * 5):
            line_list.append([x1, y1, x2, y2, abs(x1 - x2)])
            draw.line((x1, y1, x2, y2), fill="white", width=10)

    line_list.sort(key=itemgetter(1, 0, 2, 3))
    line_y_list = pd.DataFrame(
        line_list, columns=["x1", "y1", "x2", "y2", "length"]
    ).groupby("y1").sum()[["length"]]

    y_before = 0
    new_line_y_list = []
    for y, row in line_y_list.iterrows():
        if y - y_before < params.y_identify_interval:
            if len(new_line_y_list) > 0:
                new_line_y_list[-1][0].append(y)
                new_line_y_list[-1][1] += row["length"]
            else:
                new_line_y_list.append([[0, y], row["length"]])
        else:
            if len(new_line_y_list) > 0:
                new_line_y_list[-1][0] = np.mean(new_line_y_list[-1][0])
            new_line_y_list.append([[y], row["length"]])
        y_before = y
    if len(new_line_y_list) > 0:
        new_line_y_list[-1][0] = np.mean(new_line_y_list[-1][0])

    timeline_df = pd.DataFrame(new_line_y_list, columns=["y", "length"])
    timeline_df["time"] = timeline_df["y"].map(pix_to_time_fn)

    # 時刻情報を画像右側に拡張して追記
    font_size = time_length_to_pix_fn(4, True)
    if font_size is None or font_size <= 0:
        font_size = 12
    face = cv2.FONT_HERSHEY_PLAIN
    thickness = 1
    scale = cv2.getFontScaleFromHeight(face, int(font_size), thickness)
    font_pixel, baseline = cv2.getTextSize("00:00-00:00", face, scale, thickness)
    if font_pixel[0] < width:
        font_size = time_length_to_pix_fn(4 * width / font_pixel[0], True)
        if font_size is None or font_size <= 0:
            font_size = 12
        scale = cv2.getFontScaleFromHeight(face, int(font_size), thickness)
        font_pixel, baseline = cv2.getTextSize("00:00-00:00", face, scale, thickness)

    extension_width = int(font_pixel[0] * 1.2)
    extension_image = np.full((height, width + extension_width, 3), (255, 255, 255), dtype=np.uint8)
    extension_image[:, :width] = bgr_img

    for j in range(len(timeline_df) - 1):
        time_j = timeline_df.loc[j, "time"]
        time_j1 = timeline_df.loc[j + 1, "time"]
        if time_j is None or time_j1 is None:
            continue
        diff_minutes = (
            datetime.combine(datetime.today(), time_j1)
            - datetime.combine(datetime.today(), time_j)
        ).seconds / 60
        if params.y_ignoretime_threshold < diff_minutes:
            time_pix = (timeline_df.loc[j, "y"] + timeline_df.loc[j + 1, "y"]) / 2
            time_stamp = time_j.strftime('%H:%M') + "-" + time_j1.strftime('%H:%M')
            cv2.putText(
                extension_image, time_stamp,
                (width + int(width * 0.05), int(time_pix) + int(font_size * 0.6)),
                cv2.FONT_HERSHEY_PLAIN, scale, (0, 0, 0), thickness,
            )
    for _, row in timeline_df.iterrows():
        cv2.line(
            extension_image,
            (width, int(row["y"])), (width + extension_width, int(row["y"])),
            (0, 0, 0), 1,
        )

    addtime_path = img_path.replace(".png", "_addtime.png")
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmpfile:
        temp_path = tmpfile.name
        cv2.imwrite(temp_path, extension_image)
    shutil.move(temp_path, addtime_path)

    return TimelineDetectResult(
        timeline_df=timeline_df,
        addtime_image_path=addtime_path,
    )


# ---------------------------------------------------------------------------
# ステージ画像保存
# ---------------------------------------------------------------------------

def save_stage_images(
    pj_path: str,
    event_name: str,
    img_type: str,
    images_eachstage: list[Image.Image],
    images_eachstage_bbox: list[dict],
    cropped_image: Image.Image,
    crop_box: dict,
    project_info_json: dict,
    event_no: int,
    accept_flags: Optional[list[bool]] = None,
) -> dict:
    """ステージ画像を保存しproject_info_jsonを更新して返す。

    Args:
        accept_flags: Noneなら全ステージ保存。リストなら True のステージのみ保存。

    Returns:
        更新されたproject_info_json（呼び出し側が保存する）
    """
    # 遅延 import で循環を回避
    from backend_functions import project_repository as _repo

    # raw_cropped.pngの保存
    img_path = os.path.join(pj_path, event_name, img_type, "raw_cropped.png")
    cropped_image.save(img_path)

    timetable_info = _repo.get_image_entry_by_dir_name(
        project_info_json, event_no, img_type,
    )
    if timetable_info is None:
        raise KeyError(f"dir_name={img_type} not found in event_no={event_no}")
    timetable_info["raw_crop_box"] = crop_box
    parent_kind = timetable_info.get("kind")

    stage_num = 0
    for i, image_eachstage in enumerate(images_eachstage):
        if accept_flags is not None and not accept_flags[i]:
            continue
        save_path = os.path.join(pj_path, event_name, img_type, "stage_{}.png".format(stage_num))
        image_eachstage.save(save_path)
        stage_entry = {
            "stage_no": stage_num,
            "stage_name": "ステージ{}".format(stage_num),
            "bbox": images_eachstage_bbox[i],
            "kind": parent_kind,
        }
        if len(timetable_info["stage_list"]) <= stage_num:
            timetable_info["stage_list"].append(stage_entry)
        else:
            timetable_info["stage_list"][stage_num] = stage_entry
        stage_num += 1

    timetable_info["stage_num"] = stage_num
    return project_info_json


# ---------------------------------------------------------------------------
# 画像置き換え
# ---------------------------------------------------------------------------

def _rename_to_old(path: str) -> None:
    """ファイルが存在すれば拡張子の前に_oldを付けてリネームする。"""
    if not os.path.exists(path):
        return
    root, ext = os.path.splitext(path)
    old_path = root + "_old" + ext
    if os.path.exists(old_path):
        os.remove(old_path)
    os.rename(path, old_path)

def replace_stage_images_from_new_raw(
    new_image_path: str,
    pj_path: str,
    event_name: str,
    img_type: str,
    project_info_json: dict,
    event_no: int,
) -> Optional[str]:
    """新画像からbboxで各ステージ画像を切り出して置き換える。

    Returns:
        エラーメッセージ（成功時はNone）
    """
    from backend_functions import project_repository as _repo
    timetable_info = _repo.get_image_entry_by_dir_name(
        project_info_json, event_no, img_type,
    )
    if timetable_info is None:
        raise KeyError(f"dir_name={img_type} not found in event_no={event_no}")
    base_dir = os.path.join(pj_path, event_name, img_type)

    new_img = Image.open(new_image_path)

    # 既存のraw.pngとサイズ比較
    old_raw_path = os.path.join(base_dir, "raw.png")
    if os.path.exists(old_raw_path):
        old_img = Image.open(old_raw_path)
        if new_img.size != old_img.size:
            return "新しい画像のサイズ（{}）が既存画像のサイズ（{}）と異なるため、置き換えできません。".format(
                new_img.size, old_img.size
            )
        old_img.close()

    # 古いraw.pngを_oldにリネームして新画像を保存
    _rename_to_old(old_raw_path)
    new_img.save(old_raw_path)

    # raw_crop_boxが存在すればraw_cropped.pngを再生成
    if "raw_crop_box" in timetable_info:
        crop_box = timetable_info["raw_crop_box"]
        cropped = new_img.crop((
            crop_box["left"], crop_box["top"],
            crop_box["left"] + crop_box["width"],
            crop_box["top"] + crop_box["height"],
        ))
        raw_cropped_path = os.path.join(base_dir, "raw_cropped.png")
        _rename_to_old(raw_cropped_path)
        cropped.save(raw_cropped_path)

    # 各ステージのbboxで切り出して置き換え
    for stage_entry in timetable_info.get("stage_list", []):
        stage_no = stage_entry["stage_no"]
        bbox = stage_entry.get("bbox")
        if bbox is None:
            continue
        stage_img = new_img.crop((bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]))
        stage_path = os.path.join(base_dir, "stage_{}.png".format(stage_no))
        _rename_to_old(stage_path)
        stage_img.save(stage_path)

        # addtime画像が存在する場合は_oldにリネーム
        addtime_path = os.path.join(base_dir, "stage_{}_addtime.png".format(stage_no))
        _rename_to_old(addtime_path)

    return None
