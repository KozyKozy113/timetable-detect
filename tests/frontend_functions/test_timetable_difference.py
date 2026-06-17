"""変更比較（差分検出）のテスト。

`changed_area_ratio` の変化ピクセル割合算出と、
`analyze_difference_by_stage` のステージ別差分判定（bbox座標系の整合・
サイズ違いの新画像のリサイズ・スコア算出）を検証する。
"""

from __future__ import annotations

from PIL import Image

from frontend_functions import timetable_difference as td


# ---------------------------------------------------------------------------
# changed_area_ratio
# ---------------------------------------------------------------------------

def test_changed_area_ratio_zero_for_no_diff():
    """完全に同一（差分0）の領域は変化割合0%。"""
    diff = Image.new("RGB", (10, 10), color=(0, 0, 0))
    assert td.changed_area_ratio(diff) == 0.0


def test_changed_area_ratio_ignores_below_noise_floor():
    """ノイズ下限以下の微小差分は変化として数えない。"""
    diff = Image.new("RGB", (10, 10), color=(td.PIXEL_NOISE_FLOOR, 0, 0))
    assert td.changed_area_ratio(diff) == 0.0


def test_changed_area_ratio_counts_above_noise_floor():
    """ノイズ下限を超えるピクセルは変化として割合に反映される。"""
    diff = Image.new("RGB", (10, 10), color=(0, 0, 0))
    # 上半分(50px)だけ強い差分を入れる
    for y in range(5):
        for x in range(10):
            diff.putpixel((x, y), (200, 0, 0))
    assert td.changed_area_ratio(diff) == 50.0


def test_changed_area_ratio_empty_region():
    """幅または高さ0の領域は0%を返す（ゼロ除算しない）。"""
    diff = Image.new("RGB", (0, 10), color=(255, 255, 255))
    assert td.changed_area_ratio(diff) == 0.0


# ---------------------------------------------------------------------------
# analyze_difference_by_stage
# ---------------------------------------------------------------------------

def _stage(stage_no, name, box):
    left, top, right, bottom = box
    return {
        "stage_no": stage_no,
        "stage_name": name,
        "bbox": {"left": left, "top": top, "right": right, "bottom": bottom},
    }


def test_analyze_detects_per_stage_difference():
    """差分のあるステージのみスコアが立ち、ないステージは0%になる。"""
    old = Image.new("RGB", (100, 100), color=(255, 255, 255))
    new = old.copy()
    # 右半分(x>=50)のステージ領域だけを黒く塗る = 差分
    for y in range(100):
        for x in range(50, 100):
            new.putpixel((x, y), (0, 0, 0))

    stage_list = [
        _stage(0, "左", (0, 0, 50, 100)),
        _stage(1, "右", (50, 0, 100, 100)),
    ]
    result = td.analyze_difference_by_stage(new, old, stage_list)

    stages = {s["stage_no"]: s for s in result["stages"]}
    assert stages[0]["score"] == 0.0
    assert stages[1]["score"] == 100.0
    assert stages[1]["diff_crop"].size == (50, 100)
    # 既存/新規のステージ領域画像も bbox で切り出される
    assert stages[1]["old_crop"].size == (50, 100)
    assert stages[1]["new_crop"].size == (50, 100)
    # 右ステージは既存=白, 新規=黒
    assert stages[1]["old_crop"].getpixel((0, 0)) == (255, 255, 255)
    assert stages[1]["new_crop"].getpixel((0, 0)) == (0, 0, 0)


def test_analyze_resizes_new_to_old_coordinate_system():
    """サイズの異なる新画像は旧画像サイズに揃えられ、bboxが有効に働く。"""
    old = Image.new("RGB", (100, 100), color=(255, 255, 255))
    new = Image.new("RGB", (200, 200), color=(255, 255, 255))  # 旧と同内容だが2倍サイズ

    stage_list = [_stage(0, "S", (0, 0, 100, 100))]
    result = td.analyze_difference_by_stage(new, old, stage_list)

    # 同内容ならリサイズ後も差分なし
    assert result["new_image"].size == (100, 100)
    assert result["stages"][0]["score"] == 0.0


def test_analyze_skips_stage_without_bbox():
    """bbox未設定のステージは結果から除外される。"""
    old = Image.new("RGB", (50, 50), color=(255, 255, 255))
    new = old.copy()
    stage_list = [
        {"stage_no": 0, "stage_name": "no_bbox"},
        _stage(1, "withbbox", (0, 0, 50, 50)),
    ]
    result = td.analyze_difference_by_stage(new, old, stage_list)
    assert [s["stage_no"] for s in result["stages"]] == [1]


# ---------------------------------------------------------------------------
# detect_diff_regions
# ---------------------------------------------------------------------------

def test_detect_diff_regions_finds_changed_block():
    """変更されたブロックを内包する矩形が1つ検出される。"""
    old = Image.new("RGB", (200, 200), color=(255, 255, 255))
    new = old.copy()
    # 中央付近に変更ブロック
    for y in range(90, 110):
        for x in range(80, 140):
            new.putpixel((x, y), (0, 0, 0))

    regions = td.detect_diff_regions(old, new)
    assert len(regions) == 1
    x, y, w, h = regions[0]
    # 変更ブロック(80..140, 90..110)を内包している
    assert x <= 80 and y <= 90
    assert x + w >= 140 and y + h >= 110


def test_detect_diff_regions_none_when_identical():
    """同一画像では矩形は検出されない。"""
    old = Image.new("RGB", (200, 200), color=(255, 255, 255))
    assert td.detect_diff_regions(old, old.copy()) == []


def test_detect_diff_regions_filters_tiny_noise():
    """ごく小さな点ノイズは（膨張で肥大化しても）実差分ピクセル数で除外される。"""
    old = Image.new("RGB", (200, 200), color=(255, 255, 255))
    new = old.copy()
    new.putpixel((10, 10), (0, 0, 0))  # 1px のみ
    assert td.detect_diff_regions(new, old, min_changed=30) == []


# ---------------------------------------------------------------------------
# draw_diff_overlay
# ---------------------------------------------------------------------------

def test_draw_diff_overlay_returns_same_size_and_marks_region():
    """重畳画像は元サイズを保ち、矩形領域にだけ色が乗る。"""
    base = Image.new("RGB", (200, 200), color=(255, 255, 255))
    regions = [(80, 90, 60, 20)]
    out = td.draw_diff_overlay(base, regions, color=(255, 0, 0))
    assert out.size == base.size

    arr = __import__("numpy").asarray(out)
    # 領域内に赤系の画素が存在する
    region = arr[90:110, 80:140]
    assert (region[:, :, 0] > region[:, :, 2]).any()
    # 領域外は元の白のまま
    assert tuple(arr[0, 0]) == (255, 255, 255)


def test_draw_diff_overlay_no_regions_is_unchanged():
    """矩形が無ければ元画像と同一内容を返す。"""
    base = Image.new("RGB", (50, 50), color=(123, 222, 111))
    out = td.draw_diff_overlay(base, [])
    assert out.size == base.size
    assert tuple(__import__("numpy").asarray(out)[0, 0]) == (123, 222, 111)
