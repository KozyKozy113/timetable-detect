"""特典会併記形式のタイムテーブル画像生成テスト。

`_build_tokutenkai_view_json` の擬似 JSON 変換と、
`create_timetable_image` の `show_timeline_labels` / `apply_max_width_clamp` 引数の
振る舞いを検証する。
"""

from __future__ import annotations

import pytest
from PIL import Image

from frontend_functions import timetablepicture


# ---------------------------------------------------------------------------
# _build_tokutenkai_view_json
# ---------------------------------------------------------------------------

def _live(from_, to, group, tk_list):
    return {
        "グループ名": group,
        "グループ名_採用": group,
        "ライブステージ": {"from": from_, "to": to},
        "特典会": tk_list,
    }


def test_build_tokutenkai_view_single():
    """単一特典会のグループが擬似ライブ 1 件に変換される。"""
    src = {
        "ステージ名": "S1",
        "タイムテーブル": [
            _live("10:00", "10:30", "A",
                  [{"ブース": "B1", "from": "10:35", "to": "11:00"}]),
        ],
    }
    out = timetablepicture._build_tokutenkai_view_json(src)
    assert out["ステージ名"] == "S1"
    assert len(out["タイムテーブル"]) == 1
    item = out["タイムテーブル"][0]
    assert item["グループ名"] == "B1"
    assert item["グループ名_採用"] == "B1"
    # 擬似ライブの from/to はライブ枠と一致（N=1）
    assert item["ライブステージ"]["from"] == "10:00"
    assert item["ライブステージ"]["to"] == "10:30"
    # 実時刻は _display_time_* に保持
    assert item["_display_time_from"] == "10:35"
    assert item["_display_time_to"] == "11:00"


def test_build_tokutenkai_view_multiple_split():
    """複数特典会のグループが N 等分され、末尾の to がライブの to と一致する。"""
    src = {
        "タイムテーブル": [
            _live("10:00", "10:30", "A", [
                {"ブース": "B1", "from": "10:35", "to": "10:50"},
                {"ブース": "B2", "from": "10:50", "to": "11:05"},
                {"ブース": "B3", "from": "11:05", "to": "11:20"},
            ]),
        ],
    }
    out = timetablepicture._build_tokutenkai_view_json(src)
    items = out["タイムテーブル"]
    assert len(items) == 3
    assert items[0]["ライブステージ"]["from"] == "10:00"
    assert items[-1]["ライブステージ"]["to"] == "10:30"  # 末尾は live to
    # 連続性: 各枠の to == 次の from
    for prev, nxt in zip(items, items[1:]):
        assert prev["ライブステージ"]["to"] == nxt["ライブステージ"]["from"]


def test_build_tokutenkai_view_skip_empty():
    """特典会[] が空のグループはスキップされる。"""
    src = {
        "タイムテーブル": [
            _live("10:00", "10:30", "A", []),
            _live("10:30", "11:00", "B",
                  [{"ブース": "BX", "from": "10:35", "to": "11:00"}]),
        ],
    }
    out = timetablepicture._build_tokutenkai_view_json(src)
    assert len(out["タイムテーブル"]) == 1
    assert out["タイムテーブル"][0]["グループ名"] == "BX"


def test_build_tokutenkai_view_degenerate():
    """極端に短いライブ × 多数特典会でも sub_to > sub_from が保たれる。"""
    src = {
        "タイムテーブル": [
            _live("10:00", "10:05", "A", [
                {"ブース": f"B{i}", "from": "10:00", "to": "10:01"}
                for i in range(6)
            ]),
        ],
    }
    out = timetablepicture._build_tokutenkai_view_json(src)
    items = out["タイムテーブル"]
    assert len(items) == 6
    for item in items:
        # 退化禁止: from < to
        assert item["ライブステージ"]["from"] < item["ライブステージ"]["to"]


def test_build_tokutenkai_view_skip_missing_fields():
    """ブース名 / from / to のいずれかが空の特典会要素は描画対象から除外される。"""
    src = {
        "タイムテーブル": [
            _live("10:00", "10:30", "A", [
                {"ブース": "", "from": "10:00", "to": "10:30"},          # ブース無し
                {"ブース": "B1", "from": "", "to": "10:30"},              # from 無し
                {"ブース": "B2", "from": "10:00", "to": ""},              # to 無し
                {"ブース": "B3", "from": "10:00", "to": "10:30"},         # 全て有り
            ]),
        ],
    }
    out = timetablepicture._build_tokutenkai_view_json(src)
    items = out["タイムテーブル"]
    assert len(items) == 1
    assert items[0]["グループ名"] == "B3"


def test_build_tokutenkai_view_skip_all_missing():
    """全特典会の必須項目が欠けているライブはまるごとスキップされる。"""
    src = {
        "タイムテーブル": [
            _live("10:00", "10:30", "A", [
                {"ブース": "", "from": "", "to": ""},
                {"ブース": "B1", "from": "10:00", "to": ""},
            ]),
            _live("10:30", "11:00", "B", [
                {"ブース": "B2", "from": "10:30", "to": "11:00"},
            ]),
        ],
    }
    out = timetablepicture._build_tokutenkai_view_json(src)
    items = out["タイムテーブル"]
    assert len(items) == 1
    assert items[0]["グループ名"] == "B2"


# ---------------------------------------------------------------------------
# create_timetable_image の新引数
# ---------------------------------------------------------------------------

def _basic_json():
    return {
        "ステージ名": "S",
        "タイムテーブル": [
            {
                "グループ名": "アーティスト1",
                "グループ名_採用": "アーティスト1",
                "ライブステージ": {"from": "10:00", "to": "10:30"},
            },
            {
                "グループ名": "アーティスト2",
                "グループ名_採用": "アーティスト2",
                "ライブステージ": {"from": "10:30", "to": "11:00"},
            },
        ],
    }


def test_show_timeline_labels_false_omits_labels():
    """show_timeline_labels=False のとき時刻ラベル列が省略され、画像幅が縮む。"""
    base = _basic_json()
    img_with = timetablepicture.create_timetable_image(
        base, source_box_width=200,
    )
    img_without = timetablepicture.create_timetable_image(
        base, source_box_width=200, show_timeline_labels=False,
    )
    assert img_with is not None and img_without is not None
    # 時刻ラベル列が消えた分、横幅は確実に小さくなる
    assert img_without.width < img_with.width


def test_apply_max_width_clamp_false_skips_resize():
    """apply_max_width_clamp=False のとき MAX_GEN_WIDTH 超過でも縮小されない。"""
    long_name = "アーティスト名" + "あ" * 200
    json_data = {
        "ステージ名": "S",
        "タイムテーブル": [
            {
                "グループ名": long_name,
                "グループ名_採用": long_name,
                "ライブステージ": {"from": "10:00", "to": "10:30"},
            },
        ],
    }
    img_clamp = timetablepicture.create_timetable_image(
        json_data,
        time_line_spacing=120,
        source_box_width=8000,  # ありえないほど横長 → 必要幅も MAX_GEN_WIDTH 超過
        apply_max_width_clamp=True,
    )
    img_noclamp = timetablepicture.create_timetable_image(
        json_data,
        time_line_spacing=120,
        source_box_width=8000,
        apply_max_width_clamp=False,
    )
    assert img_clamp.width <= timetablepicture.MAX_GEN_WIDTH
    # クランプ無しでは MAX_GEN_WIDTH を超え得る
    assert img_noclamp.width >= img_clamp.width


def test_display_time_overrides_label():
    """_display_time_from / _to が time_text 描画に使われる（生成成功確認）。"""
    json_data = {
        "ステージ名": "S",
        "タイムテーブル": [
            {
                "グループ名": "B1",
                "グループ名_採用": "B1",
                "ライブステージ": {"from": "10:00", "to": "10:30"},
                "_display_time_from": "10:35",
                "_display_time_to": "11:00",
            },
        ],
    }
    img = timetablepicture.create_timetable_image(json_data, source_box_width=200)
    assert img is not None


# ---------------------------------------------------------------------------
# _hstack_images
# ---------------------------------------------------------------------------

def test_hstack_images_same_height():
    left = Image.new("RGB", (100, 50), "red")
    right = Image.new("RGB", (60, 50), "blue")
    combined = timetablepicture._hstack_images(left, right)
    assert combined.size == (160, 50)
    assert combined.getpixel((0, 0)) == (255, 0, 0)
    assert combined.getpixel((100, 0)) == (0, 0, 255)


def test_hstack_images_different_height_resizes_right():
    left = Image.new("RGB", (100, 50), "red")
    right = Image.new("RGB", (60, 100), "blue")
    combined = timetablepicture._hstack_images(left, right)
    assert combined.height == 50
    # 右側はアスペクト比保持で縦圧縮（60 * 50/100 = 30）
    assert combined.width == 100 + 30
