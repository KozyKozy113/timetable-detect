"""stella_panel.py (liveListPanel2.json 操作) のテスト。

bundle の挿入位置 (最初の日付昇順 / 同日は追加順)、年月エリアの作成、
複数日イベントのくくり、再採番時の再配置を検証する。
"""

import json

from backend_functions import stella_panel as sp


def _dates(*pairs):
    """{liveId: date} を作る。"""
    return {i: d for i, d in pairs}


# ---------------------------------------------------------------------------
# 新規挿入: 年月エリアの作成
# ---------------------------------------------------------------------------

def test_upsert_creates_year_and_month():
    panel = []
    sp.upsert_bundle(panel, [547, 548], "20260504", _dates((547, "20260504"), (548, "20260505")))
    assert len(panel) == 1
    y = panel[0]
    assert y["year"] == 2026
    assert y["monthEnableList"][4] is True  # 5月 (index 4)
    assert y["monthEnableList"][0] is False
    assert y["monthList"] == [{"month": 5, "liveIdlist": [[547, 548]]}]


def test_upsert_multiday_spanning_month_goes_to_first_month():
    # 4/30 開始で 5/1 に跨ぐ → 最初の日付 (4月) に全 liveId
    panel = []
    sp.upsert_bundle(panel, [600, 601], "20260430", _dates((600, "20260430"), (601, "20260501")))
    y = panel[0]
    assert y["monthList"] == [{"month": 4, "liveIdlist": [[600, 601]]}]
    assert y["monthEnableList"][3] is True  # 4月


# ---------------------------------------------------------------------------
# 月内の並び: 最初の日付昇順 / 同日は追加順
# ---------------------------------------------------------------------------

def test_insert_sorted_by_first_date():
    panel = []
    ld = _dates((10, "20260510"), (20, "20260503"), (30, "20260520"))
    sp.upsert_bundle(panel, [10], "20260510", ld)
    sp.upsert_bundle(panel, [20], "20260503", ld)  # 早い → 先頭へ
    sp.upsert_bundle(panel, [30], "20260520", ld)  # 遅い → 末尾へ
    order = panel[0]["monthList"][0]["liveIdlist"]
    assert order == [[20], [10], [30]]


def test_same_date_keeps_addition_order():
    panel = []
    ld = _dates((1, "20260505"), (2, "20260505"), (3, "20260505"))
    sp.upsert_bundle(panel, [1], "20260505", ld)
    sp.upsert_bundle(panel, [2], "20260505", ld)
    sp.upsert_bundle(panel, [3], "20260505", ld)
    order = panel[0]["monthList"][0]["liveIdlist"]
    assert order == [[1], [2], [3]]  # 同日は追加順 (末尾寄り)


def test_insert_between_existing_by_date():
    panel = []
    ld = _dates((1, "20260501"), (3, "20260520"), (2, "20260510"))
    sp.upsert_bundle(panel, [1], "20260501", ld)
    sp.upsert_bundle(panel, [3], "20260520", ld)
    sp.upsert_bundle(panel, [2], "20260510", ld)  # 1 と 3 の間
    order = panel[0]["monthList"][0]["liveIdlist"]
    assert order == [[1], [2], [3]]


# ---------------------------------------------------------------------------
# 年・月の昇順維持
# ---------------------------------------------------------------------------

def test_years_and_months_kept_sorted():
    panel = []
    ld = _dates((1, "20260705"), (2, "20250105"), (3, "20260305"))
    sp.upsert_bundle(panel, [1], "20260705", ld)
    sp.upsert_bundle(panel, [2], "20250105", ld)  # 前年 → 先頭
    sp.upsert_bundle(panel, [3], "20260305", ld)  # 2026/3 → 2026/7 の前
    assert [y["year"] for y in panel] == [2025, 2026]
    months_2026 = [m["month"] for m in panel[1]["monthList"]]
    assert months_2026 == [3, 7]


# ---------------------------------------------------------------------------
# 再採番: 既存 bundle の再配置 (event 追加で liveId が増える)
# ---------------------------------------------------------------------------

def test_reupsert_updates_existing_bundle_in_place():
    panel = []
    ld = _dates((547, "20260504"))
    sp.upsert_bundle(panel, [547], "20260504", ld)
    # event 追加: 548 が増えた (同じプロジェクト)
    ld2 = _dates((547, "20260504"), (548, "20260505"))
    sp.upsert_bundle(panel, [547, 548], "20260504", ld2)
    lst = panel[0]["monthList"][0]["liveIdlist"]
    assert lst == [[547, 548]]  # 旧 [547] は置き換わり重複しない


def test_reupsert_moves_bundle_when_first_date_earlier():
    # 既存は 5月。新たに 4月の日 (より早い) を追加 → 4月エリアへ移動
    panel = []
    ld = _dates((600, "20260510"))
    sp.upsert_bundle(panel, [600], "20260510", ld)
    ld2 = _dates((600, "20260510"), (599, "20260428"))
    sp.upsert_bundle(panel, [599, 600], "20260428", ld2)
    # 5月は空になり消える、4月に移動
    months = {m["month"]: m["liveIdlist"] for m in panel[0]["monthList"]}
    assert 5 not in months
    assert months[4] == [[599, 600]]
    assert panel[0]["monthEnableList"][4] is False  # 5月 off
    assert panel[0]["monthEnableList"][3] is True   # 4月 on


def test_empty_first_date_is_noop():
    panel = []
    sp.upsert_bundle(panel, [1], "", {})
    assert panel == []


# ---------------------------------------------------------------------------
# I/O round-trip (BOM 付き minify)
# ---------------------------------------------------------------------------

def test_write_read_roundtrip_bom(tmp_path):
    panel = [{"year": 2026, "monthEnableList": [False] * 12,
              "monthList": [{"month": 5, "liveIdlist": [[1, 2]]}]}]
    path = str(tmp_path / "liveListPanel2.json")
    sp.write_panel(path, panel)
    raw = open(path, "rb").read()
    assert raw[:3] == b"\xef\xbb\xbf"  # UTF-8 BOM
    assert b", " not in raw  # minify (スペースなし)
    assert sp.read_panel(path) == panel


def test_read_missing_returns_empty(tmp_path):
    assert sp.read_panel(str(tmp_path / "nope.json")) == []
