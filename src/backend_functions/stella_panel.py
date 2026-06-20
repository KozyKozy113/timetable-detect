"""Stella `liveListPanel2.json` (表示エリア・表示順の制御) の操作。

`liveListPanel2.json` はライブを「どの年月の表示エリアに、どの順で、複数日イベントを
どうくくって表示するか」を制御するファイル。構造:

    [
      {
        "year": 2024,
        "monthEnableList": [bool x12],   # index 0=1月 .. 11=12月。その月に表示があれば true
        "monthList": [
          {"month": 5, "liveIdlist": [[547, 548], [550], ...]},  # 各要素が 1 イベント(bundle)
          ...
        ]
      },
      ...
    ]

- **bundle**: `liveIdlist` の各要素は 1 イベント分の liveId 配列。複数日イベントは
  全日の liveId をまとめた配列 (例 `[547,548]`)、単日イベントは 1 要素 (例 `[550]`)。
  これは本アプリの `bundleId` (= 同一イベントの全 event を束ねる) に対応する。
- **配置**: イベントの **最初の日付** の年月エリアに置く (月をまたいでも最初の日付の月)。
- **並び**: 月内では各イベントの最初の日付の昇順。同一日付は追加順 (挿入は安定)。

本モジュールは I/O を伴う read/write と、純粋な `upsert_bundle()` を提供する。
GitHub への push は採番フロー (`stella_reserve`) が liveList.json と同一 commit で行う。
"""

from __future__ import annotations

import json
import os


PANEL2_FILENAME = "liveListPanel2.json"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_panel(path: str) -> list:
    """`liveListPanel2.json` を読み込んで year 配列を返す。無ければ空リスト。"""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def write_panel(path: str, panel: list) -> None:
    """`liveListPanel2.json` を 1 行 minify (UTF-8 BOM 付き) で書き出す。

    既存運用ファイルが UTF-8 BOM 付き minify のため合わせる。
    """
    with open(path, "w", encoding="utf-8-sig") as f:
        json.dump(panel, f, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# 純粋ロジック
# ---------------------------------------------------------------------------

def _bundle_first_date(bundle: list[int], live_dates: dict[int, str]) -> str:
    """bundle (liveId 配列) の最初の日付 = 含む liveId の最小日付。不明は ""。"""
    ds = [live_dates.get(int(i), "") for i in bundle]
    ds = [d for d in ds if d]
    return min(ds) if ds else ""


def _remove_existing_bundle(panel: list, live_ids: set[int]) -> None:
    """live_ids のいずれかを含む既存 bundle を panel 全体から取り除く (再採番の再配置用)。

    取り除いた結果、空になった月 / 年は削除し、monthEnableList も更新する。
    """
    for year_obj in panel:
        month_list = year_obj.get("monthList", [])
        for month_obj in month_list:
            kept = [
                b for b in month_obj.get("liveIdlist", [])
                if not (set(int(x) for x in b) & live_ids)
            ]
            month_obj["liveIdlist"] = kept
        # 空になった月を除去 + monthEnableList 更新
        non_empty_months = []
        enable = year_obj.get("monthEnableList", [False] * 12)
        present = {m["month"] for m in month_list if m["liveIdlist"]}
        for month_obj in month_list:
            if month_obj["liveIdlist"]:
                non_empty_months.append(month_obj)
        year_obj["monthList"] = non_empty_months
        year_obj["monthEnableList"] = [
            (i + 1) in present for i in range(12)
        ]
    # 空になった年を除去
    panel[:] = [y for y in panel if y.get("monthList")]


def _get_or_create_year(panel: list, year: int) -> dict:
    """year オブジェクトを取得。無ければ year 昇順の正しい位置に新規作成して返す。"""
    for y in panel:
        if y.get("year") == year:
            return y
    new_year = {
        "year": year,
        "monthEnableList": [False] * 12,
        "monthList": [],
    }
    insert_idx = len(panel)
    for i, y in enumerate(panel):
        if y.get("year") > year:
            insert_idx = i
            break
    panel.insert(insert_idx, new_year)
    return new_year


def _get_or_create_month(year_obj: dict, month: int) -> dict:
    """month オブジェクトを取得。無ければ month 昇順の正しい位置に新規作成して返す。"""
    month_list = year_obj["monthList"]
    for m in month_list:
        if m.get("month") == month:
            return m
    new_month = {"month": month, "liveIdlist": []}
    insert_idx = len(month_list)
    for i, m in enumerate(month_list):
        if m.get("month") > month:
            insert_idx = i
            break
    month_list.insert(insert_idx, new_month)
    year_obj["monthEnableList"][month - 1] = True
    return new_month


def upsert_bundle(
    panel: list,
    live_ids: list[int],
    first_date: str,
    live_dates: dict[int, str],
) -> list:
    """イベントの bundle を panel に挿入 / 更新する (純粋関数, panel をその場更新)。

    Args:
        panel: `liveListPanel2.json` の year 配列 (破壊的に更新して返す)。
        live_ids: このイベント (= プロジェクト) の全 liveId。bundle として束ねる。
        first_date: イベントの最初の日付 "YYYYMMDD"。配置先の年月を決める。
        live_dates: `{liveId: "YYYYMMDD"}`。月内の既存 bundle を並べ替える際の日付参照
            (liveList.json 由来。新規 liveId 分も含めること)。

    Returns:
        更新後の panel (引数と同一オブジェクト)。

    挙動:
        - 既存 bundle (live_ids のいずれかを含む) があれば一旦除去してから再配置
          (再採番で日付が変わり配置が移動するケースに対応)。
        - first_date の年月エリアへ、月内は最初の日付の昇順で挿入。同一日付は追加順 (末尾寄り)。
        - first_date が空なら配置先を決められないため何もしない。
    """
    if not live_ids or not first_date or len(first_date) < 6:
        return panel

    bundle = sorted(int(i) for i in live_ids)
    id_set = set(bundle)

    # 再採番等で位置が変わる場合に備え、既存 bundle を先に除去
    _remove_existing_bundle(panel, id_set)

    year = int(first_date[:4])
    month = int(first_date[4:6])

    year_obj = _get_or_create_year(panel, year)
    month_obj = _get_or_create_month(year_obj, month)
    year_obj["monthEnableList"][month - 1] = True

    # 月内を最初の日付昇順で挿入 (既存より厳密に後の日付の手前に入れる = 同日は後ろ)
    id_list = month_obj["liveIdlist"]
    insert_idx = len(id_list)
    for i, existing in enumerate(id_list):
        if _bundle_first_date(existing, live_dates) > first_date:
            insert_idx = i
            break
    id_list.insert(insert_idx, bundle)
    return panel
