"""タイムテーブル配列が時刻順でない場合の正規化に関する回帰テスト。

背景:
- json_to_df は ライブ_from でソートするが、旧実装は index を振り直さなかったため
  ④画面 (st.data_editor) に元の配列番号がそのまま index 列として表示されていた。
- 画像生成 (create_timetable_image) は配列順が時刻昇順である前提で重複インデントを
  判定するため、配列が時刻順でないと早い時刻の要素が重複と誤判定され表示位置がズレた。
  → 保存時に sort_timetable_entries で配列順を正規化する。
"""

import json

import pandas as pd

from backend_functions import timetabledata


def _out_of_order_json():
    """末尾に最も早い時刻の要素が来る、時刻順でない並びの JSON。"""
    return {
        "ステージ名": "A", "ステージID": 1,
        "タイムテーブル": [
            {"グループ名": "B", "グループ名_採用": "B",
             "ライブステージ": {"from": "16:30", "to": "16:55"},
             "出番ID": 2, "グループID": 20},
            {"グループ名": "C", "グループ名_採用": "C",
             "ライブステージ": {"from": "16:55", "to": "17:20"},
             "出番ID": 3, "グループID": 30},
            {"グループ名": "A", "グループ名_採用": "A",
             "ライブステージ": {"from": "16:05", "to": "16:30"},
             "出番ID": None, "グループID": None},  # 最も早いが配列末尾、かつ未採番
        ],
    }


def test_json_to_df_resets_index_after_sort():
    """時刻順でない配列でも index は 0..N の連番に振り直される。"""
    df = timetabledata.json_to_df(
        _out_of_order_json(), tokutenkai=False, id_assigned=True,
    )
    assert list(df.index) == list(range(len(df)))
    # ライブ_from 昇順に並んでいる
    assert df["ライブ_from"].tolist() == ["16:05", "16:30", "16:55"]


def test_sort_timetable_entries_orders_by_from():
    timetable = [
        {"グループ名": "B", "ライブステージ": {"from": "16:30", "to": "16:55"}},
        {"グループ名": "A1", "ライブステージ": {"from": "16:05", "to": "16:20"}},
        {"グループ名": "X", "ライブステージ": {"from": ""}},  # 不正時刻は末尾
        {"グループ名": "A2", "ライブステージ": {"from": "16:05", "to": "16:30"}},
    ]
    out = timetabledata.sort_timetable_entries(timetable)
    names = [e["グループ名"] for e in out]
    # 16:05 同士は元の相対順を維持 (A1 → A2)、空 from は末尾
    assert names == ["A1", "A2", "B", "X"]
    # 元リストはミューテートしない
    assert timetable[0]["グループ名"] == "B"


def test_df_to_json_then_sort_roundtrip_is_chronological_and_serializable():
    """時刻順でない JSON を ④往復 (json_to_df → df_to_json → 正規化) すると
    配列が時刻昇順になり、未採番 ID は null で JSON 直列化できる。"""
    df = timetabledata.json_to_df(
        _out_of_order_json(), tokutenkai=False, id_assigned=True,
    )
    out = timetabledata.sort_timetable_entries(timetabledata.df_to_json(df))
    froms = [e["ライブステージ"]["from"] for e in out]
    assert froms == ["16:05", "16:30", "16:55"]
    # 未採番行は null として書き出される (int(None) で TypeError 落ちしない)
    early = next(e for e in out if e["ライブステージ"]["from"] == "16:05")
    assert early["出番ID"] is None
    assert early["グループID"] is None
    # 直列化可能であること (numpy/pd.NA 混入で落ちない)
    json.dumps({"タイムテーブル": out}, ensure_ascii=False)


def test_df_to_json_null_ids_do_not_raise():
    """未採番 ID (None) を含む DataFrame でも df_to_json が例外を投げない。"""
    df = pd.DataFrame([
        {"グループ名": "A", "グループ名_採用": "A",
         "ライブ_from": "16:05", "ライブ_to": "16:30", "ライブ_長さ(分)": 25,
         "出番ID": pd.NA, "グループID": pd.NA, "備考": ""},
    ]).astype({"出番ID": "Int64", "グループID": "Int64"})
    out = timetabledata.df_to_json(df)
    assert out[0]["出番ID"] is None
    assert out[0]["グループID"] is None
