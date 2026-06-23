"""stella_export.py の単体テスト (openTime/closeTime 補完・version インクリメント・
turnList 正規化・JSON 整形出力)。"""

import json

import pandas as pd

from backend_functions import stella_export as se


def _output_df(live_from="13:01", live_to="20:30", disabled=False):
    df_stage = pd.DataFrame(
        {"ステージ名": ["メイン"], "表示順": [0], "非活性化フラグ": [disabled]},
        index=[0],
    )
    df_idol = pd.DataFrame({"グループ名_採用": ["グループA"]}, index=[0])
    df_live = pd.DataFrame({
        "出番ID": [0],
        "グループID": [0],
        "ステージID": [0],
        "ライブ_from": [live_from],
        "ライブ_to": [live_to],
    })
    return {"stage": df_stage, "idolname": df_idol, "live": df_live}


# ---------------------------------------------------------------------------
# compute_open_close_default
# ---------------------------------------------------------------------------

def test_compute_open_close_minute_nonzero_keeps_hour():
    # 13:01 → open=13, 20:30 → +1h=21:30基準 → close=22
    assert se.compute_open_close_default(_output_df("13:01", "20:30")) == ("13", "22")


def test_compute_open_close_minute_zero_subtracts_hour():
    # 13:00 → open=12, 21:00 → +1h=22:00基準 → close=23
    assert se.compute_open_close_default(_output_df("13:00", "21:00")) == ("12", "23")


def test_compute_open_close_empty_live_returns_blank():
    df = _output_df()
    df["live"] = df["live"].iloc[0:0]
    assert se.compute_open_close_default(df) == ("", "")


def test_compute_open_close_excludes_disabled_stage():
    assert se.compute_open_close_default(_output_df(disabled=True)) == ("", "")


# ---------------------------------------------------------------------------
# build_stella_json: openTime/closeTime のデフォルト補完
# ---------------------------------------------------------------------------

def test_build_fills_open_close_when_missing():
    meta = {"liveId": 547, "jsonVersion": 1, "notification": ""}
    result = se.build_stella_json(_output_df("13:01", "20:30"), meta)
    assert result["openTime"] == "13"
    assert result["closeTime"] == "22"


def test_build_keeps_explicit_open_close():
    meta = {"liveId": 547, "jsonVersion": 1, "openTime": "10", "closeTime": "23"}
    result = se.build_stella_json(_output_df("13:01", "20:30"), meta)
    assert result["openTime"] == "10"
    assert result["closeTime"] == "23"


# ---------------------------------------------------------------------------
# increment_versions_on_push: notificationVersion
# ---------------------------------------------------------------------------

def test_increment_empty_notification_no_bump():
    # 既定の _last_pushed_notification=None / notification="" で +1 しない
    meta = {"jsonVersion": 1, "notificationVersion": "1",
            "notification": "", "_last_pushed_notification": None}
    new = se.increment_versions_on_push(meta)
    assert new["notificationVersion"] == "1"
    assert new["_last_pushed_notification"] == ""


def test_increment_new_notification_bumps():
    meta = {"jsonVersion": 1, "notificationVersion": "1",
            "notification": "お知らせ", "_last_pushed_notification": None}
    new = se.increment_versions_on_push(meta)
    assert new["notificationVersion"] == "2"
    assert new["_last_pushed_notification"] == "お知らせ"


def test_increment_unchanged_notification_no_bump():
    meta = {"jsonVersion": 1, "notificationVersion": "2",
            "notification": "お知らせ", "_last_pushed_notification": "お知らせ"}
    new = se.increment_versions_on_push(meta)
    assert new["notificationVersion"] == "2"


def test_increment_json_version_always_bumps():
    meta = {"jsonVersion": 4, "notification": "",
            "_last_pushed_notification": None}
    new = se.increment_versions_on_push(meta)
    assert new["jsonVersion"] == 5


# ---------------------------------------------------------------------------
# _build_turn_list: turnId 順・欠番補完
# ---------------------------------------------------------------------------

def _df_live(rows):
    """rows: [(出番ID, グループID, ステージID, from, 長さ分), ...] → live DataFrame。"""
    return pd.DataFrame({
        "出番ID": [r[0] for r in rows],
        "グループID": [r[1] for r in rows],
        "ステージID": [r[2] for r in rows],
        "ライブ_from": [r[3] for r in rows],
        "ライブ_長さ(分)": [r[4] for r in rows],
    })


def test_turn_list_sorted_by_turn_id_not_start_time():
    # startTime が turnId と逆順でも、turnId 昇順で並ぶこと
    df = _df_live([(0, 0, 0, "15:00", 20), (1, 0, 0, "10:00", 20), (2, 0, 0, "12:00", 20)])
    turns = se._build_turn_list(df)
    assert [t["turnId"] for t in turns] == [0, 1, 2]
    assert [t["startTime"] for t in turns] == ["15:00", "10:00", "12:00"]


def test_turn_list_fills_gaps_with_min_zero():
    # 出番ID 0, 2 (1 が欠番) → 0..2 連番に補完される
    df = _df_live([(0, 0, 0, "10:00", 20), (2, 0, 0, "12:00", 20)])
    turns = se._build_turn_list(df)
    assert [t["turnId"] for t in turns] == [0, 1, 2]
    filler = turns[1]
    assert filler == {"turnId": 1, "startTime": "", "min": 0, "artId": 0, "stageId": 0}
    assert "collabArtList" not in filler and "title" not in filler


def test_turn_list_no_gap_unchanged_order_normalized():
    df = _df_live([(2, 0, 0, "12:00", 20), (0, 0, 0, "10:00", 20), (1, 0, 0, "11:00", 20)])
    turns = se._build_turn_list(df)
    assert [t["turnId"] for t in turns] == [0, 1, 2]
    assert all(t["min"] != 0 for t in turns)  # 補完レコードなし


def test_turn_list_empty_returns_empty():
    assert se._build_turn_list(pd.DataFrame()) == []


# ---------------------------------------------------------------------------
# write_stella_json: 整形 (インデント) 出力
# ---------------------------------------------------------------------------

def test_write_stella_json_is_indented_and_bom(tmp_path):
    data = {"liveId": 1, "turnList": [{"turnId": 0, "min": 20}]}
    path = se.write_stella_json(data, str(tmp_path), live_id=1)
    raw = open(path, "rb").read()
    assert raw.startswith(b"\xef\xbb\xbf")          # UTF-8 BOM 維持
    text = raw.decode("utf-8-sig")
    assert "\n  " in text                            # インデントされている
    assert text.endswith("\n")                       # 末尾改行
    assert json.loads(text) == data                  # 妥当な JSON で内容一致
