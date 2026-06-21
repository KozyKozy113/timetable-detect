"""stella_export.py の単体テスト (openTime/closeTime 補完・version インクリメント)。"""

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
