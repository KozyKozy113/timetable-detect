"""notificationData4.json お知らせ生成・重複・追記ヘルパ (stella_export) の単体テスト。"""

import json
import os

from backend_functions import stella_export as se


# ---------------------------------------------------------------------------
# build_notification_date_range
# ---------------------------------------------------------------------------

def test_date_range_single():
    assert se.build_notification_date_range(["20260627"]) == "6/27"


def test_date_range_same_month():
    assert se.build_notification_date_range(["20260627", "20260628"]) == "6/27-28"


def test_date_range_cross_month():
    assert se.build_notification_date_range(["20260627", "20260701"]) == "6/27-7/1"


def test_date_range_unordered_and_invalid_ignored():
    assert se.build_notification_date_range(["20260628", "", "x", "20260627"]) == "6/27-28"


def test_date_range_empty():
    assert se.build_notification_date_range(["", None]) == ""


# ---------------------------------------------------------------------------
# build_notification_messages
# ---------------------------------------------------------------------------

def test_messages_with_area_idol():
    jp, en = se.build_notification_messages("HERO SONIC", "6/27-28", "横浜", 2)
    assert jp == "HERO SONIC (6/27-28 @横浜 アイドル) に対応しました。"
    assert en == "HERO SONIC (6/27-28 IDLE) is supported."


def test_messages_without_area_omits_at():
    jp, en = se.build_notification_messages("ABC", "6/27", "", 2)
    assert jp == "ABC (6/27 アイドル) に対応しました。"
    assert en == "ABC (6/27 IDLE) is supported."


def test_messages_band_genre():
    jp, en = se.build_notification_messages("ROCK FES", "7/1", "渋谷", 1)
    assert "@渋谷 バンド" in jp
    assert en.endswith("(7/1 BAND) is supported.")


def test_messages_unknown_genre_defaults_idol():
    jp, _ = se.build_notification_messages("X", "7/1", "", 99)
    assert "アイドル" in jp


# ---------------------------------------------------------------------------
# build_notification_entry
# ---------------------------------------------------------------------------

def test_build_entry_shape():
    entry = se.build_notification_entry([605, 606], "6/23", "msg", "msg_en")
    assert entry == {
        "icon": "mdi-alarm-plus",
        "liveId": [605, 606],
        "date": "6/23",
        "message": "msg",
        "message_en": "msg_en",
    }


# ---------------------------------------------------------------------------
# find_duplicate_notifications
# ---------------------------------------------------------------------------

def test_find_duplicates_intersection():
    nlist = [
        {"liveId": [600, 601], "message": "a"},
        {"liveId": [605], "message": "b"},
        {"date": "6/1", "message": "old entry without liveId"},
    ]
    assert se.find_duplicate_notifications(nlist, [605, 606]) == [1]


def test_find_duplicates_none():
    nlist = [{"liveId": [600]}, {"message": "no liveId"}]
    assert se.find_duplicate_notifications(nlist, [605]) == []


def test_find_duplicates_ignores_legacy_entries():
    # liveId を持たない旧エントリは決して重複扱いしない
    nlist = [{"message": "x"}, {"liveId": []}]
    assert se.find_duplicate_notifications(nlist, [605]) == []


# ---------------------------------------------------------------------------
# prepend_notification / read / write
# ---------------------------------------------------------------------------

def test_prepend_inserts_at_head():
    data = {"notificationList": [{"liveId": [1]}, {"liveId": [2]}]}
    entry = se.build_notification_entry([605], "6/23", "m", "e")
    se.prepend_notification(data, entry)
    assert data["notificationList"][0] == entry
    assert len(data["notificationList"]) == 3


def test_prepend_with_remove_indices():
    data = {"notificationList": [{"liveId": [605]}, {"liveId": [2]}, {"liveId": [605, 9]}]}
    entry = se.build_notification_entry([605], "6/23", "m", "e")
    dups = se.find_duplicate_notifications(data["notificationList"], [605])
    assert dups == [0, 2]
    se.prepend_notification(data, entry, remove_indices=dups)
    # 重複2件は削除され、新エントリが先頭、liveId=2 のみ残る
    assert data["notificationList"][0] == entry
    assert [e.get("liveId") for e in data["notificationList"]] == [[605], [2]]


def test_read_missing_returns_empty():
    assert se.read_notification_data("/no/such/path.json") == {"notificationList": []}


def test_write_then_read_roundtrip(tmp_path):
    path = os.path.join(str(tmp_path), "notificationData4.json")
    data = {"notificationList": [se.build_notification_entry([605], "6/23", "m", "e")]}
    se.write_notification_data(path, data)
    # utf-8-sig (BOM 付き) で書かれている
    with open(path, "rb") as f:
        assert f.read(3) == b"\xef\xbb\xbf"
    assert se.read_notification_data(path) == data
