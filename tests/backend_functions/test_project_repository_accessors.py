"""project_repository.py: 新スキーマアクセサのテスト"""

import pytest

from backend_functions import project_repository as repo


def _new_schema_project(timetables: list[dict]) -> dict:
    return {
        "project_name": "p",
        "event_num": 1,
        "event_detail": [
            {
                "event_no": 0,
                "event_name": "event_1",
                "timetables": timetables,
            }
        ],
    }


def _entry(image_no: int, dir_name: str, kind: str, **extra) -> dict:
    base = {
        "image_no": image_no,
        "dir_name": dir_name,
        "display_name": dir_name,
        "kind": kind,
        "stage_num": 0,
        "stage_list": [],
    }
    base.update(extra)
    return base


def test_get_image_entry_list_returns_list():
    pij = _new_schema_project([_entry(0, "ライブ", "live")])
    assert repo.get_image_entry_list(pij, 0) == [
        _entry(0, "ライブ", "live"),
    ]


def test_get_image_entry_by_no_finds_entry():
    pij = _new_schema_project([
        _entry(0, "ライブ", "live"),
        _entry(2, "特典会", "tokutenkai"),  # 欠番 1 あり
    ])
    entry = repo.get_image_entry_by_no(pij, 0, 2)
    assert entry["dir_name"] == "特典会"


def test_get_image_entry_by_no_raises_on_missing():
    pij = _new_schema_project([_entry(0, "ライブ", "live")])
    with pytest.raises(KeyError):
        repo.get_image_entry_by_no(pij, 0, 99)


def test_get_image_entry_by_dir_name_returns_none_when_missing():
    pij = _new_schema_project([_entry(0, "ライブ", "live")])
    assert repo.get_image_entry_by_dir_name(pij, 0, "存在しない") is None


def test_get_image_no_by_dir_name():
    pij = _new_schema_project([
        _entry(0, "ライブ", "live"),
        _entry(1, "特典会", "tokutenkai"),
    ])
    assert repo.get_image_no_by_dir_name(pij, 0, "特典会") == 1
    assert repo.get_image_no_by_dir_name(pij, 0, "ない") is None


def test_next_image_no_empty():
    pij = _new_schema_project([])
    assert repo.next_image_no(pij, 0) == 0


def test_next_image_no_skips_gaps():
    """削除で欠番が出ても max+1 で発番する"""
    pij = _new_schema_project([
        _entry(0, "ライブ", "live"),
        _entry(3, "特典会", "tokutenkai"),  # 1, 2 が欠番
    ])
    assert repo.next_image_no(pij, 0) == 4


def test_find_dir_name_conflict():
    pij = _new_schema_project([_entry(5, "ライブ", "live")])
    assert repo.find_dir_name_conflict(pij, 0, "ライブ") == 5
    assert repo.find_dir_name_conflict(pij, 0, "特典会") is None


def test_get_sorted_image_no_list_orders_by_kind_then_no():
    """kind 順: live → tokutenkai → live_tokutenkai_heiki → その他、
    同 kind 内では image_no 昇順
    """
    pij = _new_schema_project([
        _entry(10, "特典会", "tokutenkai"),
        _entry(5, "ライブ特典会", "live_tokutenkai_heiki"),
        _entry(2, "ライブA", "live"),
        _entry(8, "ライブB", "live"),
        _entry(99, "未知", "unknown_kind"),
    ])
    assert repo.get_sorted_image_no_list(pij, 0) == [2, 8, 10, 5, 99]


def test_get_sorted_image_no_list_empty():
    pij = _new_schema_project([])
    assert repo.get_sorted_image_no_list(pij, 0) == []
