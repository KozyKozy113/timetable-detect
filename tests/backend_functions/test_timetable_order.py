"""timetables[] 並び順ロジック (compute_insert_index / move_* / reset_*) のテスト"""

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


def _entry(image_no: int, dir_name: str, kind: str) -> dict:
    return {
        "image_no": image_no,
        "dir_name": dir_name,
        "display_name": dir_name,
        "kind": kind,
        "stage_num": 0,
        "stage_list": [],
    }


def _dir_names(pij: dict) -> list[str]:
    return [e["dir_name"] for e in repo.get_image_entry_list(pij, 0)]


# ---------------------------------------------------------------------------
# compute_insert_index
# ---------------------------------------------------------------------------

def test_compute_insert_index_empty_event():
    pij = _new_schema_project([])
    assert repo.compute_insert_index(pij, 0, "ライブ", "live") == 0
    assert repo.compute_insert_index(pij, 0, "縁日", "tokutenkai") == 0


def test_compute_insert_index_live_goes_to_top():
    pij = _new_schema_project([
        _entry(0, "特典会", "tokutenkai"),
        _entry(1, "縁日", "tokutenkai"),
    ])
    # dir_name=="ライブ" は最優先 → 先頭
    assert repo.compute_insert_index(pij, 0, "ライブ", "live") == 0


def test_compute_insert_index_other_live_after_dir_live():
    pij = _new_schema_project([
        _entry(0, "ライブ", "live"),
        _entry(1, "特典会", "tokutenkai"),
    ])
    # kind=="live" だが dir_name!="ライブ" → バケット1 = "ライブ" の直後
    assert repo.compute_insert_index(pij, 0, "ステージA", "live") == 1


def test_compute_insert_index_tokutenkai_after_live_bucket():
    pij = _new_schema_project([
        _entry(0, "ライブ", "live"),
        _entry(1, "ステージA", "live"),
        _entry(2, "縁日", "tokutenkai"),
    ])
    # dir_name=="特典会" はバケット2 → live系の末尾(=index2)の直後
    assert repo.compute_insert_index(pij, 0, "特典会", "tokutenkai") == 2


def test_compute_insert_index_other_tokutenkai_after_dir_tokutenkai():
    pij = _new_schema_project([
        _entry(0, "ライブ", "live"),
        _entry(1, "特典会", "tokutenkai"),
    ])
    # kind=="tokutenkai" だが dir_name!="特典会" → バケット3 = "特典会" の直後
    assert repo.compute_insert_index(pij, 0, "縁日", "tokutenkai") == 2
    # live_tokutenkai_heiki もバケット3
    assert repo.compute_insert_index(pij, 0, "ライブ特典会", "live_tokutenkai_heiki") == 2


def test_compute_insert_index_other_goes_to_end():
    pij = _new_schema_project([
        _entry(0, "ライブ", "live"),
        _entry(1, "特典会", "tokutenkai"),
    ])
    # 未知 kind はバケット4 → 末尾
    assert repo.compute_insert_index(pij, 0, "謎", "unknown_kind") == 2


def test_compute_insert_index_inserts_at_same_bucket_tail():
    """同バケット内では既存末尾の直後に入る。"""
    pij = _new_schema_project([
        _entry(0, "ライブ", "live"),
        _entry(1, "特典会", "tokutenkai"),
        _entry(2, "縁日", "tokutenkai"),  # バケット3
        _entry(3, "謎", "unknown_kind"),  # バケット4
    ])
    # 新規 tokutenkai (dir!=特典会) は バケット3 の末尾の直後 = 縁日 の直後 = index 3
    assert repo.compute_insert_index(pij, 0, "限定特典", "tokutenkai") == 3


# ---------------------------------------------------------------------------
# move_timetable_up / down
# ---------------------------------------------------------------------------

def test_move_timetable_up_swaps_with_previous():
    pij = _new_schema_project([
        _entry(0, "A", "tokutenkai"),
        _entry(1, "B", "tokutenkai"),
        _entry(2, "C", "tokutenkai"),
    ])
    repo.move_timetable_up(pij, 0, 2)
    assert _dir_names(pij) == ["A", "C", "B"]


def test_move_timetable_up_at_head_is_noop():
    pij = _new_schema_project([
        _entry(0, "A", "tokutenkai"),
        _entry(1, "B", "tokutenkai"),
    ])
    repo.move_timetable_up(pij, 0, 0)
    assert _dir_names(pij) == ["A", "B"]


def test_move_timetable_down_swaps_with_next():
    pij = _new_schema_project([
        _entry(0, "A", "tokutenkai"),
        _entry(1, "B", "tokutenkai"),
        _entry(2, "C", "tokutenkai"),
    ])
    repo.move_timetable_down(pij, 0, 0)
    assert _dir_names(pij) == ["B", "A", "C"]


def test_move_timetable_down_at_tail_is_noop():
    pij = _new_schema_project([
        _entry(0, "A", "tokutenkai"),
        _entry(1, "B", "tokutenkai"),
    ])
    repo.move_timetable_down(pij, 0, 1)
    assert _dir_names(pij) == ["A", "B"]


def test_move_timetable_raises_on_missing_image_no():
    pij = _new_schema_project([_entry(0, "A", "tokutenkai")])
    with pytest.raises(KeyError):
        repo.move_timetable_up(pij, 0, 99)
    with pytest.raises(KeyError):
        repo.move_timetable_down(pij, 0, 99)


# ---------------------------------------------------------------------------
# reset_timetable_order
# ---------------------------------------------------------------------------

def test_reset_timetable_order_sorts_by_bucket_stable():
    pij = _new_schema_project([
        _entry(0, "謎", "unknown_kind"),       # バケット4
        _entry(1, "縁日", "tokutenkai"),       # バケット3
        _entry(2, "特典会", "tokutenkai"),     # バケット2
        _entry(3, "ステージA", "live"),         # バケット1
        _entry(4, "ライブ", "live"),            # バケット0
        _entry(5, "ステージB", "live"),         # バケット1 (ステージA より後)
        _entry(6, "ライブ特典会", "live_tokutenkai_heiki"),  # バケット3 (縁日 より後)
    ])
    repo.reset_timetable_order(pij, 0)
    assert _dir_names(pij) == [
        "ライブ", "ステージA", "ステージB", "特典会", "縁日", "ライブ特典会", "謎",
    ]


def test_reset_timetable_order_empty():
    pij = _new_schema_project([])
    repo.reset_timetable_order(pij, 0)
    assert _dir_names(pij) == []


# ---------------------------------------------------------------------------
# register_timetable_image との結合: insert 経路で配列順が整う
# ---------------------------------------------------------------------------

def test_register_new_uses_compute_insert_index(tmp_path):
    """新規登録 (上書きでない) は compute_insert_index で位置決定される。"""
    pij = _new_schema_project([
        _entry(0, "縁日", "tokutenkai"),  # バケット3
    ])
    pj_path = str(tmp_path)
    import os
    os.makedirs(os.path.join(pj_path, "event_1"), exist_ok=True)

    # "ライブ" を後から登録 → 先頭に挿入される
    repo.register_timetable_image(
        pj_path=pj_path,
        event_name="event_1",
        event_no=0,
        dir_name="ライブ",
        kind="live",
        img_format="通常",
        file_data=b"dummy",
        project_info_json=pij,
    )
    assert _dir_names(pij) == ["ライブ", "縁日"]

    # "特典会" を登録 → ライブ系の直後 (= index 1)
    repo.register_timetable_image(
        pj_path=pj_path,
        event_name="event_1",
        event_no=0,
        dir_name="特典会",
        kind="tokutenkai",
        img_format="通常",
        file_data=b"dummy",
        project_info_json=pij,
    )
    assert _dir_names(pij) == ["ライブ", "特典会", "縁日"]


def test_register_overwrite_keeps_position(tmp_path):
    """既存 dir_name と一致する登録は位置を維持する (現状挙動)。"""
    pij = _new_schema_project([
        _entry(0, "ライブ", "live"),
        _entry(1, "特典会", "tokutenkai"),
        _entry(2, "縁日", "tokutenkai"),
    ])
    pj_path = str(tmp_path)
    import os
    os.makedirs(os.path.join(pj_path, "event_1"), exist_ok=True)

    repo.register_timetable_image(
        pj_path=pj_path,
        event_name="event_1",
        event_no=0,
        dir_name="特典会",
        kind="tokutenkai",
        img_format="ライムライト式",
        file_data=b"dummy",
        project_info_json=pij,
    )
    # image_no=1 はそのまま、位置 (index 1) も維持
    entries = repo.get_image_entry_list(pij, 0)
    assert _dir_names(pij) == ["ライブ", "特典会", "縁日"]
    assert entries[1]["image_no"] == 1
    assert entries[1]["format"] == "ライムライト式"
