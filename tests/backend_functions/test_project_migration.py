"""project_migration.py: 旧 → 新スキーマ変換のテスト"""

import copy

from backend_functions import project_migration as _migration


# ---------------------------------------------------------------------------
# 変換: 標準キー (ライブ / 特典会 / ライブ特典会)
# ---------------------------------------------------------------------------

def _legacy_project(timetables: dict) -> dict:
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


def test_migrate_live_only():
    raw = _legacy_project({
        "ライブ": {
            "format": "通常",
            "stage_num": 2,
            "stage_list": [
                {"stage_no": 0, "stage_name": "A"},
                {"stage_no": 1, "stage_name": "B"},
            ],
        }
    })

    migrated = _migration.migrate_project_info(raw)

    tts = migrated["event_detail"][0]["timetables"]
    assert isinstance(tts, list)
    assert len(tts) == 1
    entry = tts[0]
    assert entry["image_no"] == 0
    assert entry["dir_name"] == "ライブ"
    assert entry["display_name"] == "ライブ"
    assert entry["kind"] == "live"
    assert entry["format"] == "通常"
    assert entry["stage_num"] == 2
    for stage in entry["stage_list"]:
        assert stage["kind"] == "live"


def test_migrate_live_and_tokutenkai():
    raw = _legacy_project({
        "ライブ": {
            "format": "通常",
            "stage_num": 1,
            "stage_list": [{"stage_no": 0, "stage_name": "A"}],
        },
        "特典会": {
            "format": "ライムライト式",
            "stage_num": 1,
            "stage_list": [{"stage_no": 0, "stage_name": "Booth"}],
        },
    })

    migrated = _migration.migrate_project_info(raw)
    tts = migrated["event_detail"][0]["timetables"]

    assert len(tts) == 2
    by_kind = {e["kind"]: e for e in tts}
    assert by_kind["live"]["dir_name"] == "ライブ"
    assert by_kind["live"]["format"] == "通常"
    assert by_kind["tokutenkai"]["dir_name"] == "特典会"
    assert by_kind["tokutenkai"]["format"] == "ライムライト式"
    # image_no は採番順 (0, 1)
    assert {e["image_no"] for e in tts} == {0, 1}


def test_migrate_heiki_drops_format_field():
    raw = _legacy_project({
        "ライブ特典会": {
            "format": "特典会併記",
            "stage_num": 1,
            "stage_list": [{"stage_no": 0, "stage_name": "Main"}],
        }
    })

    migrated = _migration.migrate_project_info(raw)
    entry = migrated["event_detail"][0]["timetables"][0]

    assert entry["kind"] == "live_tokutenkai_heiki"
    assert "format" not in entry
    assert entry["stage_list"][0]["kind"] == "live_tokutenkai_heiki"


# ---------------------------------------------------------------------------
# 変換: カスタム名
# ---------------------------------------------------------------------------

def test_migrate_custom_name_normal_format_becomes_tokutenkai():
    """カスタム名 + format=通常 → kind=tokutenkai (過去データ推論ルール)"""
    raw = _legacy_project({
        "縁日": {
            "format": "通常",
            "stage_num": 1,
            "stage_list": [{"stage_no": 0, "stage_name": "S"}],
        }
    })

    migrated = _migration.migrate_project_info(raw)
    entry = migrated["event_detail"][0]["timetables"][0]

    assert entry["dir_name"] == "縁日"
    assert entry["display_name"] == "縁日"
    assert entry["kind"] == "tokutenkai"
    assert entry["format"] == "通常"
    assert entry["stage_list"][0]["kind"] == "tokutenkai"


def test_migrate_custom_name_heiki_format_becomes_heiki_kind():
    raw = _legacy_project({
        "特殊併記": {
            "format": "特典会併記",
            "stage_num": 1,
            "stage_list": [{"stage_no": 0, "stage_name": "S"}],
        }
    })

    migrated = _migration.migrate_project_info(raw)
    entry = migrated["event_detail"][0]["timetables"][0]

    assert entry["kind"] == "live_tokutenkai_heiki"
    assert "format" not in entry


# ---------------------------------------------------------------------------
# 変換: 既存フィールドの保存
# ---------------------------------------------------------------------------

def test_migrate_preserves_extra_fields():
    """raw_crop_box, time_pixel など追加フィールドはそのまま継承される"""
    raw = _legacy_project({
        "ライブ": {
            "format": "通常",
            "stage_num": 1,
            "stage_list": [
                {"stage_no": 0, "stage_name": "A", "bbox": [1, 2, 3, 4]},
            ],
            "raw_crop_box": {"left": 1, "top": 2, "width": 3, "height": 4},
            "time_pixel": {"some": "data"},
        }
    })

    migrated = _migration.migrate_project_info(raw)
    entry = migrated["event_detail"][0]["timetables"][0]

    assert entry["raw_crop_box"] == {"left": 1, "top": 2, "width": 3, "height": 4}
    assert entry["time_pixel"] == {"some": "data"}
    assert entry["stage_list"][0]["bbox"] == [1, 2, 3, 4]


def test_migrate_preserves_existing_stage_kind():
    """stage_list[i] に kind が既に存在する場合は上書きしない"""
    raw = _legacy_project({
        "ライブ": {
            "format": "通常",
            "stage_num": 1,
            "stage_list": [
                {"stage_no": 0, "stage_name": "A", "kind": "tokutenkai"},
            ],
        }
    })

    migrated = _migration.migrate_project_info(raw)
    entry = migrated["event_detail"][0]["timetables"][0]
    # 親 kind は live でも、stage 側に既存値があれば尊重
    assert entry["kind"] == "live"
    assert entry["stage_list"][0]["kind"] == "tokutenkai"


# ---------------------------------------------------------------------------
# 冪等性
# ---------------------------------------------------------------------------

def test_migrate_is_idempotent():
    raw = _legacy_project({
        "ライブ": {
            "format": "通常",
            "stage_num": 1,
            "stage_list": [{"stage_no": 0, "stage_name": "A"}],
        }
    })

    once = _migration.migrate_project_info(copy.deepcopy(raw))
    twice = _migration.migrate_project_info(copy.deepcopy(once))

    assert twice == once


def test_migrate_already_new_schema_passes_through():
    """既に新スキーマの project_info に対しては何もしない"""
    new_schema = {
        "project_name": "p",
        "event_num": 1,
        "event_detail": [
            {
                "event_no": 0,
                "event_name": "event_1",
                "timetables": [
                    {
                        "image_no": 5,
                        "dir_name": "カスタム",
                        "display_name": "カスタム",
                        "kind": "live",
                        "format": "通常",
                        "stage_num": 1,
                        "stage_list": [
                            {"stage_no": 0, "stage_name": "A", "kind": "live"}
                        ],
                    }
                ],
            }
        ],
    }
    before = copy.deepcopy(new_schema)
    migrated = _migration.migrate_project_info(new_schema)
    assert migrated == before


# ---------------------------------------------------------------------------
# 多イベント
# ---------------------------------------------------------------------------

def test_migrate_multiple_events():
    raw = {
        "project_name": "p",
        "event_num": 2,
        "event_detail": [
            {
                "event_no": 0,
                "event_name": "event_1",
                "timetables": {
                    "ライブ": {
                        "format": "通常",
                        "stage_num": 1,
                        "stage_list": [{"stage_no": 0, "stage_name": "A"}],
                    }
                },
            },
            {
                "event_no": 1,
                "event_name": "event_2",
                "timetables": {
                    "特典会": {
                        "format": "通常",
                        "stage_num": 1,
                        "stage_list": [{"stage_no": 0, "stage_name": "B"}],
                    }
                },
            },
        ],
    }

    migrated = _migration.migrate_project_info(raw)

    assert isinstance(migrated["event_detail"][0]["timetables"], list)
    assert isinstance(migrated["event_detail"][1]["timetables"], list)
    assert migrated["event_detail"][0]["timetables"][0]["kind"] == "live"
    assert migrated["event_detail"][1]["timetables"][0]["kind"] == "tokutenkai"


def test_migrate_empty_timetables():
    raw = _legacy_project({})
    migrated = _migration.migrate_project_info(raw)
    assert migrated["event_detail"][0]["timetables"] == []
