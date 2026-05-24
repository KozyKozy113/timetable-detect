"""output_editor.build_stage_kind_map のテスト

⑥編集モードの D&D ラベルに種別名 (event_type/dir_name) を併記するために
stage_id → event_type のマップを構築する関数。
"""

import json
import os

from backend_functions import output_editor as _editor


def _write_stage_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def test_build_stage_kind_map_normal_kind(tmp_path):
    """通常 kind (live): トップレベル ステージID → event_type のマップ"""
    pj_path = tmp_path / "pj"
    event_dir = pj_path / "event_1" / "ライブ"
    os.makedirs(event_dir)
    _write_stage_json(event_dir / "stage_0.json", {
        "ステージ名": "A", "ステージID": 0, "タイムテーブル": [],
    })
    _write_stage_json(event_dir / "stage_1.json", {
        "ステージ名": "B", "ステージID": 1, "タイムテーブル": [],
    })

    project_info = {
        "project_name": "pj",
        "event_num": 1,
        "event_detail": [
            {
                "event_no": 0,
                "event_name": "event_1",
                "timetables": [
                    {
                        "image_no": 0,
                        "dir_name": "ライブ",
                        "display_name": "ライブ",
                        "kind": "live",
                        "stage_num": 2,
                        "stage_list": [
                            {"stage_no": 0, "stage_name": "A",
                             "kind": "live", "stage_id": 0},
                            {"stage_no": 1, "stage_name": "B",
                             "kind": "live", "stage_id": 1},
                        ],
                    },
                ],
            },
        ],
    }

    result = _editor.build_stage_kind_map(
        str(pj_path), "event_1", 0, project_info,
    )
    assert result == {0: "ライブ", 1: "ライブ"}


def test_build_stage_kind_map_heiki_kind(tmp_path):
    """heiki kind: トップレベル ステージID も 特典会[].ステージID も同じ event_type にマップ"""
    pj_path = tmp_path / "pj"
    event_dir = pj_path / "event_1" / "ライブ特典会"
    os.makedirs(event_dir)
    _write_stage_json(event_dir / "stage_0.json", {
        "ステージ名": "メイン", "ステージID": 0,
        "タイムテーブル": [
            {
                "出番ID": 0,
                "ライブステージ": {"from": "10:00", "to": "10:30"},
                "特典会": [
                    {"出番ID": 10, "ステージID": 100, "ブース": "ブースA"},
                    {"出番ID": 11, "ステージID": 101, "ブース": "ブースB"},
                ],
            },
        ],
    })

    project_info = {
        "project_name": "pj",
        "event_num": 1,
        "event_detail": [
            {
                "event_no": 0,
                "event_name": "event_1",
                "timetables": [
                    {
                        "image_no": 0,
                        "dir_name": "ライブ特典会",
                        "display_name": "ライブ特典会",
                        "kind": "live_tokutenkai_heiki",
                        "stage_num": 1,
                        "stage_list": [
                            {"stage_no": 0, "stage_name": "メイン",
                             "kind": "live_tokutenkai_heiki", "stage_id": 0},
                        ],
                    },
                ],
            },
        ],
    }

    result = _editor.build_stage_kind_map(
        str(pj_path), "event_1", 0, project_info,
    )
    assert result == {0: "ライブ特典会", 100: "ライブ特典会", 101: "ライブ特典会"}


def test_build_stage_kind_map_unknown_stage_id_not_included(tmp_path):
    """マップに含まれない stage_id (例: stage_*.json に出現しない ID) は dict に入らない"""
    pj_path = tmp_path / "pj"
    event_dir = pj_path / "event_1" / "ライブ"
    os.makedirs(event_dir)
    _write_stage_json(event_dir / "stage_0.json", {
        "ステージ名": "A", "ステージID": 0, "タイムテーブル": [],
    })

    project_info = {
        "project_name": "pj",
        "event_num": 1,
        "event_detail": [
            {
                "event_no": 0,
                "event_name": "event_1",
                "timetables": [
                    {
                        "image_no": 0,
                        "dir_name": "ライブ",
                        "display_name": "ライブ",
                        "kind": "live",
                        "stage_num": 1,
                        "stage_list": [
                            {"stage_no": 0, "stage_name": "A",
                             "kind": "live", "stage_id": 0},
                        ],
                    },
                ],
            },
        ],
    }

    result = _editor.build_stage_kind_map(
        str(pj_path), "event_1", 0, project_info,
    )
    # マスタには ID:99 があるかもしれないが stage_*.json には存在しない → マップに含まれない
    assert 99 not in result
    assert result == {0: "ライブ"}
