"""ステージID トップレベル化マイグレーションのテスト"""

import copy
import json
import os

from backend_functions import project_migration as _migration


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _build_project(tmp_path, *, dir_name="ライブ", kind="live",
                   include_stage_id_in_info=False):
    """1イベント / 1画像 / 1ステージのプロジェクトを作る"""
    stage = {"stage_no": 0, "stage_name": "メイン", "kind": kind}
    if include_stage_id_in_info:
        stage["stage_id"] = 99  # 既存値
    project_info = {
        "project_name": "p",
        "event_num": 1,
        "event_detail": [
            {
                "event_no": 0,
                "event_name": "event_1",
                "timetables": [
                    {
                        "image_no": 0,
                        "dir_name": dir_name,
                        "display_name": dir_name,
                        "kind": kind,
                        "stage_num": 1,
                        "stage_list": [stage],
                    }
                ],
            }
        ],
    }
    return project_info


# ---------------------------------------------------------------------------
# 旧形式 → 新形式の昇格
# ---------------------------------------------------------------------------

def test_migrate_promotes_per_turn_stage_id_to_toplevel(tmp_path):
    project_info = _build_project(tmp_path)
    stage_json_path = os.path.join(tmp_path, "event_1", "ライブ", "stage_0.json")
    _write_json(stage_json_path, {
        "ステージ名": "メイン",
        "タイムテーブル": [
            {"グループ名": "A", "出番ID": 0, "グループID": 0, "ステージID": 5},
            {"グループ名": "B", "出番ID": 1, "グループID": 1, "ステージID": 5},
        ],
    })

    _migration.migrate_stage_id_to_toplevel(str(tmp_path), project_info)

    data = _read_json(stage_json_path)
    # トップレベルに ステージID が昇格
    assert data["ステージID"] == 5
    # 出番粒度の ステージID は削除
    for turn in data["タイムテーブル"]:
        assert "ステージID" not in turn

    # project_info の stage_list[0].stage_id にもコピー
    stage = project_info["event_detail"][0]["timetables"][0]["stage_list"][0]
    assert stage["stage_id"] == 5


def test_migrate_is_idempotent(tmp_path):
    """新形式に対しては変更しない (冪等)"""
    project_info = _build_project(tmp_path)
    stage_json_path = os.path.join(tmp_path, "event_1", "ライブ", "stage_0.json")
    _write_json(stage_json_path, {
        "ステージ名": "メイン",
        "ステージID": 3,
        "タイムテーブル": [
            {"グループ名": "A", "出番ID": 0, "グループID": 0},
        ],
    })

    info_before = copy.deepcopy(project_info)
    _migration.migrate_stage_id_to_toplevel(str(tmp_path), project_info)
    data_after = _read_json(stage_json_path)

    assert data_after["ステージID"] == 3
    # 出番粒度には ステージID 無し (冪等)
    assert "ステージID" not in data_after["タイムテーブル"][0]
    # project_info も stage_id が無ければ補完される
    info_before["event_detail"][0]["timetables"][0]["stage_list"][0]["stage_id"] = 3
    assert project_info == info_before


def test_migrate_preserves_tokutenkai_child_stage_id(tmp_path):
    """特典会併記形式の 特典会[].ステージID (子=ブース別ID) は維持される"""
    project_info = _build_project(tmp_path, dir_name="ライブ特典会",
                                  kind="live_tokutenkai_heiki")
    stage_json_path = os.path.join(
        tmp_path, "event_1", "ライブ特典会", "stage_0.json",
    )
    _write_json(stage_json_path, {
        "ステージ名": "メイン",
        "タイムテーブル": [
            {
                "グループ名": "A",
                "出番ID": 0,
                "グループID": 0,
                "ステージID": 1,   # 親 (出番粒度) → 削除されてトップレベル化
                "ライブステージ": {"from": "10:00", "to": "10:30"},
                "特典会": [
                    {
                        "from": "10:40", "to": "11:00",
                        "ブース": "ブースA",
                        "ステージID": 11,   # 子=ブース別ID → 維持される
                    }
                ],
            }
        ],
    })

    _migration.migrate_stage_id_to_toplevel(str(tmp_path), project_info)
    data = _read_json(stage_json_path)

    assert data["ステージID"] == 1
    assert "ステージID" not in data["タイムテーブル"][0]
    # 子 特典会[].ステージID は維持
    assert data["タイムテーブル"][0]["特典会"][0]["ステージID"] == 11


def test_migrate_no_stage_id_yet_leaves_info_stage_id_absent(tmp_path):
    """IDマスタ未確定の stage_*.json (ステージID 一切無し) は、
    project_info の stage_id 補完もスキップされる。"""
    project_info = _build_project(tmp_path)
    stage_json_path = os.path.join(tmp_path, "event_1", "ライブ", "stage_0.json")
    _write_json(stage_json_path, {
        "ステージ名": "メイン",
        "タイムテーブル": [
            {"グループ名": "A", "ライブステージ": {"from": "10:00", "to": "10:30"}},
        ],
    })

    _migration.migrate_stage_id_to_toplevel(str(tmp_path), project_info)
    data = _read_json(stage_json_path)

    assert "ステージID" not in data
    # project_info にも stage_id は付かない
    stage = project_info["event_detail"][0]["timetables"][0]["stage_list"][0]
    assert "stage_id" not in stage


def test_migrate_does_not_overwrite_existing_info_stage_id(tmp_path):
    """project_info に既に stage_id がある場合は上書きしない"""
    project_info = _build_project(tmp_path, include_stage_id_in_info=True)
    stage_json_path = os.path.join(tmp_path, "event_1", "ライブ", "stage_0.json")
    _write_json(stage_json_path, {
        "ステージ名": "メイン",
        "タイムテーブル": [
            {"グループ名": "A", "出番ID": 0, "ステージID": 5},
        ],
    })

    _migration.migrate_stage_id_to_toplevel(str(tmp_path), project_info)
    stage = project_info["event_detail"][0]["timetables"][0]["stage_list"][0]
    # 既存値 99 を尊重し、JSON の 5 では上書きしない
    assert stage["stage_id"] == 99


def test_migrate_skips_missing_json(tmp_path):
    """stage_*.json が物理的に存在しないステージは無視 (エラーなし)"""
    project_info = _build_project(tmp_path)
    # JSON を作らない

    _migration.migrate_stage_id_to_toplevel(str(tmp_path), project_info)
    stage = project_info["event_detail"][0]["timetables"][0]["stage_list"][0]
    assert "stage_id" not in stage


# ---------------------------------------------------------------------------
# 特典会[].対応出番ID バックフィル
# ---------------------------------------------------------------------------

def test_backfill_corresponding_turn_id_fills_missing(tmp_path):
    """live_tokutenkai_heiki 形式で 特典会[].対応出番ID が無いとき、
    親の 出番ID がコピーされる。"""
    project_info = _build_project(tmp_path, dir_name="ライブ特典会",
                                  kind="live_tokutenkai_heiki")
    stage_json_path = os.path.join(
        tmp_path, "event_1", "ライブ特典会", "stage_0.json",
    )
    _write_json(stage_json_path, {
        "ステージ名": "メイン",
        "ステージID": 0,
        "タイムテーブル": [
            {
                "グループ名": "A", "出番ID": 7, "グループID": 0,
                "ライブステージ": {"from": "10:00", "to": "10:30"},
                "特典会": [
                    {"from": "10:40", "to": "11:00", "ブース": "ブースA"},
                    {"from": "12:00", "to": "12:30", "ブース": "ブースA"},
                ],
            }
        ],
    })

    _migration.backfill_tokutenkai_corresponding_turn_id(
        str(tmp_path), project_info,
    )
    data = _read_json(stage_json_path)

    for tk in data["タイムテーブル"][0]["特典会"]:
        assert tk["対応出番ID"] == 7


def test_backfill_is_idempotent_preserves_existing(tmp_path):
    """既に 対応出番ID が埋まっていれば上書きしない (冪等)。"""
    project_info = _build_project(tmp_path, dir_name="ライブ特典会",
                                  kind="live_tokutenkai_heiki")
    stage_json_path = os.path.join(
        tmp_path, "event_1", "ライブ特典会", "stage_0.json",
    )
    _write_json(stage_json_path, {
        "ステージ名": "メイン",
        "ステージID": 0,
        "タイムテーブル": [
            {
                "グループ名": "A", "出番ID": 7,
                "ライブステージ": {"from": "10:00", "to": "10:30"},
                "特典会": [
                    {"from": "10:40", "to": "11:00", "ブース": "ブースA",
                     "対応出番ID": 999},  # ← 既存値
                ],
            }
        ],
    })

    _migration.backfill_tokutenkai_corresponding_turn_id(
        str(tmp_path), project_info,
    )
    data = _read_json(stage_json_path)

    # 既存の 999 を上書きしない
    assert data["タイムテーブル"][0]["特典会"][0]["対応出番ID"] == 999


def test_backfill_skips_non_heiki_format(tmp_path):
    """通常形式 (kind=live) は対象外。特典会キーがなく落ちないことを確認。"""
    project_info = _build_project(tmp_path, dir_name="ライブ", kind="live")
    stage_json_path = os.path.join(tmp_path, "event_1", "ライブ", "stage_0.json")
    _write_json(stage_json_path, {
        "ステージ名": "メイン",
        "ステージID": 0,
        "タイムテーブル": [
            {"グループ名": "A", "出番ID": 0,
             "ライブステージ": {"from": "10:00", "to": "10:30"}},
        ],
    })

    before = _read_json(stage_json_path)
    _migration.backfill_tokutenkai_corresponding_turn_id(
        str(tmp_path), project_info,
    )
    after = _read_json(stage_json_path)
    assert before == after


def test_backfill_skips_when_parent_turn_id_missing(tmp_path):
    """親エントリに 出番ID がない (IDマスタ未確定) なら何もしない。"""
    project_info = _build_project(tmp_path, dir_name="ライブ特典会",
                                  kind="live_tokutenkai_heiki")
    stage_json_path = os.path.join(
        tmp_path, "event_1", "ライブ特典会", "stage_0.json",
    )
    payload = {
        "ステージ名": "メイン",
        "タイムテーブル": [
            {
                "グループ名": "A",  # 出番ID なし
                "ライブステージ": {"from": "10:00", "to": "10:30"},
                "特典会": [
                    {"from": "10:40", "to": "11:00", "ブース": "ブースA"},
                ],
            }
        ],
    }
    _write_json(stage_json_path, payload)

    _migration.backfill_tokutenkai_corresponding_turn_id(
        str(tmp_path), project_info,
    )
    data = _read_json(stage_json_path)
    assert "対応出番ID" not in data["タイムテーブル"][0]["特典会"][0]


def test_backfill_skips_missing_json(tmp_path):
    """物理的に stage_*.json が無い場合はエラーなくスキップ。"""
    project_info = _build_project(tmp_path, dir_name="ライブ特典会",
                                  kind="live_tokutenkai_heiki")
    # ファイルを作らない
    _migration.backfill_tokutenkai_corresponding_turn_id(
        str(tmp_path), project_info,
    )
