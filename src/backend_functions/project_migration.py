# TODO(post-migration): 全プロジェクトが新スキーマに移行した後、本モジュールは削除可。
# 呼び出し箇所は project_repository.get_project_json() の 1 箇所のみに限定すること。
"""
project_info.json の旧スキーマ → 新スキーマへの自動マイグレーション。

旧スキーマ (timetables: dict):
    "timetables": {
        "ライブ": {"format": "通常", "stage_num": ..., "stage_list": [...]},
        "特典会": {...},
        "ライブ特典会": {"format": "特典会併記", ...},
        "<カスタム名>": {"format": "通常" | "特典会併記", ...},
    }

新スキーマ (timetables: list):
    "timetables": [
        {
            "image_no": 0,
            "display_name": "ライブ",
            "dir_name": "ライブ",
            "kind": "live" | "tokutenkai" | "live_tokutenkai_heiki",
            "format": "通常" | "ライムライト式",     # kind=live_tokutenkai_heiki のときは省略
            "stage_num": ...,
            "stage_list": [{"stage_no": ..., "stage_name": ..., "kind": ..., ...}, ...],
            ...                                     # その他のフィールド (raw_crop_box, time_pixel 等) はそのまま継承
        },
        ...
    ]

カスタム名の kind 推論:
    - 旧 format == "特典会併記" → kind = "live_tokutenkai_heiki"
    - それ以外 (通常/ライムライト式) → kind = "tokutenkai"
      (過去データでは「その他」系画像は実質特典会扱いが多数。誤推論は許容。)
"""

from __future__ import annotations

import copy
import json
import os

KIND_LIVE = "live"
KIND_TOKUTENKAI = "tokutenkai"
KIND_LIVE_TOKUTENKAI_HEIKI = "live_tokutenkai_heiki"

_LEGACY_STANDARD_KEY_TO_KIND = {
    "ライブ": KIND_LIVE,
    "特典会": KIND_TOKUTENKAI,
    "ライブ特典会": KIND_LIVE_TOKUTENKAI_HEIKI,
}
_LEGACY_HEIKI_FORMAT = "特典会併記"


def _is_new_schema_timetables(timetables) -> bool:
    return isinstance(timetables, list)


def _infer_kind(legacy_key: str, legacy_format: str | None) -> str:
    if legacy_key in _LEGACY_STANDARD_KEY_TO_KIND:
        return _LEGACY_STANDARD_KEY_TO_KIND[legacy_key]
    if legacy_format == _LEGACY_HEIKI_FORMAT:
        return KIND_LIVE_TOKUTENKAI_HEIKI
    return KIND_TOKUTENKAI


def _migrate_timetable_entry(
    image_no: int,
    legacy_key: str,
    legacy_entry: dict,
) -> dict:
    new_entry = copy.deepcopy(legacy_entry)
    kind = _infer_kind(legacy_key, new_entry.get("format"))

    new_entry["image_no"] = image_no
    new_entry["dir_name"] = legacy_key
    new_entry["display_name"] = legacy_key
    new_entry["kind"] = kind

    if kind == KIND_LIVE_TOKUTENKAI_HEIKI:
        new_entry.pop("format", None)

    stage_list = new_entry.get("stage_list", [])
    for stage in stage_list:
        if "kind" not in stage:
            stage["kind"] = kind

    return new_entry


def _migrate_event(event_detail: dict) -> dict:
    timetables = event_detail.get("timetables")
    if timetables is None:
        return event_detail
    if _is_new_schema_timetables(timetables):
        return event_detail

    new_timetables: list[dict] = []
    for image_no, (legacy_key, legacy_entry) in enumerate(timetables.items()):
        new_timetables.append(
            _migrate_timetable_entry(image_no, legacy_key, legacy_entry)
        )
    event_detail["timetables"] = new_timetables
    return event_detail


def migrate_project_info(project_info_json: dict) -> dict:
    """旧スキーマの project_info.json を新スキーマに変換して返す。

    新スキーマであればそのまま返す（冪等）。
    入力 dict は破壊的に変更される。
    """
    event_detail_list = project_info_json.get("event_detail", [])
    for event_detail in event_detail_list:
        _migrate_event(event_detail)
    return project_info_json


# ---------------------------------------------------------------------------
# ステージID のトップレベル化マイグレーション
# ---------------------------------------------------------------------------
# 旧:
#   stage_*.json: 出番粒度 タイムテーブル[i]["ステージID"]
#   project_info.stage_list[i]: stage_id なし
# 新:
#   stage_*.json: トップレベル "ステージID"
#                 (特典会併記形式では子=特典会[].ステージID もあわせて維持)
#   project_info.stage_list[i]: stage_id (= 親ライブステージID)
#
# 判別: stage_*.json のトップレベル "ステージID" の有無で旧/新を区別する。
# 冪等性: 新形式であれば変更しない。

def _migrate_stage_json_file(json_path: str) -> int | None:
    """単一の stage_*.json をトップレベル化する。冪等。

    Returns:
        ステージID (トップレベル) を確定したら int、未確定/欠落なら None。
    """
    if not os.path.exists(json_path):
        return None
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    timetable = data.get("タイムテーブル", [])

    # 既にトップレベルに ステージID がある場合 → 冪等 (出番粒度の余剰を念のため掃除)
    if "ステージID" in data and data["ステージID"] is not None:
        changed = False
        for turn in timetable:
            if "ステージID" in turn:
                del turn["ステージID"]
                changed = True
        if changed:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        try:
            return int(data["ステージID"])
        except (ValueError, TypeError):
            return None

    # 旧形式: 出番粒度から拾い上げてトップレベル化
    top_stage_id: int | None = None
    for turn in timetable:
        if "ステージID" in turn and turn["ステージID"] is not None:
            try:
                top_stage_id = int(turn["ステージID"])
                break
            except (ValueError, TypeError):
                continue

    changed = False
    if top_stage_id is not None:
        data["ステージID"] = top_stage_id
        changed = True
    # 出番粒度の ステージID を全削除
    for turn in timetable:
        if "ステージID" in turn:
            del turn["ステージID"]
            changed = True

    if changed:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    return top_stage_id


# ---------------------------------------------------------------------------
# 特典会[].対応出番ID 補完マイグレーション
# ---------------------------------------------------------------------------
# live_tokutenkai_heiki 形式の stage_*.json 内、特典会[]要素に「対応出番ID」を補完する。
# 編集モード(Phase 4)で特典会行を JSON に書き戻す際に、親ライブ出番との対応を取るキー。
# 親エントリの 出番ID を 特典会[j] にコピーするだけ。冪等。

def _backfill_corresponding_turn_id_in_file(json_path: str) -> bool:
    """単一の stage_*.json について 特典会[].対応出番ID を補完する。冪等。

    Returns: 変更があれば True。
    """
    if not os.path.exists(json_path):
        return False
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    changed = False
    for turn in data.get("タイムテーブル", []):
        parent_turn_id = turn.get("出番ID")
        if parent_turn_id is None:
            continue
        tk_list = turn.get("特典会")
        if not isinstance(tk_list, list):
            continue
        for tk in tk_list:
            if not isinstance(tk, dict):
                continue
            existing = tk.get("対応出番ID")
            if existing is not None:
                # 既に埋まっていれば触らない (誤った値の上書きを避ける)
                continue
            try:
                tk["対応出番ID"] = int(parent_turn_id)
                changed = True
            except (ValueError, TypeError):
                continue
    if changed:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    return changed


def backfill_tokutenkai_corresponding_turn_id(
    pj_path: str, project_info_json: dict,
) -> dict:
    """既存プロジェクトの live_tokutenkai_heiki 形式 stage_*.json に
    特典会[].対応出番ID を補完する。冪等。

    project_info_json 自体は変更しない (stage_*.json のみ書き換え)。
    """
    event_detail_list = project_info_json.get("event_detail", [])
    for event_detail in event_detail_list:
        event_name = event_detail.get("event_name")
        if event_name is None:
            continue
        timetables = event_detail.get("timetables", [])
        if not isinstance(timetables, list):
            continue
        for entry in timetables:
            if entry.get("kind") != KIND_LIVE_TOKUTENKAI_HEIKI:
                continue
            dir_name = entry.get("dir_name")
            stage_num = entry.get("stage_num", 0)
            if dir_name is None:
                continue
            for stage_no in range(stage_num):
                json_path = os.path.join(
                    pj_path, event_name, dir_name, f"stage_{stage_no}.json",
                )
                _backfill_corresponding_turn_id_in_file(json_path)
    return project_info_json


def migrate_stage_id_to_toplevel(pj_path: str, project_info_json: dict) -> dict:
    """既存 stage_*.json のステージID出番粒度 → トップレベル昇格と、
    project_info.json の stage_list[i].stage_id 補完を行う。冪等。

    入力 project_info_json は破壊的に変更される。
    project_info.json 自体の永続化は呼び出し側の責務。
    """
    event_detail_list = project_info_json.get("event_detail", [])
    for event_detail in event_detail_list:
        event_name = event_detail.get("event_name")
        if event_name is None:
            continue
        timetables = event_detail.get("timetables", [])
        if not isinstance(timetables, list):
            continue
        for entry in timetables:
            dir_name = entry.get("dir_name")
            stage_list = entry.get("stage_list", [])
            if dir_name is None or not stage_list:
                continue
            for stage_no, stage in enumerate(stage_list):
                json_path = os.path.join(
                    pj_path, event_name, dir_name, f"stage_{stage_no}.json",
                )
                top_stage_id = _migrate_stage_json_file(json_path)
                # project_info への補完: 既に stage_id が無く、JSON 側に確定値があれば書き込む
                if "stage_id" not in stage and top_stage_id is not None:
                    stage["stage_id"] = top_stage_id
    return project_info_json
