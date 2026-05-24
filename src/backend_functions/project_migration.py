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
