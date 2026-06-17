"""
プロジェクトデータの読み書きを担うリポジトリ層。

streamlitに依存しない純粋な関数群。
引数でデータを受け取り、戻り値でデータを返す。
"""

import json
import os
import shutil
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from backend_functions import project_migration as _migration


# ---------------------------------------------------------------------------
# Phase 3: Stella メタデータ デフォルト値
# ---------------------------------------------------------------------------

def _default_stella_metadata() -> dict:
    """新規プロジェクト / 後方互換補完用の stella_metadata 初期値。

    `liveId` / `bundleId` / `jsonVersion` は Phase 4 / Push 成功時に
    初めてセットされるため、本ブロックには含めない (キー欠損で表現)。
    """
    return {
        "openTime": "",
        "closeTime": "",
        "notificationVersion": "1",
        "notification": "",
        "_last_pushed_notification": None,
    }


def get_stella_metadata(project_info_json: dict, event_no: int) -> dict:
    """指定イベントの stella_metadata を返す (欠損時は既定値で補完)。"""
    ev = project_info_json["event_detail"][event_no]
    if "stella_metadata" not in ev or ev["stella_metadata"] is None:
        ev["stella_metadata"] = _default_stella_metadata()
    return ev["stella_metadata"]


def set_stella_metadata(
    project_info_json: dict, event_no: int, metadata: dict,
) -> None:
    """指定イベントの stella_metadata を上書きする。

    既存の内部フィールド (`_last_pushed_notification` 等) は明示的に
    上書きしない限り保持する。
    """
    current = get_stella_metadata(project_info_json, event_no)
    current.update(metadata)


def ensure_stella_metadata(project_info_json: dict) -> dict:
    """`event_detail[i].stella_metadata` が無いイベントに既定値を補完する。

    既存プロジェクトの後方互換用 (Phase 3-2)。
    """
    for ev in project_info_json.get("event_detail", []):
        if "stella_metadata" not in ev or ev["stella_metadata"] is None:
            ev["stella_metadata"] = _default_stella_metadata()
        else:
            for k, v in _default_stella_metadata().items():
                ev["stella_metadata"].setdefault(k, v)
    return project_info_json


# ---------------------------------------------------------------------------
# Group A: 純粋アクセサ（読み取りのみ、I/Oなし）
# ---------------------------------------------------------------------------

def get_event_name(project_info_json: dict, event_no: int) -> str:
    return project_info_json["event_detail"][event_no]["event_name"]


def get_event_name_list(project_info_json: dict) -> list[str]:
    return [d["event_name"] for d in project_info_json["event_detail"]]


def get_event_type_list(project_info_json: dict, event_no: int) -> list[str]:
    """イベントごとに存在する画像の dir_name リストを kind 順で返す。

    並び: live → tokutenkai → live_tokutenkai_heiki → その他、
          同 kind 内は image_no 昇順。
    """
    image_no_list = get_sorted_image_no_list(project_info_json, event_no)
    return [
        get_image_entry_by_no(project_info_json, event_no, image_no)["dir_name"]
        for image_no in image_no_list
    ]


def get_event_no_by_event_name(project_info_json: dict, event_name: str) -> int | None:
    for event_detail in project_info_json["event_detail"]:
        if event_detail["event_name"] == event_name:
            return event_detail["event_no"]
    return None


def get_stage_name_list(project_info_json: dict, event_no: int, img_type: str) -> list[str]:
    """img_type は dir_name。新スキーマの list から該当エントリを引いて stage_name のリストを返す。"""
    entry = get_image_entry_by_dir_name(project_info_json, event_no, img_type)
    if entry is None:
        raise KeyError(f"dir_name={img_type} not found in event_no={event_no}")
    return [stage_info["stage_name"] for stage_info in entry["stage_list"]]


def get_stage_name(project_info_json: dict, event_no: int, img_type: str, stage_no: int) -> str:
    """img_type は dir_name。"""
    entry = get_image_entry_by_dir_name(project_info_json, event_no, img_type)
    if entry is None:
        raise KeyError(f"dir_name={img_type} not found in event_no={event_no}")
    return entry["stage_list"][stage_no]["stage_name"]


def get_stage_id(
    project_info_json: dict, event_no: int, img_type: str, stage_no: int,
) -> int | None:
    """stage_list[stage_no].stage_id を取得。未確定/欠落時は None。"""
    entry = get_image_entry_by_dir_name(project_info_json, event_no, img_type)
    if entry is None:
        return None
    stage_list = entry.get("stage_list", [])
    if stage_no >= len(stage_list):
        return None
    return stage_list[stage_no].get("stage_id")


def set_stage_id(
    project_info_json: dict, event_no: int, img_type: str, stage_no: int,
    stage_id: int,
) -> None:
    """stage_list[stage_no].stage_id を設定する。"""
    entry = get_image_entry_by_dir_name(project_info_json, event_no, img_type)
    if entry is None:
        raise KeyError(f"dir_name={img_type} not found in event_no={event_no}")
    entry["stage_list"][stage_no]["stage_id"] = int(stage_id)


def get_stage_id_list(
    project_info_json: dict, event_no: int, img_type: str,
) -> list[int | None]:
    """stage_list の stage_id を順に返す。欠落は None。"""
    entry = get_image_entry_by_dir_name(project_info_json, event_no, img_type)
    if entry is None:
        raise KeyError(f"dir_name={img_type} not found in event_no={event_no}")
    return [s.get("stage_id") for s in entry["stage_list"]]


def set_stage_name(
    project_info_json: dict, event_no: int, img_type: str, stage_no: int,
    stage_name: str,
) -> None:
    """stage_list[stage_no].stage_name を設定する。"""
    entry = get_image_entry_by_dir_name(project_info_json, event_no, img_type)
    if entry is None:
        raise KeyError(f"dir_name={img_type} not found in event_no={event_no}")
    entry["stage_list"][stage_no]["stage_name"] = stage_name


# ---------------------------------------------------------------------------
# Group A': 新スキーマ (timetables: list) 用アクセサ
# ---------------------------------------------------------------------------
# Phase 3 で旧 get_event_type_list / get_stage_name_list / get_stage_name を
# これらに置き換える。それまでは並存。


def get_image_entry_list(project_info_json: dict, event_no: int) -> list[dict]:
    """指定イベントの timetables (画像エントリのリスト) を返す。"""
    return project_info_json["event_detail"][event_no]["timetables"]


def get_image_entry_by_no(
    project_info_json: dict, event_no: int, image_no: int,
) -> dict:
    """image_no で画像エントリを引く。存在しない場合 KeyError。"""
    for entry in get_image_entry_list(project_info_json, event_no):
        if entry["image_no"] == image_no:
            return entry
    raise KeyError(f"image_no={image_no} not found in event_no={event_no}")


def get_image_entry_by_dir_name(
    project_info_json: dict, event_no: int, dir_name: str,
) -> dict | None:
    """dir_name で画像エントリを引く。なければ None。"""
    for entry in get_image_entry_list(project_info_json, event_no):
        if entry["dir_name"] == dir_name:
            return entry
    return None


def get_image_no_by_dir_name(
    project_info_json: dict, event_no: int, dir_name: str,
) -> int | None:
    """dir_name から image_no を引く。なければ None。"""
    entry = get_image_entry_by_dir_name(project_info_json, event_no, dir_name)
    return entry["image_no"] if entry is not None else None


def next_image_no(project_info_json: dict, event_no: int) -> int:
    """新規 image_no を採番する。欠番は残し、(現在の最大値 + 1) を返す。"""
    entries = get_image_entry_list(project_info_json, event_no)
    if not entries:
        return 0
    return max(e["image_no"] for e in entries) + 1


def find_dir_name_conflict(
    project_info_json: dict, event_no: int, dir_name: str,
) -> int | None:
    """同一 dir_name の既存画像があれば、その image_no を返す。なければ None。"""
    return get_image_no_by_dir_name(project_info_json, event_no, dir_name)


def get_sorted_image_no_list(
    project_info_json: dict, event_no: int,
) -> list[int]:
    """timetables[] の配列順そのままで image_no リストを返す。"""
    return [e["image_no"] for e in get_image_entry_list(project_info_json, event_no)]


# ---------------------------------------------------------------------------
# Group A'': 並び順ロジック
# ---------------------------------------------------------------------------
# timetables[] の配列順 = 表示順。新規登録の挿入位置決定とリセット時の再ソートで
# 以下のバケット定義を使う。
#
#   0: dir_name == "ライブ"
#   1: kind == "live" (dir_name != "ライブ")
#   2: dir_name == "特典会"
#   3: kind in ("tokutenkai", "live_tokutenkai_heiki") (dir_name != "特典会")
#   4: その他

_DIR_NAME_LIVE = "ライブ"
_DIR_NAME_TOKUTENKAI = "特典会"
_TOKUTENKAI_KINDS = ("tokutenkai", "live_tokutenkai_heiki")


def _bucket_index(entry: dict) -> int:
    dir_name = entry.get("dir_name", "")
    kind = entry.get("kind", "")
    if dir_name == _DIR_NAME_LIVE:
        return 0
    if kind == "live":
        return 1
    if dir_name == _DIR_NAME_TOKUTENKAI:
        return 2
    if kind in _TOKUTENKAI_KINDS:
        return 3
    return 4


def compute_insert_index(
    project_info_json: dict, event_no: int, dir_name: str, kind: str,
) -> int:
    """新規エントリの挿入先 index を返す。

    バケット定義に従い、同じバケットの末尾の直後（無ければ自バケットより
    優先度の高いバケット末尾の直後、それも無ければ先頭）を返す。
    """
    entries = get_image_entry_list(project_info_json, event_no)
    target_bucket = _bucket_index({"dir_name": dir_name, "kind": kind})
    insert_idx = 0
    for i, e in enumerate(entries):
        if _bucket_index(e) <= target_bucket:
            insert_idx = i + 1
    return insert_idx


def move_timetable_up(
    project_info_json: dict, event_no: int, image_no: int,
) -> None:
    """同一イベント内で image_no エントリを 1 つ前と swap する。先頭なら何もしない。"""
    entries = get_image_entry_list(project_info_json, event_no)
    for i, e in enumerate(entries):
        if e["image_no"] == image_no:
            if i > 0:
                entries[i - 1], entries[i] = entries[i], entries[i - 1]
            return
    raise KeyError(f"image_no={image_no} not found in event_no={event_no}")


def move_timetable_down(
    project_info_json: dict, event_no: int, image_no: int,
) -> None:
    """同一イベント内で image_no エントリを 1 つ後と swap する。末尾なら何もしない。"""
    entries = get_image_entry_list(project_info_json, event_no)
    for i, e in enumerate(entries):
        if e["image_no"] == image_no:
            if i < len(entries) - 1:
                entries[i], entries[i + 1] = entries[i + 1], entries[i]
            return
    raise KeyError(f"image_no={image_no} not found in event_no={event_no}")


def reset_timetable_order(
    project_info_json: dict, event_no: int,
) -> None:
    """timetables[] をデフォルトバケット順 (同バケット内は既存配列順) で並べ直す。"""
    entries = get_image_entry_list(project_info_json, event_no)
    # sorted は stable なので、同バケット内では元の出現順が保たれる
    entries.sort(key=_bucket_index)


# ---------------------------------------------------------------------------
# Group A: 旧スキーマアクセサ (Phase 3 で削除予定)
# ---------------------------------------------------------------------------

def get_ticket_urls_for_event(project_info_json: dict, event_name: str) -> list[str]:
    """指定イベントに紐づくチケットURLリストを取得する"""
    if "ticket_urls" not in project_info_json:
        return []

    ticket_urls_config = project_info_json["ticket_urls"]
    scope = ticket_urls_config.get("scope", "project")

    if scope == "project":
        return ticket_urls_config.get("urls", [])
    else:
        for event in project_info_json["event_detail"]:
            if event["event_name"] == event_name:
                return event.get("ticket_urls", [])
        return []


# ---------------------------------------------------------------------------
# Group B: ファイルI/O
# ---------------------------------------------------------------------------

def get_project_json(pj_path: str) -> dict:
    """project_info.jsonを読み込んで返す。

    読み込み時に旧スキーマの timetables (dict) を新スキーマ (list) に自動変換する。
    また、stage_*.json のステージIDのトップレベル化と project_info.stage_list[i].stage_id
    補完も同時に行う。変換結果は次回 save_project_json 時にディスクへ書き戻される。
    """
    json_path = os.path.join(pj_path, "project_info.json")
    with open(json_path, "r", encoding="utf-8") as f:
        project_info_json = json.load(f)
    # TODO(post-migration): 全プロジェクト移行後、本呼び出しごと削除可
    project_info_json = _migration.migrate_project_info(project_info_json)
    project_info_json = _migration.migrate_stage_id_to_toplevel(pj_path, project_info_json)
    project_info_json = _migration.backfill_tokutenkai_corresponding_turn_id(
        pj_path, project_info_json,
    )
    project_info_json = ensure_stella_metadata(project_info_json)
    return project_info_json


def save_project_json(pj_path: str, json_data: dict) -> None:
    """project_info.jsonに書き込む（タイムスタンプ更新は呼び出し側の責務）"""
    json_path = os.path.join(pj_path, "project_info.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)


def update_timestamp(
    project_master: pd.DataFrame,
    pj_name: str,
    data_path: str,
) -> pd.DataFrame:
    """project_masterのupdated_atを現在の日本時間で更新し、CSVに保存して返す"""
    jst = ZoneInfo("Asia/Tokyo")
    now_jst = datetime.now(jst)
    updated_at = now_jst.strftime("%Y/%m/%d %H:%M:%S.%f")
    project_master.loc[pj_name, "updated_at"] = updated_at
    project_master.to_csv(os.path.join(data_path, "master", "projects_master.csv"))
    return project_master


# ---------------------------------------------------------------------------
# Group C: データロジック（UIから分離）
# ---------------------------------------------------------------------------

def create_project_data(
    data_path: str,
    pj_name: str,
    project_master: pd.DataFrame,
    project_master_s3: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """新規プロジェクトを作成する。

    Returns:
        (project_info_json, updated_project_master, updated_project_master_s3)
    """
    pj_dir = os.path.join(data_path, "projects", pj_name)
    os.makedirs(pj_dir, exist_ok=True)

    created_at = datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")
    project_master.loc[pj_name] = [created_at, created_at, "フェス", 1]
    project_master_s3.loc[pj_name] = [created_at, created_at, "フェス", 1]
    project_master.to_csv(os.path.join(data_path, "master", "projects_master.csv"))
    project_master_s3.to_csv(os.path.join(data_path, "master", "projects_master_s3.csv"))

    project_info_json = {
        "project_name": pj_name,
        "event_num": 1,
        "ticket_urls": {
            "scope": "project",
            "urls": [],
        },
        "event_detail": [
            {
                "event_no": 0,
                "event_name": "event_1",
                "ticket_urls": [],
                "timetables": [],
                "stella_metadata": _default_stella_metadata(),
            }
        ],
    }
    json_path = os.path.join(pj_dir, "project_info.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(project_info_json, f, indent=4, ensure_ascii=False)

    return project_info_json, project_master, project_master_s3


def delete_project_data(
    data_path: str,
    pj_name: str,
    project_master: pd.DataFrame,
    project_master_s3: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ローカルのプロジェクト本体ディレクトリと両マスタCSVから該当行を削除する。

    対象が存在しないステップは個別にスキップするため、リトライ時の冪等性を持つ。

    Returns:
        (updated_project_master, updated_project_master_s3)
    """
    pj_dir = os.path.join(data_path, "projects", pj_name)
    if os.path.isdir(pj_dir):
        shutil.rmtree(pj_dir)
    if pj_name in project_master.index:
        project_master = project_master.drop(index=pj_name)
    if pj_name in project_master_s3.index:
        project_master_s3 = project_master_s3.drop(index=pj_name)
    project_master.to_csv(os.path.join(data_path, "master", "projects_master.csv"))
    project_master_s3.to_csv(os.path.join(data_path, "master", "projects_master_s3.csv"))
    return project_master, project_master_s3


def apply_project_setting(
    data_path: str,
    pj_name: str,
    project_info_json: dict,
    project_master: pd.DataFrame,
    event_type: str,
    event_num: int,
) -> tuple[dict, pd.DataFrame]:
    """プロジェクト設定（イベント形式・数）を適用する。

    Returns:
        (updated_project_info_json, updated_project_master)
    """
    updated_at = datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")
    project_master.loc[pj_name, "event_type"] = event_type
    project_master.loc[pj_name, "event_num"] = event_num
    project_master.loc[pj_name, "updated_at"] = updated_at
    project_master.to_csv(os.path.join(data_path, "master", "projects_master.csv"))

    project_info_json["event_num"] = event_num
    project_info_json["event_detail"] = project_info_json["event_detail"][:event_num]
    for i in range(event_num):
        os.makedirs(
            os.path.join(data_path, "projects", pj_name, "event_" + str(i + 1)),
            exist_ok=True,
        )
        if len(project_info_json["event_detail"]) - 1 < i:
            project_info_json["event_detail"].append(
                {
                    "event_no": i,
                    "event_name": "event_{}".format(i + 1),
                    "ticket_urls": [],
                    "timetables": [],
                    "stella_metadata": _default_stella_metadata(),
                }
            )

    pj_path = os.path.join(data_path, "projects", pj_name)
    save_project_json(pj_path, project_info_json)

    return project_info_json, project_master


def register_timetable_image(
    pj_path: str,
    event_name: str,
    event_no: int,
    dir_name: str,
    kind: str,
    img_format: str | None,
    file_data: bytes,
    project_info_json: dict,
) -> dict:
    """画像ファイルを保存しproject_info_jsonを更新して返す。

    dir_name: 保存先サブフォルダ名 (= UI 表示名)
    kind: "live" | "tokutenkai" | "live_tokutenkai_heiki"
    img_format: kind=live_tokutenkai_heiki のときは None。それ以外は "通常" / "ライムライト式"。

    既存の同 dir_name エントリがあれば image_no を再利用して上書き、なければ末尾に append。

    NOTE: 上書き判定 (find_dir_name_conflict) と派生物クリーンアップは呼び出し側 (UI 層) の責務。
    本関数は project_info_json のエントリと raw.png のみを書き換える。
    """
    img_dir = os.path.join(pj_path, event_name, dir_name)
    os.makedirs(img_dir, exist_ok=True)
    img_path = os.path.join(img_dir, "raw.png")
    with open(img_path, "wb") as f:
        f.write(file_data)

    existing_image_no = get_image_no_by_dir_name(
        project_info_json, event_no, dir_name,
    )
    if existing_image_no is not None:
        image_no = existing_image_no
    else:
        image_no = next_image_no(project_info_json, event_no)

    new_entry: dict = {
        "image_no": image_no,
        "dir_name": dir_name,
        "display_name": dir_name,
        "kind": kind,
        "stage_num": 0,
        "stage_list": [],
    }
    if kind != _migration.KIND_LIVE_TOKUTENKAI_HEIKI:
        new_entry["format"] = img_format

    entries = get_image_entry_list(project_info_json, event_no)
    replaced = False
    for i, e in enumerate(entries):
        if e["image_no"] == image_no:
            entries[i] = new_entry
            replaced = True
            break
    if not replaced:
        insert_idx = compute_insert_index(
            project_info_json, event_no, dir_name, kind,
        )
        entries.insert(insert_idx, new_entry)

    return project_info_json


_DERIVATIVE_FILE_PATTERNS = ("raw_cropped.png", "stage_", "raw_old", "raw_cropped_old", "all_stages")


def cleanup_image_artifacts(pj_path: str, event_name: str, dir_name: str) -> None:
    """指定画像フォルダ配下の派生物を削除する。raw.png 含むフォルダ内全ファイルを消す。

    上書き登録時に古いステージ画像・OCR JSON が残らないようにするための関数。
    フォルダ自体は削除せず、中身のみクリーンアップする。
    """
    img_dir = os.path.join(pj_path, event_name, dir_name)
    if not os.path.isdir(img_dir):
        return
    for fname in os.listdir(img_dir):
        fpath = os.path.join(img_dir, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)


def delete_timetable_image(
    project_info_json: dict,
    event_no: int,
    img_type: str,
) -> dict:
    """指定 dir_name の画像エントリを削除して返す (image_no は欠番として残る)"""
    entries = get_image_entry_list(project_info_json, event_no)
    project_info_json["event_detail"][event_no]["timetables"] = [
        e for e in entries if e["dir_name"] != img_type
    ]
    return project_info_json


def build_ticket_urls_data(
    project_info_json: dict,
    scope: str,
    urls_data: dict,
) -> dict:
    """チケットURL設定をproject_info_jsonに適用して返す。

    Args:
        scope: "プロジェクト共通" or "イベントごと"
        urls_data: scope=="プロジェクト共通" → {"project": [url, ...]}
                   scope=="イベントごと"    → {"event_0": [url,...], ...}
    """
    if "ticket_urls" not in project_info_json:
        project_info_json["ticket_urls"] = {"scope": "project", "urls": []}

    if scope == "プロジェクト共通":
        project_info_json["ticket_urls"] = {
            "scope": "project",
            "urls": urls_data.get("project", []),
        }
        for event in project_info_json["event_detail"]:
            event["ticket_urls"] = []
    else:
        project_info_json["ticket_urls"] = {
            "scope": "event",
            "urls": [],
        }
        for i, event in enumerate(project_info_json["event_detail"]):
            event["ticket_urls"] = urls_data.get(f"event_{i}", [])

    return project_info_json
