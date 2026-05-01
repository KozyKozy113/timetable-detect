"""
プロジェクトデータの読み書きを担うリポジトリ層。

streamlitに依存しない純粋な関数群。
引数でデータを受け取り、戻り値でデータを返す。
"""

import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd


# ---------------------------------------------------------------------------
# Group A: 純粋アクセサ（読み取りのみ、I/Oなし）
# ---------------------------------------------------------------------------

def get_event_name(project_info_json: dict, event_no: int) -> str:
    return project_info_json["event_detail"][event_no]["event_name"]


def get_event_name_list(project_info_json: dict) -> list[str]:
    return [d["event_name"] for d in project_info_json["event_detail"]]


def get_event_type_list(project_info_json: dict, event_no: int) -> list[str]:
    """ライブ、特典会を先頭にしつつイベントごとに存在する画像種別のリストを返す"""
    event_type_all = list(
        project_info_json["event_detail"][event_no]["timetables"].keys()
    )
    event_type_list = []
    for event_type in ["ライブ", "特典会", "ライブ特典会"]:
        if event_type in event_type_all:
            event_type_list.append(event_type)
    for event_type in event_type_all:
        if event_type not in event_type_list:
            event_type_list.append(event_type)
    return event_type_list


def get_event_no_by_event_name(project_info_json: dict, event_name: str) -> int | None:
    for event_detail in project_info_json["event_detail"]:
        if event_detail["event_name"] == event_name:
            return event_detail["event_no"]
    return None


def get_stage_name_list(project_info_json: dict, event_no: int, img_type: str) -> list[str]:
    return [
        stage_info["stage_name"]
        for stage_info in project_info_json["event_detail"][event_no]["timetables"][img_type]["stage_list"]
    ]


def get_stage_name(project_info_json: dict, event_no: int, img_type: str, stage_no: int) -> str:
    return project_info_json["event_detail"][event_no]["timetables"][img_type]["stage_list"][stage_no]["stage_name"]


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
    """project_info.jsonを読み込んで返す"""
    json_path = os.path.join(pj_path, "project_info.json")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


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
                "timetables": {},
            }
        ],
    }
    json_path = os.path.join(pj_dir, "project_info.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(project_info_json, f, indent=4, ensure_ascii=False)

    return project_info_json, project_master, project_master_s3


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
                    "timetables": {},
                }
            )

    pj_path = os.path.join(data_path, "projects", pj_name)
    save_project_json(pj_path, project_info_json)

    return project_info_json, project_master


def register_timetable_image(
    pj_path: str,
    event_name: str,
    event_no: int,
    resolved_img_type: str,
    img_format: str,
    file_data: bytes,
    project_info_json: dict,
) -> dict:
    """画像ファイルを保存しproject_info_jsonを更新して返す。

    resolved_img_type: 保存先のディレクトリ名（"ライブ", "特典会", "ライブ特典会", カスタム名等）
    img_format: timetableのformat値
    """
    img_dir = os.path.join(pj_path, event_name, resolved_img_type)
    os.makedirs(img_dir, exist_ok=True)
    img_path = os.path.join(img_dir, "raw.png")
    with open(img_path, "wb") as f:
        f.write(file_data)

    project_info_json["event_detail"][event_no]["timetables"][resolved_img_type] = {
        "format": img_format,
        "stage_num": 0,
        "stage_list": [],
    }
    return project_info_json


def delete_timetable_image(
    project_info_json: dict,
    event_no: int,
    img_type: str,
) -> dict:
    """指定画像種別をproject_info_jsonから削除して返す"""
    del project_info_json["event_detail"][event_no]["timetables"][img_type]
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
