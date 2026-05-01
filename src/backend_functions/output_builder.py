"""
出力データの構築・エクスポート。

ステージマスタ・アーティストマスタ・出番データの組み立て、
ID確定、Excelエクスポート、グループ名マスタ更新を行う。
Streamlitに依存しない。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from openpyxl import Workbook

from backend_functions import timetabledata, idolname, s3access
from backend_functions import project_repository as repo


# ---------------------------------------------------------------------------
# ID確定
# ---------------------------------------------------------------------------

def determine_id_master(
    output_df: dict[str, dict[str, pd.DataFrame]],
    pj_path: str,
    project_info_json: dict,
) -> None:
    """ステージマスタ・グループマスタ・出番マスタのIDを確定させ、CSV/JSONに保存する。"""
    event_list = repo.get_event_name_list(project_info_json)
    for event_name in event_list:
        output_path = os.path.join(pj_path, event_name)
        output_df[event_name]["stage"].to_csv(os.path.join(output_path, "master_stage.csv"))
        output_df[event_name]["idolname"].to_csv(os.path.join(output_path, "master_idolname.csv"))
        turn_id_data = output_df[event_name]["live"]
        turn_id_data.to_csv(os.path.join(output_path, "turn_id_data.csv"))

        event_no = repo.get_event_no_by_event_name(project_info_json, event_name)
        event_type_list = repo.get_event_type_list(project_info_json, event_no)
        for event_type in event_type_list:
            tgt_event_type_info = project_info_json["event_detail"][event_no]["timetables"][event_type]
            for stage_no in range(tgt_event_type_info["stage_num"]):
                stage_name = repo.get_stage_name(project_info_json, event_no, event_type, stage_no)
                json_path = os.path.join(output_path, event_type, f"stage_{stage_no}.json")
                if os.path.exists(json_path):
                    with open(json_path, encoding="utf-8") as f:
                        json_data = json.load(f)
                    json_data = timetabledata.id_apply_to_json(
                        json_data, turn_id_data, stage_name,
                        tgt_event_type_info["format"] == "特典会併記",
                    )
                    with open(json_path, "w", encoding="utf8") as f:
                        json.dump(json_data, f, indent=4, ensure_ascii=False)


# ---------------------------------------------------------------------------
# S3保存
# ---------------------------------------------------------------------------

def save_to_s3(pj_name: str) -> None:
    """プロジェクトデータをS3にアップロードする。"""
    s3access.put_project_data(pj_name)


# ---------------------------------------------------------------------------
# Excel出力
# ---------------------------------------------------------------------------

def export_excel(
    output_df: dict[str, dict[str, pd.DataFrame]],
    pj_path: str,
    event_list: list[str],
) -> str:
    """Excel形式でデータを出力し、出力パスを返す。"""
    output_path = os.path.join(pj_path, "output.xlsx")
    wb = Workbook()
    for event_name in event_list:
        for df_type, position in zip(["stage", "idolname", "live"], [(1, 1), (5, 1), (8, 1)]):
            save_dataframe_to_excel(wb, event_name, output_df[event_name][df_type], position)
    default_sheet = wb["Sheet"]
    wb.remove(default_sheet)
    wb.save(output_path)
    return output_path


def save_dataframe_to_excel(
    wb: Workbook,
    sheet_name: str,
    df: pd.DataFrame,
    position: tuple[int, int],
) -> None:
    """DataFrameをExcelワークブックの指定位置に書き込む。"""
    existing_sheets = wb.sheetnames
    if sheet_name not in existing_sheets:
        ws = wb.create_sheet(title=sheet_name)
    else:
        ws = wb[sheet_name]
    for i, row in enumerate(df.itertuples(), start=position[1]):
        for j, value in enumerate(row, start=position[0]):
            ws.cell(row=i + 1, column=j, value=value)
    for j, header in enumerate(df.columns, start=position[0]):
        ws.cell(row=position[1], column=j + 1, value=header)
    ws.cell(row=position[1], column=position[0], value=df.index.name)


# ---------------------------------------------------------------------------
# グループ名マスタ
# ---------------------------------------------------------------------------

def listup_new_idolname(
    output_df: dict[str, dict[str, pd.DataFrame]],
    event_list: list[str],
) -> pd.DataFrame:
    """新しく出現したグループ名をリストアップする。"""
    idol_name_all = []
    for event_name in event_list:
        idol_name_all.extend(list(output_df[event_name]["idolname"]["グループ名_採用"]))
    new_idol_name = idolname.detect_new_data(list(set(idol_name_all)))
    return pd.DataFrame({
        "追加": [True for _ in range(len(new_idol_name))],
        "グループ名": new_idol_name,
    }).sort_values(by="グループ名").reset_index(drop=True)


def update_master_idolname(
    df_new_idolname: pd.DataFrame,
    data_path: str,
) -> None:
    """新しく出現したグループ名をマスタに追加し、S3にアップロードする。"""
    new_idolname = list(df_new_idolname[df_new_idolname["追加"]]["グループ名"])
    idolname.add_new_data_file(new_idolname)

    json_path = os.path.join(data_path, "master/master_version_s3.json")
    with open(json_path, 'r', encoding='utf-8') as f:
        master_version_s3 = json.load(f)
    jst = ZoneInfo("Asia/Tokyo")
    now_jst = datetime.now(jst)
    updated_at = now_jst.strftime('%Y/%m/%d %H:%M:%S.%f')
    master_version_s3["idolname_embedding_data.csv"] = updated_at
    master_version_s3["idolname_latest.csv"] = updated_at
    with open(json_path, "w", encoding="utf8") as f:
        json.dump(master_version_s3, f, indent=4, ensure_ascii=False)

    s3_prefix = "master"
    s3access.upload_s3_file(s3_prefix, "master_version_s3.json", os.path.join(data_path, "master/master_version_s3.json"))
    s3access.upload_s3_file(s3_prefix, "idolname_embedding_data.csv", os.path.join(data_path, "master/idolname_embedding_data.csv"))
    s3access.upload_s3_file(s3_prefix, "idolname_latest.csv", os.path.join(data_path, "master/idolname_latest.csv"))
