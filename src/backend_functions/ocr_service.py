"""
OCR実行サービス層。

GPT-OCRによるタイムテーブル読み取り、グループ名補正、
ステージ名取得などの業務ロジック。
Streamlitに依存しない。
"""

from __future__ import annotations

import concurrent.futures
import copy
import json
import os
from typing import Optional

import pandas as pd

from backend_functions import gpt_ocr, idolname, timetabledata
from backend_functions import project_repository as repo
from backend_functions import time_axis as _time_axis
from backend_functions import image_processing as _imgproc
from backend_functions.ticket_scraper import get_performers_list_from_ticket_urls
from frontend_functions import timetablepicture


# ---------------------------------------------------------------------------
# OCR実行
# ---------------------------------------------------------------------------

def run_ocr_single_stage(
    mode: str,
    stage_no: int,
    user_prompt: str,
    pj_path: str,
    event_name: str,
    img_type: str,
    project_info_json: dict,
    ticket_urls: Optional[list[str]] = None,
) -> dict:
    """1ステージのOCR実行。結果JSONを保存して返す。

    mode: "normal" | "tokutenkai" | "notime"
    """
    user_prompt_full = "この画像のタイムテーブルをJSONデータとして出力して。" + user_prompt

    if mode == "normal":
        img_path = os.path.join(pj_path, event_name, img_type, f"stage_{stage_no}.png")
        return_json = gpt_ocr.getocr_fes_timetable_structured(img_path, user_prompt_full, ticket_urls=ticket_urls)
    elif mode == "tokutenkai":
        img_path = os.path.join(pj_path, event_name, img_type, f"stage_{stage_no}.png")
        return_json = gpt_ocr.getocr_fes_withtokutenkai_timetable_structured(img_path, user_prompt_full, ticket_urls=ticket_urls)
    elif mode == "notime":
        img_path = os.path.join(pj_path, event_name, img_type, f"stage_{stage_no}_addtime.png")
        if img_type == "ライブ":
            return_json = gpt_ocr.getocr_fes_timetable_notime_structured(img_path, user_prompt_full, ticket_urls=ticket_urls)
        elif img_type == "特典会" or "特典会" in img_type:
            return_json = gpt_ocr.getocr_fes_timetable_notime_structured(img_path, user_prompt_full, live=False, ticket_urls=ticket_urls)
        else:
            return_json = gpt_ocr.getocr_fes_timetable_notime_structured(img_path, user_prompt_full, ticket_urls=ticket_urls)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if "タイムテーブル" not in return_json.keys():
        return_json["タイムテーブル"] = []

    # ステージ名付与
    event_no = repo.get_event_no_by_event_name(project_info_json, event_name)
    stage_name = repo.get_stage_name(project_info_json, event_no, img_type, stage_no)
    return_json["ステージ名"] = stage_name

    # JSON保存
    json_path = os.path.join(pj_path, event_name, img_type, f"stage_{stage_no}.json")
    with open(json_path, "w", encoding="utf8") as f:
        json.dump(return_json, f, indent=4, ensure_ascii=False)

    return return_json


def run_ocr_all_stages(
    mode: str,
    user_prompt: str,
    pj_path: str,
    event_name: str,
    img_type: str,
    project_info_json: dict,
    stage_num: int,
    ticket_urls: Optional[list[str]] = None,
    ensure_addtime_fn=None,
    max_workers: int = 10,
) -> None:
    """全ステージの並列OCR実行。

    Args:
        ensure_addtime_fn: mode="notime"時にaddtime画像が無い場合に呼ぶコールバック。
            シグネチャ: fn(stage_no: int) -> None
            通常はapp.py側の detect_timeline_onlyonestage を渡す。
    """
    stage_nums = list(range(stage_num))
    pij_copy = copy.deepcopy(project_info_json)

    # addtime画像の事前生成（session_state依存のため外部コールバック経由）
    if mode == "notime" and ensure_addtime_fn is not None:
        for i in stage_nums:
            img_path = os.path.join(pj_path, event_name, img_type, f"stage_{i}_addtime.png")
            if not os.path.exists(img_path):
                ensure_addtime_fn(i)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                run_ocr_single_stage,
                mode, i, user_prompt,
                pj_path, event_name, img_type, pij_copy, ticket_urls,
            )
            for i in stage_nums
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


# ---------------------------------------------------------------------------
# グループ名補正
# ---------------------------------------------------------------------------

def correct_idol_names_single(
    stage_no: int,
    pj_path: str,
    event_name: str,
    img_type: str,
    use_confirmed_list: bool,
    confirmed_list: Optional[list[str]] = None,
    ticket_performers: Optional[list[str]] = None,
) -> None:
    """1ステージのグループ名を補正する。JSONファイルを直接更新。"""
    json_path = os.path.join(pj_path, event_name, img_type, f"stage_{stage_no}.json")
    if not os.path.exists(json_path):
        return

    with open(json_path, encoding="utf-8") as f:
        timetable_json = json.load(f)

    if "タイムテーブル" not in timetable_json or len(timetable_json["タイムテーブル"]) == 0:
        return

    for item in timetable_json["タイムテーブル"]:
        if use_confirmed_list and confirmed_list:
            item['グループ名_採用'] = idolname.get_name_by_inlist(item["グループ名"], confirmed_list)
        elif ticket_performers:
            item['グループ名_採用'] = idolname.get_name_by_levenshtein_and_vector_with_hint(item["グループ名"], ticket_performers)
        else:
            item['グループ名_採用'] = idolname.get_name_by_levenshtein_and_vector(item["グループ名"])

    with open(json_path, "w", encoding="utf8") as f:
        json.dump(timetable_json, f, indent=4, ensure_ascii=False)


def correct_idol_names_all(
    pj_path: str,
    event_name: str,
    img_type: str,
    stage_num: int,
    use_confirmed_list: bool,
    confirmed_list: Optional[list[str]] = None,
    ticket_performers: Optional[list[str]] = None,
) -> None:
    """全ステージのグループ名を補正する。"""
    for i in range(stage_num):
        correct_idol_names_single(i, pj_path, event_name, img_type, use_confirmed_list, confirmed_list, ticket_performers)


def get_idolname_confirmed_list(
    pj_path: str,
    event_name: str,
    project_info_json: dict,
) -> list[str]:
    """確定したグループ名一覧を全ステージJSONから集約して返す。"""
    idolname_confirmed_list = []
    event_no = repo.get_event_no_by_event_name(project_info_json, event_name)
    event_type_list = repo.get_event_type_list(project_info_json, event_no)
    for event_type in event_type_list:
        entry = repo.get_image_entry_by_dir_name(project_info_json, event_no, event_type)
        if entry is None:
            continue
        for stage_info in entry["stage_list"]:
            stage_no = stage_info["stage_no"]
            json_path = os.path.join(pj_path, event_name, event_type, f"stage_{stage_no}.json")
            if not os.path.exists(json_path):
                continue
            with open(json_path, encoding="utf-8") as f:
                timetable_json = json.load(f)
            if "タイムテーブル" not in timetable_json or len(timetable_json["タイムテーブル"]) == 0:
                continue
            for group_stage in timetable_json["タイムテーブル"]:
                if "グループ名_採用" in group_stage:
                    if isinstance(group_stage["グループ名_採用"], str) and len(group_stage["グループ名_採用"]) > 0:
                        idolname_confirmed_list.append(group_stage["グループ名_採用"])
    return list(set(idolname_confirmed_list))


# ---------------------------------------------------------------------------
# ステージ名
# ---------------------------------------------------------------------------

def detect_stage_names(
    pj_path: str,
    event_name: str,
    img_type: str,
    stage_num: int,
    user_prompt: str,
    project_info_json: dict,
) -> dict:
    """OCRでステージ名を読み取り、project_info_jsonを更新して返す。"""
    img_path = os.path.join(pj_path, event_name, img_type, "raw_cropped.png")
    try:
        stage_list, rule = gpt_ocr.getocr_fes_stagelist_structured(img_path, stage_num, user_prompt)
        if len(stage_list) < stage_num:
            raise IndexError
        if rule in ["数字", "アルファベット"]:
            prefix_flag = True
        else:
            prefix_flag = False
        event_no = repo.get_event_no_by_event_name(project_info_json, event_name)
        for i in range(stage_num):
            if prefix_flag:
                if "特典会" in img_type:
                    stage_name = "特典会" + str(stage_list[i])
                else:
                    stage_name = img_type + str(stage_list[i])
            else:
                stage_name = str(stage_list[i])
            entry = repo.get_image_entry_by_dir_name(project_info_json, event_no, img_type)
            if entry is None:
                raise KeyError(f"dir_name={img_type} not found in event_no={event_no}")
            entry["stage_list"][i]["stage_name"] = stage_name
    except Exception:
        print("ステージ名がうまく取得できませんでした")
    return project_info_json


def set_stage_name(
    pj_path: str,
    event_name: str,
    img_type: str,
    stage_no: int,
    stage_name: str,
    project_info_json: dict,
) -> dict:
    """ステージ名を設定し、project_info_jsonを更新して返す。"""
    event_no = repo.get_event_no_by_event_name(project_info_json, event_name)
    entry = repo.get_image_entry_by_dir_name(project_info_json, event_no, img_type)
    if entry is None:
        raise KeyError(f"dir_name={img_type} not found in event_no={event_no}")
    entry["stage_list"][stage_no]["stage_name"] = stage_name

    json_path = os.path.join(pj_path, event_name, img_type, f"stage_{stage_no}.json")
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            json_old = json.load(f)
    else:
        json_old = {}
    json_old["ステージ名"] = stage_name
    with open(json_path, "w", encoding="utf8") as f:
        json.dump(json_old, f, indent=4, ensure_ascii=False)

    return project_info_json


# ---------------------------------------------------------------------------
# データ操作
# ---------------------------------------------------------------------------

def booth_name_add_prefix(
    df_timetable: pd.DataFrame,
    stage_name: str,
) -> pd.DataFrame:
    """ブース名にステージ名を接頭辞として付加して返す。"""
    df_timetable["ブース"] = df_timetable["ブース"].apply(
        lambda x: x if x.startswith(stage_name) else stage_name + x
    )
    return df_timetable


def save_timetable_data(
    stage_no: int,
    df_timetable: pd.DataFrame,
    stage_name: str,
    pj_path: str,
    event_name: str,
    img_type: str,
    is_tokutenkai_heiki: bool,
) -> pd.DataFrame:
    """タイムテーブルDataFrameをJSONに変換して保存。更新後のDFを返す。"""
    json_path = os.path.join(pj_path, event_name, img_type, f"stage_{stage_no}.json")
    json_timetable = timetabledata.df_to_json(df_timetable)
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            json_old = json.load(f)
    else:
        json_old = {}
    json_old["タイムテーブル"] = json_timetable
    json_old["ステージ名"] = stage_name
    updated_df = timetabledata.json_to_df(json_old, is_tokutenkai_heiki)
    with open(json_path, "w", encoding="utf8") as f:
        json.dump(json_old, f, indent=4, ensure_ascii=False)
    return updated_df


# ---------------------------------------------------------------------------
# 画像生成
# ---------------------------------------------------------------------------

def generate_timetable_picture(
    stage_no: int,
    pj_path: str,
    event_name: str,
    img_type: str,
    time_match: bool,
    time_axis_converter: Optional[_time_axis.TimeAxisConverter] = None,
) -> Optional[str]:
    """読み取り結果からタイムテーブル画像を生成して保存。出力パスを返す。"""
    from datetime import datetime

    json_path = os.path.join(pj_path, event_name, img_type, f"stage_{stage_no}.json")
    if not os.path.exists(json_path):
        return None
    output_path = json_path.replace(".json", "_timetable.png")
    with open(json_path, encoding="utf-8") as f:
        json_data = json.load(f)

    if time_match and time_axis_converter is not None:
        if "タイムテーブル" not in json_data or len(json_data["タイムテーブル"]) == 0:
            return None
        time_format = "%H:%M"
        try:
            start_time = min(
                datetime.strptime(live["ライブステージ"]["from"], time_format)
                for live in json_data["タイムテーブル"]
            ).time()
        except (ValueError, KeyError):
            return None
        start_time = start_time.replace(minute=0)
        start_margin = time_axis_converter.time_to_pix(start_time)
        time_line_spacing = time_axis_converter.time_length_to_pix(30, False)
        timetable_image = timetablepicture.create_timetable_image(json_data, start_margin, time_line_spacing)
    else:
        timetable_image = timetablepicture.create_timetable_image(json_data)

    if timetable_image is not None:
        timetable_image.save(output_path)
    return output_path


# ---------------------------------------------------------------------------
# バッチ処理
# ---------------------------------------------------------------------------

def run_batch_ocr(
    event_list: list[str],
    project_info_json: dict,
    pj_path: str,
    together_targets: dict[str, bool],
    ocr_stage: bool,
    ocr_timetable: bool,
    correct: bool,
    correct_in_confirmed: bool,
    ocr_stage_prompt: str,
    ocr_user_prompt: str,
    use_ticket_urls: bool,
    ensure_addtime_fn=None,
    get_ticket_urls_fn=None,
) -> dict:
    """一括OCR実行（get_timetabledata_together相当）。

    Args:
        together_targets: {"event_name/img_type": True/False} 実行対象フラグ
        ensure_addtime_fn: addtime画像事前生成用コールバック
        get_ticket_urls_fn: イベント名からticket_urlsを取得する関数

    Returns:
        更新されたproject_info_json
    """
    for i, event_name in enumerate(event_list):
        event_no = repo.get_event_no_by_event_name(project_info_json, event_name)
        event_type_list = repo.get_event_type_list(project_info_json, event_no)

        # 確定リスト取得
        confirmed_list = None
        use_confirmed = False
        if correct_in_confirmed:
            confirmed_list = get_idolname_confirmed_list(pj_path, event_name, project_info_json)
            if len(confirmed_list) > 0:
                use_confirmed = True

        # チケットURL取得
        ticket_urls = None
        if use_ticket_urls and get_ticket_urls_fn:
            ticket_urls = get_ticket_urls_fn(event_name)
            if ticket_urls and len(ticket_urls) == 0:
                ticket_urls = None

        for event_type in event_type_list:
            target_key = f"{event_name}/{event_type}"
            if target_key not in together_targets or not together_targets[target_key]:
                continue

            timetable_info = repo.get_image_entry_by_dir_name(
                project_info_json, event_no, event_type,
            )
            if timetable_info is None:
                continue
            stage_num = timetable_info["stage_num"]

            if ocr_stage:
                project_info_json = detect_stage_names(
                    pj_path, event_name, event_type, stage_num, ocr_stage_prompt, project_info_json
                )

            if ocr_timetable:
                if timetable_info.get("kind") == "live_tokutenkai_heiki":
                    mode = "tokutenkai"
                elif timetable_info.get("format") == "ライムライト式":
                    mode = "notime"
                else:
                    mode = "normal"
                run_ocr_all_stages(
                    mode, ocr_user_prompt, pj_path, event_name, event_type,
                    project_info_json, stage_num, ticket_urls,
                    ensure_addtime_fn=ensure_addtime_fn,
                )

            if correct:
                ticket_performers = get_performers_list_from_ticket_urls(ticket_urls)
                correct_idol_names_all(
                    pj_path, event_name, event_type, stage_num,
                    use_confirmed, confirmed_list,
                    ticket_performers=ticket_performers,
                )

    return project_info_json
