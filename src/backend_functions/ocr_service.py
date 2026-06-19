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
from PIL import Image as _PILImage

from backend_functions import gpt_ocr, idolname, timetabledata
from backend_functions import project_repository as repo
from backend_functions import time_axis as _time_axis
from backend_functions import image_processing as _imgproc
from backend_functions import event_timetable_picture as _etp
from backend_functions import timetable_layout as _ttl
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


def autodetect_collab_single(
    stage_no: int,
    pj_path: str,
    event_name: str,
    img_type: str,
    clear_turn_id: bool = False,
) -> None:
    """1ステージのコラボ出番を自動検出し、JSONファイルを直接更新する。

    同じ ライブ_from を持つ行群に同一の コラボグループID を採番する
    (timetabledata.autodetect_collab_groups と同一ロジック)。
    既に コラボグループID が入っている行は変更しない (冪等)。
    """
    json_path = os.path.join(pj_path, event_name, img_type, f"stage_{stage_no}.json")
    if not os.path.exists(json_path):
        return

    with open(json_path, encoding="utf-8") as f:
        timetable_json = json.load(f)

    items = timetable_json.get("タイムテーブル")
    if not items:
        return

    # autodetect_collab_groups の採番ロジックを再利用するため、判定に必要な列だけの
    # 最小 DataFrame を json の並び順で構築する (json_to_df のフル往復による
    # ソート・duration再計算・特典会行展開といった副作用を避ける)。
    rows = []
    for item in items:
        live_stage = item.get("ライブステージ")
        live_from = live_stage.get("from", "") or "" if isinstance(live_stage, dict) else ""
        rows.append({
            "ライブ_from": live_from,
            "コラボグループID": item.get("コラボグループID"),
            "出番ID": item.get("出番ID"),
        })
    df = pd.DataFrame(rows)
    updated = timetabledata.autodetect_collab_groups(df, clear_turn_id=clear_turn_id)

    # 位置インデックスで json item へ書き戻す。
    for pos, item in enumerate(items):
        cgid = updated.iloc[pos]["コラボグループID"]
        item["コラボグループID"] = None if pd.isna(cgid) else int(cgid)
        if clear_turn_id:
            tid = updated.iloc[pos]["出番ID"]
            item["出番ID"] = None if pd.isna(tid) else int(tid)

    with open(json_path, "w", encoding="utf8") as f:
        json.dump(timetable_json, f, indent=4, ensure_ascii=False)


def autodetect_collab_all(
    pj_path: str,
    event_name: str,
    img_type: str,
    stage_num: int,
    clear_turn_id: bool = False,
) -> None:
    """全ステージのコラボ出番を自動検出する。"""
    for i in range(stage_num):
        autodetect_collab_single(i, pj_path, event_name, img_type, clear_turn_id)


def adopt_raw_idol_names_single(
    stage_no: int,
    pj_path: str,
    event_name: str,
    img_type: str,
) -> None:
    """1ステージの全アイテムについて グループ名_採用 を グループ名 (raw) で上書きする。"""
    json_path = os.path.join(pj_path, event_name, img_type, f"stage_{stage_no}.json")
    if not os.path.exists(json_path):
        return

    with open(json_path, encoding="utf-8") as f:
        timetable_json = json.load(f)

    if "タイムテーブル" not in timetable_json or len(timetable_json["タイムテーブル"]) == 0:
        return

    for item in timetable_json["タイムテーブル"]:
        raw = item.get("グループ名")
        item["グループ名_採用"] = raw if isinstance(raw, str) else ""

    with open(json_path, "w", encoding="utf8") as f:
        json.dump(timetable_json, f, indent=4, ensure_ascii=False)


def adopt_raw_idol_names_all(
    pj_path: str,
    event_name: str,
    img_type: str,
    stage_num: int,
) -> None:
    """指定 (event, img_type) の全ステージに対して raw → 採用 をコピーする。"""
    for i in range(stage_num):
        adopt_raw_idol_names_single(i, pj_path, event_name, img_type)


def adopt_raw_idol_names_event(
    pj_path: str,
    event_name: str,
    event_no: int,
    project_info_json: dict,
) -> None:
    """イベント配下の全 (event_type, stage) に対して raw → 採用 をコピーする。"""
    for event_type in repo.get_event_type_list(project_info_json, event_no):
        entry = repo.get_image_entry_by_dir_name(project_info_json, event_no, event_type)
        if entry is None:
            continue
        stage_num = int(entry.get("stage_num", 0) or 0)
        adopt_raw_idol_names_all(pj_path, event_name, event_type, stage_num)


def check_event_has_empty_adopted_idol_names(
    pj_path: str,
    event_name: str,
    event_no: int,
    project_info_json: dict,
    only_event_types: list[str] | None = None,
) -> bool:
    """イベント配下の全 stage JSON を走査し、グループ名_採用 が空/None のレコードが
    1 件でもあれば True を返す。

    `only_event_types` を指定すると、その種別のみを走査対象とする
    (採番済み種別だけのビルド可否判定に使う)。None で全種別。
    """
    event_type_list = repo.get_event_type_list(project_info_json, event_no)
    if only_event_types is not None:
        _only = set(only_event_types)
        event_type_list = [et for et in event_type_list if et in _only]
    for event_type in event_type_list:
        entry = repo.get_image_entry_by_dir_name(project_info_json, event_no, event_type)
        if entry is None:
            continue
        stage_num = int(entry.get("stage_num", 0) or 0)
        for stage_no in range(stage_num):
            json_path = os.path.join(
                pj_path, event_name, event_type, f"stage_{stage_no}.json",
            )
            if not os.path.exists(json_path):
                continue
            try:
                with open(json_path, encoding="utf-8") as f:
                    timetable_json = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            for item in timetable_json.get("タイムテーブル", []):
                v = item.get("グループ名_採用")
                if v is None or (isinstance(v, str) and v.strip() == ""):
                    return True
    return False


def get_idolname_confirmed_list(
    pj_path: str,
    event_name: str,
    project_info_json: dict,  # noqa: ARG001 (シグネチャ互換のため保持)
) -> list[str]:
    """確定したグループ名一覧を master_idolname.csv から集約して返す。

    「確定」の判定は IDマスタ確定済フラグ (3点CSVの存在) に従う
    (`repo.event_ids_assigned`)。未確定なら空リストを返し、呼び出し側で
    通常の全マスタ補正へフォールバックさせる。
    """
    if not repo.event_ids_assigned(pj_path, event_name):
        return []
    idolname_csv = os.path.join(pj_path, event_name, "master_idolname.csv")
    try:
        df = pd.read_csv(idolname_csv)
    except (OSError, ValueError, pd.errors.ParserError):
        return []
    if "グループ名_採用" not in df.columns:
        return []
    names = [str(n).strip() for n in df["グループ名_採用"].dropna() if str(n).strip()]
    return list(set(names))


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
    project_info_json: Optional[dict] = None,
    event_no: Optional[int] = None,
) -> Optional[str]:
    """読み取り結果からタイムテーブル画像を生成して保存。出力パスを返す。

    project_info_json が渡され、対応 image entry の kind == "live_tokutenkai_heiki"
    であれば、ライブ列画像と特典会列画像を別生成して横並びに合体する。
    """
    json_path = os.path.join(pj_path, event_name, img_type, f"stage_{stage_no}.json")
    if not os.path.exists(json_path):
        return None
    output_path = json_path.replace(".json", "_timetable.png")
    with open(json_path, encoding="utf-8") as f:
        json_data = json.load(f)

    # 特典会併記形式の判定
    is_heiki = False
    if project_info_json is not None:
        try:
            ev_no = event_no
            if ev_no is None:
                ev_no = repo.get_event_no_by_event_name(project_info_json, event_name)
            entry = repo.get_image_entry_by_dir_name(project_info_json, ev_no, img_type)
            is_heiki = entry is not None and entry.get("kind") == "live_tokutenkai_heiki"
        except Exception:
            is_heiki = False

    def _combine_or_passthrough(live_img, *, source_box_width_for_tk=None,
                                start_margin=None, time_line_spacing=None,
                                image_height=None):
        """is_heiki なら特典会列を生成して合体し、MAX_GEN_WIDTH を呼び出し側で判定。"""
        if not is_heiki:
            return live_img
        if live_img is None:
            return None
        tk_json = timetablepicture._build_tokutenkai_view_json(json_data)
        if not tk_json.get("タイムテーブル"):
            # 特典会が一件もなければライブ列のみ（必要なら MAX_GEN_WIDTH 縮小）
            if live_img.width > timetablepicture.MAX_GEN_WIDTH:
                scale = timetablepicture.MAX_GEN_WIDTH / live_img.width
                new_h = max(1, int(round(live_img.height * scale)))
                live_img = live_img.resize(
                    (timetablepicture.MAX_GEN_WIDTH, new_h), _PILImage.LANCZOS,
                )
            return live_img
        tk_kwargs = dict(
            box_color="lightblue",
            show_timeline_labels=False,
            apply_max_width_clamp=False,
        )
        if start_margin is not None:
            tk_kwargs["start_margin"] = start_margin
        if time_line_spacing is not None:
            tk_kwargs["time_line_spacing"] = time_line_spacing
        if image_height is not None:
            tk_kwargs["image_height"] = image_height
        if source_box_width_for_tk is not None:
            tk_kwargs["source_box_width"] = source_box_width_for_tk
        tk_img = timetablepicture.create_timetable_image(tk_json, **tk_kwargs)
        if tk_img is None:
            return live_img
        combined = timetablepicture._hstack_images(live_img, tk_img)
        if combined.width > timetablepicture.MAX_GEN_WIDTH:
            scale = timetablepicture.MAX_GEN_WIDTH / combined.width
            new_h = max(1, int(round(combined.height * scale)))
            combined = combined.resize(
                (timetablepicture.MAX_GEN_WIDTH, new_h), _PILImage.LANCZOS,
            )
        return combined

    layout: Optional[_ttl.StageLayout] = None
    stage_img_path = os.path.join(pj_path, event_name, img_type, f"stage_{stage_no}.png")
    if time_match and time_axis_converter is not None and os.path.exists(stage_img_path):
        with _PILImage.open(stage_img_path) as _src:
            source_size = _src.size
        # Y シフト込みのレイアウト (生成側・UI 側で共有)
        layout = _ttl.compute_stage_layout(json_data, source_size, time_axis_converter)
        if layout is None:
            return None

        # 併記モードでは元画像内にライブ列・特典会列が等幅で含まれている前提
        live_source_box_width = (
            layout.source_box_width / 2 if is_heiki else layout.source_box_width
        )

        live_img = timetablepicture.create_timetable_image(
            json_data,
            start_margin=layout.start_margin,
            time_line_spacing=layout.time_line_spacing,
            image_height=layout.image_height,
            source_box_width=live_source_box_width,
            apply_max_width_clamp=not is_heiki,
        )
        timetable_image = _combine_or_passthrough(
            live_img,
            source_box_width_for_tk=live_source_box_width,
            start_margin=layout.start_margin,
            time_line_spacing=layout.time_line_spacing,
            image_height=layout.image_height,
        )
    else:
        live_img = timetablepicture.create_timetable_image(
            json_data,
            apply_max_width_clamp=not is_heiki,
        )
        timetable_image = _combine_or_passthrough(live_img)

    if timetable_image is not None:
        save_kwargs = {}
        if layout is not None:
            # UI 側で source-aligned 領域に揃えるためのレイアウト情報を PNG メタデータに埋め込む。
            # クランプ前 (=計算上) の px 値を入れる。UI 側は実保存画像高との比で
            # クランプ係数を再構成する。
            from PIL.PngImagePlugin import PngInfo
            pnginfo = PngInfo()
            pnginfo.add_text("layout_version", "v2")
            pnginfo.add_text("image_height_pre_clamp", str(layout.image_height))
            pnginfo.add_text("source_aligned_height_px", str(layout.source_aligned_height_px))
            pnginfo.add_text("top_extension_px", str(layout.top_extension_px))
            pnginfo.add_text("bottom_extension_px", str(layout.bottom_extension_px))
            save_kwargs["pnginfo"] = pnginfo
        timetable_image.save(output_path, **save_kwargs)
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
    autodetect_collab: bool,
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

            # コラボ検出は ライブ_from (OCR結果) のみに依存するため最後に実行する。
            if autodetect_collab:
                autodetect_collab_all(pj_path, event_name, event_type, stage_num)

    return project_info_json
