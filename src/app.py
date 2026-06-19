import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from streamlit_cropper import st_cropper

import os

from PIL import Image
import pandas as pd
from datetime import datetime
from datetime import time as dttime
from datetime import timedelta
import json

from backend_functions import s3access, timetabledata
from backend_functions import project_repository as _repo
from backend_functions import time_axis as _time_axis
from backend_functions import image_processing as _imgproc
from backend_functions import ocr_service as _ocr
from backend_functions import output_builder as _output
from backend_functions import event_timetable_picture as _etp
from backend_functions.ticket_scraper import get_performers_list_from_ticket_urls
from html import escape as _html_escape
from frontend_functions import stage_reorder as _stage_reorder
from app_state import AppState
from workflow import ProjectWorkflow, ImageWorkflow, OcrWorkflow, OutputWorkflow

st.set_page_config(
    page_title="タイムテーブル読み取りアプリ",
    layout="wide",
    menu_items={
        'Get Help': 'https://www.google.com',
        'Report a bug': "https://www.google.com",
        'About': """
        # アイドル対バンタイムテーブル読み取りツール
        ライブ管理アプリへの搭載用のデータを生成できます。
        """
     })

DIR_PATH = os.path.dirname(__file__)
DATA_PATH = DIR_PATH +"/../data"

_project_wf = ProjectWorkflow(data_path=DATA_PATH)
_image_wf = ImageWorkflow(data_path=DATA_PATH)
_ocr_wf = OcrWorkflow(data_path=DATA_PATH)
_output_wf = OutputWorkflow(data_path=DATA_PATH)

# --- AppState 初期化 ---
if "app_state" not in st.session_state:
    s3access.get_master()
    _init_state = AppState()
    _init_state.project.project_master = pd.read_csv(os.path.join(DATA_PATH, "master", "projects_master.csv"), index_col=0)
    _init_state.project.project_master_s3 = pd.read_csv(os.path.join(DATA_PATH, "master", "projects_master_s3.csv"), index_col=0)
    st.session_state.app_state = _init_state

app_state: AppState = st.session_state.app_state

# ====================================================================
# [DEBUG-BUG] イベント切替時の state 追跡用 一時デバッグ
# 修正完了後は `grep -n "DEBUG-BUG" src/app.py` で全箇所を削除すること
# ====================================================================
st.session_state["_dbg_rerun"] = st.session_state.get("_dbg_rerun", 0) + 1
print(
    f"\n[DEBUG-BUG] ====== rerun #{st.session_state['_dbg_rerun']} ======\n"
    f"  ss.crop_tgt_event       = {st.session_state.get('crop_tgt_event')!r}\n"
    f"  app.crop.crop_tgt_event = {app_state.crop.crop_tgt_event!r}\n"
    f"  ss.crop_tgt_img_type        = {st.session_state.get('crop_tgt_img_type')!r}\n"
    f"  app.crop.crop_tgt_img_type  = {app_state.crop.crop_tgt_img_type!r}\n"
    f"  app.crop.cropped_image is None = {app_state.crop.cropped_image is None}\n"
    f"  ss.ocr_tgt_event        = {st.session_state.get('ocr_tgt_event')!r}\n"
    f"  app.ocr.ocr_tgt_event   = {app_state.ocr.ocr_tgt_event!r}\n"
    f"  ss.ocr_tgt_img_type         = {st.session_state.get('ocr_tgt_img_type')!r}\n"
    f"  app.ocr.ocr_tgt_img_type    = {app_state.ocr.ocr_tgt_img_type!r}\n"
    f"  ss.ocr_tgt_stage_num    = {st.session_state.get('ocr_tgt_stage_num')!r}",
    flush=True,
)
# ====================================================================

def _sync_to_session(state: AppState) -> None:
    """AppStateの一部をUIウィジェットが参照するsession_stateキーに同期する"""
    st.session_state.pj_name = state.project.pj_name
    st.session_state.exist_pj_name = state.project.pj_name
    st.session_state.pj_path = state.project.pj_path
    st.session_state.project_info_json = state.project.project_info_json
    st.session_state.project_master = state.project.project_master
    st.session_state.project_master_s3 = state.project.project_master_s3
    st.session_state.images_eachstage = state.crop.images_eachstage
    st.session_state.images_eachstage_bbox = state.crop.images_eachstage_bbox
    st.session_state.crop_box = state.crop.crop_box
    st.session_state.cropped_image = state.crop.cropped_image
    st.session_state.stage_line_list = state.crop.stage_line_list
    if state.crop.crop_tgt_event is not None:
        st.session_state.crop_tgt_event = state.crop.crop_tgt_event
    if state.crop.crop_tgt_img_type is not None:
        st.session_state.crop_tgt_img_type = state.crop.crop_tgt_img_type
    if state.ocr.ocr_tgt_event is not None:
        st.session_state.ocr_tgt_event = state.ocr.ocr_tgt_event
    if state.ocr.ocr_tgt_img_type is not None:
        st.session_state.ocr_tgt_img_type = state.ocr.ocr_tgt_img_type
    st.session_state.time_axis_detect = state.ocr.time_axis_detect
    st.session_state.timeline_eachstage = state.ocr.timeline_eachstage

pj_name_list = pd.concat((app_state.project.project_master_s3, app_state.project.project_master)).sort_values(by="updated_at",ascending=False).index.to_list()
pj_name_list = list(dict.fromkeys(pj_name_list))

def make_project(pj_name=None):
    if pj_name is None:
        pj_name = st.session_state.new_pj_name
    result = _project_wf.create_project(pj_name, app_state, pj_name_list)
    if not result.success:
        st.toast(result.error, icon="🚨")
    else:
        _sync_to_session(app_state)

def set_project(pj_name):
    result = _project_wf.load_project(pj_name, app_state)
    if result.success:
        _sync_to_session(app_state)

def request_delete_project():
    """削除ボタン押下時。確認ダイアログ表示状態に遷移する。"""
    st.session_state["_pending_delete_confirm"] = True
    st.session_state.pop("_delete_confirm_text", None)

def confirm_delete_project():
    """確認ダイアログで「削除を実行する」が押された時の本処理。"""
    pj_name = app_state.project.pj_name
    result = _project_wf.delete_project(pj_name, app_state)
    st.session_state.pop("_pending_delete_confirm", None)
    st.session_state.pop("_delete_confirm_text", None)
    if not result.success:
        st.toast(result.error, icon="🚨")
        return
    _sync_to_session(app_state)
    # サイドバーの選択状態と pj_name_list の再評価のため、関連 session_state を削除
    st.session_state.pop("exist_pj_name", None)
    st.toast(f"プロジェクト「{pj_name}」を削除しました", icon="🗑️")
    st.rerun()

def cancel_delete_project():
    """確認ダイアログのキャンセル。"""
    st.session_state.pop("_pending_delete_confirm", None)
    st.session_state.pop("_delete_confirm_text", None)

def save_ticket_urls():
    """チケットURL設定を保存する"""
    scope = st.session_state.ticket_url_scope

    # UIからURL文字列を収集してdict化
    if scope == "プロジェクト共通":
        urls_text = st.session_state.get("ticket_urls_project", "")
        urls_data = {"project": [u.strip() for u in urls_text.split("\n") if u.strip()]}
    else:
        urls_data = {}
        for i in range(len(app_state.project.project_info_json["event_detail"])):
            urls_text = st.session_state.get(f"ticket_urls_event_{i}", "")
            urls_data[f"event_{i}"] = [u.strip() for u in urls_text.split("\n") if u.strip()]

    result = _project_wf.save_ticket_urls(app_state, scope, urls_data)
    if result.success:
        _sync_to_session(app_state)

def get_ticket_urls_for_event(event_name: str) -> list:
    """指定イベントに紐づくチケットURLリストを取得する"""
    return _repo.get_ticket_urls_for_event(app_state.project.project_info_json, event_name)

def determine_project_setting():
    result = _project_wf.update_project_setting(
        app_state, st.session_state.event_type, int(st.session_state.event_num)
    )
    if result.success:
        _sync_to_session(app_state)

def get_image(img_path):
    return _imgproc.get_image(img_path)

_UI_IMG_TYPE_OPTIONS = (
    "ライブ",
    "特典会",
    "ライブ特典会併記",
    "両方(特典会別添え)",
    "その他(ライブ)",
    "その他(特典会)",
    "その他(ライブ特典会併記)",
)

# UI ラベル → 登録すべき (kind, dir_name) ペアのリスト。
# 「両方(特典会別添え)」は同一画像を 2 エントリとして登録する。
# 「その他(...)」は dir_name=None を返し、呼び出し側でカスタム名に差し替える。
_UI_IMG_TYPE_TO_REGISTRATIONS: dict[str, list[tuple[str, str | None]]] = {
    "ライブ":                ([("live", "ライブ")]),
    "特典会":                ([("tokutenkai", "特典会")]),
    "ライブ特典会併記":      ([("live_tokutenkai_heiki", "ライブ特典会")]),
    "両方(特典会別添え)":    ([("live", "ライブ"), ("tokutenkai", "特典会")]),
    "その他(ライブ)":        ([("live", None)]),
    "その他(特典会)":        ([("tokutenkai", None)]),
    "その他(ライブ特典会併記)": ([("live_tokutenkai_heiki", None)]),
}


def _resolve_registrations(ui_img_type: str, alternative: str) -> list[tuple[str, str]]:
    """UI ラベルと alternative 文字列から登録対象のリスト [(kind, dir_name), ...] を返す"""
    out: list[tuple[str, str]] = []
    for kind, fixed_dir_name in _UI_IMG_TYPE_TO_REGISTRATIONS[ui_img_type]:
        dir_name = fixed_dir_name if fixed_dir_name is not None else alternative
        out.append((kind, dir_name))
    return out


def _is_heiki_ui_type(ui_img_type: str) -> bool:
    return ui_img_type in ("ライブ特典会併記", "その他(ライブ特典会併記)")


def determine_timetable_image():
    file = st.session_state.uploaded_image
    file_data = file.read()
    ui_img_type = st.session_state.img_type
    alternative = st.session_state.get("img_type_alternative", "")
    event_name = st.session_state.img_event_name
    overwrite = st.session_state.pop("_img_register_force_overwrite", False)

    try:
        registrations = _resolve_registrations(ui_img_type, alternative)
    except KeyError:
        st.toast(f"不明な種別: {ui_img_type}", icon="🚨")
        return

    event_no = get_event_no_by_event_name(event_name)
    # 衝突検出 (上書きフラグがなければ確認待ちにする)
    if not overwrite:
        conflicts = [
            dir_name for _kind, dir_name in registrations
            if _repo.find_dir_name_conflict(
                app_state.project.project_info_json, event_no, dir_name,
            ) is not None
        ]
        if conflicts:
            st.session_state._img_register_pending = {
                "ui_img_type": ui_img_type,
                "alternative": alternative,
                "event_name": event_name,
                "img_format": st.session_state.img_format,
                "file_data": file_data,
                "conflict_dir_names": conflicts,
            }
            return

    img_format = None if _is_heiki_ui_type(ui_img_type) else st.session_state.img_format

    last_dir_name = None
    for kind, dir_name in registrations:
        # ↑で読んだ file_data を 2 回 register する場合のためにシークし直す必要があるが、
        # bytes はシーク不要なのでそのまま使える
        result = _project_wf.register_image(
            app_state,
            event_name=event_name,
            kind=kind,
            img_format=img_format,
            dir_name=dir_name,
            file_data=file_data,
            overwrite=overwrite,
        )
        if not result.success:
            st.toast(result.error or "登録失敗", icon="🚨")
            return
        last_dir_name = dir_name

    _sync_to_session(app_state)
    if last_dir_name is not None:
        app_state.crop.crop_tgt_event = event_name
        app_state.crop.crop_tgt_img_type = last_dir_name
        app_state.ocr.ocr_tgt_event = event_name
        app_state.ocr.ocr_tgt_img_type = last_dir_name
        st.session_state.crop_tgt_event = event_name
        st.session_state.crop_tgt_img_type = last_dir_name
        st.session_state.ocr_tgt_event = event_name
        st.session_state.ocr_tgt_img_type = last_dir_name
    st.toast("画像を登録しました", icon="✅")


def _confirm_overwrite_and_register():
    """上書き確認モーダルで「上書きする」が押されたときに呼ばれる"""
    pending = st.session_state.get("_img_register_pending")
    if pending is None:
        return
    st.session_state.img_event_name = pending["event_name"]
    st.session_state.img_type = pending["ui_img_type"]
    st.session_state.img_type_alternative = pending["alternative"]
    st.session_state.img_format = pending["img_format"]
    # uploaded_image は session 上に既にあるためそのまま使う
    st.session_state._img_register_force_overwrite = True
    st.session_state.pop("_img_register_pending", None)
    determine_timetable_image()


def _cancel_overwrite():
    st.session_state.pop("_img_register_pending", None)

def move_timetable_up_cb(event_no, image_no):
    result = _project_wf.move_timetable_up(app_state, event_no, image_no)
    if not result.success:
        st.toast(result.error or "並び替えに失敗しました", icon="🚨")
        return
    _sync_to_session(app_state)


def move_timetable_down_cb(event_no, image_no):
    result = _project_wf.move_timetable_down(app_state, event_no, image_no)
    if not result.success:
        st.toast(result.error or "並び替えに失敗しました", icon="🚨")
        return
    _sync_to_session(app_state)


def reset_timetable_order_cb(event_no):
    result = _project_wf.reset_timetable_order(app_state, event_no)
    if not result.success:
        st.toast(result.error or "並び順リセットに失敗しました", icon="🚨")
        return
    _sync_to_session(app_state)
    st.toast("並び順をリセットしました", icon="✅")


def delete_uploaded_image(img_event_no, img_type):
    result = _project_wf.delete_image(app_state, img_event_no, img_type)
    if not result.success:
        st.toast(result.error, icon="🚨")
        return
    remaining = result.data["remaining_types"]
    if len(remaining) == 0:
        app_state.crop.crop_tgt_img_type = None
        app_state.ocr.ocr_tgt_img_type = None
        st.session_state.crop_tgt_img_type = None
        st.session_state.ocr_tgt_img_type = None
    else:
        if st.session_state.get("crop_tgt_img_type") == img_type:
            app_state.crop.crop_tgt_img_type = remaining[0]
            st.session_state.crop_tgt_img_type = remaining[0]
        if st.session_state.get("ocr_tgt_img_type") == img_type:
            app_state.ocr.ocr_tgt_img_type = remaining[0]
            st.session_state.ocr_tgt_img_type = remaining[0]
    _sync_to_session(app_state)
    st.toast("画像を削除しました", icon="✅")

def get_event_name(event_no):
    return _repo.get_event_name(app_state.project.project_info_json, event_no)

def get_event_name_list():
    return _repo.get_event_name_list(app_state.project.project_info_json)

def get_event_type_list(event_no):
    return _repo.get_event_type_list(app_state.project.project_info_json, event_no)

def get_event_no_by_event_name(event_name):
    return _repo.get_event_no_by_event_name(app_state.project.project_info_json, event_name)

def get_stage_name_list(event_no,img_type):
    return _repo.get_stage_name_list(app_state.project.project_info_json, event_no, img_type)

def get_stage_name(event_no,img_type,stage_no):
    return _repo.get_stage_name(app_state.project.project_info_json, event_no, img_type, stage_no)

def set_crop_image():
    # [DEBUG-BUG] callback 発火タイミングと前後の state 値を記録
    print(
        f"[DEBUG-BUG] >>> set_crop_image() FIRED (rerun #{st.session_state.get('_dbg_rerun')})\n"
        f"    ss.crop_tgt_event    = {st.session_state.get('crop_tgt_event')!r}\n"
        f"    ss.crop_tgt_img_type = {st.session_state.get('crop_tgt_img_type')!r}\n"
        f"    app.crop.crop_tgt_event    (before) = {app_state.crop.crop_tgt_event!r}\n"
        f"    app.crop.crop_tgt_img_type (before) = {app_state.crop.crop_tgt_img_type!r}",
        flush=True,
    )
    app_state.crop.crop_tgt_event = st.session_state.crop_tgt_event
    if "crop_tgt_img_type" in st.session_state:
        app_state.crop.crop_tgt_img_type = st.session_state.crop_tgt_img_type
    app_state.crop.images_eachstage = []
    app_state.crop.images_eachstage_bbox = []
    st.session_state.images_eachstage = []
    st.session_state.images_eachstage_bbox = []
    # [DEBUG-BUG]
    print(
        f"[DEBUG-BUG] <<< set_crop_image() DONE | "
        f"app.crop.crop_tgt_event={app_state.crop.crop_tgt_event!r}, "
        f"app.crop.crop_tgt_img_type={app_state.crop.crop_tgt_img_type!r}",
        flush=True,
    )

def get_x_freq(image, stage_num):
    return _imgproc.get_x_freq(image, stage_num)

def get_xpoint(image, stage_num):
    return _imgproc.get_xpoint(image, stage_num)

def detect_stageline(image):#ステージ領域を特定する縦線を取得
    params = _imgproc.StageLineDetectParams(
        x_minlength_rate=st.session_state.x_minlength_rate,
        x_edge_threshold_1=st.session_state.x_edge_threshold_1,
        x_edge_threshold_2=st.session_state.x_edge_threshold_2,
        x_hough_threshold=st.session_state.x_hough_threshold,
        x_hough_gap=st.session_state.x_hough_gap,
        x_identify_interval=st.session_state.x_identify_interval,
    )
    _image_wf.detect_stage_lines(
        app_state, image, params,
        st.session_state.crop_tgt_event, st.session_state.crop_tgt_img_type,
    )
    _sync_to_session(app_state)

def get_image_eachstage_byocr(image, stage_num):
    return _imgproc.get_image_eachstage_byocr(image, stage_num)

def get_image_eachstage_for_croppedimage_byevenly():
    _image_wf.split_evenly(
        app_state, app_state.crop.cropped_image,
        st.session_state.devide_stage_num,
        st.session_state.crop_tgt_event, st.session_state.crop_tgt_img_type,
    )
    _sync_to_session(app_state)

def determine_image_eachstage():
    _image_wf.save_stage_images(
        app_state,
        st.session_state.crop_tgt_event, st.session_state.crop_tgt_img_type,
    )
    _sync_to_session(app_state)

def determine_image_eachstage_without_nocheck():
    accept_flags = [
        st.session_state["each_stage_accept_{}".format(i)]
        for i in range(len(app_state.crop.images_eachstage))
    ]
    _image_wf.save_stage_images(
        app_state,
        st.session_state.crop_tgt_event, st.session_state.crop_tgt_img_type,
        accept_flags=accept_flags,
    )
    _sync_to_session(app_state)

def save_time_pixel(time_start, top, height, total_duration):
    _image_wf.save_time_axis(
        app_state, time_start, top, height, total_duration,
        st.session_state.crop_tgt_event, st.session_state.crop_tgt_img_type,
    )
    _sync_to_session(app_state)

def set_ocr_image():
    # [DEBUG-BUG] callback 発火タイミングと前後の state 値を記録
    print(
        f"[DEBUG-BUG] >>> set_ocr_image() FIRED (rerun #{st.session_state.get('_dbg_rerun')})\n"
        f"    ss.ocr_tgt_event    = {st.session_state.get('ocr_tgt_event')!r}\n"
        f"    ss.ocr_tgt_img_type = {st.session_state.get('ocr_tgt_img_type')!r}\n"
        f"    app.ocr.ocr_tgt_event    (before) = {app_state.ocr.ocr_tgt_event!r}\n"
        f"    app.ocr.ocr_tgt_img_type (before) = {app_state.ocr.ocr_tgt_img_type!r}",
        flush=True,
    )
    app_state.ocr.ocr_tgt_event = st.session_state.ocr_tgt_event
    if "ocr_tgt_img_type" in st.session_state:
        app_state.ocr.ocr_tgt_img_type = st.session_state.ocr_tgt_img_type
    app_state.ocr.time_axis_detect = None
    app_state.ocr.timeline_eachstage = []
    st.session_state.time_axis_detect = None
    st.session_state.timeline_eachstage = []
    # [DEBUG-BUG]
    print(
        f"[DEBUG-BUG] <<< set_ocr_image() DONE | "
        f"app.ocr.ocr_tgt_event={app_state.ocr.ocr_tgt_event!r}, "
        f"app.ocr.ocr_tgt_img_type={app_state.ocr.ocr_tgt_img_type!r}",
        flush=True,
    )

def _get_time_axis_converter():
    """現在のOCR対象イベント/画像種別のTimeAxisConverterを取得する。未設定ならNone。"""
    event_no = get_event_no_by_event_name(st.session_state.ocr_tgt_event)
    return _time_axis.TimeAxisConverter.from_project_info(
        app_state.project.project_info_json, event_no, st.session_state.ocr_tgt_img_type
    )

def pix_to_time(pix):
    converter = _get_time_axis_converter()
    if converter is None:
        st.warning("基準時間を設定してください")
        return None
    return converter.pix_to_time(pix)

def time_to_pix(tgt_time):
    converter = _get_time_axis_converter()
    if converter is None:
        return None
    return converter.time_to_pix(tgt_time)

def time_length_to_pix(minutes, int_flag=True):
    converter = _get_time_axis_converter()
    if converter is None:
        return None
    return converter.time_length_to_pix(minutes, int_flag)

def get_timetabledata_onestage(mode, stage_no, user_prompt, ticket_urls=None):
    _ocr_wf.run_ocr_single(
        app_state, mode, stage_no, user_prompt,
        st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type,
        ticket_urls, ensure_addtime_fn=detect_timeline_onlyonestage,
    )
    output_timetable_picture_onlyonestage(stage_no)

def get_timetabledata_allstages(mode, user_prompt, ticket_urls=None):
    _ocr_wf.run_ocr_all(
        app_state, mode, user_prompt, st.session_state.ocr_tgt_stage_num,
        st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type,
        ticket_urls, ensure_addtime_fn=detect_timeline_onlyonestage,
    )
    for i in range(st.session_state.ocr_tgt_stage_num):
        _generate_stage_timetable_picture(i)
    _regenerate_current_event_type_images()

def get_timetabledata_allstages_with_ticket_urls(mode, user_prompt):
    ticket_urls = None
    if st.session_state.get("use_ticket_urls", False):
        ticket_urls = get_ticket_urls_for_event(st.session_state.ocr_tgt_event)
        if len(ticket_urls) == 0:
            ticket_urls = None
    get_timetabledata_allstages(mode, user_prompt, ticket_urls)

def get_timetabledata_onestage_with_ticket_urls(mode, stage_no, user_prompt):
    ticket_urls = None
    if st.session_state.get("use_ticket_urls", False):
        ticket_urls = get_ticket_urls_for_event(st.session_state.ocr_tgt_event)
        if len(ticket_urls) == 0:
            ticket_urls = None
    get_timetabledata_onestage(mode, stage_no, user_prompt, ticket_urls)

def detect_timeline_onlyonestage(stage_no):
    if len(st.session_state.timeline_eachstage) != st.session_state.ocr_tgt_stage_num:
        st.session_state.timeline_eachstage = [None for _ in range(st.session_state.ocr_tgt_stage_num)]

    converter = _get_time_axis_converter()
    if converter is None:
        st.warning("基準時間を設定してください")
        return

    params = _imgproc.TimelineDetectParams(
        y_minlength_rate=st.session_state.y_minlength_rate,
        y_edge_threshold_1=st.session_state.y_edge_threshold_1,
        y_edge_threshold_2=st.session_state.y_edge_threshold_2,
        y_hough_threshold=st.session_state.y_hough_threshold,
        y_hough_gap=st.session_state.y_hough_gap,
        y_identify_interval=st.session_state.y_identify_interval,
        y_ignoretime_threshold=st.session_state.y_ignoretime_threshold,
    )
    result = _ocr_wf.detect_timeline(
        app_state, stage_no,
        st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type,
        params, converter,
    )
    if result.data is not None:
        st.session_state.timeline_eachstage[stage_no - 1] = result.data.timeline_df
    _sync_to_session(app_state)

def detect_timeline_eachstage():
    if len(st.session_state.timeline_eachstage) != st.session_state.ocr_tgt_stage_num:
        st.session_state.timeline_eachstage = [None for _ in range(st.session_state.ocr_tgt_stage_num)]
    for i in range(st.session_state.ocr_tgt_stage_num):
        detect_timeline_onlyonestage(i)

def idolname_correct_onlyonestage(stage_no):
    confirmed_list = None
    if st.session_state.correct_idolname_in_confirmed_list:
        confirmed_list = get_idolname_confirmed_list()
    _ocr_wf.correct_idol_names_single(
        app_state, stage_no,
        st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type,
        st.session_state.correct_idolname_in_confirmed_list, confirmed_list,
    )
    output_timetable_picture_onlyonestage(stage_no)

def idolname_correct_eachstage():
    confirmed_list = None
    if st.session_state.correct_idolname_in_confirmed_list:
        confirmed_list = get_idolname_confirmed_list()
        if len(confirmed_list) == 0:
            st.session_state.correct_idolname_in_confirmed_list = False
            confirmed_list = None
    # チケットサイト出演者を突合候補として取得
    ticket_urls = get_ticket_urls_for_event(st.session_state.ocr_tgt_event)
    ticket_performers = get_performers_list_from_ticket_urls(ticket_urls)
    _ocr_wf.correct_idol_names_all(
        app_state,
        st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type,
        st.session_state.ocr_tgt_stage_num,
        st.session_state.correct_idolname_in_confirmed_list, confirmed_list,
        ticket_performers=ticket_performers,
    )
    for i in range(st.session_state.ocr_tgt_stage_num):
        _generate_stage_timetable_picture(i)
    _regenerate_current_event_type_images()

def adopt_raw_idolname_eachstage():
    """④: 現在の (event, img_type) 配下の全ステージで グループ名_採用 ← グループ名 (raw)。"""
    _ocr_wf.adopt_raw_idol_names_all(
        app_state,
        st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type,
        st.session_state.ocr_tgt_stage_num,
    )
    for i in range(st.session_state.ocr_tgt_stage_num):
        _generate_stage_timetable_picture(i)
    _regenerate_current_event_type_images()

def adopt_raw_idolname_event(event_name: str):
    """⑥: 指定イベント配下の全 (event_type, stage) で グループ名_採用 ← グループ名 (raw)。"""
    _ocr_wf.adopt_raw_idol_names_event(app_state, event_name)

def get_idolname_confirmed_list():
    return _ocr.get_idolname_confirmed_list(
        app_state.project.pj_path, st.session_state.ocr_tgt_event, app_state.project.project_info_json,
    )

def get_stagelist(user_prompt):
    _ocr_wf.detect_stage_names(
        app_state, user_prompt,
        st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type,
        st.session_state.ocr_tgt_stage_num,
    )
    _sync_to_session(app_state)

def set_stage_name(stage_no, stage_name):
    _ocr_wf.set_stage_name(
        app_state, stage_no, stage_name,
        st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type,
    )
    _sync_to_session(app_state)

def booth_name_add_prefix_onlyonestage(stage_no):
    result = _ocr_wf.booth_name_add_prefix(
        app_state, stage_no, st.session_state.df_timetables[stage_no],
        st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type,
    )
    st.session_state.df_timetables[stage_no] = result.data
    save_timetable_data_onlyonestage(stage_no)

def booth_name_add_prefix_eachstage():
    for i in range(st.session_state.ocr_tgt_stage_num):
        booth_name_add_prefix_onlyonestage(i)

def get_timetabledata_together():
    event_list = get_event_name_list()
    together_targets = {}
    for i, event_name in enumerate(event_list):
        event_type_list = get_event_type_list(i)
        for event_type in event_type_list:
            key = f"together_{event_name}/{event_type}"
            if key in st.session_state and st.session_state[key]:
                together_targets[f"{event_name}/{event_type}"] = True
    options = {
        "ocr_stage": st.session_state.together_ocr_stage,
        "ocr_timetable": st.session_state.together_ocr_timetable,
        "correct": st.session_state.toghther_correct,
        "correct_in_confirmed": st.session_state.correct_idolname_in_confirmed_list_toghther,
        "ocr_stage_prompt": st.session_state.ocr_stage_user_prompt_together,
        "ocr_user_prompt": st.session_state.ocr_user_prompt_together,
        "use_ticket_urls": st.session_state.get("together_use_ticket_urls", True),
    }
    _ocr_wf.run_batch(
        app_state, together_targets, options,
        ensure_addtime_fn=detect_timeline_onlyonestage,
        get_ticket_urls_fn=get_ticket_urls_for_event,
    )
    # バッチOCR後に各ステージのタイムテーブル画像を生成
    touched_event_type_pairs: set[tuple[str, str]] = set()
    touched_events: set[str] = set()
    for target_key in together_targets:
        if not together_targets[target_key]:
            continue
        event_name, img_type = target_key.split("/")
        event_no = get_event_no_by_event_name(event_name)
        timetable_info = _repo.get_image_entry_by_dir_name(
            app_state.project.project_info_json, event_no, img_type,
        )
        converter = _time_axis.TimeAxisConverter.from_project_info(
            app_state.project.project_info_json, event_no, img_type,
        )
        for stage_no in range(timetable_info["stage_num"]):
            _ocr_wf.generate_timetable_picture(
                app_state, stage_no, event_name, img_type,
                st.session_state.ocr_output_picture_time_match, converter,
            )
        touched_event_type_pairs.add((event_name, img_type))
        touched_events.add(event_name)
    # 触れた (event, img_type) ペアの種別単位画像を一括再生成し、
    # 各 event の種別横断画像は最後に 1 回だけ再生成する。
    for event_name, img_type in touched_event_type_pairs:
        _output_wf.regenerate_event_type_images(
            app_state, event_name, img_type, include_cross_type=False,
        )
    for event_name in touched_events:
        _output_wf.regenerate_event_cross_image(app_state, event_name)
    _sync_to_session(app_state)

def _save_timetable_data_onestage_inner(stage_no):
    """1ステージ保存 + 個別画像生成のみ。集約画像の再生成は呼び出し側で行う。

    保存前バリデーション (出番ID 他ステージ衝突等) で失敗した場合は
    session_state にエラーメッセージを保存し、False を返す。
    """
    stage_name = st.session_state["stage_name_stage{}".format(stage_no)]
    is_tokutenkai_heiki = st.session_state.ocr_tgt_image_info.get("kind") == "live_tokutenkai_heiki"
    result = _ocr_wf.save_timetable_data(
        app_state, stage_no, st.session_state.df_timetables[stage_no], stage_name,
        st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type,
        is_tokutenkai_heiki,
    )
    err_key = "save_timetable_error_stage{}".format(stage_no)
    if not result.success:
        st.session_state[err_key] = result.error
        return False
    st.session_state.pop(err_key, None)
    st.session_state.df_timetables[stage_no] = result.data
    set_stage_name(stage_no, stage_name)
    _generate_stage_timetable_picture(stage_no)
    return True

def save_timetable_data_onlyonestage(stage_no):
    if _save_timetable_data_onestage_inner(stage_no):
        _regenerate_current_event_type_images()
        _recommit_current_event()

def autodetect_collab_onlyonestage(stage_no):
    """コラボグループを自動検出 (同じ ライブ_from の行をグループ化) → 保存。"""
    df = st.session_state.df_timetables[stage_no]
    updated = timetabledata.autodetect_collab_groups(df)
    st.session_state.df_timetables[stage_no] = updated
    save_timetable_data_onlyonestage(stage_no)

def autodetect_collab_eachstage():
    """全ステージでコラボグループを自動検出 → 保存。"""
    clear_turn_id = bool(st.session_state.get("autodetect_collab_clear_turn_id", False))
    any_success = False
    for i in range(st.session_state.ocr_tgt_stage_num):
        df = st.session_state.df_timetables[i]
        updated = timetabledata.autodetect_collab_groups(df, clear_turn_id=clear_turn_id)
        st.session_state.df_timetables[i] = updated
        if _save_timetable_data_onestage_inner(i):
            any_success = True
    if any_success:
        _regenerate_current_event_type_images()
        # 全 stage の json 書き込み後に、イベント全体の編集後情報で1回だけ判定する。
        _recommit_current_event()

def save_timetable_data_eachstage():
    any_success = False
    for i in range(st.session_state.ocr_tgt_stage_num):
        if _save_timetable_data_onestage_inner(i):
            any_success = True
    if any_success:
        _regenerate_current_event_type_images()
        # 全 stage の json 書き込み後に、イベント全体の編集後情報で1回だけ判定する。
        _recommit_current_event()

def _recommit_current_event():
    """④保存後、編集中イベント/種別の採番状態に応じてIDマスタを再確定する。

    - 未採番種別なら workflow 側で no-op (json保存のみ)。
    - 内部整合異常はブロック (`recommit_error` に格納し st.error 表示)。
    - master差分は非ブロック通知 (`recommit_notice` に格納し st.warning 表示)。
    """
    result = _output_wf.recommit_event_ids_after_edit(
        app_state,
        st.session_state.ocr_tgt_event,
        st.session_state.ocr_tgt_img_type,
    )
    if not result.success:
        st.session_state["recommit_error"] = result.error
        st.session_state.pop("recommit_notice", None)
        return
    st.session_state.pop("recommit_error", None)
    notices = (result.data or {}).get("notices") or []
    if notices:
        st.session_state["recommit_notice"] = notices
    else:
        st.session_state.pop("recommit_notice", None)
    _sync_to_session(app_state)

def _generate_stage_timetable_picture(stage_no):
    """個別ステージのタイテ画像のみ生成 (集約画像の再生成は呼び出し側で行う)。"""
    converter = _get_time_axis_converter()
    _ocr_wf.generate_timetable_picture(
        app_state, stage_no,
        st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type,
        st.session_state.ocr_output_picture_time_match, converter,
    )

def _regenerate_current_event_type_images():
    """現在編集中の (event, img_type) の集約画像を再生成。"""
    _output_wf.regenerate_event_type_images(
        app_state,
        st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type,
    )

def output_timetable_picture_onlyonestage(stage_no):
    _generate_stage_timetable_picture(stage_no)
    _regenerate_current_event_type_images()

def output_timetable_picture_eachstage():
    for i in range(st.session_state.ocr_tgt_stage_num):
        _generate_stage_timetable_picture(i)
    _regenerate_current_event_type_images()

def output_difference_image(new_image):
    result = _image_wf.output_difference_image(
        app_state, new_image,
        st.session_state.diff_tgt_event, st.session_state.diff_tgt_img_type,
    )
    if not result.success:
        st.warning(result.error)
        st.session_state._diff_result = None
        return
    st.session_state._diff_result = result.data

def replace_stage_images_from_new_raw(new_image):#新しい画像から既存のbbox座標でステージ画像を切り出して置き換える
    result = _image_wf.replace_stage_images_from_new_raw(
        app_state, new_image,
        st.session_state.diff_tgt_event, st.session_state.diff_tgt_img_type,
    )
    if not result.success:
        st.warning(result.error)
        return
    _sync_to_session(app_state)

def save_to_s3():
    _output_wf.save_to_s3(app_state)

def output_data_for_stella():
    _output_wf.export_excel(app_state)

def listup_new_idolname():
    _output_wf.listup_new_idolname(app_state)

def update_master_idolname(df_new_idolname):
    _output_wf.update_idol_name_master(app_state, df_new_idolname)


# ===========================================================================
# レンダー関数
# ===========================================================================

def render_project_setting():
    """①プロジェクトの設定"""
    st.markdown("#### ①プロジェクトの設定")
    if app_state.project.pj_name is None:
        st.info("サイドバーからプロジェクトを選択または作成してください")
        return
    st.text("選択中のプロジェクト："+app_state.project.pj_name)
    col_project_setting = st.columns(2)
    with col_project_setting[0]:
        st.info(
"""イベント形式を選択してください。
・フェス：複数のステージが同時進行するイベント
・対バン：ステージが一つのみのイベント
※Stella向けなので、一旦フェスのみに対応して作っています
""")
        event_type_options = ("対バン", "フェス")
        event_type_idx = 0
        if app_state.project.event_type in event_type_options:
            event_type_idx = event_type_options.index(app_state.project.event_type)
        st.radio("イベント形式", event_type_options, index=event_type_idx, key="event_type", horizontal=True)
    with col_project_setting[1]:
        st.info(
"""イベント数を入力してください。
イベント数とは、チケットの販売単位に相当する概念です。
複数日にわたって開かれるフェスや、昼夜で別イベントとして開かれる対バンの場合、その数を記入してください。
""")
        st.number_input("イベント数", min_value=1, step=1, value=app_state.project.event_num, key="event_num")
    if st.session_state.event_type is not None and st.session_state.event_num is not None:
        st.button(label="プロジェクト設定を反映する", on_click=determine_project_setting, type="primary")

    # チケットURL設定
    with st.expander("チケットURL設定（出演者情報取得用）"):
        st.info(
"""チケットサイトのURLを登録すると、タイムテーブル読み取り時に出演者リストを自動取得してOCRの精度向上に活用できます。
・対応サイト：TicketDive, LivePocket, tiget など
・紐づけ単位を「イベントごと」に変更すると、プロジェクト共通のURL設定はクリアされます（逆も同様）
・複数URLを登録する場合は、1行に1URLずつ入力してください
""")
        current_scope = "project"
        current_project_urls = []
        if "ticket_urls" in app_state.project.project_info_json:
            current_scope = app_state.project.project_info_json["ticket_urls"].get("scope", "project")
            current_project_urls = app_state.project.project_info_json["ticket_urls"].get("urls", [])

        scope_options = ("プロジェクト共通", "イベントごと")
        default_scope_index = 0 if current_scope == "project" else 1
        st.radio("チケットURLの紐づけ単位", scope_options, index=default_scope_index, key="ticket_url_scope", horizontal=True)

        if st.session_state.ticket_url_scope == "プロジェクト共通":
            default_urls = "\n".join(current_project_urls) if current_scope == "project" else ""
            st.text_area("チケットサイトURL（1行に1つずつ）", value=default_urls, key="ticket_urls_project", height=100)
        else:
            event_list = get_event_name_list()
            for i, event_name in enumerate(event_list):
                event_urls = []
                if current_scope == "event":
                    event_data = app_state.project.project_info_json["event_detail"][i]
                    event_urls = event_data.get("ticket_urls", [])
                default_urls = "\n".join(event_urls)
                st.text_area(f"{event_name} のチケットサイトURL", value=default_urls, key=f"ticket_urls_event_{i}", height=80)

        st.button("チケットURL設定を保存", on_click=save_ticket_urls, type="secondary")

    # ---- プロジェクト削除 ----
    st.divider()
    with st.expander("⚠️ プロジェクトの削除", expanded=False):
        st.warning(
            f"「{app_state.project.pj_name}」のローカル・S3 上の全データと "
            "プロジェクトマスタの該当行を完全に削除します。**この操作は取り消せません。**"
        )
        if app_state.output.edits:
            st.info("⑥出力確認・編集タブで編集中の作業コピーがあります。"
                    "保存または破棄してから削除してください。")
            st.button(
                "このプロジェクトを削除",
                key="btn_request_delete_project",
                disabled=True,
            )
        elif not st.session_state.get("_pending_delete_confirm"):
            st.button(
                "このプロジェクトを削除",
                key="btn_request_delete_project",
                on_click=request_delete_project,
                type="secondary",
            )
        else:
            st.error(f"本当に「{app_state.project.pj_name}」を削除しますか?")
            st.text_input(
                f"確認のため、プロジェクト名「{app_state.project.pj_name}」を入力してください",
                key="_delete_confirm_text",
            )
            confirm_cols = st.columns(2)
            with confirm_cols[0]:
                st.button(
                    "削除を実行する",
                    key="btn_confirm_delete_project",
                    on_click=confirm_delete_project,
                    type="primary",
                    disabled=(
                        st.session_state.get("_delete_confirm_text")
                        != app_state.project.pj_name
                    ),
                )
            with confirm_cols[1]:
                st.button(
                    "キャンセル",
                    key="btn_cancel_delete_project",
                    on_click=cancel_delete_project,
                )


def render_image_upload():
    """②タイムテーブル画像の登録"""
    st.markdown("#### ②タイムテーブル画像の登録")
    event_list = get_event_name_list()
    pij = app_state.project.project_info_json

    tab_labels = [
        f"{name}（{len(pij['event_detail'][i]['timetables'])}枚）"
        for i, name in enumerate(event_list)
    ]
    tabs = st.tabs(tab_labels)
    for i, (event_name, tab) in enumerate(zip(event_list, tabs)):
        with tab:
            entries = _repo.get_image_entry_list(pij, i)
            header_col = st.columns((5, 1))
            with header_col[0]:
                st.markdown(f"###### 登録済み画像：{len(entries)}枚")
            with header_col[1]:
                st.button(
                    "並び順をリセット",
                    key=f"reset_order_event_{i}",
                    on_click=reset_timetable_order_cb, args=(i,),
                    help="このイベントの並び順をデフォルトに戻す",
                    disabled=(len(entries) < 2),
                )
            if not entries:
                st.info("画像がまだ登録されていません")
                continue
            col_all_files = st.columns(len(entries))
            for entry_idx, entry in enumerate(entries):
                img_type = entry["dir_name"]
                image_no = entry["image_no"]
                img_path = os.path.join(app_state.project.pj_path, event_name, img_type, "raw.png")
                if not os.path.exists(img_path):
                    continue
                with col_all_files[entry_idx]:
                    col_uploaded_image = st.columns(3)
                    with col_uploaded_image[0]:
                        st.button(
                            "↑", key=f"move_up_uploaded_image_{i}_{image_no}",
                            on_click=move_timetable_up_cb, args=(i, image_no),
                            disabled=(entry_idx == 0),
                        )
                    with col_uploaded_image[1]:
                        st.button(
                            "↓", key=f"move_down_uploaded_image_{i}_{image_no}",
                            on_click=move_timetable_down_cb, args=(i, image_no),
                            disabled=(entry_idx == len(entries) - 1),
                        )
                    with col_uploaded_image[2]:
                        st.button(
                            "削除", key=f"delete_uploaded_image_{i}_{image_no}",
                            on_click=delete_uploaded_image, args=(i, img_type),
                        )
                    st.markdown(f"- {img_type}")
                    image = get_image(img_path)
                    st.image(image)

    col_file_uploader = st.columns((3, 1))
    with col_file_uploader[0]:
        st.file_uploader("読み取りたいタイムテーブル画像をアップロードしてください。"
                                , type=["jpg", "jpeg", "png", "jfif"]
                                , key="uploaded_image")
        if st.session_state.uploaded_image is not None:
            st.image(
                st.session_state.uploaded_image,
                use_container_width=True
            )
    with col_file_uploader[1]:
        if st.session_state.uploaded_image is not None:
            st.info(
    """画像が何の情報を示しているか入力してください。
    ・イベント：複数イベントがある場合（2days開催など）、どのイベントの情報であるか
    ・種別：画像に載っている情報の種類
    ・形式：タイムテーブルの形式（通常：時間が各枠に記載 / ライムライト式：時間軸が枠外に記載）
    """)
            event_list = get_event_name_list()
            st.selectbox("イベント", event_list,index=0, key="img_event_name")
            st.radio("種別", _UI_IMG_TYPE_OPTIONS, key="img_type", horizontal=True)
            st.text_input("その他の種別 (フォルダ名)", key="img_type_alternative")
            heiki_selected = _is_heiki_ui_type(st.session_state.img_type)
            st.radio("形式", ("通常", "ライムライト式"), key="img_format", horizontal=True, disabled=heiki_selected)
            other_selected = st.session_state.img_type in (
                "その他(ライブ)", "その他(特典会)", "その他(ライブ特典会併記)",
            )
            alt_empty = not st.session_state.get("img_type_alternative", "")
            register_disabled = other_selected and alt_empty
            st.button(
                label="画像を登録する",
                on_click=determine_timetable_image,
                type="primary",
                disabled=register_disabled,
            )

            # 上書き確認モーダル: determine_timetable_image が衝突を検出すると pending を立てる
            pending = st.session_state.get("_img_register_pending")
            if pending is not None:
                conflict_list = "、".join(pending["conflict_dir_names"])
                st.warning(
                    f"このイベントには既に「{conflict_list}」の画像が登録されています。\n"
                    "上書きしますか？（既存の画像・分割結果・OCR結果は失われます）"
                )
                col_confirm = st.columns(2)
                with col_confirm[0]:
                    st.button(
                        "上書きする", on_click=_confirm_overwrite_and_register,
                        key="btn_confirm_overwrite", type="primary",
                    )
                with col_confirm[1]:
                    st.button(
                        "キャンセル", on_click=_cancel_overwrite,
                        key="btn_cancel_overwrite",
                    )


def render_crop_section():
    """③タイムテーブル画像の切り取り"""
    st.markdown("#### ③タイムテーブル画像の切り取り")
    st.info(
"""元のタイムテーブル画像に加工などを施して、読み取りが行えるための準備をします。
大きく分けて3つのステップで準備します。
- （ⅰ）まず必要最小限の領域に画像を切り出します。
- （ⅱ）次に複数あるであろうステージごとに画像を分割します。
- （ⅲ）最後に基準となる時間軸の位置を指定します（推奨だがオプション）。
""")

    event_list = get_event_name_list()
    default_crop_event_idx = 0
    if app_state.crop.crop_tgt_event in event_list:
        default_crop_event_idx = event_list.index(app_state.crop.crop_tgt_event)
    st.selectbox("イベント", event_list, index=default_crop_event_idx, key="crop_tgt_event", on_change=set_crop_image)
    crop_tgt_event_no = get_event_no_by_event_name(st.session_state.crop_tgt_event)
    event_type_list = get_event_type_list(crop_tgt_event_no)
    if len(event_type_list) == 0:
        st.warning("画像を登録するか他のイベントを選択してください")
        return
    default_crop_type_idx = 0
    if app_state.crop.crop_tgt_img_type in event_type_list:
        default_crop_type_idx = event_type_list.index(app_state.crop.crop_tgt_img_type)
    st.selectbox("種別", event_type_list, index=default_crop_type_idx, key="crop_tgt_img_type", on_change=set_crop_image)
    img_path = os.path.join(app_state.project.pj_path, st.session_state.crop_tgt_event, st.session_state.crop_tgt_img_type, "raw.png")
    # ==================== [DEBUG-BUG] ③ 画像読込直前 ====================
    _dbg_img_exists = os.path.exists(img_path)
    st.caption(
        f"🔬 [③DEBUG] rerun #{st.session_state.get('_dbg_rerun')} | "
        f"ss.crop_tgt_event={st.session_state.crop_tgt_event!r} | "
        f"app.crop.crop_tgt_event={app_state.crop.crop_tgt_event!r} | "
        f"ss.crop_tgt_img_type={st.session_state.get('crop_tgt_img_type')!r} | "
        f"img_path exists={_dbg_img_exists} | "
        f"app.crop.cropped_image is None={app_state.crop.cropped_image is None} | "
        f"len(app.crop.images_eachstage)={len(app_state.crop.images_eachstage)}"
    )
    print(
        f"[DEBUG-BUG] ③ render_crop_section after selectbox: "
        f"ss.crop_tgt_event={st.session_state.crop_tgt_event!r}, "
        f"app.crop.crop_tgt_event={app_state.crop.crop_tgt_event!r}, "
        f"img_path={img_path!r}, exists={_dbg_img_exists}",
        flush=True,
    )
    if not _dbg_img_exists:
        print(f"[DEBUG-BUG] ③ img_path DOES NOT EXIST → 以下の描画ブロックを SKIP するため、前回の描画が画面上に残ります", flush=True)
    # =====================================================================
    if os.path.exists(img_path):
        image = Image.open(img_path)
        image_info = _repo.get_image_entry_by_dir_name(
            app_state.project.project_info_json,
            crop_tgt_event_no,
            st.session_state.crop_tgt_img_type,
        )

        with st.container():# タイムテーブルに関係する領域を切り出す
            st.markdown("""###### ③（ⅰ）タイムテーブルに関係する領域を切り出す""")
            st.info(
"""まず、タイムテーブルに関係する領域の切り出しを行います。
必要十分な領域を指定してください。
- 上下については、出演枠だけでなくステージ名の部分まで含めることを推奨します（ステージ名自動読み取りが可能になります）。
- 左右については、時間軸の部分まで含めることを推奨します（時間軸の設定で使います）。
    - 特にライムライト形式のタイテでは出演枠から時間が分からないため、時間軸の部分が必須になります。
- また、このあとのステップでステージの均等割を行う場合には、それに適した領域に切り出してください。
""")
            st.markdown("""###### 切り出し設定""")
            col_cropimage_first_setting = st.columns(2)
            with col_cropimage_first_setting[0]:
                box_color = st.color_picker(label="Box Color", value='#0000FF', key="crop_box_coler")
            with col_cropimage_first_setting[1]:
                stroke_width = st.number_input(label="Box Thickness", value=1, step=1, key="crop_stroke_width")
            crop_box = st_cropper(image,
                            box_color=box_color,
                            stroke_width=stroke_width,
                            return_type="box")
            crop_left, crop_top, crop_width, crop_height = tuple(map(int, crop_box.values()))
            app_state.crop.cropped_image = image.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))
            app_state.crop.crop_box = {"left": crop_left, "top": crop_top, "width": crop_width, "height": crop_height}
            st.markdown("""###### 切り出し結果""")
            col_cropimage_first_result = st.columns([1,2,1])
            with col_cropimage_first_result[1]:
                st.image(app_state.crop.cropped_image,use_container_width=True)

        with st.container():# ステージごとにタイムテーブル領域を分割する
            st.markdown("""###### ③（ⅱ）ステージごとにタイムテーブル領域を分割する""")
            st.info(
"""次に、切り出した領域を複数あるステージごとに分割します。
分割の方法は2種類あり、「縦線検出による分割」と「均等幅での分割」です。
各ステージの幅が概ね均等であり、また間に時間軸なども等しく入っているor入っていない場合は、「均等幅での分割」を推奨します。
しかしそうでない場合などには、「縦線を機械的に検出してそれに基づいた分割」を行います。
手間はかかりますが、縦線検出法ではステージに関係ある情報の部分のみを取得できるというメリットがあります。
画像を見て、どちらのパターンがふさわしいかを判断して作業を行ってください。
いずれかの方法で分割を行い、採用不採用を決めたら確定ボタンを押してステージごとの画像を確定してください。
""")
            split_button = st.columns(2)
            with split_button[0]:
                st.info(
"""###### 縦線検出による分割
- エッジ検出とハフ変換という手法によって、縦に長いラインを画像から発見し、そこで画像を分割します。
- 分割の精度は100%ではありませんが、アルゴリズムのパラメータ変更である程度対応できるようになります。
- パラメータ変更でも対応できなかったときの場合に、手動で分割併合が出来る機能を実装予定です。
- 分割された領域の中には、ステージに該当しない余白領域などが存在する可能性がありますので、採用不採用をチェックしてください。
""")
                st.button("縦ラインの自動抽出による分割",on_click=detect_stageline,args=(app_state.crop.cropped_image,),type="primary",help="縦に長いラインを機械的に画像から発見し、そこで画像を分割します。うまく行かない場合はパラメータを変更してください。")
                with st.expander("縦ライン抽出のパラメータ"):
                    st.info(
"""- 「エッジ抽出の閾値」は、特定の色と色の間の線が検出できていない場合に下げると良いです。劇的に下げてもOKです。
- 「ハフ変換の閾値」は、全体的に検出できていない場合に下げると良いです。
- 「抽出したエッジを伸ばす際の閾値」は、検出された線が途切れ途切れになったり短かったりする場合に下げると良いです。「エッジ抽出の閾値」以下に設定してください。
- 「ハフ変換で許容する線分の飛び」は、どの程度の破線を許容するかのパラメータです。連続した文字などの上に不要な線分が検出されてしまっている場合に上げると良いです。
- 「抽出線分の長さ（元画像の縦に対する比率）」は、短い線分を除くためのパラメータです。不要な短い線分が検出されている場合に上げると良いです。
- 「同一視する線分の許容誤差幅」は、線分の位置がほぼ同じ場合にそれらをまとめ上げる範囲についてのパラメータです。
""")
                    st.slider('エッジ抽出の閾値', value=285, min_value=1, max_value=500, step=1, key="x_edge_threshold_2", on_change=detect_stageline,args=(st.session_state.cropped_image,))
                    st.slider('抽出したエッジを伸ばす際の閾値', value=130, min_value=1, max_value=500, step=1, key="x_edge_threshold_1", on_change=detect_stageline,args=(st.session_state.cropped_image,))
                    st.slider('ハフ変換の閾値', value=100, min_value=1, max_value=500, step=1, key="x_hough_threshold", on_change=detect_stageline,args=(st.session_state.cropped_image,))
                    st.slider('ハフ変換で許容する線分の飛び', value=1, min_value=0, max_value=100, step=1, key="x_hough_gap", on_change=detect_stageline,args=(st.session_state.cropped_image,))
                    st.slider('抽出線分の長さ（元画像の縦に対する比率）', value=0.01, min_value=0.0, max_value=1.0, step=0.001, key="x_minlength_rate", on_change=detect_stageline,args=(st.session_state.cropped_image,))
                    st.slider('同一視する線分の許容誤差幅', value=5, min_value=1, max_value=30, step=1, key="x_identify_interval", on_change=detect_stageline,args=(st.session_state.cropped_image,))
            with split_button[1]:
                st.info(
"""###### 均等幅での分割
- ステージ数を入力し、横幅が均等になるように画像をステージの数だけ分割します。
- 均等にステージが表現されている場合はこちらの手法が安定します。
- 分割位置がきれいでない場合、一つ前の工程での領域抽出を修正すると良いかもしれません。
- 前後5%ぶん余裕をとっているので、隣り合う画像に重複する部分が残ります。
""")
                st.button("画像を均等な横幅で指定数に分割",on_click=get_image_eachstage_for_croppedimage_byevenly,type="primary")
                st.number_input("ステージ数",1,step=1,key="devide_stage_num")
            # 縦線抽出結果の表示
            if app_state.crop.annotated_image is not None:
                st.image(app_state.crop.annotated_image, caption="縦線抽出結果")
            determine_image_eachstage_button_area_1 = st.container()
            if len(app_state.crop.images_eachstage) > 0 or "images_eachstage" in st.session_state:
                stage_num = len(app_state.crop.images_eachstage)
                if stage_num >0:
                    each_stage_area = st.container(height=500)
                    determine_image_eachstage_button_area_2 = st.container()
                    with determine_image_eachstage_button_area_1:
                        st.button("ステージごとの画像を確定",on_click=determine_image_eachstage_without_nocheck,type="primary",key="determine_image_eachstage_button_1")
                    with each_stage_area:
                        col_eachstage = st.columns(stage_num)
                        for i in range(stage_num):
                            with col_eachstage[i]:
                                st.checkbox("採用",key="each_stage_accept_{}".format(i),value=True)
                                st.image(app_state.crop.images_eachstage[i])
                    with determine_image_eachstage_button_area_2:
                        st.button("ステージごとの画像を確定",on_click=determine_image_eachstage_without_nocheck,key="determine_image_eachstage_button_2")

        with st.container():# 基準時間を設定する
            st.markdown("""###### ③（ⅲ）基準時間を設定する（オプション）""")
            st.info(
"""最後に、画像のどのラインが何時に相当するかを指定します。
画像内の枠線をドラッグして、上端と下端が基準となる時刻になるよう調整してください。
ライムライト式のタイテでは、この情報を元に時刻の読み取りを行います。
それ以外の形式のタイテでも、読み取り結果と元画像の比較において時間軸を合わせるのに使います。
ただし画像によっては時間軸が線形でない（同じ時間の長さが同じ縦幅ではない）場合があるので、その場合は指定しないでください。
- 横幅は特に関係ないので、上端と下端だけを調整してください。
- 上下幅は出演枠が存在している範囲でセットしてください。余白部分は微妙に間隔が違ったりします。
- 上端と下端に対応する時刻を入力して、画像上のどの位置がどの時刻かを対応づけてください。
- 最後に「時間軸の設定を確定する」ボタンを押してください。
""")
            cropped_img_path = os.path.join(app_state.project.pj_path, st.session_state.crop_tgt_event, st.session_state.crop_tgt_img_type, "raw_cropped.png")
            if os.path.exists(cropped_img_path):
                cropped_image = Image.open(cropped_img_path)
                col_timeaxis_setting = st.columns(2)
                with col_timeaxis_setting[0]:
                    box_color = st.color_picker(label="Box Color", value='#0000FF', key="timeaxis_box_coler")
                with col_timeaxis_setting[1]:
                    stroke_width = st.number_input(label="Box Thickness", value=1, step=1, key="timeaxs_stroke_width")
                rect = st_cropper(cropped_image,
                                box_color=box_color,
                                stroke_width=stroke_width,
                                return_type="box")
                left, top, width, height = tuple(map(int, rect.values()))
                col_timeaxis_bottom = st.columns([1,2])
                with col_timeaxis_bottom[0]:
                    st.image(cropped_image.crop((left, top, left+width, top+height)),use_container_width=True)
                with col_timeaxis_bottom[1]:
                    st.slider('上端時間', value=dttime(10), key="time_start", step=timedelta(minutes=5))
                    st.slider('下端時間', value=dttime(20), key="time_finish", step=timedelta(minutes=5))
                    total_duration = (datetime(2024,1,1,st.session_state.time_finish.hour,st.session_state.time_finish.minute)-datetime(2024,1,1,st.session_state.time_start.hour,st.session_state.time_start.minute)).seconds/60
                    if st.session_state.time_finish.hour < st.session_state.time_start.hour:
                        st.warning("上端時間より下端時間が早くなっています。深夜イベントなどで日を跨ぐ場合はそのまま実行可能ですが、そうでない場合は修正してください。")
                    if "time_pixel" not in image_info:
                        st.warning("基準時間を確定してください")
                        st.button("基準時間を確定する",on_click=save_time_pixel, args=(st.session_state.time_start, top, height, total_duration), type="primary")
                    else:
                        st.button("基準時間を更新する",on_click=save_time_pixel, args=(st.session_state.time_start, top, height, total_duration))


def render_ocr_section():
    """④タイムテーブル画像の読み取り"""
    st.markdown("#### ④タイムテーブル画像の読み取り")
    # 保存時のID再確定の結果 (ブロックエラー / 要確認通知) を表示する。
    _recommit_error = st.session_state.get("recommit_error")
    if _recommit_error:
        st.error(_recommit_error)
    _recommit_notice = st.session_state.get("recommit_notice")
    if _recommit_notice:
        st.warning(
            "IDマスタを更新しました。以下は前回確定時からの変更点です（要確認）:\n"
            + "\n".join(f"  - {m}" for m in _recommit_notice),
        )
    event_list = get_event_name_list()
    with st.expander("まとめて読み取りを実施"):
        st.info("""ライムライト式の画像の時刻推定も同時にまとめて行えますが、画像によって時刻の基準位置が違う場合が多いので、できるだけ先に時刻の読み取りだけ別途それぞれの画像で行うことを強く推奨します。""")
        col_toghether_ocr = st.columns([1,1,3])
        with col_toghether_ocr[0]:
            for i,event_name in enumerate(event_list):
                event_type_list_tmp = get_event_type_list(i)
                for event_type in event_type_list_tmp:
                    image_info_tmp = _repo.get_image_entry_by_dir_name(
                        app_state.project.project_info_json, i, event_type,
                    )
                    stage_num_tmp = image_info_tmp["stage_num"]
                    if stage_num_tmp>0:
                        st.checkbox(event_list[i]+"/"+event_type,key="together_"+event_list[i]+"/"+event_type,value=True)
        with col_toghether_ocr[1]:
            st.checkbox("ステージ名の読み取り",key="together_ocr_stage",value=True)
            st.checkbox("タイムテーブルの読み取り",key="together_ocr_timetable",value=True)
            st.checkbox("グループ名の修正（マスタ参照）",key="toghther_correct",value=True)
            st.checkbox("既に確定したタイテ種別で採用したグループ名一覧の中からグループ名を選ぶ",key="correct_idolname_in_confirmed_list_toghther",help="""例えばライブのタイムテーブルを先に作成し、後から特典会のタイムテーブルを作成する際に、ライブのタイムテーブルデータで「グループ名_採用」に入力したグループ名の一覧を候補として、特典会のタイムテーブルデータでも「グループ名_採用」への修正を行うことが出来ます。
この処理はイベントごとに切り分けて行われるため、day1はday1の中で候補を用意してグループ名を修正し、day2はday2でまた別になります。
ライブと特典会、あるいは他の種別についてはどのような順番でもよく、「全種別を通じて既に『グループ名_修正』に入力されているグループ一覧」が候補になります。
どの種別においても一つもグループ名を確定していない場合は、通常通り全グループリストから出力されます。""")
            st.checkbox("チケットサイトの出演者情報を読み取りに使用する",key="together_use_ticket_urls",value=True,help="各イベントに登録されたチケットURLから出演者情報を取得し、読み取り精度を向上させます")
        with col_toghether_ocr[2]:
            st.text_input("ステージ名読み取りの追加指示",key="ocr_stage_user_prompt_together")
            st.text_input("タイムテーブル読み取りの追加指示",key="ocr_user_prompt_together")
            st.button("まとめて実行",on_click=get_timetabledata_together)

    # session_state に既存値があると selectbox の index= と二重指定になり警告が出るため、
    # 事前に session_state を初期化し index= は渡さない
    if st.session_state.get("ocr_tgt_event") not in event_list:
        st.session_state.ocr_tgt_event = (
            app_state.ocr.ocr_tgt_event
            if app_state.ocr.ocr_tgt_event in event_list
            else event_list[0]
        )
    st.selectbox("イベント", event_list, key="ocr_tgt_event", on_change=set_ocr_image)
    ocr_tgt_event_no = get_event_no_by_event_name(st.session_state.ocr_tgt_event)
    event_type_list = get_event_type_list(ocr_tgt_event_no)
    if len(event_type_list) == 0:
        st.warning("画像を登録するか他のイベントを選択してください")
        return
    if st.session_state.get("ocr_tgt_img_type") not in event_type_list:
        st.session_state.ocr_tgt_img_type = (
            app_state.ocr.ocr_tgt_img_type
            if app_state.ocr.ocr_tgt_img_type in event_type_list
            else event_type_list[0]
        )
    st.selectbox("種別", event_type_list, key="ocr_tgt_img_type", on_change=set_ocr_image)

    # ==================== [DEBUG-BUG] ④ event/type selectbox 直後 ====================
    st.caption(
        f"🔬 [④DEBUG-A] rerun #{st.session_state.get('_dbg_rerun')} | "
        f"ss.ocr_tgt_event={st.session_state.ocr_tgt_event!r} (no={ocr_tgt_event_no}) | "
        f"app.ocr.ocr_tgt_event={app_state.ocr.ocr_tgt_event!r} | "
        f"ss.ocr_tgt_img_type={st.session_state.get('ocr_tgt_img_type')!r} | "
        f"event_type_list={event_type_list}"
    )
    print(
        f"[DEBUG-BUG] ④ after selectbox: "
        f"ss.ocr_tgt_event={st.session_state.ocr_tgt_event!r}, no={ocr_tgt_event_no}, "
        f"ss.ocr_tgt_img_type={st.session_state.get('ocr_tgt_img_type')!r}, "
        f"event_type_list={event_type_list}",
        flush=True,
    )
    # ==================================================================================

    # チケットサイト情報使用オプション
    ticket_urls = get_ticket_urls_for_event(st.session_state.ocr_tgt_event)
    if len(ticket_urls) > 0:
        st.checkbox("チケットサイトの出演者情報を読み取りに使用する", value=True, key="use_ticket_urls",
                    help=f"登録済みURL: {', '.join(ticket_urls)}")
    else:
        st.session_state.use_ticket_urls = False

    img_path = os.path.join(app_state.project.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "raw.png")
    # ==================== [DEBUG-BUG] ④ img_path 存在チェック前 ====================
    _dbg_img_exists = os.path.exists(img_path)
    print(
        f"[DEBUG-BUG] ④ img_path={img_path!r}, exists={_dbg_img_exists}",
        flush=True,
    )
    if not _dbg_img_exists:
        st.error(
            f"🔬 [④DEBUG] img_path が存在しません: `{img_path}`\n\n"
            "→ 以下の描画ブロックを SKIP するため、画面上には前回 rerun の描画が残ります"
        )
        print(f"[DEBUG-BUG] ④ img_path DOES NOT EXIST → 以下の描画ブロックを SKIP", flush=True)
    # =================================================================================
    if os.path.exists(img_path):
        image = Image.open(img_path)
        st.session_state.ocr_tgt_image_info = _repo.get_image_entry_by_dir_name(
            app_state.project.project_info_json,
            ocr_tgt_event_no,
            st.session_state.ocr_tgt_img_type,
        )
        st.session_state.ocr_tgt_stage_num = st.session_state.ocr_tgt_image_info["stage_num"]
        # ==================== [DEBUG-BUG] ④ stage_num set 後 ====================
        st.caption(
            f"🔬 [④DEBUG-B] ocr_tgt_image_info.stage_num={st.session_state.ocr_tgt_stage_num} | "
            f"image_info keys={list((st.session_state.ocr_tgt_image_info or {}).keys())}"
        )
        print(
            f"[DEBUG-BUG] ④ stage_num={st.session_state.ocr_tgt_stage_num}, "
            f"image_info dir_name={st.session_state.ocr_tgt_image_info.get('dir_name') if st.session_state.ocr_tgt_image_info else None}",
            flush=True,
        )
        # =========================================================================
        if st.session_state.ocr_tgt_stage_num<=0:
            print(f"[DEBUG-BUG] ④ EARLY RETURN at stage_num<=0 → 前回描画が残る恐れあり", flush=True)
            st.warning("各ステージの画像を確定してください")
            return

        with st.container():# 各ステージ情報の読み取り
            st.markdown("""###### 各ステージ情報の読み取り""")
            stage_name_list = get_stage_name_list(ocr_tgt_event_no,st.session_state.ocr_tgt_img_type)
            # ==================== [DEBUG-BUG] ④ stage_name_list 取得後 ====================
            st.caption(
                f"🔬 [④DEBUG-C] stage_name_list={stage_name_list} (len={len(stage_name_list)})"
            )
            print(
                f"[DEBUG-BUG] ④ stage_name_list={stage_name_list} (len={len(stage_name_list)})",
                flush=True,
            )
            # ================================================================================
            if st.session_state.ocr_tgt_image_info.get("format")=="ライムライト式":
                st.info(
"""アーティストごとに時間が書いていないタイムテーブルの場合、横線を検出して時間を推定します。
その後推定した時刻を画像に書き込み、そこからタイムテーブル情報を生成します。
画像の色合いによって横線の検出の難易度が変わるので、パラメータをいじってください。
- 「エッジ抽出の閾値」は、特定の色と色の間の線が検出できていない場合に下げると良いです。劇的に下げてもOKです。
- 「ハフ変換の閾値」は、全体的に検出できていない場合に下げると良いです。
- 「抽出したエッジを伸ばす際の閾値」は、検出された線が途切れ途切れになったり短かったりする場合に下げると良いです。
- 「ハフ変換で許容する線分の飛び」は、どの程度の破線を許容するかのパラメータです。連続した文字などの上に不要な線分が検出されてしまっている場合に上げると良いです。
- 「抽出線分の長さ（元画像の縦に対する比率）」は、短い線分を除くためのパラメータです。不要な短い線分が検出されている場合に上げると良いです。
- 「同一視する線分の許容誤差幅」は、線分の位置がほぼ同じ場合にそれらをまとめ上げる範囲についてのパラメータです。
- 「無視する時間幅」パラメータは、その値（分）以下の長さしかない横線と横線の間には時刻を書き込まないようにするものです

※「全ステージの画像を読み取りを実施」をいきなり押すと、先に横線の時刻の読み取りを裏で実行してからタイムテーブル情報の読み取りを行います
※「全ステージの横線の時刻の読み取りを実施」を押して、正しく時刻が取得できたことを画像で確認してから、「全ステージの画像を読み取りを実施」を押すことを推奨します
""")
                st.text_input("ステージ名読み取りの追加指示",key="ocr_stage_user_prompt")
                st.button("ステージ名の読み取りを実施",on_click=get_stagelist,args=(st.session_state.ocr_stage_user_prompt,),type="primary")
                st.button("全ステージの横線の時刻の読み取りを実施",on_click=detect_timeline_eachstage,type="primary")
                st.text_input("タイムテーブル読み取りの追加指示",key="ocr_user_prompt")
                st.button("全ステージのタイムテーブルを読み取りを実施", on_click=get_timetabledata_allstages_with_ticket_urls, args=("notime", st.session_state.ocr_user_prompt), type="primary")
                with st.expander("時間ライン抽出のパラメータ"):
                    st.slider('無視する時間幅（分）（以下）', value=5, min_value=0, max_value=60, step=5, key="y_ignoretime_threshold")
                    st.slider('エッジ抽出の閾値', value=150, min_value=1, max_value=500, step=1, key="y_edge_threshold_2")
                    st.slider('抽出したエッジを伸ばす際の閾値（エッジ抽出の閾値以下にする）', value=80, min_value=1, max_value=500, step=1, key="y_edge_threshold_1")
                    st.slider('ハフ変換の閾値', value=60, min_value=1, max_value=500, step=1, key="y_hough_threshold")
                    st.slider('ハフ変換で許容する線分の飛び', value=1, min_value=0, max_value=100, step=1, key="y_hough_gap")
                    st.slider('抽出線分の長さ（元画像の縦に対する比率）', value=0.05, min_value=0.0, max_value=1.0, step=0.01, key="y_minlength_rate")
                    st.slider('同一視する線分の許容誤差幅', value=5, min_value=1, max_value=30, step=1, key="y_identify_interval")
                tmp_timeline = st.container()#暫定
            elif st.session_state.ocr_tgt_image_info.get("kind")=="live_tokutenkai_heiki":
                st.text_input("ステージ名読み取りの追加指示",key="ocr_stage_user_prompt")
                st.button("ステージ名の読み取りを実施",on_click=get_stagelist,args=(st.session_state.ocr_stage_user_prompt,),type="primary")
                st.text_input("タイムテーブル読み取りの追加指示",key="ocr_user_prompt")
                st.button("全ステージのタイムテーブルをそれぞれ読み取り実施", on_click=get_timetabledata_allstages_with_ticket_urls, args=("tokutenkai", st.session_state.ocr_user_prompt), type="primary")
                st.button("全ステージの特典会ブース名にステージ名を接頭辞として付与する",on_click=booth_name_add_prefix_eachstage)
            else:
                st.text_input("ステージ名読み取りの追加指示",key="ocr_stage_user_prompt")
                st.button("ステージ名の読み取りを実施",on_click=get_stagelist,args=(st.session_state.ocr_stage_user_prompt,),type="primary")
                st.text_input("タイムテーブル読み取りの追加指示",key="ocr_user_prompt")
                st.button("全ステージのタイムテーブルをそれぞれ読み取り実施", on_click=get_timetabledata_allstages_with_ticket_urls, args=("normal", st.session_state.ocr_user_prompt), type="primary")
            #個別ステージ
            with st.container():#表示設定
                if st.session_state.ocr_tgt_image_info.get("format")=="ライムライト式":
                    img_path_tmp = os.path.join(app_state.project.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "stage_{}_addtime.png".format(0))
                    if not os.path.exists(img_path_tmp):
                        img_path_tmp = os.path.join(app_state.project.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "stage_{}.png".format(0))
                else:
                    img_path_tmp = os.path.join(app_state.project.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "stage_{}.png".format(0))
                if os.path.exists(img_path_tmp):
                    image_tmp = Image.open(img_path_tmp)
                img_path_tmp_output = os.path.join(app_state.project.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "stage_{}_timetable.png".format(0))
                if os.path.exists(img_path_tmp_output):
                    image_tmp_output = Image.open(img_path_tmp_output)

                ocr_eachimage_width_default = 40
                st.slider('タイテ元画像の表示幅（%）', value=ocr_eachimage_width_default, min_value=1, max_value=99, step=1, key="ocr_eachimage_width")
                ocr_show_setting_col = st.columns([1,1,1,1,1])
                with ocr_show_setting_col[0]:
                    st.checkbox("画像の縦スクロール表示", value=True, key="ocr_eachimage_scroll")
                with ocr_show_setting_col[1]:
                    st.checkbox("読み取り結果の画像の時間軸を元画像に合わせる", value=app_state.ocr.ocr_output_picture_time_match, key="ocr_output_picture_time_match",on_change=output_timetable_picture_eachstage , help="""「時間軸の設定」で指定したラインに合わせて画像を生成します。
このチェックボックスのオンオフまたはステージの編集結果の保存で画像が更新されます。""")

            stage_tabs = st.tabs(stage_name_list)
            st.session_state.df_timetables = []
            for i in range(st.session_state.ocr_tgt_stage_num):
                with stage_tabs[i]:
                    if st.session_state.ocr_tgt_image_info.get("format")=="ライムライト式":
                        img_path = os.path.join(app_state.project.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "stage_{}_addtime.png".format(i))
                        if not os.path.exists(img_path):
                            img_path = os.path.join(app_state.project.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "stage_{}.png".format(i))
                    else:
                        img_path = os.path.join(app_state.project.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "stage_{}.png".format(i))
                    if os.path.exists(img_path):
                        image = Image.open(img_path)
                    img_path_output = os.path.join(app_state.project.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "stage_{}_timetable.png".format(i))
                    if os.path.exists(img_path_output):
                        image_output = Image.open(img_path_output)
                    ocr_col = st.columns([st.session_state.ocr_eachimage_width,100-st.session_state.ocr_eachimage_width])
                    with ocr_col[0]:
                        if st.session_state.ocr_output_picture_time_match:
                            if os.path.exists(img_path_output):
                                st.markdown("""###### タイテ画像+読み取り結果画像""")
                                # PNG メタデータ (layout_version=v2) がある場合は source-aligned 領域基準で配置。
                                # 旧フォーマット PNG (v2 メタなし) は従来の uniform リサイズで並べる。
                                if image_output.info.get("layout_version") == "v2":
                                    image_height_pc = int(image_output.info["image_height_pre_clamp"])
                                    src_aligned_pc = int(image_output.info["source_aligned_height_px"])
                                    top_ext_pc = int(image_output.info["top_extension_px"])
                                    bot_ext_pc = int(image_output.info["bottom_extension_px"])
                                    clamp_scale = image_output.height / max(1, image_height_pc)
                                    src_aligned_in_saved = src_aligned_pc * clamp_scale
                                    # source-aligned 領域を image.height に揃えるスケール
                                    scale = image.height / max(1, src_aligned_in_saved)
                                    new_out_w = max(1, round(image_output.width * scale))
                                    new_out_h = max(1, round(image_output.height * scale))
                                    image_output = image_output.resize((new_out_w, new_out_h), Image.LANCZOS)
                                    scaled_top = round(top_ext_pc * clamp_scale * scale)
                                    scaled_bot = round(bot_ext_pc * clamp_scale * scale)
                                    new_width = image.width + image_output.width
                                    new_height = image.height + scaled_top + scaled_bot
                                    new_image = Image.new("RGB", (new_width, new_height), "white")
                                    new_image.paste(image, (0, scaled_top))
                                    new_image.paste(image_output, (image.width, 0))
                                else:
                                    # 旧フォーマット: 出力画像を image.height に uniform リサイズして並べる
                                    if image_output.height != image.height:
                                        resized_w = round(image_output.width * image.height / image_output.height)
                                        image_output = image_output.resize((resized_w, image.height), Image.LANCZOS)
                                    new_width = image.width + image_output.width
                                    new_height = image.height
                                    new_image = Image.new("RGB", (new_width, new_height), "white")
                                    new_image.paste(image, (0, 0))
                                    new_image.paste(image_output, (image.width, 0))
                                if st.session_state.ocr_eachimage_scroll:
                                    with st.container(height=800):
                                        st.image(new_image,use_container_width=True)
                                else:
                                    with st.container():
                                        st.image(new_image,use_container_width=True)
                            else:
                                ocr_image_col = st.columns(2)
                                with ocr_image_col[0]:
                                    st.markdown("""###### タイテ画像""")
                                    if st.session_state.ocr_eachimage_scroll:
                                        with st.container(height=700):
                                            if os.path.exists(img_path):
                                                st.image(image,use_container_width=True)
                                    else:
                                        with st.container():
                                            if os.path.exists(img_path):
                                                st.image(image,use_container_width=True)
                                with ocr_image_col[1]:
                                    st.markdown("""###### 読み取り結果画像""")
                                    st.warning("各ステージの読み取りを行うとタイムテーブル画像が表示されます")
                        else:
                            ocr_image_col = st.columns(2)
                            with ocr_image_col[0]:
                                st.markdown("""###### タイテ画像""")
                                if st.session_state.ocr_eachimage_scroll:
                                    with st.container(height=700):
                                        if os.path.exists(img_path):
                                            st.image(image,use_container_width=True)
                                else:
                                    with st.container():
                                        if os.path.exists(img_path):
                                            st.image(image,use_container_width=True)
                            with ocr_image_col[1]:
                                st.markdown("""###### 読み取り結果画像""")
                                if os.path.exists(img_path_output):
                                    if st.session_state.ocr_eachimage_scroll:
                                        with st.container(height=700):
                                            st.image(image_output,use_container_width=True)
                                    else:
                                        with st.container():
                                            st.image(image_output,use_container_width=True)
                                else:
                                    st.warning("各ステージの読み取りを行うとタイムテーブル画像が表示されます")

                    with ocr_col[1]:
                        st.markdown("""###### 読み取り結果""")
                        stage_name = get_stage_name(ocr_tgt_event_no,st.session_state.ocr_tgt_img_type,i)
                        st.text_input("ステージ名",value=stage_name,key="stage_name_stage{}".format(i))
                        if st.session_state.ocr_tgt_image_info.get("format")=="ライムライト式":
                            st.button("このステージの横線の時刻の読み取りを実施",on_click=detect_timeline_onlyonestage,args=(i,),key="button_ocr_timeline_stage{}".format(i))
                            st.text_input("タイムテーブル読み取りの追加指示",key="ocr_user_prompt_stage{}".format(i))
                            st.button("このステージのタイムテーブルの読み取りを実施", on_click=get_timetabledata_onestage_with_ticket_urls, args=("notime", i, st.session_state["ocr_user_prompt_stage{}".format(i)]), key="button_ocr_stage{}".format(i))
                        elif st.session_state.ocr_tgt_image_info.get("kind")=="live_tokutenkai_heiki":
                            st.text_input("タイムテーブル読み取りの追加指示",key="ocr_user_prompt_stage{}".format(i))
                            st.button("このステージのタイムテーブルの読み取りを実施",on_click=get_timetabledata_onestage_with_ticket_urls,args=("tokutenkai",i,st.session_state["ocr_user_prompt_stage{}".format(i)]),key="button_ocr_stage{}".format(i))
                            st.button("特典会ブース名にステージ名を接頭辞として付与する",on_click=booth_name_add_prefix_onlyonestage,args=(i,),key="booth_name_add_prefix_stage{}".format(i))
                        else:
                            st.text_input("タイムテーブル読み取りの追加指示",key="ocr_user_prompt_stage{}".format(i))
                            st.button("このステージのタイムテーブルの読み取りを実施",on_click=get_timetabledata_onestage_with_ticket_urls,args=("normal",i,st.session_state["ocr_user_prompt_stage{}".format(i)]),key="button_ocr_stage{}".format(i))
                        json_path = os.path.join(app_state.project.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "stage_{}.json".format(i))
                        if os.path.exists(json_path):
                            with open(json_path, encoding="utf-8") as f:
                                return_json = json.load(f)
                            tokutenkai = st.session_state.ocr_tgt_image_info.get("kind")=="live_tokutenkai_heiki"
                            # ID系カラムの表示は種別(img_type)単位の採番状態で判定する。
                            # 例: ライブ採番後に追加された特典会は、特典会をリビルドする
                            #     までID未採番として扱い、ID列を出さない。
                            id_assigned = _repo.img_type_ids_assigned(
                                app_state.project.project_info_json,
                                ocr_tgt_event_no, st.session_state.ocr_tgt_img_type,
                            )
                            if not return_json.get("タイムテーブル"):
                                # OCR結果が空のステージ: data_editor がカラム無しで
                                # 行追加不能にならないよう、正規カラムの空dfを供給する。
                                return_json_df = timetabledata.empty_timetable_df(tokutenkai, id_assigned)
                            else:
                                return_json_df = timetabledata.json_to_df(
                                    return_json, tokutenkai=tokutenkai, id_assigned=id_assigned,
                                )
                            edited_df = st.data_editor(return_json_df, key="timetabledata_stage{}".format(i), num_rows="dynamic")

                            edited_df["ステージ名"]=stage_name
                            st.session_state.df_timetables.append(edited_df)

                            if st.button("このステージのグループ名を修正（マスタ参照）",key="button_correct_idolname_stage{}_confirm".format(i)):
                                st.warning('「グループ名_採用」が上書きされます。本当に処理を実行しますか？')
                                st.button("OK",on_click=idolname_correct_onlyonestage,args=(i,),key="button_correct_idolname_stage{}".format(i))
                            st.button(
                                "コラボ出番を自動検出",
                                on_click=autodetect_collab_onlyonestage,
                                args=(i,),
                                key="button_autodetect_collab_stage{}".format(i),
                                help="同じ ライブ_from を持つ行を1つのコラボ出番として `コラボグループID` を採番します。既に値が入っている行は変更しません。",
                            )
                            st.button("このステージの編集結果を保存",on_click=save_timetable_data_onlyonestage,args=(i,),key="button_save_timetable_stage{}".format(i))
                            save_err = st.session_state.get("save_timetable_error_stage{}".format(i))
                            if save_err:
                                st.error(save_err)

            st.checkbox("既に確定したタイテ種別で採用したグループ名一覧の中からグループ名を選ぶ", value=app_state.ocr.correct_idolname_in_confirmed_list, key="correct_idolname_in_confirmed_list",help="""例えばライブのタイムテーブルを先に作成し、後から特典会のタイムテーブルを作成する際に、ライブのタイムテーブルデータで「グループ名_採用」に入力したグループ名の一覧を候補として、特典会のタイムテーブルデータでも「グループ名_採用」への修正を行うことが出来ます。
この処理はイベントごとに切り分けて行われるため、day1はday1の中で候補を用意してグループ名を修正し、day2はday2でまた別になります。
ライブと特典会、あるいは他の種別についてはどのような順番でもよく、IDマスタを確定済みのイベントでは「master_idolname」に登録されたグループ一覧が候補になります。
IDマスタが未確定（master_stage / master_idolname / turn_id_data の3点が未保存）の場合は、通常通り全グループリストから出力されます。""")
            if st.button("全ステージのグループ名を修正（マスタ参照）",key="button_correct_idolname_confirm"):
                st.warning('「グループ名_採用」が上書きされます。本当に処理を実行しますか？')
                st.button('OK', on_click=idolname_correct_eachstage,key="button_correct_idolname")
            if st.button("読み取ったグループ名をそのまま採用する",key="button_adopt_raw_idolname_confirm",
                         help="全ステージの「グループ名_採用」を「グループ名(OCR)」と同じ値で上書きします。"):
                st.warning('「グループ名_採用」が「グループ名(OCR)」と同じ値で上書きされます。本当に処理を実行しますか？')
                st.button('OK', on_click=adopt_raw_idolname_eachstage,key="button_adopt_raw_idolname")
            st.checkbox(
                "コラボ判定された行の既存の出番IDをクリアする",
                value=False,
                key="autodetect_collab_clear_turn_id",
                help="今回の実行で新たに `コラボグループID` を採番した行について、既に入力されている `出番ID` を空にします。既存IDが入っている行 (=自動検出対象外の行) には影響しません。",
            )
            st.button(
                "全ステージのコラボ出番を自動検出",
                on_click=autodetect_collab_eachstage,
                key="button_autodetect_collab",
                help="全ステージで、同じ ライブ_from を持つ行を1つのコラボ出番として `コラボグループID` を採番します。既に値が入っている行は変更しません。",
            )
            st.button("全ステージの編集結果を保存",on_click=save_timetable_data_eachstage,key="button_save_timetable")
            # 各ステージの保存エラーを集約表示
            _save_errors = [
                (i, st.session_state.get("save_timetable_error_stage{}".format(i)))
                for i in range(st.session_state.ocr_tgt_stage_num)
            ]
            _save_errors = [(i, e) for i, e in _save_errors if e]
            if _save_errors:
                st.error(
                    "以下のステージで保存に失敗しました:\n"
                    + "\n".join(f"--- stage_{i} ---\n{e}" for i, e in _save_errors)
                )


def render_comparison_section():
    """⑤タイムテーブル画像の追加・変更"""
    st.markdown("""#### ⑤タイムテーブル画像の追加・変更
- 読み取りを行った後にタイムテーブルが変更になった場合に、画像の変更点のみを読み取って修正してくれる機能をいつか実装します""")
    event_list = get_event_name_list()
    timetable_compare_setting_col = st.columns(2)
    timetable_compare_col = st.columns(2)
    with timetable_compare_setting_col[0]:
        st.file_uploader("読み取りたいタイムテーブル画像をアップロードしてください。"
                                , type=["jpg", "jpeg", "png", "jfif"]
                                , key="uploaded_image_updated")
    with timetable_compare_col[0]:
        if st.session_state.uploaded_image_updated is not None:
            st.caption("新規画像")
            st.image(
                st.session_state.uploaded_image_updated,
                use_container_width=True
            )
    with timetable_compare_setting_col[1]:
        st.selectbox("イベント", event_list,index=0,key="diff_tgt_event")
        diff_tgt_event_no = get_event_no_by_event_name(st.session_state.diff_tgt_event)
        event_type_list = get_event_type_list(diff_tgt_event_no)
        if len(event_type_list) == 0:
            st.warning("画像を登録するか他のイベントを選択してください")
            return
        st.selectbox("種別", event_type_list,index=0,key="diff_tgt_img_type")
        st.button("差分画像を出力する",on_click=output_difference_image,args=(st.session_state.uploaded_image_updated,))
        if st.session_state.uploaded_image_updated is not None:
            st.button("画像を置き換える",on_click=replace_stage_images_from_new_raw,args=(st.session_state.uploaded_image_updated,))
    diff_result = st.session_state.get("_diff_result")
    with timetable_compare_col[1]:
        if diff_result is not None:
            st.caption("既存画像")
            st.image(diff_result["old_image"], use_container_width=True)
    if diff_result is not None:
        _render_diff_result(diff_result)


# ステージを「差分あり」とみなす変化ピクセル割合(%)の初期しきい値。
# 差分は稀である前提のもと、ごく小さな変化以外は拾えるよう低めに設定。
_DIFF_STAGE_THRESHOLD_DEFAULT = 0.1


def _render_diff_result(diff_result):
    """全体差分画像・ステージ別差分サマリ・差分ありステージの横並びを描画する。"""
    st.divider()
    st.caption("差分（既存 vs 新規）")
    st.image(diff_result["diff_image"], use_container_width=True)

    stages = diff_result.get("stages") or []
    if not stages:
        st.info("ステージのbbox情報がないため、ステージ別の差分判定はできません。")
        return

    st.markdown("##### ステージ別の差分")
    threshold = st.slider(
        "差分とみなすしきい値（変化ピクセル割合 %）",
        min_value=0.0, max_value=5.0,
        value=_DIFF_STAGE_THRESHOLD_DEFAULT, step=0.01,
        format="%.2f",
        key="_diff_stage_threshold",
        help="この割合を超えて変化したステージを「差分あり」と判定します。値を下げるほど敏感になります。",
    )

    df_summary = pd.DataFrame([
        {
            "ステージ": f"#{s['stage_no']} {s['stage_name']}",
            "変化割合(%)": round(s["score"], 2),
            "判定": "⚠️ 差分あり" if s["score"] >= threshold else "✅ 差分なし",
        }
        for s in stages
    ])
    st.dataframe(df_summary, hide_index=True, use_container_width=True)

    diff_stages = [s for s in stages if s["score"] >= threshold]
    if not diff_stages:
        st.success("しきい値を超える差分のあるステージはありません。")
        return

    st.markdown(f"**差分のありそうなステージ（{len(diff_stages)}件）**")
    st.caption("差分のあった領域を矩形＋ハッチングで表示しています。")
    for s in diff_stages:
        n_regions = len(s.get("regions") or [])
        st.markdown(f"###### #{s['stage_no']} {s['stage_name']}（{s['score']:.2f}% / 差分領域 {n_regions}件）")
        cols = st.columns(2)
        with cols[0]:
            st.caption("既存")
            st.image(s["old_overlay"], use_container_width=True)
        with cols[1]:
            st.caption("新規")
            st.image(s["new_overlay"], use_container_width=True)


def _sync_group_select_from_id(state_key_id, state_key_name, id_list, name_list):
    chosen_id = st.session_state[state_key_id]
    st.session_state[state_key_name] = name_list[id_list.index(chosen_id)]


def _sync_group_select_from_name(state_key_id, state_key_name, id_list, name_list):
    chosen_name = st.session_state[state_key_name]
    st.session_state[state_key_id] = id_list[name_list.index(chosen_name)]


def _render_group_appearance_selector(event_name: str, df_appearances):
    """グループID／グループ名の連動 selectbox を描画し、選択グループの出番一覧を表示する。"""
    df_groups = (
        df_appearances[["グループID", "グループ名"]]
        .drop_duplicates()
        .sort_values("グループID")
    )
    id_list = df_groups["グループID"].tolist()
    name_list = df_groups["グループ名"].tolist()

    state_key_id = f"appearance_sel_id_{event_name}"
    state_key_name = f"appearance_sel_name_{event_name}"

    # 初期化 / 失効した state の補正
    if st.session_state.get(state_key_id) not in id_list:
        st.session_state[state_key_id] = id_list[0]
        st.session_state[state_key_name] = name_list[0]

    sel_cols = st.columns(2)
    with sel_cols[0]:
        st.selectbox(
            "グループID", id_list,
            key=state_key_id,
            on_change=_sync_group_select_from_id,
            args=(state_key_id, state_key_name, id_list, name_list),
        )
    with sel_cols[1]:
        st.selectbox(
            "グループ名", name_list,
            key=state_key_name,
            on_change=_sync_group_select_from_name,
            args=(state_key_id, state_key_name, id_list, name_list),
        )

    selected_id = st.session_state[state_key_id]
    df_filtered = df_appearances[df_appearances["グループID"] == selected_id]
    st.dataframe(df_filtered)


def _render_event_aggregations(event_name: str, data):
    """集計表示 (view / editor 共通)"""
    st.divider()
    st.markdown("##### 集計情報")
    aggr_cols = st.columns(2)
    with aggr_cols[0]:
        st.markdown("**出演枠時間の頻度分布**")
        st.dataframe(data["duration_distribution"])
    with aggr_cols[1]:
        st.markdown("**グループ別出演回数**")
        sort_mode = st.radio(
            "並び順",
            ["合計回数(降順)", "グループID(昇順)"],
            key=f"group_count_sort_{event_name}",
            horizontal=True,
        )
        df_group = data["group_count"]
        if sort_mode == "合計回数(降順)":
            df_group = df_group.sort_values(
                ["合計", "ライブ出演回数"], ascending=[False, False],
            )
        st.dataframe(df_group)

    st.markdown("**出演時間が重複しているグループ**")
    df_overlap = data["overlap_alerts"]
    if len(df_overlap) == 0:
        st.success("重複なし")
    else:
        st.warning(f"{len(df_overlap)}件の重複があります")
        st.dataframe(df_overlap)

    st.markdown("**特定グループの出番一覧**")
    df_appearances = data["group_appearances"]
    if len(df_appearances) == 0:
        st.info("出演グループがありません")
    else:
        _render_group_appearance_selector(event_name, df_appearances)


def _resolve_stage_color(color_name: str, preset: dict[str, tuple[str, str]]) -> tuple[str, str] | None:
    """`カラー名` を (bg, fg) に解決する。プリセット名 / `#bg-#fg` カスタム両対応。"""
    if not isinstance(color_name, str) or color_name == "":
        return None
    if color_name in preset:
        return preset[color_name]
    # カスタム指定: `#RRGGBB-#RRGGBB`
    if color_name.startswith("#") and "-" in color_name:
        parts = color_name.split("-")
        if len(parts) == 2 and parts[0].startswith("#") and parts[1].startswith("#"):
            return parts[0], parts[1]
    return None


def _style_stage_color_column(df_stage):
    """ステージマスタDF の `カラー名` セルに実背景色を当てた Styler を返す。"""
    if "カラー名" not in df_stage.columns:
        return df_stage
    preset = _etp.load_color_preset()

    def _cell_style(v):
        resolved = _resolve_stage_color(v, preset)
        if resolved is None:
            return ""
        bg, fg = resolved
        return f"background-color: {bg}; color: {fg}; text-align: center;"

    return df_stage.style.map(_cell_style, subset=["カラー名"])


_CUSTOM_COLOR_KEY = "自由に色を設定 (カスタム)"


def _render_stage_color_editor(event_name: str) -> None:
    """ステージカラー編集セクション (編集モード内、ステージマスタ data_editor の下)。

    各ステージごとに 1 行:
        ステージ名 | プルダウン (プリセット27色 + 「自由に色を設定」) | カスタム時のカラーピッカー | 実色プレビュー

    「自由に色を設定」を選ぶと背景色/文字色の `st.color_picker` が出現し、
    結果は `#bg-#fg` 形式で `カラー名` に保存される。
    プリセット選択時は色名 (`stage-red` 等) がそのまま保存される。
    """
    edits = app_state.output.edits.get(event_name)
    if edits is None or edits.get("stage") is None:
        return
    df_stage = edits["stage"]
    if "カラー名" not in df_stage.columns:
        df_stage["カラー名"] = ""

    preset = _etp.load_color_preset()
    preset_names = list(_output._PRESET_COLOR_NAMES)
    options = preset_names + [_CUSTOM_COLOR_KEY]

    st.markdown("###### ステージカラー (編集中)")
    st.caption(
        "各ステージのカラーを選択します。プリセット27色のいずれかを選ぶか、"
        "「自由に色を設定」でカラーピッカーから背景色・文字色を直接指定できます。"
    )

    for sid, row in df_stage.sort_values("表示順").iterrows():
        current = str(row.get("カラー名") or "").strip()
        custom_parsed = _etp._parse_custom_color(current) if current else None
        is_custom = custom_parsed is not None and current not in preset

        # selectbox の初期 index 決定
        if is_custom or not current:
            default_idx = options.index(_CUSTOM_COLOR_KEY) if is_custom else 0
        elif current in preset_names:
            default_idx = preset_names.index(current)
        else:
            # プリセットでもカスタムでもない不明値 → 強制的にカスタムに寄せる
            default_idx = options.index(_CUSTOM_COLOR_KEY)

        cols = st.columns([2, 3, 2, 2, 2])
        with cols[0]:
            st.write(f"**#{int(sid)}** {row.get('ステージ名', '')}")
        with cols[1]:
            chosen = st.selectbox(
                "カラー", options=options,
                index=default_idx,
                key=f"stage_color_sel_{event_name}_{sid}",
                label_visibility="collapsed",
            )

        if chosen == _CUSTOM_COLOR_KEY:
            # デフォルト bg/fg を決定
            if custom_parsed is not None:
                default_bg, default_fg = custom_parsed
            elif current in preset:
                default_bg, default_fg = preset[current]
            else:
                default_bg, default_fg = "#EA749E", "#FFFFFF"
            with cols[2]:
                bg = st.color_picker(
                    "背景色", value=default_bg,
                    key=f"stage_color_bg_{event_name}_{sid}",
                    label_visibility="collapsed",
                )
            with cols[3]:
                fg = st.color_picker(
                    "文字色", value=default_fg,
                    key=f"stage_color_fg_{event_name}_{sid}",
                    label_visibility="collapsed",
                )
            new_value = f"{bg}-{fg}"
            preview_label = f"{bg} / {fg}"
        else:
            # プリセット選択
            new_value = chosen
            bg, fg = preset.get(chosen, ("#FFFFFF", "#000000"))
            preview_label = chosen
            with cols[2]:
                st.empty()
            with cols[3]:
                st.empty()

        with cols[4]:
            st.markdown(
                f'<div style="background:{bg};color:{fg};padding:6px 12px;'
                f'border-radius:4px;text-align:center;border:1px solid #ccc;'
                f'font-size:12px;line-height:1.4;">{_html_escape(preview_label)}</div>',
                unsafe_allow_html=True,
            )

        # edits["stage"] (= df_stage) を直接更新
        df_stage.at[sid, "カラー名"] = new_value


def _render_event_output_view(event_name: str, data):
    """編集モードOFF: 読み取り専用表示

    `data["stage"]` / `data["live"]` は非活性化を含む master 全行を保持しているため、
    表示時に `非活性化フラグ` でフィルタする。
    """
    df_stage_master = data["stage"]
    disabled_stage_ids: set = set()
    if "非活性化フラグ" in df_stage_master.columns:
        disabled_stage_ids = set(
            df_stage_master[df_stage_master["非活性化フラグ"]].index.tolist()
        )
        df_stage_view = df_stage_master[~df_stage_master["非活性化フラグ"]]
    else:
        df_stage_view = df_stage_master

    with st.expander("ステージマスタ", expanded=True):
        st.dataframe(_style_stage_color_column(df_stage_view), use_container_width=True)
    with st.expander("演者マスタ", expanded=True):
        idol_view = data["idolname"].rename(columns={"グループ名_採用": "グループ名"})
        st.dataframe(idol_view, use_container_width=True)
    with st.expander("出番マスタ", expanded=True):
        live_view = data["live"].drop(columns=["グループ名_raw"], errors="ignore")
        if disabled_stage_ids and "ステージID" in live_view.columns:
            live_view = live_view[~live_view["ステージID"].isin(disabled_stage_ids)]
        st.dataframe(live_view, use_container_width=True)
    _render_event_aggregations(event_name, data)
    _render_event_combined_pictures(event_name)


def _regenerate_event_all_images(event_name: str):
    _output_wf.regenerate_all_event_images(app_state, event_name)


def _render_event_combined_pictures(event_name: str):
    """全ステージ統合タイテ画像セクション (閲覧モードのみ)。"""
    pj_path = app_state.project.pj_path
    if not pj_path:
        return
    pij = app_state.project.project_info_json
    event_no = _repo.get_event_no_by_event_name(pij, event_name)
    if event_no is None:
        return

    st.divider()
    st.markdown("##### 全ステージ統合タイテ画像")
    st.button(
        "今すぐ再生成",
        on_click=_regenerate_event_all_images,
        args=(event_name,),
        key=f"regen_event_imgs_{event_name}",
        help="ステージ並び順・カラー等の変更を画像に反映する場合に使用します。",
    )

    # 選択肢: 種別ごと (variant ごと) + 種別横断
    event_type_list = _repo.get_event_type_list(pij, event_no)
    options: list[tuple[str, str, str]] = []  # (label, kind_key, path)
    for img_type in event_type_list:
        entry = _repo.get_image_entry_by_dir_name(pij, event_no, img_type)
        if entry is None:
            continue
        kind = entry.get("kind")
        if kind == "live":
            options.append((f"{img_type}", f"{img_type}/live",
                            os.path.join(pj_path, event_name, img_type, "all_stages_live.png")))
        elif kind == "tokutenkai":
            options.append((f"{img_type}", f"{img_type}/tokutenkai",
                            os.path.join(pj_path, event_name, img_type, "all_stages_tokutenkai.png")))
        elif kind == "live_tokutenkai_heiki":
            options.append((f"{img_type} (ライブ列)", f"{img_type}/live",
                            os.path.join(pj_path, event_name, img_type, "all_stages_live.png")))
            options.append((f"{img_type} (特典会列)", f"{img_type}/tokutenkai",
                            os.path.join(pj_path, event_name, img_type, "all_stages_tokutenkai.png")))
    options.append(("全体", "_event_",
                    os.path.join(pj_path, event_name, "all_stages.png")))

    if not options:
        return

    labels = [o[0] for o in options]
    sel_key = f"combined_pic_sel_{event_name}"
    selected = st.radio(
        "表示する画像", labels, key=sel_key, horizontal=True,
    )
    sel_idx = labels.index(selected)
    _, _, img_path = options[sel_idx]
    if os.path.exists(img_path):
        _render_scrollable_image(img_path)
        try:
            with open(img_path, "rb") as f:
                st.download_button(
                    "画像をダウンロード", data=f.read(),
                    file_name=os.path.basename(img_path),
                    mime="image/png",
                    key=f"dl_{event_name}_{sel_idx}",
                )
        except OSError:
            pass
    else:
        st.info("画像が未生成です。「今すぐ再生成」ボタンを押してください。")


def _render_scrollable_image(img_path: str, viewport_height_px: int = 900):
    """画像をネイティブ解像度で横/縦スクロール可能なコンテナに表示する。

    Streamlit の `st.image(use_container_width=True)` だと巨大画像が
    コンテナ幅まで縮小されて視認性が下がるため、独自 HTML を埋め込む。
    """
    import base64
    try:
        with open(img_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
    except OSError:
        return
    html = (
        '<div style="overflow:auto;'
        f'max-height:{viewport_height_px}px;'
        'border:1px solid #ddd;background:white;">'
        f'<img src="data:image/png;base64,{data}" '
        'style="display:block;max-width:none;" />'
        '</div>'
    )
    st.components.v1.html(html, height=viewport_height_px + 20, scrolling=False)


def _on_enter_edit_mode(event_name: str):
    result = _output_wf.enter_output_edit_mode(app_state, event_name)
    if not result.success:
        st.warning(result.error)


def _on_cancel_edit_mode(event_name: str):
    _output_wf.cancel_output_edit_mode(app_state, event_name)


def _on_save_edits(event_name: str):
    result = _output_wf.save_output_edits(app_state, event_name)
    if not result.success:
        st.session_state[f"output_edit_error_{event_name}"] = result.error
    else:
        st.session_state.pop(f"output_edit_error_{event_name}", None)
        # 保存成功時はトグル / 編集UI関連のキーを掃除して通常画面に戻す。
        # 次回 render で st.toggle が再インスタンス化される前に False をセットする。
        st.session_state[f"output_edit_mode_{event_name}"] = False
        for k in [f"save_edits_{event_name}", f"cancel_edits_{event_name}"]:
            st.session_state.pop(k, None)
        for k in [k for k in list(st.session_state.keys())
                  if k == f"stage_editor_{event_name}"
                  or k == f"sortable_stage_{event_name}"
                  or k == f"idolname_editor_{event_name}"
                  or k == f"live_editor_{event_name}"
                  or k.startswith(f"stage_color_sel_{event_name}_")
                  or k.startswith(f"stage_color_bg_{event_name}_")
                  or k.startswith(f"stage_color_fg_{event_name}_")]:
            st.session_state.pop(k, None)


def _render_event_output_editor(event_name: str, data):
    """編集モードON: 編集UI (ステージ / グループ / 出番マスタ)"""
    edits = app_state.output.edits.get(event_name)
    if edits is None:
        # 初期化漏れ防止
        _output_wf.enter_output_edit_mode(app_state, event_name)
        edits = app_state.output.edits.get(event_name)
        if edits is None:
            st.error("編集モードの初期化に失敗しました")
            return

    err = st.session_state.get(f"output_edit_error_{event_name}")
    if err:
        st.error(err)

    st.markdown("###### ステージマスタ (編集中)")
    sorted_stage = edits["stage"].sort_values("表示順")
    kind_map = edits.get("stage_kind_map", {})

    stage_cols = st.columns([1, 2])

    # --- 左: D&D 並び替えエリア ---
    with stage_cols[0]:
        if len(sorted_stage) >= 2:
            st.markdown("**ドラッグで並び替え**")
            from streamlit_sortables import sort_items
            labels = [
                _stage_reorder.make_stage_dnd_label(sid, row, kind_map)
                for sid, row in sorted_stage.iterrows()
            ]
            id_by_label = dict(zip(labels, sorted_stage.index.tolist()))
            new_labels = sort_items(
                labels,
                direction="vertical",
                key=f"sortable_stage_{event_name}",
            )
            if new_labels != labels:
                new_order_ids = [id_by_label[lab] for lab in new_labels]
                _stage_reorder.apply_stage_reorder(edits["stage"], new_order_ids)
                st.rerun()

    # --- 右: st.data_editor (ステージ名 / 短縮 / 非活性化 編集) ---
    # カラー名は別途 _render_stage_color_editor で行ごとのプルダウン編集する
    with stage_cols[1]:
        # 後方互換: 旧マスタに ステージ名_短縮 / カラー名 が無いケースを保護
        if "ステージ名_短縮" not in sorted_stage.columns:
            sorted_stage = sorted_stage.copy()
            sorted_stage["ステージ名_短縮"] = sorted_stage["ステージ名"]
        if "カラー名" not in sorted_stage.columns:
            sorted_stage = sorted_stage.copy()
            sorted_stage["カラー名"] = ""

        stage_display_df = (
            sorted_stage[[
                "ステージ名", "ステージ名_短縮",
                "特典会フラグ", "非活性化フラグ",
            ]]
            .reset_index()
            .rename(columns={sorted_stage.index.name or "index": "ステージID"})
        )
        edited_stage = st.data_editor(
            stage_display_df,
            column_config={
                "ステージID": st.column_config.NumberColumn("ステージID", disabled=True),
                "ステージ名": st.column_config.TextColumn("ステージ名", required=True),
                "ステージ名_短縮": st.column_config.TextColumn(
                    "ステージ名(短縮)",
                    help="Stella stageNameShort 用。空ならステージ名と同値で出力されます。",
                ),
                "特典会フラグ": st.column_config.CheckboxColumn("特典会", disabled=True),
                "非活性化フラグ": st.column_config.CheckboxColumn("非活性化"),
            },
            num_rows="fixed",
            hide_index=True,
            key=f"stage_editor_{event_name}",
            use_container_width=True,
        )
        # ステージID を index に戻し、表示順 / カラー名 (別UI管理) を補って書き戻す
        edited_stage = edited_stage.set_index("ステージID")
        edited_stage["表示順"] = edits["stage"]["表示順"]
        # カラー名は data_editor で扱わないため、既存値を明示的に保持して書き戻す
        if "カラー名" in edits["stage"].columns:
            edited_stage["カラー名"] = edits["stage"]["カラー名"]
        edits["stage"] = edited_stage

    # --- ステージカラー編集 (data_editor から独立) ---
    _render_stage_color_editor(event_name)

    # --- グループマスタ編集 (Phase 3) ---
    # グループID (index) は通常列に降ろして disabled 化し、誤編集を防ぐ
    st.markdown("###### グループマスタ (編集中)")
    idolname_show = edits["idolname"].reset_index()
    edited_idolname = st.data_editor(
        idolname_show,
        column_config={
            "グループID": st.column_config.NumberColumn("グループID", disabled=True),
            "グループ名_採用": st.column_config.TextColumn("グループ名", required=True),
        },
        num_rows="fixed",
        hide_index=True,
        key=f"idolname_editor_{event_name}",
        use_container_width=True,
    )
    edits["idolname"] = edited_idolname.set_index("グループID")

    # --- 出番マスタ編集 (Phase 4 / Phase 5) ---
    st.markdown("###### 出番マスタ (編集中)")
    group_options = [int(gid) for gid in edits["idolname"].index]
    group_label_map = {
        int(gid): f"{int(gid)}: {name}"
        for gid, name in edits["idolname"]["グループ名_採用"].items()
    }
    stage_options = [int(sid) for sid in edits["stage"].index]
    stage_label_map: dict[int, str] = {}
    for sid, stage_row in edits["stage"].iterrows():
        suffix = " [特典会]" if bool(stage_row.get("特典会フラグ", False)) else ""
        stage_label_map[int(sid)] = f"{int(sid)}: {stage_row['ステージ名']}{suffix}"
    # 出番ID (index) は通常列に降ろして disabled 化し、誤編集を防ぐ
    # グループ名_raw は UI 上は非表示 (Excel 出力には output_df 側で残る)
    live_show = edits["live"].reset_index().drop(columns=["グループ名_raw"], errors="ignore")
    edited_live = st.data_editor(
        live_show,
        column_config={
            "出番ID": st.column_config.NumberColumn("出番ID", disabled=True),
            "ステージID": st.column_config.SelectboxColumn(
                "ステージID", options=stage_options,
                format_func=lambda x: stage_label_map.get(int(x), str(x))
                if x is not None else "",
                required=True,
                help="変更すると エントリが別 stage_*.json (同 kind / 同 特典会フラグ) に移動します",
            ),
            "ステージ名": st.column_config.TextColumn("ステージ名", disabled=True),
            "グループ名": st.column_config.TextColumn(
                "グループ名", disabled=True,
                help="グループマスタを編集すると追従します",
            ),
            "特典会フラグ": st.column_config.CheckboxColumn("特典会", disabled=True),
            "グループID": st.column_config.SelectboxColumn(
                "グループID", options=group_options,
                format_func=lambda x: group_label_map.get(int(x), str(x))
                if x is not None else "",
                required=True,
            ),
            "ライブ_from": st.column_config.TextColumn(
                "from", required=True, help="HH:MM 形式",
            ),
            "ライブ_to": st.column_config.TextColumn(
                "to", disabled=True,
                help="from + 長さ(分) から自動算出 (保存時に再計算)",
            ),
            "ライブ_長さ(分)": st.column_config.NumberColumn(
                "長さ(分)", min_value=1, step=1, required=True,
            ),
            "対応出番ID": st.column_config.NumberColumn(
                "対応出番ID", step=1,
                help="特典会行のみ。対応するライブ行の出番ID。"
                     "変更すると 特典会要素が別ライブに付け替えられます (ファイル跨ぎ可)。",
            ),
            "備考": st.column_config.TextColumn("備考"),
        },
        num_rows="fixed",
        hide_index=True,
        key=f"live_editor_{event_name}",
        use_container_width=True,
    )
    edits["live"] = edited_live.set_index("出番ID")

    btn_cols = st.columns([1, 1, 6])
    with btn_cols[0]:
        st.button(
            "保存", key=f"save_edits_{event_name}",
            on_click=_on_save_edits, args=(event_name,),
            type="primary",
        )
    with btn_cols[1]:
        st.button(
            "キャンセル", key=f"cancel_edits_{event_name}",
            on_click=_on_cancel_edit_mode, args=(event_name,),
        )

    _render_event_aggregations(event_name, data)


def render_output_section():
    """⑥タイムテーブル情報の出力"""
    st.markdown("#### ⑥タイムテーブル情報の出力")

    event_list = get_event_name_list()
    pj_path = app_state.project.pj_path
    pij = app_state.project.project_info_json

    # 各イベントについて グループ名_採用 の欠損チェック。
    # 欠損ありのイベントはデータビルドを行わずアラート + 一括採用ボタンを表示する。
    event_has_empty: dict[str, bool] = {}
    valid_events: list[str] = []
    for event_name in event_list:
        event_no = _repo.get_event_no_by_event_name(pij, event_name)
        if event_no is None:
            event_has_empty[event_name] = False
            valid_events.append(event_name)
            continue
        has_empty = _ocr.check_event_has_empty_adopted_idol_names(
            pj_path, event_name, event_no, pij,
        )
        event_has_empty[event_name] = has_empty
        if not has_empty:
            valid_events.append(event_name)

    # 欠損なしイベントのみビルドする
    output_df: dict[str, dict] = {}
    for event_name in valid_events:
        event_no = _repo.get_event_no_by_event_name(pij, event_name)
        if event_no is None:
            output_df[event_name] = {}
            continue
        data = _output.build_event_output(pj_path, event_name, event_no, pij)
        output_df[event_name] = data if data is not None else {}
    app_state.output.output_df = output_df

    # 自動IDマスタ確定: 未確定 (= ファイル欠落 or 新規IDあり) のイベントが存在すれば
    # ボタン操作なしで採番結果を永続化する。同一 rerun 内の二重発火を session_state で抑止。
    events_needing_commit = _output_wf.list_events_with_unconfirmed_ids(app_state)
    if events_needing_commit and not st.session_state.get("_auto_id_committing"):
        # ID内部整合異常のあるイベントは永続化対象から除外し、エラー表示する。
        # master差分は非ブロック通知として警告表示する (永続化前に旧 master を読む)。
        committable_events: list[str] = []
        diff_notices: list[str] = []
        for event_name in events_needing_commit:
            data = output_df.get(event_name) or {}
            if not data:
                continue
            anomalies = _output.detect_id_anomalies(data)
            if anomalies:
                st.error(
                    f"イベント「{event_name}」でID異常を検出したため永続化を中止しました。"
                    "④読み取りタブで重複IDを修正してください:\n"
                    + "\n".join(f"  - {m}" for m in anomalies),
                )
                continue
            output_path = os.path.join(pj_path, event_name)
            diff_notices += [
                f"{event_name} / {m}"
                for m in _output.detect_master_diff(data, output_path)
            ]
            committable_events.append(event_name)

        if committable_events:
            st.session_state["_auto_id_committing"] = True
            try:
                with st.spinner("IDマスタを確定中..."):
                    commit_result = _output_wf.determine_id_master(
                        app_state, target_event_names=committable_events,
                    )
            finally:
                st.session_state["_auto_id_committing"] = False
            if not commit_result.success:
                st.error(
                    f"IDマスタの自動確定に失敗しました: {commit_result.error}"
                )
                return
            if diff_notices:
                st.warning(
                    "IDマスタを更新しました。以下は前回確定時からの変更点です（要確認）:\n"
                    + "\n".join(f"  - {m}" for m in diff_notices),
                )
            # 確定後の最新状態を反映するため output_df を再ビルド
            for event_name in valid_events:
                event_no = _repo.get_event_no_by_event_name(pij, event_name)
                if event_no is None:
                    continue
                data = _output.build_event_output(pj_path, event_name, event_no, pij)
                output_df[event_name] = data if data is not None else {}
            app_state.output.output_df = output_df

    event_tabs = st.tabs(event_list)
    for event_name, event_tab in zip(event_list, event_tabs):
        with event_tab:
            if event_has_empty.get(event_name):
                st.error(
                    "「グループ名_採用」が未入力のレコードが含まれているため、"
                    "このイベントのデータは表示できません。"
                    "下記ボタンで「読み取ったグループ名」をそのまま採用するか、"
                    "④読み取りタブで個別に修正してください。",
                )
                st.button(
                    "読み取ったグループ名をそのまま採用する",
                    on_click=adopt_raw_idolname_event, args=(event_name,),
                    key=f"button_adopt_raw_idolname_event_{event_name}",
                    help="このイベント配下の全 (種別, ステージ) について"
                         "「グループ名_採用」を「グループ名(OCR)」と同じ値で上書きします。",
                )
                continue
            data = output_df.get(event_name)
            if not data:
                continue
            # 自動確定により IDマスタは常に確定済み → 編集モードトグルを常時表示
            toggle_key = f"output_edit_mode_{event_name}"
            prev_state = bool(st.session_state.get(toggle_key, False))
            edit_mode = st.toggle("編集モード", key=toggle_key)
            # トグル変化を検知して enter/cancel を呼ぶ
            if edit_mode and not prev_state:
                _on_enter_edit_mode(event_name)
            elif not edit_mode and prev_state:
                _on_cancel_edit_mode(event_name)

            if edit_mode:
                _render_event_output_editor(event_name, data)
            else:
                _render_event_output_view(event_name, data)

    st.button("プロジェクトデータをクラウドにアップロード ※通信料・保存料が発生するので留意",
              on_click=save_to_s3)
    if st.button("Excelデータを出力", on_click=output_data_for_stella):
        file_path = os.path.join(app_state.project.pj_path, "output.xlsx")
        with open(file_path, "rb") as file:
            excel_data = file.read()
        st.download_button(
            "ファイルをダウンロード",
            data=excel_data,
            file_name="{}.xlsx".format(app_state.project.pj_name),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    _render_stella_section(valid_events)


# ===========================================================================
# ⑥ Stella連携セクション (メタデータ / JSON出力)
# ===========================================================================

def _render_stella_section(event_list: list[str]) -> None:
    """Stellaメタデータ / JSON出力 を描画する。
    ステージ詳細 (ステージ名_短縮 / カラー名) は ⑥編集モードのステージマスタに統合済。
    GitHub 連携は Phase 6 で別途実装。
    """
    st.divider()
    st.markdown("#### Stella連携")
    for event_name in event_list:
        with st.expander(f"{event_name}: Stellaメタデータ / JSON出力", expanded=False):
            _render_stella_metadata_form(event_name)
            _render_stella_export_form(event_name)


def _compute_stella_open_close_default(event_name: str) -> tuple[str, str]:
    """全出番の最早開始 / 最遅終了から openTime / closeTime を算出する。

    - openTime: min("ライブ_from") の H:MM について、MM==0 なら H-1、MM>=1 なら H
                (例 11:01～12:00 → 11、12:01～13:00 → 12)
    - closeTime: max("ライブ_to") の H:MM について、常に H+1
                (例 21:00～21:59 → 22、23:00～23:59 → 24)
    算出不能な場合は ("", "") を返す。
    """
    data = (app_state.output.output_df or {}).get(event_name)
    if not data:
        return ("", "")
    live = data.get("live")
    if live is None or len(live) == 0:
        return ("", "")
    # 非活性化ステージに紐づく出番は openTime/closeTime 算出から除外する
    df_stage_master = data.get("stage")
    if df_stage_master is not None and "非活性化フラグ" in df_stage_master.columns:
        disabled_stage_ids = set(
            df_stage_master[df_stage_master["非活性化フラグ"]].index.tolist()
        )
        if disabled_stage_ids and "ステージID" in live.columns:
            live = live[~live["ステージID"].isin(disabled_stage_ids)]
            if len(live) == 0:
                return ("", "")

    def _to_minutes(s) -> int | None:
        if s is None or (isinstance(s, float) and pd.isna(s)) or s == "":
            return None
        try:
            h, m = str(s).split(":")
            return int(h) * 60 + int(m)
        except (ValueError, AttributeError):
            return None

    from_series = live["ライブ_from"] if "ライブ_from" in live.columns else []
    to_series = live["ライブ_to"] if "ライブ_to" in live.columns else []
    from_min = [m for m in (_to_minutes(v) for v in from_series) if m is not None]
    end_min = [m for m in (_to_minutes(v) for v in to_series) if m is not None]
    if not from_min or not end_min:
        return ("", "")

    fmin = min(from_min)
    tmax = max(end_min)
    fh, fm = divmod(fmin, 60)
    th, tm = divmod(tmax, 60)
    open_h = fh - 1 if fm == 0 else fh
    close_h = th + 1
    return (str(open_h), str(close_h))


def _render_stella_metadata_form(event_name: str) -> None:
    """openTime / closeTime / notification 等 を編集する。"""
    st.markdown("###### メタデータ")
    pij = app_state.project.project_info_json
    event_no = _repo.get_event_no_by_event_name(pij, event_name)
    if event_no is None:
        st.warning("event_no が解決できません")
        return
    metadata = _repo.get_stella_metadata(pij, event_no)

    open_default, close_default = _compute_stella_open_close_default(event_name)
    saved_open = str(metadata.get("openTime", ""))
    saved_close = str(metadata.get("closeTime", ""))
    open_initial = saved_open if saved_open else open_default
    close_initial = saved_close if saved_close else close_default

    cols = st.columns(2)
    with cols[0]:
        st.text_input(
            "openTime (時のみ, 例 '12')",
            value=open_initial,
            key=f"stella_open_{event_name}",
            help=f"全出番の最早開始から自動算出: {open_default or '算出不能'}",
        )
    with cols[1]:
        st.text_input(
            "closeTime (時のみ, 例 '23')",
            value=close_initial,
            key=f"stella_close_{event_name}",
            help=f"全出番の最遅終了から自動算出: {close_default or '算出不能'}",
        )
    st.text_area(
        "notification (更新通知メッセージ)",
        value=str(metadata.get("notification", "")),
        key=f"stella_notif_{event_name}",
    )
    st.caption(
        f"現在の notificationVersion: {metadata.get('notificationVersion', '1')} "
        f"(Push時に notification 変更で +1)"
    )

    def _save():
        _output_wf.save_stella_metadata(app_state, event_name, {
            "openTime": st.session_state[f"stella_open_{event_name}"],
            "closeTime": st.session_state[f"stella_close_{event_name}"],
            "notification": st.session_state[f"stella_notif_{event_name}"],
        })

    st.button(
        "メタデータを保存", on_click=_save,
        key=f"stella_meta_save_{event_name}",
    )


def _render_stella_export_form(event_name: str) -> None:
    """Stella JSON を生成 → プレビュー / ダウンロード。"""
    st.markdown("###### JSON 出力")
    state_key = f"stella_json_preview_{event_name}"

    def _build():
        result = _output_wf.build_stella_json(app_state, event_name)
        if not result.success:
            st.session_state[state_key] = None
            st.session_state[f"{state_key}_err"] = result.error
        else:
            st.session_state[state_key] = result.data
            st.session_state.pop(f"{state_key}_err", None)

    st.button(
        "Stella JSONを生成", on_click=_build,
        key=f"stella_export_build_{event_name}",
    )

    err = st.session_state.get(f"{state_key}_err")
    if err:
        st.error(err)
    preview = st.session_state.get(state_key)
    if preview is not None:
        st.caption(
            f"artList: {len(preview.get('artList', []))}件 / "
            f"stageList: {len(preview.get('stageList', []))}件 / "
            f"turnList: {len(preview.get('turnList', []))}件"
        )
        with st.expander("JSONプレビュー", expanded=False):
            st.code(json.dumps(preview, ensure_ascii=False, indent=2), language="json")
        live_id = preview.get("liveId")
        fname = f"live{live_id}.json" if live_id is not None else "live_unassigned.json"
        st.download_button(
            "Stella JSONをダウンロード",
            data=json.dumps(preview, ensure_ascii=False, separators=(",", ":")),
            file_name=fname,
            mime="application/json",
            key=f"stella_export_dl_{event_name}",
        )


def render_master_update_section():
    """⑦マスタのアップデート"""
    st.markdown("#### ⑦マスタのアップデート")
    st.button("新規登場の「グループ名_採用」をリストアップ",on_click=listup_new_idolname)
    if app_state.output.new_idolname is not None:
        df_new_idolname = st.data_editor(app_state.output.new_idolname,num_rows="dynamic")
        st.button("チェックしたグループ名をマスタに追加", on_click=update_master_idolname, args=(df_new_idolname,))


# ===========================================================================
# サイドバー + メインディスパッチ
# ===========================================================================

with st.sidebar:
    st.title(
        "タイムテーブル読み取りアプリ",
        help=(
            "**タイムテーブル画像を構造化データに変換します**\n\n"
            "- ライブのタイムテーブル画像をアップロードしてください\n"
            "- ライブの形式を選択して、OCRや生成AIを活用し構造化データに変換します\n"
            "- 精度は100%ではないので、人の手で修正を行ってください\n"
            "- 最終的に出来上がった構造化データは、現時点ではStella用に形式を変換してダウンロードすることができます"
        ),
    )
    st.divider()
    st.markdown("### プロジェクト")
    col_makepj = st.columns((5,1))
    with col_makepj[0]:
        st.text_input(label="新しいプロジェクト名", placeholder="入力してください", key="new_pj_name")
    with col_makepj[1]:
        st.button(label="作成", on_click=make_project)
    col_setpj = st.columns((5,1))
    with col_setpj[0]:
        st.selectbox("既存のプロジェクト一覧"
                                , pj_name_list
                                , placeholder = "プロジェクトを選択または作成してください"
                                , key="exist_pj_name")
    with col_setpj[1]:
        st.button(label="呼出", on_click=set_project, args=(st.session_state.exist_pj_name,))

    if app_state.project.pj_name is not None:
        st.success(f"選択中: {app_state.project.pj_name}")

    st.divider()
    page = st.radio("処理フェーズ", [
        "①設定", "②画像登録", "③画像切り取り",
        "④読み取り", "⑤変更比較", "⑥出力確認・編集", "⑦マスタ更新",
    ], key="nav_page")

# === ページ遷移警告 (編集中の未保存データ保護) ===
def _has_unsaved_edits() -> bool:
    """編集中の作業コピーが元データから実際に変更されているかを判定する。
    値の変化がなければ False (= 編集モード ON にしただけなら警告しない)。"""
    def _df_changed(original_df, current_df, cols) -> bool:
        if original_df is None or current_df is None:
            return False
        cols = [c for c in cols
                if c in original_df.columns and c in current_df.columns]
        if not cols:
            return False
        o = original_df[cols].sort_index()
        c = current_df[cols].sort_index()
        return not o.equals(c)

    for ev, edits in app_state.output.edits.items():
        original = app_state.output.output_df.get(ev) or {}
        try:
            if _df_changed(original.get("stage"), edits.get("stage"),
                           ("ステージ名", "ステージ名_短縮", "カラー名",
                            "表示順", "非活性化フラグ")):
                return True
            if _df_changed(original.get("idolname"), edits.get("idolname"),
                           ("グループ名_採用",)):
                return True
            if _df_changed(original.get("live"), edits.get("live"),
                           ("ライブ_from", "ライブ_長さ(分)", "グループID", "備考")):
                return True
            # 対応出番ID は output_df["live"] に存在しないため、
            # edits["_live_baseline"] (enter 時のスナップショット) と比較する
            baseline = edits.get("_live_baseline")
            current_live = edits.get("live")
            if baseline is not None and current_live is not None \
                    and "対応出番ID" in current_live.columns:
                try:
                    o = baseline["対応出番ID"].sort_index()
                    c = current_live["対応出番ID"].sort_index()
                    if not o.equals(c):
                        return True
                except Exception:
                    return True
        except Exception:
            return True
    return False


def _clear_edit_mode_widgets():
    """編集モード関連のウィジェット session_state キーを掃除する。
    次回 render 時にトグル / 入力欄が初期値から始まるようにする。"""
    keys_to_drop = [
        k for k in list(st.session_state.keys())
        if (
            k.startswith("output_edit_mode_")
            or k.startswith("stage_name_")
            or k.startswith("stage_disabled_")
            or k.startswith("up_")
            or k.startswith("dn_")
            or k.startswith("save_edits_")
            or k.startswith("cancel_edits_")
            or k.startswith("output_edit_error_")
            or k.startswith("idolname_editor_")
            or k.startswith("live_editor_")
            or k.startswith("stage_color_sel_")
            or k.startswith("stage_color_bg_")
            or k.startswith("stage_color_fg_")
        )
    ]
    for k in keys_to_drop:
        st.session_state.pop(k, None)


def _save_all_pending_edits():
    """全イベントの編集を保存する。バリデーションエラーがあれば中断。"""
    failed = []
    for ev in list(app_state.output.edits.keys()):
        result = _output_wf.save_output_edits(app_state, ev)
        if not result.success:
            failed.append(f"{ev}: {result.error}")
    if failed:
        st.session_state["nav_warn_error"] = "\n".join(failed)
        return False
    st.session_state.pop("nav_warn_error", None)
    return True


def _discard_all_pending_edits():
    app_state.output.edits.clear()


_dirty_pending_edits = _has_unsaved_edits()
_pending_nav_change = (
    _dirty_pending_edits
    and app_state.ui.last_page is not None
    and page != app_state.ui.last_page
)

if _pending_nav_change:
    st.warning("未保存の編集があります。どうしますか？")
    err = st.session_state.get("nav_warn_error")
    if err:
        st.error(err)
    warn_cols = st.columns(3)
    with warn_cols[0]:
        if st.button("保存して移動", key="nav_warn_save"):
            if _save_all_pending_edits():
                _clear_edit_mode_widgets()
                app_state.ui.last_page = page
                st.rerun()
    with warn_cols[1]:
        if st.button("破棄して移動", key="nav_warn_discard"):
            _discard_all_pending_edits()
            _clear_edit_mode_widgets()
            st.session_state.pop("nav_warn_error", None)
            app_state.ui.last_page = page
            st.rerun()
    with warn_cols[2]:
        if st.button("キャンセル(前ページに留まる)", key="nav_warn_cancel"):
            st.session_state["nav_page"] = app_state.ui.last_page
            st.rerun()
    st.stop()

# 警告がなければ last_page を現在ページに同期し、編集モードを解除する。
# ページ遷移時は毎回通常画面に戻す方針。
if app_state.ui.last_page is not None and page != app_state.ui.last_page:
    _discard_all_pending_edits()
    _clear_edit_mode_widgets()
app_state.ui.last_page = page

# === メインエリア ===
if app_state.project.pj_name is None and page != "①設定":
    st.info("サイドバーからプロジェクトを選択または作成してください")
elif page == "①設定":
    render_project_setting()
elif page == "②画像登録":
    render_image_upload()
elif page == "③画像切り取り":
    render_crop_section()
elif page == "④読み取り":
    render_ocr_section()
elif page == "⑤変更比較":
    render_comparison_section()
elif page == "⑥出力確認・編集":
    render_output_section()
elif page == "⑦マスタ更新":
    render_master_update_section()
