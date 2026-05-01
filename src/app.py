import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from streamlit_cropper import st_cropper
# from streamlit_cropperjs import st_cropperjs
# from streamlit_drawable_canvas import st_canvas

import os
import shutil
import tempfile
import copy
from operator import itemgetter
import concurrent.futures

from PIL import Image, ImageDraw
import cv2
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
# from openpyxl import Workbook  # moved to output_builder
from io import BytesIO
from datetime import datetime
from datetime import time as dttime
from datetime import timedelta
# from zoneinfo import ZoneInfo  # moved to output_builder
import json

from backend_functions import s3access, timetabledata
from backend_functions import project_repository as _repo
from backend_functions import time_axis as _time_axis
from backend_functions import image_processing as _imgproc
from backend_functions import ocr_service as _ocr
from backend_functions.ticket_scraper import get_performers_list_from_ticket_urls
from app_state import AppState
from workflow import ProjectWorkflow, ImageWorkflow, OcrWorkflow, OutputWorkflow

st.set_page_config(
    page_title="タイムテーブル読み取りアプリ", 
    # page_icon=image, 
    layout="wide", 
    # initial_sidebar_state="auto", 
    menu_items={
        'Get Help': 'https://www.google.com',
        'Report a bug': "https://www.google.com",
        'About': """
        # アイドル対バンタイムテーブル読み取りツール
        ライブ管理アプリへの搭載用のデータを生成できます。
        """
     })

st.title("タイムテーブル読み取りアプリ")
st.markdown(
"""### タイムテーブル画像を構造化データに変換します
- ライブのタイムテーブル画像をアップロードしてください
- ライブの形式を選択して、OCRや生成AIを活用し構造化データに変換します
- 精度は100%ではないので、人の手で修正を行ってください
- 最終的に出来上がった構造化データは、現時点ではStella用に形式を変換してダウンロードすることができます 
""")

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

def _sync_to_session(state: AppState) -> None:
    """AppStateの一部をUIウィジェットが参照するsession_stateキーに同期する"""
    st.session_state.pj_name = state.project.pj_name
    st.session_state.exist_pj_name = state.project.pj_name
    st.session_state.pj_path = state.project.pj_path
    st.session_state.project_info_json = state.project.project_info_json
    st.session_state.project_master = state.project.project_master
    st.session_state.project_master_s3 = state.project.project_master_s3
    st.session_state.event_type = state.project.event_type
    st.session_state.event_num = state.project.event_num
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
        with project_setting:
            st.error(result.error)
    else:
        _sync_to_session(app_state)

def set_project(pj_name):
    result = _project_wf.load_project(pj_name, app_state)
    if result.success:
        _sync_to_session(app_state)

def get_project_json():
    return _repo.get_project_json(app_state.project.pj_path)

def update_project_timestamp():
    app_state.project.project_master = _repo.update_timestamp(
        app_state.project.project_master, app_state.project.pj_name, DATA_PATH
    )

def set_project_json(json_data):
    _repo.save_project_json(app_state.project.pj_path, json_data)
    app_state.project.project_info_json = json_data
    st.session_state.project_info_json = json_data
    update_project_timestamp()

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

@st.cache_data
def get_image(img_path):
    return Image.open(img_path)

def determine_timetable_image():
    file = st.session_state.uploaded_image
    file_data = file.read()
    result = _project_wf.register_image(
        app_state,
        event_name=st.session_state.img_event_name,
        img_type=st.session_state.img_type,
        img_format=st.session_state.img_format,
        file_data=file_data,
        img_type_alternative=st.session_state.get("img_type_alternative", ""),
    )
    if result.success:
        _sync_to_session(app_state)
        resolved = result.data["resolved_img_type"]
        app_state.crop.crop_tgt_event = st.session_state.img_event_name
        app_state.crop.crop_tgt_img_type = resolved
        app_state.ocr.ocr_tgt_event = st.session_state.img_event_name
        app_state.ocr.ocr_tgt_img_type = resolved
        st.session_state.crop_tgt_event = st.session_state.img_event_name
        st.session_state.crop_tgt_img_type = resolved
        st.session_state.ocr_tgt_event = st.session_state.img_event_name
        st.session_state.ocr_tgt_img_type = resolved
        with col_file_uploader[1]:
            st.success("画像を登録しました")

def delete_uploaded_image(img_event_no, img_type):
    pij = app_state.project.project_info_json
    _repo.delete_timetable_image(pij, img_event_no, img_type)
    _repo.save_project_json(app_state.project.pj_path, pij)
    update_project_timestamp()
    next_img_type = get_event_type_list(img_event_no)
    if len(next_img_type)==0:
        app_state.crop.crop_tgt_img_type = None
        app_state.ocr.ocr_tgt_img_type = None
        st.session_state.crop_tgt_img_type = None
        st.session_state.ocr_tgt_img_type = None
    else:
        if st.session_state.get("crop_tgt_img_type") == img_type:
            app_state.crop.crop_tgt_img_type = next_img_type[0]
            st.session_state.crop_tgt_img_type = next_img_type[0]
        if st.session_state.get("ocr_tgt_img_type") == img_type:
            app_state.ocr.ocr_tgt_img_type = next_img_type[0]
            st.session_state.ocr_tgt_img_type = next_img_type[0]
    with col_file_uploader[1]:
        st.success("画像を削除しました")

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
    app_state.crop.images_eachstage = []
    app_state.crop.images_eachstage_bbox = []
    st.session_state.images_eachstage = []
    st.session_state.images_eachstage_bbox = []

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
    result = _image_wf.detect_stage_lines(
        app_state, image, params,
        st.session_state.crop_tgt_event, st.session_state.crop_tgt_img_type,
    )
    _sync_to_session(app_state)
    with edge_result:
        st.image(result.data.annotated_image, caption="縦線抽出結果")

def get_image_eachstage_byocr(image, stage_num):
    return _imgproc.get_image_eachstage_byocr(image, stage_num)

def get_image_eachstage_for_linecroppedimage_byocr():
    st.session_state.images_eachstage = []
    st.session_state.images_eachstage_bbox = []
    crop_offset_x = st.session_state.crop_box["left"]
    crop_offset_y = st.session_state.crop_box["top"]
    crop_bottom = crop_offset_y + st.session_state.crop_box["height"]
    for i,line_cropped_image in enumerate(st.session_state.line_cropped_images):
        if st.session_state["stage_num_{}".format(i)]>0:
            images, bboxes = get_image_eachstage_byocr(line_cropped_image, st.session_state["stage_num_{}".format(i)])
            st.session_state.images_eachstage += images
            for bbox in bboxes:
                st.session_state.images_eachstage_bbox.append({
                    "left": bbox["left"] + crop_offset_x,
                    "top": crop_offset_y,
                    "right": bbox["right"] + crop_offset_x,
                    "bottom": crop_bottom
                })

def get_image_eachstage_for_linecroppedimage_byevenly(): #使ってない
    st.session_state.images_eachstage = []
    st.session_state.images_eachstage_bbox = []
    crop_offset_x = st.session_state.crop_box["left"]
    crop_offset_y = st.session_state.crop_box["top"]
    crop_bottom = crop_offset_y + st.session_state.crop_box["height"]
    for i,line_cropped_image in enumerate(st.session_state.line_cropped_images):
        stage_num = st.session_state["stage_num_{}".format(i)]
        if stage_num>0:
            width, height = line_cropped_image.size
            segment_width = width / stage_num
            for j in range(stage_num):
                left = round(max(0, (j - 0.05) * segment_width))
                right = round(min((j + 1.05) * segment_width, width))
                segment = line_cropped_image.crop((left, 0, right, height))
                st.session_state.images_eachstage.append(segment)
                st.session_state.images_eachstage_bbox.append({
                    "left": left + crop_offset_x,
                    "top": crop_offset_y,
                    "right": right + crop_offset_x,
                    "bottom": crop_bottom
                })

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
    app_state.ocr.time_axis_detect = None
    app_state.ocr.timeline_eachstage = []
    st.session_state.time_axis_detect = None
    st.session_state.timeline_eachstage = []

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
        output_timetable_picture_onlyonestage(i)

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
        output_timetable_picture_onlyonestage(i)

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
    for target_key in together_targets:
        event_name, img_type = target_key.split("/")
        event_no = get_event_no_by_event_name(event_name)
        timetable_info = app_state.project.project_info_json["event_detail"][event_no]["timetables"][img_type]
        converter = _time_axis.TimeAxisConverter.from_project_info(
            app_state.project.project_info_json, event_no, img_type,
        )
        for stage_no in range(timetable_info["stage_num"]):
            _ocr_wf.generate_timetable_picture(
                app_state, stage_no, event_name, img_type,
                st.session_state.ocr_output_picture_time_match, converter,
            )
    _sync_to_session(app_state)

def save_timetable_data_onlyonestage(stage_no):
    stage_name = st.session_state["stage_name_stage{}".format(stage_no)]
    is_tokutenkai_heiki = st.session_state.ocr_tgt_image_info["format"] == "特典会併記"
    result = _ocr_wf.save_timetable_data(
        app_state, stage_no, st.session_state.df_timetables[stage_no], stage_name,
        st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type,
        is_tokutenkai_heiki,
    )
    st.session_state.df_timetables[stage_no] = result.data
    set_stage_name(stage_no, stage_name)
    output_timetable_picture_onlyonestage(stage_no)

def save_timetable_data_eachstage():
    for i in range(st.session_state.ocr_tgt_stage_num):
        save_timetable_data_onlyonestage(i)

def output_timetable_picture_onlyonestage(stage_no):
    converter = _get_time_axis_converter()
    _ocr_wf.generate_timetable_picture(
        app_state, stage_no,
        st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type,
        st.session_state.ocr_output_picture_time_match, converter,
    )

def output_timetable_picture_eachstage():
    for i in range(st.session_state.ocr_tgt_stage_num):
        output_timetable_picture_onlyonestage(i)

def output_difference_image(new_image):
    result = _image_wf.output_difference_image(
        new_image, app_state.project.pj_path,
        st.session_state.diff_tgt_event, st.session_state.diff_tgt_img_type,
    )
    with timetable_compare_col[1]:
        st.image(result.data)

def replace_stage_images_from_new_raw(new_image):#新しい画像から既存のbbox座標でステージ画像を切り出して置き換える
    result = _image_wf.replace_stage_images_from_new_raw(
        app_state, new_image,
        st.session_state.diff_tgt_event, st.session_state.diff_tgt_img_type,
    )
    if not result.success:
        st.warning(result.error)
        return
    _sync_to_session(app_state)

# def get_all_stage_info():#全ステージ情報の出力 #暫定
#     all_stage_df = pd.concat(st.session_state.df_timetables).reset_index(drop=True)
#     with all_stage_info:
#         st.dataframe(all_stage_df,hide_index=True)

def determine_id_master():
    _output_wf.determine_id_master(app_state)
    _sync_to_session(app_state)

def save_to_s3():
    _output_wf.save_to_s3(app_state)

def output_data_for_stella():
    _output_wf.export_excel(app_state)

def listup_new_idolname():
    _output_wf.listup_new_idolname(app_state)

def update_master_idolname(df_new_idolname):
    _output_wf.update_idol_name_master(app_state, df_new_idolname)

project_setting = st.container()#プロジェクト設定
with project_setting:
    st.markdown("""#### ①プロジェクトの設定""")
    col_project_setting = st.columns(3)
    project_setting_determine = st.container()
with col_project_setting[0]:
    st.info(
"""既存のプロジェクトを呼び出すか、新しくプロジェクトを作成してください。  
一つのプロジェクトは一つのイベントに紐付きます。  
""")
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
    if app_state.project.pj_name is None:
        st.stop()
    st.text("選択中のプロジェクト："+app_state.project.pj_name)
with col_project_setting[1]:
    st.info(
"""イベント形式を選択してください。  
・フェス：複数のステージが同時進行するイベント  
・対バン：ステージが一つのみのイベント  
※Stella向けなので、一旦フェスのみに対応して作っています  
""")
    st.radio("イベント形式", ("対バン", "フェス"), key="event_type", horizontal=True)
with col_project_setting[2]:
    st.info(
"""イベント数を入力してください。  
イベント数とは、チケットの販売単位に相当する概念です。  
複数日にわたって開かれるフェスや、昼夜で別イベントとして開かれる対バンの場合、その数を記入してください。  
""")
    st.number_input("イベント数", min_value=1, step=1, key="event_num")
with project_setting_determine:
    if st.session_state.event_type is not None and st.session_state.event_num is not None:
        st.button(label="プロジェクト設定を反映する", on_click=determine_project_setting, type="primary")

# チケットURL設定
ticket_url_setting = st.container()
with ticket_url_setting:
    with st.expander("チケットURL設定（出演者情報取得用）"):
        st.info(
"""チケットサイトのURLを登録すると、タイムテーブル読み取り時に出演者リストを自動取得してOCRの精度向上に活用できます。
・対応サイト：TicketDive, LivePocket, tiget など
・紐づけ単位を「イベントごと」に変更すると、プロジェクト共通のURL設定はクリアされます（逆も同様）
・複数URLを登録する場合は、1行に1URLずつ入力してください
""")
        # 現在の設定を読み込み
        current_scope = "project"
        current_project_urls = []
        if "ticket_urls" in st.session_state.project_info_json:
            current_scope = st.session_state.project_info_json["ticket_urls"].get("scope", "project")
            current_project_urls = st.session_state.project_info_json["ticket_urls"].get("urls", [])

        scope_options = ("プロジェクト共通", "イベントごと")
        default_scope_index = 0 if current_scope == "project" else 1
        st.radio("チケットURLの紐づけ単位", scope_options, index=default_scope_index, key="ticket_url_scope", horizontal=True)

        if st.session_state.ticket_url_scope == "プロジェクト共通":
            # プロジェクト共通URL入力欄
            default_urls = "\n".join(current_project_urls) if current_scope == "project" else ""
            st.text_area("チケットサイトURL（1行に1つずつ）", value=default_urls, key="ticket_urls_project", height=100)
        else:
            # イベントごとのURL入力欄
            event_list = get_event_name_list()
            for i, event_name in enumerate(event_list):
                event_urls = []
                if current_scope == "event":
                    event_data = app_state.project.project_info_json["event_detail"][i]
                    event_urls = event_data.get("ticket_urls", [])
                default_urls = "\n".join(event_urls)
                st.text_area(f"{event_name} のチケットサイトURL", value=default_urls, key=f"ticket_urls_event_{i}", height=80)

        st.button("チケットURL設定を保存", on_click=save_ticket_urls, type="secondary")

st.divider()

file_uploader = st.container()#タイテ画像の登録
with file_uploader:
    st.markdown("""#### ②タイムテーブル画像の登録""")
    all_files_raw =  st.container(height=200)
    col_file_uploader = st.columns((3,1))
with all_files_raw:
    st.markdown("""###### 登録済みタイムテーブル画像一覧""")
    image_num = 0
    event_list = get_event_name_list()
    for i in range(len(event_list)):
        image_num += len(app_state.project.project_info_json["event_detail"][i]["timetables"])
        # image_num += len(os.listdir(os.path.join(st.session_state.pj_path, "event_{}".format(i))))
    st.markdown("- 画像数：{}".format(str(image_num)))
    if image_num <1:
        image_num=1
    col_all_files = st.columns(image_num)
    image_idx = 0
    for i, event_name in enumerate(event_list):
        event_type_list = get_event_type_list(i)
        for img_type in event_type_list:
            img_path = os.path.join(app_state.project.pj_path, event_name, img_type, "raw.png")
            if os.path.exists(img_path):
                with col_all_files[image_idx]:
                    col_uploaded_image = st.columns(2)
                    with col_uploaded_image[0]:
                        st.markdown("- {}/{}".format(event_name, img_type))
                    with col_uploaded_image[1]:
                        st.button("削除",key="delete_uploaded_image_{}".format(image_idx),on_click=delete_uploaded_image,args=(i, img_type))
                    image = get_image(img_path)
                    st.image(image)
            image_idx += 1
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
        # st.number_input("イベントNo.", min_value=1, max_value=st.session_state.project_info_json["event_num"], step=1, key="img_event_no")
        st.radio("種別", ("ライブ", "特典会", "両方(特典会別添え)", "両方(特典会併記)","その他", "その他(特典会併記)"), key="img_type", horizontal=True)
        st.text_input("その他の種別", key="img_type_alternative")
        if st.session_state.img_type != "両方(特典会併記)" and st.session_state.img_type != "その他(特典会併記)":
            st.radio("形式", ("通常", "ライムライト式"), key="img_format", horizontal=True)
        else:
            st.radio("形式", ("通常", "ライムライト式"), key="img_format", horizontal=True, disabled=True)
        if st.session_state.img_type in ["その他", "その他(特典会併記)"] and (st.session_state.img_type_alternative == "" or st.session_state.img_type_alternative is None):
            st.button(label="画像を登録する", on_click=determine_timetable_image, type="primary", disabled=True)
        else:
            st.button(label="画像を登録する", on_click=determine_timetable_image, type="primary")

if image_idx == 0:#画像登録確認
    st.warning("画像を登録してください")
    st.stop()
st.divider()

timetable_crop = st.container()#タイテ画像の切り取り
with timetable_crop:
    st.markdown("""#### ③タイムテーブル画像の切り取り""")
    st.info(
"""元のタイムテーブル画像に加工などを施して、読み取りが行えるための準備をします。  
大きく分けて3つのステップで準備します。  
- （ⅰ）まず必要最小限の領域に画像を切り出します。  
- （ⅱ）次に複数あるであろうステージごとに画像を分割します。  
- （ⅲ）最後に基準となる時間軸の位置を指定します（推奨だがオプション）。  
""")

    event_list = get_event_name_list()
    st.selectbox("イベント", event_list,index=0,key="crop_tgt_event",on_change=set_crop_image)
    crop_tgt_event_no = get_event_no_by_event_name(st.session_state.crop_tgt_event)
    event_type_list = get_event_type_list(crop_tgt_event_no)
    if len(event_type_list) == 0:
        st.warning("画像を登録するか他のイベントを選択してください")
        st.stop()
    st.selectbox("種別", event_type_list,index=0,key="crop_tgt_img_type",on_change=set_crop_image)
    img_path = os.path.join(app_state.project.pj_path, st.session_state.crop_tgt_event, st.session_state.crop_tgt_img_type, "raw.png")
    if os.path.exists(img_path):
        image = Image.open(img_path)
        image_info = app_state.project.project_info_json["event_detail"][crop_tgt_event_no]["timetables"][st.session_state.crop_tgt_img_type]
        # image_info = st.session_state.pj_timetable_master[
        #     (st.session_state.pj_timetable_master["event_no"]==int(st.session_state.crop_tgt_event.split("_")[1]))
        #     & (st.session_state.pj_timetable_master["image_type"]==st.session_state.crop_tgt_img_type)
        # ].iloc[0]

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
            col_cropimage_first = st.columns([2,1])
            with col_cropimage_first[0]:
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
            with col_cropimage_first[1]:
                st.markdown("""###### 切り出し結果""")
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
            edge_result = st.container()
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
                                #画像の幅（最頻値近辺）で採用不採用のデフォルト値を切り替えても良いかも
                                st.checkbox("採用",key="each_stage_accept_{}".format(i),value=True)
                                st.image(app_state.crop.images_eachstage[i])
                    with determine_image_eachstage_button_area_2:
                        st.button("ステージごとの画像を確定",on_click=determine_image_eachstage_without_nocheck,key="determine_image_eachstage_button_2")


            # if len(st.session_state.images_eachstage)>0:
            #     col_cropimage_eachstage = st.columns([img.size[0] for img in st.session_state.images_eachstage])
            #     for i,eachstage_image in enumerate(st.session_state.images_eachstage):
            #         with col_cropimage_eachstage[i]:
            #             st.image(eachstage_image)

            #     st.button("ステージごとの画像分割を確定",on_click=determine_image_eachstage, type="primary")

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
                col_timeaxis = st.columns([2,1])
                with col_timeaxis[0]:
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
                with col_timeaxis[1]:
                    st.image(cropped_image.crop((left, top, left+width, top+height)),use_container_width=True)
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

# timetable_crop = st.container()#タイテ画像の切り取り
# with timetable_crop:
#     st.markdown("""#### タイムテーブル画像の切り取り""")
#     st.info(
# """複数ステージが一つの画像に存在している場合、各ステージごとに画像を切り分けます。  
# この時、各ステージの幅が概ね均等であり、また間に時間軸なども等しく入っているor入っていない場合は、  
# 画像を均等に分割することによりステージごとの画像を取得できます。  
# しかし、そうでない場合には「均等割」できるいくつかの小さい画像に分割する必要があります。  
# 画像を見て、どちらのパターンかを判断して作業を行ってください。  
# """)
#     event_list = os.listdir(st.session_state.pj_path)
#     st.selectbox("イベント", event_list,index=0,key="crop_tgt_event")
#     st.selectbox("種別", ["ライブ","特典会"],index=0,key="crop_tgt_img_type")
#     img_path = os.path.join(app_state.project.pj_path, st.session_state.crop_tgt_event, st.session_state.crop_tgt_img_type, "raw.png")
#     if os.path.exists(img_path):
#         image = Image.open(img_path)
#         image_info = st.session_state.pj_timetable_master[
#             (st.session_state.pj_timetable_master["event_no"]==int(st.session_state.crop_tgt_event.split("_")[1]))
#             & (st.session_state.pj_timetable_master["image_type"]==st.session_state.crop_tgt_img_type)
#         ].iloc[0]

#         with st.container():# タイムテーブルに関係する領域を切り出す
#             st.markdown("""###### タイムテーブルに関係する領域を切り出す""")
#             st.info(
# """まず、タイムテーブルに関係する領域の切り出しを行います。  
# 必要十分な領域を指定してください。  
# ・ライムライト式の場合、時間軸の情報は含める必要があります（後で使います）  
# ・上下にステージ名などの情報は含まれていても構いません  
# ・そのまま一枚の画像として均等割を行う場合にはそれに適した領域に切り出してください  
# """)
#             col_cropimage_first = st.columns(2)
#             with col_cropimage_first[0]:
#                 st.session_state.cropped_image = st_cropper(image)
#             with col_cropimage_first[1]:
#                 st.image(st.session_state.cropped_image,use_container_width=True)

#         with st.container():# 「均等割」できるようにタイムテーブル領域を分割する
#             st.markdown("""###### 「均等割」できるようにタイムテーブル領域を分割する""")
#             st.info(
# """次に、タイムテーブルが均等でない場合、各々が均等な領域になるよう分割します。  
# 分割が不要な場合、この工程を無視して進んでください。  
# 分割が必要な場合、「分割を行う」をONにして分割を行ってください。  
  
# ドラッグで縦線を入力し、それに従って分割を行います。  
# ・実際には、縦線ではなく縦線の始点の横位置で分割します。線はイメージです  
# ・なので、斜めな線でも、画像の上から下まで線を引かなくとも、問題はありません  
# ・右側に表示される分割結果を参照して、適宜線を修正してください  
# ・なお画像の表示比率がおかしかったり欠けていたりするかもしれませんが、一旦気にせず進めてください  
  
# 分割が終了したら、各領域に何個のステージが含まれてているかを入力します。  
# ・タイムテーブル以外の情報（時間軸領域など）しか含まない領域を無視したい場合、ステージ数を0にすることで実現できます  
# ・なお次のステップにおいて、均等割ではなく矩形抽出による自動調整分割もできるので、必ずしも時間軸領域を無視せずとも動きますが、精度は100%ではないので、無視できるならしたほうがよいです  
# ・このステップで完全人力でステージ1つずつへの分割を行うこともできます。時間はかかりますがそれが一番確実です  
# """)
#             line_cropping = st.toggle("分割を行う")
#             if line_cropping:
#                 col_cropimage_line = st.columns(2)
#                 with col_cropimage_line[0]:
#                     canvas_width = 800
#                     canvas_rate = canvas_width/st.session_state.cropped_image.size[0]
#                     canvas_height = int(st.session_state.cropped_image.size[1]*canvas_rate)
#                     resized_image = st.session_state.cropped_image.resize((canvas_height, canvas_width))
#                     canvas_result = st_canvas(
#                         fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
#                         stroke_width=3,
#                         # stroke_color=stroke_color,
#                         # background_color=bg_color,
#                         background_image=resized_image,
#                         update_streamlit=True,
#                         width=canvas_width,
#                         height=canvas_height,
#                         drawing_mode="line",
#                         # point_display_radius=1,
#                         key="crop_line",
#                     )
#                 with col_cropimage_line[1]:
#                     if canvas_result.json_data is not None:
#                         objects = pd.json_normalize(canvas_result.json_data["objects"])
#                         # for col in objects.select_dtypes(include=["object"]).columns:
#                         #     objects[col] = objects[col].astype("str")
#                         # st.dataframe(objects)

#                         st.session_state.line_cropped_images = []
#                         line_x_before = 0
#                         for i,row in objects.iterrows():
#                             line_x = row["left"]+row["width"]/2
#                             st.session_state.line_cropped_images.append(st.session_state.cropped_image.crop((line_x_before/canvas_rate, 0, line_x/canvas_rate, st.session_state.cropped_image.size[1])))
#                             line_x_before = line_x
#                         st.session_state.line_cropped_images.append(st.session_state.cropped_image.crop((line_x_before/canvas_rate, 0, st.session_state.cropped_image.size[0], st.session_state.cropped_image.size[1])))

#                         col_cropimage_line_after = st.columns([max(img.size[0],st.session_state.cropped_image.size[0]/len(st.session_state.line_cropped_images)/2) for img in st.session_state.line_cropped_images])
#                         for i,line_cropped_image in enumerate(st.session_state.line_cropped_images):
#                             with col_cropimage_line_after[i]:
#                                 st.image(line_cropped_image)
#                                 st.number_input("ステージ数",0,value=1,step=1,key="stage_num_{}".format(i))
#             else:
#                 st.session_state.line_cropped_images = [st.session_state.cropped_image]
#                 st.number_input("ステージ数",1,step=1,key="stage_num_0")

#         with st.container():# 各ステージに1枚の画像が対応するよう分割する
#             st.markdown("""###### 各ステージに1枚の画像が対応するよう分割する""")
#             st.info(
# """最後に、ステージ一つひとつに一枚の画像が対応するよう、画像を分割します。  
# 画像の分割は、均等でないタイムテーブル領域を分割した場合にはそれぞれに対して、  
# 分割していない場合にはタイムテーブルに関係する領域として切り出した1枚に対して、  
# 指定されたステージ数への分割が実施されます。  
# 分割の方法は下記の通りです。どちらかを選んでボタンを押してください。  
# 1. OCRにより矩形を抽出し、ステージの幅を推定して分割する  
# 1. 画像を均等な横幅になるよう分割する  

# なお先述の通りステージ数が1の場合はそのまま一つのステージの画像として採用されます。  
# """)
#             # st.button("ステージごとの画像を抽出",on_click=get_image_eachstage_byocr,args=(cropped_image, st.session_state.stage_num))
#             st.button("OCR自動分割",on_click=get_image_eachstage_for_linecroppedimage_byocr)
#             st.button("横幅均等分割",on_click=get_image_eachstage_for_linecroppedimage_byevenly)

#             if len(st.session_state.images_eachstage)>0:
#                 col_cropimage_eachstage = st.columns([img.size[0] for img in st.session_state.images_eachstage])
#                 for i,eachstage_image in enumerate(st.session_state.images_eachstage):
#                     with col_cropimage_eachstage[i]:
#                         st.image(eachstage_image)

#                 st.button("ステージごとの画像分割を確定",on_click=determine_image_eachstage, type="primary")
st.divider()

timetable_ocr = st.container()#タイテ画像の読み取り
with timetable_ocr:
    st.markdown("""#### ④タイムテーブル画像の読み取り""")
    event_list = get_event_name_list()
    with st.expander("まとめて読み取りを実施"):
        st.info("""ライムライト式の画像の時刻推定も同時にまとめて行えますが、画像によって時刻の基準位置が違う場合が多いので、できるだけ先に時刻の読み取りだけ別途それぞれの画像で行うことを強く推奨します。""")
        col_toghether_ocr = st.columns([1,1,3])
        with col_toghether_ocr[0]:
            for i,event_name in enumerate(event_list):
                event_type_list = get_event_type_list(i)
                for event_type in event_type_list:
                    image_info = app_state.project.project_info_json["event_detail"][i]["timetables"][event_type]
                    stage_num = image_info["stage_num"]
                    if stage_num>0:
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

    st.selectbox("イベント", event_list,index=0,key="ocr_tgt_event",on_change=set_ocr_image)#変わった時にst.session_state.timeline_eachstageなどをリセット
    ocr_tgt_event_no = get_event_no_by_event_name(st.session_state.ocr_tgt_event)
    event_type_list = get_event_type_list(ocr_tgt_event_no)
    if len(event_type_list) == 0:
        st.warning("画像を登録するか他のイベントを選択してください")
        st.stop()
    st.selectbox("種別", event_type_list,index=0,key="ocr_tgt_img_type",on_change=set_ocr_image)#同上

    # チケットサイト情報使用オプション
    ticket_urls = get_ticket_urls_for_event(st.session_state.ocr_tgt_event)
    if len(ticket_urls) > 0:
        st.checkbox("チケットサイトの出演者情報を読み取りに使用する", value=True, key="use_ticket_urls",
                    help=f"登録済みURL: {', '.join(ticket_urls)}")
    else:
        st.session_state.use_ticket_urls = False

    img_path = os.path.join(app_state.project.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "raw.png")
    if os.path.exists(img_path):
        image = Image.open(img_path)
        st.session_state.ocr_tgt_image_info = app_state.project.project_info_json["event_detail"][ocr_tgt_event_no]["timetables"][st.session_state.ocr_tgt_img_type]
        st.session_state.ocr_tgt_stage_num = st.session_state.ocr_tgt_image_info["stage_num"]
        if st.session_state.ocr_tgt_stage_num<=0:
            st.warning("各ステージの画像を確定してください")
            st.stop()
#             cropped_img_path = os.path.join(app_state.project.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "raw_cropped.png")
#             if os.path.exists(cropped_img_path):# 時間軸の設定
#                 cropped_image = Image.open(cropped_img_path)
#                 with st.container():
#                     st.markdown("""###### 時間軸の設定""")
#                     st.info(
# """時間軸の基準となる位置を指定してください。  
# ・ライムライト式の画像では、この情報を元に時刻の読み取りを行います  
# ・全ての画像形式で、読み取り結果の画像化においてこの情報を元に時間軸を合わせます  
# """)
#                     col_timeaxis = st.columns([2,1])
#                     with col_timeaxis[0]:
#                         # canvas_height = st.number_input("表示する縦幅",min_value=100,step=100,value=800)
#                         canvas_height = 800
#                         canvas_rate = canvas_height/cropped_image.size[1]
#                         canvas_width = int(cropped_image.size[0]*canvas_rate)
#                         resized_image = cropped_image.resize((canvas_width, canvas_height))
#                         canvas_result = st_canvas(
#                             fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
#                             stroke_width=1,
#                             # stroke_color=stroke_color,
#                             # background_color=bg_color,
#                             background_image=resized_image,
#                             update_streamlit=True,
#                             width=canvas_width,
#                             height=canvas_height,
#                             drawing_mode="rect",
#                             # point_display_radius=1,
#                             key="time_axis_detect"
#                         )
#                     with col_timeaxis[1]:
#                         # st.time_input("開始時間", value=dttime(10), key=None, step=300)
#                         # st.time_input("終了時間", value=dttime(20), key=None, step=300)
#                         st.slider('開始時間', value=dttime(10), key="time_start", step=timedelta(minutes=5))
#                         st.slider('終了時間', value=dttime(20), key="time_finish", step=timedelta(minutes=5))
#                         total_duration = (datetime(2024,1,1,st.session_state.time_finish.hour,st.session_state.time_finish.minute)-datetime(2024,1,1,st.session_state.time_start.hour,st.session_state.time_start.minute)).seconds/60
#                         if st.session_state.time_finish.hour < st.session_state.time_start.hour:
#                             st.warning("終了時間が開始時間よりも早くなっています。深夜イベントなどで日を跨ぐ場合はそのまま実行可能ですが、そうでない場合は修正してください。")
#                         if canvas_result.json_data is not None:
#                             objects = pd.json_normalize(canvas_result.json_data["objects"])
#                             # for col in objects.select_dtypes(include=["object"]).columns:
#                             #     objects[col] = objects[col].astype("str")
#                             # st.dataframe(objects)
#                             try:
#                                 resized_start_pix = objects.iloc[0]["top"]
#                                 resized_total_pix = objects.iloc[0]["height"]
#                                 st.session_state.start_pix = resized_start_pix/canvas_rate
#                                 st.session_state.total_pix = resized_total_pix/canvas_rate
#                                 st.session_state.total_duration = total_duration 

#                                 # from PIL import ImageDraw
#                                 # lines = [
#                                 #     ((0, resized_start_pix/canvas_rate), (cropped_image.size[0], resized_start_pix/canvas_rate)),  # (x1, y1), (x2, y2)
#                                 #     ((0, (resized_start_pix+resized_total_pix)/canvas_rate), (cropped_image.size[0], (resized_start_pix+resized_total_pix)/canvas_rate))
#                                 # ]

#                                 # # 直線を描画する関数
#                                 # def draw_lines_on_image(image, lines, color="red", width=3):
#                                 #     draw = ImageDraw.Draw(image)
#                                 #     for line in lines:
#                                 #         draw.line(line, fill=color, width=width)
#                                 #     return image

#                                 # # 画像のコピーを作成し、直線を描画
#                                 # image_with_lines = cropped_image.copy()
#                                 # image_with_lines = draw_lines_on_image(image_with_lines, lines)

#                                 # # 直線を描画した画像を表示
#                                 # st.image(image_with_lines,use_container_width=True)

#                             except IndexError:
#                                 st.warning("画像上で時間範囲をドラッグしてください")

        with st.container():# 各ステージ情報の読み取り
            st.markdown("""###### 各ステージ情報の読み取り""")
            stage_name_list = get_stage_name_list(ocr_tgt_event_no,st.session_state.ocr_tgt_img_type)
            if st.session_state.ocr_tgt_image_info["format"]=="ライムライト式":
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
                # st.number_input("ステージ番号",0,st.session_state.ocr_tgt_stage_num-1,key="ocr_tgt_stage_no")
                # st.selectbox("ステージ",stage_name_list,index=0,key="ocr_tgt_stage_no")
                # st.button("あるステージのみ横線の時刻の読み取りを実施",on_click=detect_timeline_onlyonestage,args=(st.session_state.ocr_tgt_stage_no,))
                # st.button("あるステージのみタイムテーブルの読み取りを実施",on_click=get_timetabledata_onlyonestage_notime,args=(st.session_state.ocr_tgt_stage_no,st.session_state.ocr_user_prompt))
                with st.expander("時間ライン抽出のパラメータ"):
                    # st.slider('白黒二値化の閾値（0はグレースケールのまま実施）', value=0, min_value=0, max_value=255, step=1, key="y_binary_threshold")#, on_change=detect_timeline_onlyonestage,args=(st.session_state.ocr_tgt_stage_no,))
                    st.slider('無視する時間幅（分）（以下）', value=5, min_value=0, max_value=60, step=5, key="y_ignoretime_threshold")#, on_change=detect_timeline_onlyonestage,args=(st.session_state.ocr_tgt_stage_no,))
                    st.slider('エッジ抽出の閾値', value=150, min_value=1, max_value=500, step=1, key="y_edge_threshold_2")#, on_change=detect_timeline_onlyonestage,args=(st.session_state.ocr_tgt_stage_no,))
                    st.slider('抽出したエッジを伸ばす際の閾値（エッジ抽出の閾値以下にする）', value=80, min_value=1, max_value=500, step=1, key="y_edge_threshold_1")#, on_change=detect_timeline_onlyonestage,args=(st.session_state.ocr_tgt_stage_no,))
                    st.slider('ハフ変換の閾値', value=60, min_value=1, max_value=500, step=1, key="y_hough_threshold")#, on_change=detect_timeline_onlyonestage,args=(st.session_state.ocr_tgt_stage_no,))
                    st.slider('ハフ変換で許容する線分の飛び', value=1, min_value=0, max_value=100, step=1, key="y_hough_gap")#, on_change=detect_timeline_onlyonestage,args=(st.session_state.ocr_tgt_stage_no,))
                    st.slider('抽出線分の長さ（元画像の縦に対する比率）', value=0.05, min_value=0.0, max_value=1.0, step=0.01, key="y_minlength_rate")#, on_change=detect_timeline_onlyonestage,args=(st.session_state.ocr_tgt_stage_no,))
                    st.slider('同一視する線分の許容誤差幅', value=5, min_value=1, max_value=30, step=1, key="y_identify_interval")#, on_change=detect_timeline_onlyonestage,args=(st.session_state.ocr_tgt_stage_no,))
                tmp_timeline = st.container()#暫定
            elif st.session_state.ocr_tgt_image_info["format"]=="特典会併記":
                st.text_input("ステージ名読み取りの追加指示",key="ocr_stage_user_prompt")
                st.button("ステージ名の読み取りを実施",on_click=get_stagelist,args=(st.session_state.ocr_stage_user_prompt,),type="primary")
                st.text_input("タイムテーブル読み取りの追加指示",key="ocr_user_prompt")
                st.button("全ステージのタイムテーブルをそれぞれ読み取り実施", on_click=get_timetabledata_allstages_with_ticket_urls, args=("tokutenkai", st.session_state.ocr_user_prompt), type="primary")
                # st.number_input("ステージ番号",0,st.session_state.ocr_tgt_stage_num-1,key="ocr_tgt_stage_no")
                # st.button("あるステージのみタイムテーブルの読み取りを実施",on_click=get_timetabledata_withtokutenkai_onlyonestage,args=(st.session_state.ocr_tgt_stage_no,st.session_state.ocr_user_prompt))
                st.button("全ステージの特典会ブース名にステージ名を接頭辞として付与する",on_click=booth_name_add_prefix_eachstage)
            else:
                st.text_input("ステージ名読み取りの追加指示",key="ocr_stage_user_prompt")
                st.button("ステージ名の読み取りを実施",on_click=get_stagelist,args=(st.session_state.ocr_stage_user_prompt,),type="primary")
                st.text_input("タイムテーブル読み取りの追加指示",key="ocr_user_prompt")
                st.button("全ステージのタイムテーブルをそれぞれ読み取り実施", on_click=get_timetabledata_allstages_with_ticket_urls, args=("normal", st.session_state.ocr_user_prompt), type="primary")
                # st.number_input("ステージ番号",0,st.session_state.ocr_tgt_stage_num-1,key="ocr_tgt_stage_no")
                # st.button("あるステージのみタイムテーブルの読み取りを実施",on_click=get_timetabledata_onlyonestage,args=(st.session_state.ocr_tgt_stage_no,st.session_state.ocr_user_prompt))

            #個別ステージ
            with st.container():#表示設定
                if st.session_state.ocr_tgt_image_info["format"]=="ライムライト式":
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

                
                ocr_eachimage_width_default = 40#int((image_tmp.size[0])*100/(image_tmp.size[0]+600))
                # output_eachimage_width_default = int(ocr_eachimage_width_default*100/(100-ocr_eachimage_width_default))
                st.slider('タイテ元画像の表示幅（%）', value=ocr_eachimage_width_default, min_value=1, max_value=99, step=1, key="ocr_eachimage_width")#, on_change=detect_timeline_onlyonestage,args=(st.session_state.ocr_tgt_stage_no,))
                # ocr_slider_col = st.columns([st.session_state.ocr_eachimage_width,100-st.session_state.ocr_eachimage_width])
                # with ocr_slider_col[1]:
                #     st.slider('タイテ出力画像の表示幅（%）', value=st.session_state.ocr_eachimage_width, min_value=0, max_value=100-st.session_state.ocr_eachimage_width, step=1, key="output_eachimage_width")#, on_change=detect_timeline_onlyonestage,args=(st.session_state.ocr_tgt_stage_no,))
                ocr_show_setting_col = st.columns([1,1,1,1,1])
                with ocr_show_setting_col[0]:
                    st.checkbox("画像の縦スクロール表示", value=True, key="ocr_eachimage_scroll")
                with ocr_show_setting_col[1]:
                    st.checkbox("読み取り結果の画像の時間軸を元画像に合わせる", value=True, key="ocr_output_picture_time_match",on_change=output_timetable_picture_eachstage , help="""「時間軸の設定」で指定したラインに合わせて画像を生成します。  
このチェックボックスのオンオフまたはステージの編集結果の保存で画像が更新されます。""")

            stage_tabs = st.tabs(stage_name_list)
            st.session_state.df_timetables = []
            for i in range(st.session_state.ocr_tgt_stage_num):
                with stage_tabs[i]:
                    if st.session_state.ocr_tgt_image_info["format"]=="ライムライト式":
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
                                new_width = image.width + image_output.width
                                new_height = max(image.height, image_output.height)
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
                        if st.session_state.ocr_tgt_image_info["format"]=="ライムライト式":
                            st.button("このステージの横線の時刻の読み取りを実施",on_click=detect_timeline_onlyonestage,args=(i,),key="button_ocr_timeline_stage{}".format(i))
                            st.text_input("タイムテーブル読み取りの追加指示",key="ocr_user_prompt_stage{}".format(i))
                            st.button("このステージのタイムテーブルの読み取りを実施", on_click=get_timetabledata_onestage_with_ticket_urls, args=("notime", i, st.session_state["ocr_user_prompt_stage{}".format(i)]), key="button_ocr_stage{}".format(i))
                        elif st.session_state.ocr_tgt_image_info["format"]=="特典会併記":
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
                            if st.session_state.ocr_tgt_image_info["format"]=="特典会併記":
                                return_json_df = timetabledata.json_to_df(return_json)
                            else:
                                return_json_df = timetabledata.json_to_df(return_json,tokutenkai=False)
                            # st.dataframe(return_json_df)
                            edited_df = st.data_editor(return_json_df, key="timetabledata_stage{}".format(i), num_rows="dynamic")
                            
                            # edited_df["ステージID"]=i
                            edited_df["ステージ名"]=stage_name
                            st.session_state.df_timetables.append(edited_df)

                            # st.button("このステージのグループ名を修正（マスタ参照）",on_click=idolname_correct_onlyonestage,args=(i,),key="button_correct_idolname_stage{}".format(i))
                            if st.button("このステージのグループ名を修正（マスタ参照）",key="button_correct_idolname_stage{}_confirm".format(i)):
                                st.warning('「グループ名_採用」が上書きされます。本当に処理を実行しますか？')
                                st.button("OK",on_click=idolname_correct_onlyonestage,args=(i,),key="button_correct_idolname_stage{}".format(i))
                            st.button("このステージの編集結果を保存",on_click=save_timetable_data_onlyonestage,args=(i,),key="button_save_timetable_stage{}".format(i))

            # st.button("全ステージのグループ名を修正（マスタ参照）",on_click=idolname_correct_eachstage,key="button_correct_idolname")
            st.checkbox("既に確定したタイテ種別で採用したグループ名一覧の中からグループ名を選ぶ",key="correct_idolname_in_confirmed_list",help="""例えばライブのタイムテーブルを先に作成し、後から特典会のタイムテーブルを作成する際に、ライブのタイムテーブルデータで「グループ名_採用」に入力したグループ名の一覧を候補として、特典会のタイムテーブルデータでも「グループ名_採用」への修正を行うことが出来ます。  
この処理はイベントごとに切り分けて行われるため、day1はday1の中で候補を用意してグループ名を修正し、day2はday2でまた別になります。  
ライブと特典会、あるいは他の種別についてはどのような順番でもよく、「全種別を通じて既に『グループ名_修正』に入力されているグループ一覧」が候補になります。  
どの種別においても一つもグループ名を確定していない場合は、通常通り全グループリストから出力されます。""")
            if st.button("全ステージのグループ名を修正（マスタ参照）",key="button_correct_idolname_confirm"):
                st.warning('「グループ名_採用」が上書きされます。本当に処理を実行しますか？')
                st.button('OK', on_click=idolname_correct_eachstage,key="button_correct_idolname")
            st.button("全ステージの編集結果を保存",on_click=save_timetable_data_eachstage,key="button_save_timetable")

            # st.button("全ステージのデータを表形式で出力",on_click=get_all_stage_info,type="primary")
            # all_stage_info = st.container()
st.divider()

timetable_change = st.container()#タイテ画像の追加・変更
with timetable_change:
    st.markdown("""#### ⑤タイムテーブル画像の追加・変更
- 読み取りを行った後にタイムテーブルが変更になった場合に、画像の変更点のみを読み取って修正してくれる機能をいつか実装します""")
    timetable_compare_setting_col = st.columns(2)
    timetable_compare_col = st.columns(2)
    with timetable_compare_setting_col[0]:
        st.file_uploader("読み取りたいタイムテーブル画像をアップロードしてください。"
                                , type=["jpg", "jpeg", "png", "jfif"]
                                , key="uploaded_image_updated")
    with timetable_compare_col[0]:
        if st.session_state.uploaded_image_updated is not None:
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
            st.stop()
        st.selectbox("種別", event_type_list,index=0,key="diff_tgt_img_type")#同上
        st.button("差分画像を出力する",on_click=output_difference_image,args=(st.session_state.uploaded_image_updated,))
        if st.session_state.uploaded_image_updated is not None:
            st.button("画像を置き換える",on_click=replace_stage_images_from_new_raw,args=(st.session_state.uploaded_image_updated,))

st.divider()

timetable_output = st.container()#タイテ情報の出力
app_state.output.output_df = {}
with timetable_output:
    st.markdown("""#### ⑥タイムテーブル情報の出力""")

    event_list = get_event_name_list()
    event_tabs = st.tabs(event_list)
    for i, event_tab in enumerate(event_tabs):
        app_state.output.output_df[event_list[i]]={}
        with event_tab:#イベントごとに出力を作る
            edit_tgt_event_no = get_event_no_by_event_name(event_list[i])
            event_type_list = get_event_type_list(edit_tgt_event_no)
            output_path =  os.path.join(app_state.project.pj_path, event_list[i])
            if os.path.exists(os.path.join(output_path, "master_stage.csv")):
                stage_master_df = pd.read_csv(os.path.join(output_path, "master_stage.csv"), index_col=0)
                stage_master = json.loads(stage_master_df.T.to_json())
                tokutenkai_timetable = []
                stage_id = int(max(stage_master_df.index))+1
            else:
                stage_master = {}
                tokutenkai_timetable = []
                stage_id = 0
            if os.path.exists(os.path.join(output_path, "master_idolname.csv")):
                idolname_master_df = pd.read_csv(os.path.join(output_path, "master_idolname.csv"), index_col=0).rename(columns={"グループ名":"グループ名_採用"})
                artist_id = int(max(idolname_master_df.index))+1
            else:
                idolname_master_df = pd.DataFrame(columns=["グループID","グループ名_採用"]).set_index("グループID")
                artist_id = 0
            # if os.path.exists(os.path.join(output_path, "turn_id_data.csv")):
            #     live_master_df = pd.read_csv(os.path.join(output_path, "turn_id_data.csv"), index_col=0)

            stage_master_tokutenkai = {}
            event_timetable_all = []
            for event_type in event_type_list:#全種別をまとめる
                tgt_event_type_info = app_state.project.project_info_json["event_detail"][edit_tgt_event_no]["timetables"][event_type]
                stage_name_list = get_stage_name_list(edit_tgt_event_no,event_type)
                tokutenkai_flg = event_type=="特典会"
                for j in range(tgt_event_type_info["stage_num"]):
                    json_path = os.path.join(st.session_state.pj_path, event_list[i], event_type, "stage_{}.json".format(j))
                    if os.path.exists(json_path):
                        try:
                            with open(json_path, encoding="utf-8") as f:
                                edit_tgt_json = json.load(f)
                            if tgt_event_type_info["format"]=="特典会併記":#特典会併記タイテは分離してライブのみをまず扱う
                                df_edit_tgt = timetabledata.json_to_df(edit_tgt_json, tokutenkai=True)
                                df_edit_live, df_edit_tokutenkai = timetabledata.devide_df_live_tokutenkai(df_edit_tgt)
                                df_edit_tokutenkai = df_edit_tokutenkai[ df_edit_tokutenkai['ライブ_長さ(分)'].notnull() & (df_edit_tokutenkai['ライブ_長さ(分)'] != '') ]
                                # st.dataframe(df_edit_live)
                                # st.dataframe(df_edit_tokutenkai)
                                tokutenkai_timetable.append(df_edit_tokutenkai)
                            else:
                                df_edit_live = timetabledata.json_to_df(edit_tgt_json, tokutenkai=False)
                            df_edit_live = df_edit_live[ df_edit_live['ライブ_長さ(分)'].notnull() & (df_edit_live['ライブ_長さ(分)'] != '') ]
                            df_edit_live = df_edit_live.copy()
                            for k,v in stage_master.items():#既にID確定済のステージの場合はそれを採用
                                if v["ステージ名"]==stage_name_list[j]:
                                    this_stage_id = k
                                    break
                            else:
                                this_stage_id = stage_id
                                stage_master[this_stage_id]={"ステージ名":stage_name_list[j],"特典会フラグ":tokutenkai_flg}
                                stage_id += 1
                            if "ステージID" not in df_edit_live.columns:
                                df_edit_live["ステージID"]=None
                                df_edit_live["ステージ名"]=None
                            df_edit_live.loc[:,"ステージID"]=this_stage_id
                            df_edit_live.loc[:,"ステージ名"]=stage_name_list[j]
                            event_timetable_all.append(df_edit_live)
                        except KeyError:
                            pass
            if len(tokutenkai_timetable)>0:#特典会併記タイテの場合の特典会情報の処理
                df_tokutenkai = pd.concat((tokutenkai_timetable)).reset_index(drop=True)
                if "ステージID" in df_tokutenkai.columns:
                    df_tokutenkai = df_tokutenkai.drop(columns=["ステージID"])
                booth_name_list = df_tokutenkai["ステージ名"].drop_duplicates().tolist()
                for j, booth_name in enumerate(booth_name_list):
                    for k,v in stage_master.items():
                        if v["ステージ名"]==booth_name:
                            this_stage_id = k
                            break
                    else:
                        this_stage_id = stage_id
                        stage_master[this_stage_id]={"ステージ名":booth_name,"特典会フラグ":True}
                        stage_id += 1
                    stage_master_tokutenkai[this_stage_id]={"ステージ名":booth_name,"特典会フラグ":True}
            if len(event_timetable_all)==0:
                st.stop()
            df_stage = pd.DataFrame.from_dict(stage_master, orient='index')
            df_stage.index.name = "ステージID"
            df_stage_tokutenkai = pd.DataFrame.from_dict(stage_master_tokutenkai, orient='index')
            df_stage_tokutenkai.index.name = "ステージID"
            df_live = pd.concat((event_timetable_all)).reset_index(drop=True)
            if len(tokutenkai_timetable)>0:
                df_tokutenkai = pd.merge(df_tokutenkai,df_stage_tokutenkai.reset_index().drop("特典会フラグ",axis=1),on="ステージ名",how="left")
                df_live = pd.concat((df_live, df_tokutenkai)).reset_index(drop=True)
            ## ここまでがステージマスタ作成
            ## ここからアーティストマスタ作成
            df_idolname = pd.DataFrame(df_live["グループ名_採用"].drop_duplicates().sort_values().reset_index(drop=True))
            df_idolname = df_idolname[~df_idolname["グループ名_採用"].isin(idolname_master_df["グループ名_採用"])].reset_index(drop=True)
            df_idolname.index = df_idolname.index + artist_id
            df_idolname.index.name = 'グループID'
            df_idolname = pd.concat((idolname_master_df,df_idolname))
            ## ここまでがアーティストマスタ作成
            ## ここから出番データ作成
            if "グループID" in df_live.columns:
                df_live = df_live.drop(columns=["グループID"])
            df_live = pd.merge(df_live,df_idolname.reset_index(),on="グループ名_採用",how="left").rename(columns={"グループ名":"グループ名_raw","グループ名_採用":"グループ名"})
            if "出番ID" in df_live:
                turn_id = int(df_live["出番ID"].max())+1
                for row_id, row in df_live.iterrows():
                    try:
                        df_live.loc[row_id,"出番ID"]=int(row["出番ID"])
                    except ValueError:
                        df_live.loc[row_id,"出番ID"]=turn_id
                        turn_id+=1
                df_live["出番ID"]=df_live["出番ID"].astype(int)
                df_live.set_index("出番ID",inplace=True)
            else:
                df_live.reset_index(drop=True,inplace=True)
                df_live.index.name = "出番ID"
            output_cols = st.columns([1,1,3])
            with output_cols[0]:
                st.dataframe(df_stage)
            with output_cols[1]:
                st.dataframe(df_idolname)
            with output_cols[2]:
                st.dataframe(df_live[["ライブ_from","ライブ_長さ(分)","グループID","ステージID","グループ名_raw","グループ名","ステージ名","備考"]])
            app_state.output.output_df[event_list[i]]["stage"]=df_stage
            app_state.output.output_df[event_list[i]]["idolname"]=df_idolname
            app_state.output.output_df[event_list[i]]["live"]=df_live[["ライブ_from","ライブ_長さ(分)","グループID","ステージID","グループ名_raw","グループ名","ステージ名","備考"]]
            # st.dataframe(df_live[["グループID","グループ名","グループ名_採用","ライブ_from","ライブ_to","ライブ_長さ(分)","ステージ名","ステージID","備考"]])

    st.button("IDマスタを確定",on_click=determine_id_master)
    st.button("プロジェクトデータをクラウドにアップロード ※通信料・保存料が発生するので留意",on_click=save_to_s3)
    if st.button("Excelデータを出力",on_click=output_data_for_stella):#全イベントのタイテをシートに分けてExcelで出力
        file_path =  os.path.join(app_state.project.pj_path, "output.xlsx")
        with open(file_path, "rb") as file:
            excel_data = file.read()
        st.download_button("ファイルをダウンロード",data=excel_data, file_name="{}.xlsx".format(app_state.project.pj_name), mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.divider()

idolname_add = st.container()

with idolname_add:
    st.markdown("""#### ⑦マスタのアップデート""")
    st.button("新規登場の「グループ名_採用」をリストアップ",on_click=listup_new_idolname)
    if app_state.output.new_idolname is not None:
        df_new_idolname = st.data_editor(app_state.output.new_idolname,num_rows="dynamic")
        st.button("チェックしたグループ名をマスタに追加", on_click=update_master_idolname, args=(df_new_idolname,))