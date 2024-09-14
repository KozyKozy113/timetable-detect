import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from streamlit_cropper import st_cropper

import os
from PIL import Image
# import cv2
import numpy as np
import json

import gptocr
import timetabledata

st.set_page_config(
    page_title="ライムライトのフェス用タイムテーブル読み取り", 
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

DIR_PATH = os.path.dirname(__file__)
DATA_PATH = DIR_PATH +"/../../data"
INPUT_PATH =  os.path.join(DATA_PATH, "input")
OUTPUT_PATH =  os.path.join(DATA_PATH, "output/projects")

def upload_file():
    file = st.session_state.timetable_image
    print(type(file))
    print(dir(file))
    if file is not None:
        img_path = os.path.join(INPUT_PATH, file.name)
        with open(img_path, 'wb') as f:
            f.write(file.read())
        project_make(file.name)

def project_make(filename=None):
    if filename is None:
        filename = st.session_state.timetable_image_filename
    st.session_state.pjfolder = os.path.join(OUTPUT_PATH, filename.split(".")[0])
    json_path = os.path.join(st.session_state.pjfolder, "timetable.json")
    if os.path.exists(json_path):
        with open(json_path,"r",encoding = "utf8") as f:
            return_json = json.load(f)
        st.session_state.json_timetable = return_json
        st.session_state.df_timetable = timetabledata.json_to_df(return_json)
    else:
        if "json_timetable" in st.session_state:
            del st.session_state.json_timetable
            del st.session_state.df_timetable
    json_path = os.path.join(st.session_state.pjfolder, "timetable_fesinfo.json")
    if os.path.exists(json_path):
        with open(json_path,"r",encoding = "utf8") as f:
            return_json = json.load(f)
        st.session_state.json_timetable_fesinfo = return_json
        st.session_state.fes_flag=True
    else:
        if "json_timetable_fesinfo" in st.session_state:
            del st.session_state.json_timetable_fesinfo
        st.session_state.fes_flag=False
    json_path = os.path.join(st.session_state.pjfolder, "timetable_stage0.json")
    if os.path.exists(json_path):
        i=0
        return_json_by_stage=[]
        df_timetables=[]
        while True:
            json_path = os.path.join(st.session_state.pjfolder, "timetable_stage{}.json".format(i))
            if os.path.exists(json_path):
                with open(json_path,"r",encoding = "utf8") as f:
                    return_json = json.load(f)
                return_json_by_stage.append(return_json)
                df_timetables.append(timetabledata.json_to_df(return_json,tokutenkai=False))
                i+=1
            else:
                break
        st.session_state.json_timetable_fes = return_json_by_stage
        st.session_state.df_timetable_fes = df_timetables
    else:
        if "json_timetable_fes" in st.session_state:
            del st.session_state.json_timetable_fes
            del st.session_state.df_timetable_fes
    
def ocr_taiban():
    filename = st.session_state.timetable_image_filename
    os.makedirs(st.session_state.pjfolder, exist_ok=True)
    image_path = os.path.join(INPUT_PATH, filename)
    prompt_user = "この画像のタイムテーブルをデータ化して"
    return_json = gptocr.getocr_taiban(image_path, prompt_user)
    json_path = os.path.join(st.session_state.pjfolder, "timetable.json")
    with open(json_path,"w",encoding = "utf8") as f:
        json.dump(return_json, f, indent = 4, ensure_ascii = False)
    st.session_state.json_timetable = return_json
    st.session_state.df_timetable = timetabledata.json_to_df(return_json)

def ocr_fes_first():
    filename = st.session_state.timetable_image_filename
    os.makedirs(os.path.join(st.session_state.pjfolder,"cropped"), exist_ok=True)
    image_path = os.path.join(INPUT_PATH, filename)
    return_json = gptocr.getocr_fes_info(image_path)
    json_path = os.path.join(st.session_state.pjfolder, "timetable_fesinfo.json")
    with open(json_path,"w",encoding = "utf8") as f:
        json.dump(return_json, f, indent = 4, ensure_ascii = False)
    st.session_state.json_timetable_fesinfo = return_json
    st.session_state.fes_flag=True

def ocr_fes_second():
    filename = st.session_state.timetable_image_filename
    fes_timetable_image_path = os.path.join(st.session_state.pjfolder, "cropped/all.jpg")
    st.session_state.cropped_image.save(fes_timetable_image_path)
    #croppedの中の分割したやつを全部一度削除したほうがよい

    return_json_by_stage = []
    df_timetables = []
    width, height = st.session_state.cropped_image.size
    segment_width = width // st.session_state.stage_num
    for i in range(st.session_state.stage_num):
        left = i * segment_width
        right = (i + 1) * segment_width if i < stage_num - 1 else width
        segment = st.session_state.cropped_image.crop((left, 0, right, height))
        fes_timetable_image_path = os.path.join(st.session_state.pjfolder, "cropped/{}.jpg".format(i))
        segment.save(fes_timetable_image_path)
        return_json = gptocr.getocr_fes_timetable(fes_timetable_image_path)
        return_json["ステージ名"] = st.session_state.stage_names[i]
        return_json_by_stage.append(return_json)
        df_timetables.append(timetabledata.json_to_df(return_json,tokutenkai=False))
        json_path = os.path.join(st.session_state.pjfolder, "timetable_stage{}.json".format(i))
        with open(json_path,"w",encoding = "utf8") as f:
            json.dump(return_json, f, indent = 4, ensure_ascii = False)
    st.session_state.json_timetable_fes = return_json_by_stage
    st.session_state.df_timetable_fes = df_timetables
    # st.success(f'Cropped image saved as {save_path}')

def idolname_correct():
    if not st.session_state.fes_flag:
        df_timetable = st.session_state.df_timetable
        for i,row in df_timetable.iterrows():
            group_name_correct = timetabledata.get_name_list(row["グループ名"])
            if not group_name_correct[0]:
                df_timetable.at[i,"グループ名"] = group_name_correct[1]
        st.session_state.df_timetable = df_timetable
    else:
        df_timetables = st.session_state.df_timetable_fes
        df_timetables_corrected = []
        for df_timetable in df_timetables:
            for i,row in df_timetable.iterrows():
                group_name_correct = timetabledata.get_name_list(row["グループ名"])
                if not group_name_correct[0]:
                    df_timetable.at[i,"グループ名"] = group_name_correct[1]
            df_timetables_corrected.append(df_timetable)
        st.session_state.df_timetable_fes = df_timetables_corrected

if "fes_flag" not in st.session_state:
    st.session_state.fes_flag = False

st.file_uploader("読み取りたいタイムテーブル画像をアップロードしてください。"
                        , type=["jpg", "jpeg", "png"]
                        , on_change=upload_file
                        , key="timetable_image")
        
options = os.listdir(INPUT_PATH)
default_index = 0

if st.session_state.timetable_image is not None:
    default_option = st.session_state.timetable_image.name
    for i, file_name in enumerate(options):
        if file_name == default_option:
            default_index = i
            break
 
selected_option = st.selectbox("読み取りを行う画像ファイルを選択してください。"
                               , options
                               , on_change = project_make
                               , index = default_index
                               , placeholder = "画像をアップロードまたは画像ファイルをリストから選択"
                               , key="timetable_image_filename")
if "pjfolder" not in st.session_state:
    project_make()

image_col, timetable_col = st.columns([2,3])
if st.session_state.timetable_image_filename is not None:
    filename = st.session_state.timetable_image_filename

    with image_col:
        ocr_button = st.columns(2)
        with ocr_button[0]:
            st.button(label="対バンOCRを実行する",on_click=ocr_taiban)
        with ocr_button[1]:
            st.button(label="フェスOCR(Step1)を実行する",on_click=ocr_fes_first)
        img_path = os.path.join(INPUT_PATH, filename)
        image = Image.open(img_path)
        if not st.session_state.fes_flag:
            st.image(
                image, #caption="元となるタイムテーブル画像",
                use_column_width=True
            )
        else:
            st.write("必要十分なタイムテーブル領域を選択してください。ステージ数で縦割りして読み取りを実施します。")
            st.button(label="フェスOCR(Step2)を実行する",on_click=ocr_fes_second)
            st.session_state.cropped_image = st_cropper(image)
            st.image(st.session_state.cropped_image, caption='Cropped Image')

    if not st.session_state.fes_flag:
        # json_path = os.path.join(OUTPUT_PATH, "{file_name}/timetable.json".format(file_name=filename.split(".")[0]))
        if "json_timetable" in st.session_state:
            return_json = st.session_state.json_timetable

            gb = GridOptionsBuilder.from_dataframe(st.session_state.df_timetable)
            gb.configure_default_column(editable=True)
            grid_options = gb.build()

            with timetable_col:
                st.button(label="アイドル名を自動修正する",on_click=idolname_correct)
                live_title = st.text_input(label="ライブ名", value=return_json["ライブ名"])
                live_place = st.text_input(label="会場名", value=return_json["会場名"])
                live_date = st.text_input(label="日付", value=return_json["日付"])
                grid_response = AgGrid(st.session_state.df_timetable, gridOptions=grid_options, height=800 ,editable=True)
            st.session_state.df_timetable = grid_response['data']#編集後のデータ

    else:
        # json_path_info = os.path.join(OUTPUT_PATH, "{file_name}/timetable_fesinfo.json".format(file_name=filename.split(".")[0]))
        if "json_timetable_fesinfo" in st.session_state:
            return_json_fesinfo = st.session_state.json_timetable_fesinfo

            with timetable_col:
                st.button(label="アイドル名を自動修正する",on_click=idolname_correct)
                live_title = st.text_input(label="ライブ名", value=return_json_fesinfo["ライブ名"])
                live_place = st.text_input(label="会場名", value=return_json_fesinfo["会場名"])
                live_date = st.text_input(label="日付", value=return_json_fesinfo["日付"])
                stage_num = len(return_json_fesinfo["ステージ名"])
                st.number_input(label="ステージ数", value=stage_num, key="stage_num", step=1)
                stage_name_cols = st.columns(st.session_state.stage_num)
                stage_names = ["" for i in range(st.session_state.stage_num)]
                for i in range(st.session_state.stage_num):
                    with stage_name_cols[i]:
                        if i < len(return_json_fesinfo["ステージ名"]):
                            stage_names[i] = st.text_input(label="ステージ{}".format(i+1), value=return_json_fesinfo["ステージ名"][i])
                        else:
                            stage_names[i] = st.text_input(label="ステージ{}".format(i+1), placeholder="ステージ名を入力")
                st.session_state.stage_names = stage_names

        if "json_timetable_fes" in st.session_state:
            return_json_bystage = st.session_state.json_timetable_fes
            df_timetables = st.session_state.df_timetable_fes
            stage_names = [return_json["ステージ名"] for return_json in return_json_bystage]
            with timetable_col:
                stage_name_tabs = st.tabs(stage_names)
                for i in range(len(stage_names)):
                    return_json = return_json_bystage[i]["タイムテーブル"]
                    with stage_name_tabs[i]:
                        gb = GridOptionsBuilder.from_dataframe(df_timetables[i])
                        gb.configure_default_column(min_column_width=1000, editable=True)#width指定うまくいかない
                        grid_options = gb.build()
                        grid_response = AgGrid(df_timetables[i], gridOptions=grid_options, height=800 ,editable=True)
                        st.session_state.df_timetable_fes[i] = grid_response['data']
