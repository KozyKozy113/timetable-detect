import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
import os
from PIL import Image
import json

import gptocr
import timetabledata

st.set_page_config(
    page_title="対バンタイムテーブル読み取り", 
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
DATA_PATH = DIR_PATH +"/../data"
INPUT_PATH =  os.path.join(DATA_PATH, "input")
OUTPUT_PATH =  os.path.join(DATA_PATH, "output")

def upload_file():
    file = st.session_state.timetable_image
    print(type(file))
    print(dir(file))
    if file is not None:
        img_path = os.path.join(INPUT_PATH, file.name)
        with open(img_path, 'wb') as f:
            f.write(file.read())

def ocr():
    filename = st.session_state.timetable_image_filename
    image_path = os.path.join(INPUT_PATH, filename)
    prompt_user = "この画像のタイムテーブルをデータ化して"
    return_json = gptocr.getocr_taiban(image_path, prompt_user)
    json_path = os.path.join(OUTPUT_PATH, "timetable_taiban/{file_name}.json".format(file_name=filename.split(".")[0]))
    with open(json_path,"w",encoding = "utf8") as f:
        json.dump(return_json, f, indent = 4, ensure_ascii = False)

st.file_uploader("読み取りたい対バンのタイムテーブル画像をアップロードしてください。"
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
                               , index = default_index
                               , placeholder = "画像をアップロードまたは画像ファイルをリストから選択"
                               , key="timetable_image_filename")

image_col, timetable_col = st.columns(2)
if st.session_state.timetable_image_filename is not None:
    filename = st.session_state.timetable_image_filename

    with image_col:
        st.button(label="OCRを実行する",on_click=ocr)
        img_path = os.path.join(INPUT_PATH, filename)
        image = Image.open(img_path)
        st.image(
            image, caption="元となるタイムテーブル画像",
            use_column_width=True
        )

    json_path = os.path.join(OUTPUT_PATH, "timetable_taiban/{file_name}.json".format(file_name=filename.split(".")[0]))
    if os.path.exists(json_path):
        with open(json_path,"r",encoding = "utf8") as f:
            return_json = json.load(f)
        df_timetable = timetabledata.json_to_df(return_json)

        gb = GridOptionsBuilder.from_dataframe(df_timetable)
        gb.configure_default_column(editable=True)
        grid_options = gb.build()

        with timetable_col:
            st.button(label="OCRを実行する",on_click=ocr)
            live_title = st.text_input(label="ライブ名", value=return_json["ライブ名"])
            live_place = st.text_input(label="会場名", value=return_json["会場名"])
            live_date = st.text_input(label="日付", value=return_json["日付"])
            grid_response = AgGrid(df_timetable, gridOptions=grid_options, height=800 ,editable=True)

        # 編集後のデータフレームを取得
        edited_df = grid_response['data']

        # 編集後のデータフレームを表示
        # st.write("Edited data:")
        # st.write(edited_df)