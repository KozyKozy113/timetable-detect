import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from streamlit_cropper import st_cropper
from streamlit_drawable_canvas import st_canvas

import os
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import time as dttime
from datetime import timedelta
import json

# import gptocr
# import timetabledata

from backend_functions import gpt_ocr, timetabledata

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

if "pj_name" not in st.session_state:
    st.session_state.pj_name = None
    st.session_state.project_master = pd.read_csv(os.path.join(DATA_PATH, "master", "projects_master.csv"), index_col=0)
    st.session_state.timetable_image_master = pd.read_csv(os.path.join(DATA_PATH, "master", "timetable_image_master.csv"))
pj_name_list = st.session_state.project_master.index.to_list()[::-1]

def make_project(pj_name=None):
    if pj_name is None:
        pj_name = st.session_state.new_pj_name
    if pj_name in pj_name_list:
        with project_setting:
            st.error("既に存在するプロジェクトです")
    else:
        created_at = datetime.now().strftime('%Y/%m/%d %H:%M:%S.%f')
        st.session_state.project_master.loc[pj_name] = [created_at,"",1]
        st.session_state.project_master.to_csv(os.path.join(DATA_PATH, "master", "projects_master.csv"))
        set_project(pj_name)

def set_project(pj_name):
    st.session_state.pj_name = pj_name
    st.session_state.exist_pj_name = pj_name
    os.makedirs(os.path.join(DATA_PATH, "projects", st.session_state.pj_name), exist_ok=True)
    st.session_state.pj_path = DATA_PATH +"/projects/"+pj_name
    project_info = st.session_state.project_master.loc[st.session_state.pj_name]
    if not pd.isna(project_info["event_type"]):
        st.session_state.event_type = project_info["event_type"]
    else:
        st.session_state.event_type = "対バン"
    if not pd.isna(project_info["event_num"]):
        st.session_state.event_num = project_info["event_num"]
    else:
        st.session_state.event_num = 1
    st.session_state.pj_timetable_master = st.session_state.timetable_image_master[st.session_state.timetable_image_master["project_name"]==st.session_state.pj_name]
    #その他以後の変数も初期化等する必要あり
    st.session_state.images_eachstage=[]#実際にはロードできるならロードする

def determine_project_setting():
    st.session_state.project_master.loc[st.session_state.pj_name,"event_type"] = st.session_state.event_type
    st.session_state.project_master.loc[st.session_state.pj_name,"event_num"] = st.session_state.event_num
    st.session_state.project_master.to_csv(os.path.join(DATA_PATH, "master", "projects_master.csv"))
    for i in range(1,1+st.session_state.event_num):
        os.makedirs(os.path.join(DATA_PATH, "projects", st.session_state.pj_name, "event_"+str(i)), exist_ok=True)
        # os.makedirs(os.path.join(DATA_PATH, "projects", st.session_state.pj_name, "event_"+str(i), "ライブ"), exist_ok=True)
        # os.makedirs(os.path.join(DATA_PATH, "projects", st.session_state.pj_name, "event_"+str(i), "特典会"), exist_ok=True)
        # os.makedirs(os.path.join(DATA_PATH, "projects", st.session_state.pj_name, "event_"+str(i), "両方"), exist_ok=True)

def determine_timetable_image():
    file = st.session_state.uploaded_image
    img_path = os.path.join(st.session_state.pj_path, "event_"+str(st.session_state.img_event_no), st.session_state.img_type)
    os.makedirs(img_path, exist_ok=True)
    img_path = os.path.join(img_path, "raw.png")
    with open(img_path, 'wb') as f:
        f.write(file.read())
    st.session_state.timetable_image_master = pd.concat((st.session_state.timetable_image_master,pd.DataFrame([[st.session_state.pj_name,st.session_state.event_type,st.session_state.img_event_no,st.session_state.img_type,st.session_state.img_format,1,1]],columns=st.session_state.timetable_image_master.columns)))
    st.session_state.timetable_image_master.drop_duplicates(subset=["project_name","event_no","image_type"],inplace=True,keep="last")
    st.session_state.timetable_image_master.to_csv(os.path.join(DATA_PATH, "master", "timetable_image_master.csv"), index=False)
    st.session_state.pj_timetable_master = st.session_state.timetable_image_master[st.session_state.timetable_image_master["project_name"]==st.session_state.pj_name]
    with col_file_uploader[1]:
        st.success("画像を登録しました")
    st.session_state.crop_tgt_event = "event_"+str(st.session_state.img_event_no)
    st.session_state.crop_tgt_img_type = st.session_state.img_type
    st.session_state.ocr_tgt_event = "event_"+str(st.session_state.img_event_no)
    st.session_state.ocr_tgt_img_type = st.session_state.img_type

def get_x_freq(image, stage_num):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#グレースケールへの変換   
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)#エッジ検出
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#輪郭検出
    #各輪郭を囲む矩形を取得し、抽出
    rectangles = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        rectangles.append((x, y, w, h))

    df_rectangle = pd.DataFrame(rectangles,columns=["横位置","縦位置","横幅","縦幅"])

    fig, ax = plt.subplots()
    min_width = image.shape[1]/stage_num/3
    x_appear = pd.Series(df_rectangle[df_rectangle["横幅"]>min_width]["横位置"])
    # Streamlitでヒストグラムを表示
    # ax.hist(x_appear, bins=100, edgecolor='black')
    # with timetable_ocr:
    #     st.pyplot(fig)
    return x_appear.value_counts()

def get_xpoint(image, stage_num):
    x_freq = get_x_freq(np.array(image), stage_num)
    response = gpt_ocr.get_xpoint(x_freq, stage_num)

    return json.loads(response.choices[0].message.content)["xpoint"]
    
    with timetable_ocr:
        st.write(json.loads(response.choices[0].message.content)["xpoint"])

def get_image_eachstage_byocr(image, stage_num):
    if stage_num ==1:
        return [image]
    for _ in range(3):
        try:
            xpoints = get_xpoint(image, stage_num)
            break
        except ValueError as e:
            continue
    else:
        raise ValueError
    print(stage_num)
    print(xpoints)
    images_eachstage = []
    for i in range(stage_num):
        x_from = xpoints[i]
        if i < stage_num - 1:
            x_to = xpoints[i+1]
        else:
            x_to = image.width
        width=x_to-x_from
        if i>0:
            x_from = x_from-width/10
        else:
            x_from = 0
        rect_img = image.crop((x_from, 0, x_to, image.height))
        # img_path = os.path.join(st.session_state.pj_path, st.session_state.crop_tgt_event, st.session_state.crop_tgt_img_type, "stage_{}.png".format(i+1))
        # rect_img.save(img_path)
        images_eachstage.append(rect_img)
    return images_eachstage

def get_image_eachstage_for_linecroppedimage_byocr():
    st.session_state.images_eachstage = []
    for i,line_cropped_image in enumerate(st.session_state.line_cropped_images):
        if st.session_state["stage_num_{}".format(i)]>0:
            st.session_state.images_eachstage += get_image_eachstage_byocr(line_cropped_image, st.session_state["stage_num_{}".format(i)])

def get_image_eachstage_for_linecroppedimage_byevenly():
    st.session_state.images_eachstage = []
    for i,line_cropped_image in enumerate(st.session_state.line_cropped_images):
        stage_num = st.session_state["stage_num_{}".format(i)]
        if stage_num>0:
            width, height = line_cropped_image.size
            segment_width = width // stage_num
            for j in range(stage_num):
                left = j * segment_width - segment_width*0.05
                right = (j + 1.05) * segment_width if j < stage_num - 1 else width
                segment = line_cropped_image.crop((left, 0, right, height))
                st.session_state.images_eachstage.append(segment)

def determine_image_eachstage():
    img_path = os.path.join(st.session_state.pj_path, st.session_state.crop_tgt_event, st.session_state.crop_tgt_img_type, "raw_cropped.png")
    st.session_state.cropped_image.save(img_path)
    for i, image_eachstag in enumerate(st.session_state.images_eachstage):
        img_path = os.path.join(st.session_state.pj_path, st.session_state.crop_tgt_event, st.session_state.crop_tgt_img_type, "stage_{}.png".format(i+1))
        image_eachstag.save(img_path)

    st.session_state.timetable_image_master.loc[
        (st.session_state.timetable_image_master["project_name"]==st.session_state.pj_name)
        &(st.session_state.timetable_image_master["event_no"]==int(st.session_state.crop_tgt_event.split("_")[1]))
        &(st.session_state.timetable_image_master["image_type"]==st.session_state.crop_tgt_img_type),"stage_num"]=len(st.session_state.images_eachstage)
    st.session_state.timetable_image_master.to_csv(os.path.join(DATA_PATH, "master", "timetable_image_master.csv"), index=False)
    st.session_state.pj_timetable_master = st.session_state.timetable_image_master[st.session_state.timetable_image_master["project_name"]==st.session_state.pj_name]

def get_timetabledata_eachstage(stage_num):
    for i in range(1,stage_num+1):
        img_path = os.path.join(st.session_state.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "stage_{}.png".format(i))
        return_json = gpt_ocr.getocr_fes_timetable(img_path)
        return_json["ステージ名"] = "ステージ{}".format(i)#st.session_state.stage_names[i]
        json_path = os.path.join(st.session_state.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "stage_{}.json".format(i))
        with open(json_path,"w",encoding = "utf8") as f:
            json.dump(return_json, f, indent = 4, ensure_ascii = False)
def get_timetabledata_onlyonestage(stage_no):
    img_path = os.path.join(st.session_state.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "stage_{}.png".format(stage_no))
    return_json = gpt_ocr.getocr_fes_timetable(img_path)
    return_json["ステージ名"] = "ステージ{}".format(stage_no)#st.session_state.stage_names[i]
    json_path = os.path.join(st.session_state.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "stage_{}.json".format(stage_no))
    with open(json_path,"w",encoding = "utf8") as f:
        json.dump(return_json, f, indent = 4, ensure_ascii = False)
def get_timetabledata_eachstage_notime(stage_num):
    for i in range(1,stage_num+1):
        img_path = os.path.join(st.session_state.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "stage_{}.png".format(i))
        return_json = gpt_ocr.getocr_fes_timetable_notime(img_path)
        return_json["ステージ名"] = "ステージ{}".format(i)#st.session_state.stage_names[i]
        json_path = os.path.join(st.session_state.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "stage_{}.json".format(i))
        with open(json_path,"w",encoding = "utf8") as f:
            json.dump(return_json, f, indent = 4, ensure_ascii = False)
    #時刻の読み取りを別途行う

#アイドル名の修正
def idolname_correct(stage_num):
    for i in range(stage_num):
        json_path = os.path.join(st.session_state.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "stage_{}.json".format(i+1))
        if os.path.exists(json_path):
            with open(json_path, encoding="utf-8") as f:
                timetable_json = json.load(f)
            for item in timetable_json["タイムテーブル"]:
                group_name_correct = timetabledata.get_name_list(item["グループ名"])
                if not group_name_correct[0]:
                    item['グループ名_修正候補'] = group_name_correct[1]
                else:
                    item['グループ名_修正候補'] = item["グループ名"]
            with open(json_path,"w",encoding = "utf8") as f:
                json.dump(timetable_json, f, indent = 4, ensure_ascii = False)



project_setting = st.container()#プロジェクト設定
with project_setting:
    st.markdown("""#### プロジェクトの設定""")
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
    if st.session_state.pj_name is None:
        st.stop()
    st.text("選択中のプロジェクト："+st.session_state.pj_name)
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

file_uploader = st.container()#タイテ画像の登録
with file_uploader:
    st.markdown("""#### タイムテーブル画像の登録""")
    all_files_raw =  st.container(height=200)
    col_file_uploader = st.columns((3,1))
with all_files_raw:
    st.markdown("""###### 登録済みタイムテーブル画像一覧""")
    col_all_files = st.columns(int(2*st.session_state.event_num))
    for i in range(1,st.session_state.event_num+1):
        img_path = os.path.join(st.session_state.pj_path, "event_{}".format(i), "ライブ", "raw.png")
        if os.path.exists(img_path):
            with col_all_files[(i-1)*2]:
                image = Image.open(img_path)
                st.image(image)
        img_path = os.path.join(st.session_state.pj_path, "event_{}".format(i), "特典会", "raw.png")
        if os.path.exists(img_path):
            with col_all_files[(i-1)*2+1]:
                image = Image.open(img_path)
                st.image(image)
with col_file_uploader[0]:
    st.file_uploader("読み取りたいタイムテーブル画像をアップロードしてください。"
                            , type=["jpg", "jpeg", "png"]
                            , key="uploaded_image")
    if st.session_state.uploaded_image is not None:
        st.image(
            st.session_state.uploaded_image,
            use_column_width=True
        )
with col_file_uploader[1]:
    if st.session_state.uploaded_image is not None:
        st.info(
    """画像が何の情報を示しているか入力してください。  
    ・イベントNo.：複数イベントがある場合、どの何番目のイベントの情報であるか  
    ・種別：画像に載っている情報の種類  
    ・形式：タイムテーブルの形式（通常：時間が各枠に記載 / ライムライト式：時間軸が枠外に記載）
    """)
        st.number_input("イベントNo.", min_value=1, max_value=st.session_state.event_num, step=1, key="img_event_no")
        # st.radio("種別", ("ライブ", "特典会", "両方"), key="img_type", horizontal=True)
        st.radio("種別", ("ライブ", "特典会"), key="img_type", horizontal=True)
        st.radio("形式", ("通常", "ライムライト式"), key="img_format", horizontal=True)
        st.button(label="画像を登録する", on_click=determine_timetable_image, type="primary")

timetable_crop = st.container()#タイテ画像の切り取り
with timetable_crop:
    st.markdown("""#### タイムテーブル画像の切り取り""")
    st.info(
"""複数ステージが一つの画像に存在している場合、各ステージごとに画像を切り分けます。  
この時、各ステージの幅が概ね均等であり、また間に時間軸なども等しく入っているor入っていない場合は、  
画像を均等に分割することによりステージごとの画像を取得できます。  
しかし、そうでない場合には「均等割」できるいくつかの小さい画像に分割する必要があります。  
画像を見て、どちらのパターンかを判断して作業を行ってください。  
""")
    event_list = os.listdir(st.session_state.pj_path)
    st.selectbox("イベント", event_list,index=0,key="crop_tgt_event")
    st.selectbox("種別", ["ライブ","特典会"],index=0,key="crop_tgt_img_type")
    img_path = os.path.join(st.session_state.pj_path, st.session_state.crop_tgt_event, st.session_state.crop_tgt_img_type, "raw.png")
    if os.path.exists(img_path):
        image = Image.open(img_path)
        image_info = st.session_state.pj_timetable_master[
            (st.session_state.pj_timetable_master["event_no"]==int(st.session_state.crop_tgt_event.split("_")[1]))
            & (st.session_state.pj_timetable_master["image_type"]==st.session_state.crop_tgt_img_type)
        ].iloc[0]

        with st.container():# タイムテーブルに関係する領域を切り出す
            st.markdown("""###### タイムテーブルに関係する領域を切り出す""")
            st.info(
"""まず、タイムテーブルに関係する領域の切り出しを行います。  
必要十分な領域を指定してください。  
・ライムライト式の場合、時間軸の情報は含める必要があります（後で使います）  
・上下にステージ名などの情報は含まれていても構いません  
・そのまま一枚の画像として均等割を行う場合にはそれに適した領域に切り出してください  
""")
            col_cropimage_first = st.columns(2)
            with col_cropimage_first[0]:
                st.session_state.cropped_image = st_cropper(image)
            with col_cropimage_first[1]:
                st.image(st.session_state.cropped_image,use_column_width=True)

        with st.container():# 「均等割」できるようにタイムテーブル領域を分割する
            st.markdown("""###### 「均等割」できるようにタイムテーブル領域を分割する""")
            st.info(
"""次に、タイムテーブルが均等でない場合、各々が均等な領域になるよう分割します。  
分割が不要な場合、この工程を無視して進んでください。  
分割が必要な場合、「分割を行う」をONにして分割を行ってください。  
  
ドラッグで縦線を入力し、それに従って分割を行います。  
・実際には、縦線ではなく縦線の始点の横位置で分割します。線はイメージです  
・なので、斜めな線でも、画像の上から下まで線を引かなくとも、問題はありません  
・右側に表示される分割結果を参照して、適宜線を修正してください  
・なお画像の表示比率がおかしかったり欠けていたりするかもしれませんが、一旦気にせず進めてください  
  
分割が終了したら、各領域に何個のステージが含まれてているかを入力します。  
・タイムテーブル以外の情報（時間軸領域など）しか含まない領域を無視したい場合、ステージ数を0にすることで実現できます  
・なお次のステップにおいて、均等割ではなく矩形抽出による自動調整分割もできるので、必ずしも時間軸領域を無視せずとも動きますが、精度は100%ではないので、無視できるならしたほうがよいです  
・このステップで完全人力でステージ1つずつへの分割を行うこともできます。時間はかかりますがそれが一番確実です  
""")
            line_cropping = st.toggle("分割を行う")
            if line_cropping:
                col_cropimage_line = st.columns(2)
                with col_cropimage_line[0]:
                    canvas_width = 800
                    canvas_rate = canvas_width/st.session_state.cropped_image.size[0]
                    canvas_height = int(st.session_state.cropped_image.size[1]*canvas_rate)
                    resized_image = st.session_state.cropped_image.resize((canvas_height, canvas_width))
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                        stroke_width=3,
                        # stroke_color=stroke_color,
                        # background_color=bg_color,
                        background_image=resized_image,
                        update_streamlit=True,
                        width=canvas_width,
                        height=canvas_height,
                        drawing_mode="line",
                        # point_display_radius=1,
                        key="crop_line",
                    )
                with col_cropimage_line[1]:
                    if canvas_result.json_data is not None:
                        objects = pd.json_normalize(canvas_result.json_data["objects"])
                        # for col in objects.select_dtypes(include=["object"]).columns:
                        #     objects[col] = objects[col].astype("str")
                        # st.dataframe(objects)

                        st.session_state.line_cropped_images = []
                        line_x_before = 0
                        for i,row in objects.iterrows():
                            line_x = row["left"]+row["width"]/2
                            st.session_state.line_cropped_images.append(st.session_state.cropped_image.crop((line_x_before/canvas_rate, 0, line_x/canvas_rate, st.session_state.cropped_image.size[1])))
                            line_x_before = line_x
                        st.session_state.line_cropped_images.append(st.session_state.cropped_image.crop((line_x_before/canvas_rate, 0, st.session_state.cropped_image.size[0], st.session_state.cropped_image.size[1])))

                        col_cropimage_line_after = st.columns([max(img.size[0],st.session_state.cropped_image.size[0]/len(st.session_state.line_cropped_images)/2) for img in st.session_state.line_cropped_images])
                        for i,line_cropped_image in enumerate(st.session_state.line_cropped_images):
                            with col_cropimage_line_after[i]:
                                st.image(line_cropped_image)
                                st.number_input("ステージ数",0,value=1,step=1,key="stage_num_{}".format(i))
            else:
                st.session_state.line_cropped_images = [st.session_state.cropped_image]
                st.number_input("ステージ数",1,step=1,key="stage_num_0")

        with st.container():# 各ステージに1枚の画像が対応するよう分割する
            st.markdown("""###### 各ステージに1枚の画像が対応するよう分割する""")
            st.info(
"""最後に、ステージ一つひとつに一枚の画像が対応するよう、画像を分割します。  
画像の分割は、均等でないタイムテーブル領域を分割した場合にはそれぞれに対して、  
分割していない場合にはタイムテーブルに関係する領域として切り出した1枚に対して、  
指定されたステージ数への分割が実施されます。  
分割の方法は下記の通りです。どちらかを選んでボタンを押してください。  
1. OCRにより矩形を抽出し、ステージの幅を推定して分割する  
1. 画像を均等な横幅になるよう分割する  

なお先述の通りステージ数が1の場合はそのまま一つのステージの画像として採用されます。  
""")
            # st.button("ステージごとの画像を抽出",on_click=get_image_eachstage_byocr,args=(cropped_image, st.session_state.stage_num))
            st.button("OCR自動分割",on_click=get_image_eachstage_for_linecroppedimage_byocr)
            st.button("横幅均等分割",on_click=get_image_eachstage_for_linecroppedimage_byevenly)

            if len(st.session_state.images_eachstage)>0:
                col_cropimage_eachstage = st.columns([img.size[0] for img in st.session_state.images_eachstage])
                for i,eachstage_image in enumerate(st.session_state.images_eachstage):
                    with col_cropimage_eachstage[i]:
                        st.image(eachstage_image)

                st.button("ステージごとの画像分割を確定",on_click=determine_image_eachstage, type="primary")

timetable_ocr = st.container()#タイテ画像の読み取り
with timetable_ocr:
    st.markdown("""#### タイムテーブル画像の読み取り""")
    event_list = os.listdir(st.session_state.pj_path)
    st.selectbox("イベント", event_list,index=0,key="ocr_tgt_event")
    st.selectbox("種別", ["ライブ","特典会"],index=0,key="ocr_tgt_img_type")
    img_path = os.path.join(st.session_state.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "raw.png")
    if os.path.exists(img_path):
    # if st.session_state.crop_tgt_event is not None and st.session_state.crop_tgt_img_type is not None:
        image = Image.open(img_path)
        image_info = st.session_state.pj_timetable_master[
            (st.session_state.pj_timetable_master["event_no"]==int(st.session_state.ocr_tgt_event.split("_")[1]))
            & (st.session_state.pj_timetable_master["image_type"]==st.session_state.ocr_tgt_img_type)
        ].iloc[0]
        stage_num = image_info["stage_num"]
        if False:#image_info["image_format"]=="ライムライト式":#一旦諦め
            cropped_img_path = os.path.join(st.session_state.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "raw_cropped.png")
            if os.path.exists(cropped_img_path):
                cropped_image = Image.open(cropped_img_path)
                with st.container():# 時間軸の設定（ライムライト式の場合のみ実施）
                    st.markdown("""###### 時間軸の設定（ライムライト式の場合のみ実施）""")
                    st.info(
"""時間軸の基準となる位置を指定してください。  
・画像上でドラッグして長方形を描画します  
・横幅は特に関係ないので、適当で大丈夫です  
・縦位置は出演枠が存在している領域でセットしてください。余白部分は微妙に間隔が違ったりします
・上端と下端に対応する時刻を入力して、画像上のどの位置がどの時刻か対応づけます  
・複数の長方形が描画できしまいますが、一番最初のものが採用されてしまいます  
・変更したい時は、画像下の削除ボタンやundoボタンを使ってください  
・なお画像の表示比率がおかしかったり欠けていたりするかもしれませんが、一旦気にせず進めてください  
""")
                    col_timeaxis = st.columns(2)
                    with col_timeaxis[0]:
                        # canvas_height = st.number_input("表示する縦幅",min_value=100,step=100,value=800)
                        canvas_height = 800
                        canvas_rate = canvas_height/cropped_image.size[1]
                        canvas_width = int(cropped_image.size[0]*canvas_rate)
                        resized_image = cropped_image.resize((canvas_width, canvas_height))
                        canvas_result = st_canvas(
                            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                            stroke_width=1,
                            # stroke_color=stroke_color,
                            # background_color=bg_color,
                            background_image=resized_image,
                            update_streamlit=True,
                            width=canvas_width,
                            height=canvas_height,
                            drawing_mode="rect",
                            # point_display_radius=1,
                            key="time_axis_detect",
                        )

                    with col_timeaxis[1]:
                        # st.time_input("開始時間", value=dttime(10), key=None, step=300)
                        # st.time_input("終了時間", value=dttime(20), key=None, step=300)
                        st.slider('開始時間', value=dttime(10), key="time_start", step=timedelta(minutes=5))
                        st.slider('終了時間', value=dttime(20), key="time_finish", step=timedelta(minutes=5))
                        total_duration = (datetime(2024,1,1,st.session_state.time_finish.hour,st.session_state.time_finish.minute)-datetime(2024,1,1,st.session_state.time_start.hour,st.session_state.time_start.minute)).seconds/60
                        if st.session_state.time_finish.hour < st.session_state.time_start.hour:
                            st.warning("終了時間が開始時間よりも早くなっています。深夜イベントなどで日を跨ぐ場合はそのまま実行可能ですが、そうでない場合は修正してください。")
                        if canvas_result.json_data is not None:
                            objects = pd.json_normalize(canvas_result.json_data["objects"])
                            # for col in objects.select_dtypes(include=["object"]).columns:
                            #     objects[col] = objects[col].astype("str")
                            # st.dataframe(objects)
                            try:
                                resized_start_pix = objects.iloc[0]["top"]
                                resized_total_pix = objects.iloc[0]["height"]

                                # from PIL import ImageDraw
                                # lines = [
                                #     ((0, resized_start_pix/canvas_rate), (cropped_image.size[0], resized_start_pix/canvas_rate)),  # (x1, y1), (x2, y2)
                                #     ((0, (resized_start_pix+resized_total_pix)/canvas_rate), (cropped_image.size[0], (resized_start_pix+resized_total_pix)/canvas_rate))
                                # ]

                                # # 直線を描画する関数
                                # def draw_lines_on_image(image, lines, color="red", width=3):
                                #     draw = ImageDraw.Draw(image)
                                #     for line in lines:
                                #         draw.line(line, fill=color, width=width)
                                #     return image

                                # # 画像のコピーを作成し、直線を描画
                                # image_with_lines = cropped_image.copy()
                                # image_with_lines = draw_lines_on_image(image_with_lines, lines)

                                # # 直線を描画した画像を表示
                                # st.image(image_with_lines, use_column_width=True)

                            except IndexError:
                                st.warning("画像上で時間範囲をドラッグしてください")
                # st.write(canvas_rate)
                # st.write(resized_start_pix/canvas_rate)
                # st.write(resized_total_pix/canvas_rate)
                # st.write(total_duration)#分

        with st.container():# 各ステージ情報の読み取り
            st.markdown("""###### 各ステージ情報の読み取り""")
            st.info(
"""ライムライト形式における時間抽出、もう少し時間ください！  
希望は見えているので、竜王までには時間までできるとよいなと思っています…！  
現状、タイムライト形式で出力されている時間は嘘なので使わないでください。  
ステージの連番、0スタートと1スタートどっちがいいか分からずごちゃまぜになってますがご容赦ください。  
""")

            # st.button("全ステージの画像をそれぞれ読み取り実施",on_click=get_timetabledata_eachstage_notime,args=(stage_num,),type="primary")
            st.button("全ステージの画像をそれぞれ読み取り実施",on_click=get_timetabledata_eachstage,args=(stage_num,),type="primary")
            st.number_input("ステージ番号",1,stage_num,key="ocr_tgt_stage_no")
            st.button("あるステージのみ読み取りを実施",on_click=get_timetabledata_onlyonestage,args=(st.session_state.ocr_tgt_stage_no,))
            stage_tabs = st.tabs(["ステージ"+str(i+1) for i in range(stage_num)])

            st.session_state.df_timetables = []
            for i in range(stage_num):
                with stage_tabs[i]:
                    ocr_col = st.columns([1,19])
                    with ocr_col[0]:
                        img_path = os.path.join(st.session_state.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "stage_{}.png".format(i+1))
                        if os.path.exists(img_path):
                            image = Image.open(img_path)
                            st.image(image, use_column_width=True)
                    with ocr_col[1]:
                        json_path = os.path.join(st.session_state.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "stage_{}.json".format(i+1))
                        if os.path.exists(json_path):
                            with open(json_path, encoding="utf-8") as f:
                                return_json = json.load(f)
                            return_json_df = timetabledata.json_to_df(return_json,tokutenkai=False)
                            st.dataframe(return_json_df)
                            return_json_df["ステージNo."]=i
                            st.session_state.df_timetables.append(return_json_df)

            #全ステージ情報の出力 #暫定
            def get_all_stage_info():
                all_stage_df = pd.concat(st.session_state.df_timetables).reset_index(drop=True)
                with all_stage_info:
                    st.dataframe(all_stage_df,hide_index=True)

            st.button("グループ名を修正",on_click=idolname_correct,args=(stage_num,),type="primary")
            st.button("全ステージのデータを表形式で出力",on_click=get_all_stage_info,type="primary")
            all_stage_info = st.container()

# st.session_state.event_type#フェスか対バンか
# st.session_state.event_num#イベント数
# st.session_state.pj_timetable_master#各イベントのライブ、特典会の情報