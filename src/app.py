import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from streamlit_cropper import st_cropper
from streamlit_drawable_canvas import st_canvas

import os
import shutil
import tempfile
from operator import itemgetter

from PIL import Image, ImageDraw
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
        st.session_state.project_master.loc[pj_name] = [created_at,"フェス",1]
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
    for i in range(1,st.session_state.event_num+1):
        os.makedirs(os.path.join(st.session_state.pj_path, "event_{}".format(i)), exist_ok=True)
    st.session_state.pj_timetable_master = st.session_state.timetable_image_master[st.session_state.timetable_image_master["project_name"]==st.session_state.pj_name]
    #その他以後の変数も初期化等する必要あり
    set_crop_image()
    set_ocr_image()
    # st.session_state.images_eachstage=[]#実際にはロードできるならロードする
    # st.session_state.timeline_eachstage=[]#同上

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
    if st.session_state.img_type in ["ライブ","特典会"]:
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
    elif st.session_state.img_type == "両方(特典会別添え)":
        file_image = file.read()
        for img_type in ["ライブ","特典会"]:
            img_path = os.path.join(st.session_state.pj_path, "event_"+str(st.session_state.img_event_no), img_type)
            os.makedirs(img_path, exist_ok=True)
            img_path = os.path.join(img_path, "raw.png")
            with open(img_path, 'wb') as f:
                f.write(file_image)
            st.session_state.timetable_image_master = pd.concat((st.session_state.timetable_image_master,pd.DataFrame([[st.session_state.pj_name,st.session_state.event_type,st.session_state.img_event_no,img_type,st.session_state.img_format,1,1]],columns=st.session_state.timetable_image_master.columns)))
        st.session_state.timetable_image_master.drop_duplicates(subset=["project_name","event_no","image_type"],inplace=True,keep="last")
        st.session_state.timetable_image_master.to_csv(os.path.join(DATA_PATH, "master", "timetable_image_master.csv"), index=False)
        st.session_state.pj_timetable_master = st.session_state.timetable_image_master[st.session_state.timetable_image_master["project_name"]==st.session_state.pj_name]
        st.session_state.crop_tgt_event = "event_"+str(st.session_state.img_event_no)
        st.session_state.crop_tgt_img_type = "ライブ"
        st.session_state.ocr_tgt_event = "event_"+str(st.session_state.img_event_no)
        st.session_state.ocr_tgt_img_type = "ライブ"
        with col_file_uploader[1]:
            st.success("画像を登録しました")
    elif st.session_state.img_type == "両方(特典会併記)":
        img_type = "ライブ"
        img_path = os.path.join(st.session_state.pj_path, "event_"+str(st.session_state.img_event_no), img_type)
        os.makedirs(img_path, exist_ok=True)
        img_path = os.path.join(img_path, "raw.png")
        with open(img_path, 'wb') as f:
            f.write(file.read())
        st.session_state.timetable_image_master = pd.concat((st.session_state.timetable_image_master,pd.DataFrame([[st.session_state.pj_name,st.session_state.event_type,st.session_state.img_event_no,img_type,"特典会併記",1,1]],columns=st.session_state.timetable_image_master.columns)))
        st.session_state.timetable_image_master.drop_duplicates(subset=["project_name","event_no","image_type"],inplace=True,keep="last")
        st.session_state.timetable_image_master.to_csv(os.path.join(DATA_PATH, "master", "timetable_image_master.csv"), index=False)
        st.session_state.pj_timetable_master = st.session_state.timetable_image_master[st.session_state.timetable_image_master["project_name"]==st.session_state.pj_name]
        with col_file_uploader[1]:
            st.success("画像を登録しました")
        st.session_state.crop_tgt_event = "event_"+str(st.session_state.img_event_no)
        st.session_state.crop_tgt_img_type = img_type
        st.session_state.ocr_tgt_event = "event_"+str(st.session_state.img_event_no)
        st.session_state.ocr_tgt_img_type = img_type
    elif st.session_state.img_type == "その他":
        img_type = st.session_state.img_type_alternative
        img_path = os.path.join(st.session_state.pj_path, "event_"+str(st.session_state.img_event_no), img_type)
        os.makedirs(img_path, exist_ok=True)
        img_path = os.path.join(img_path, "raw.png")
        with open(img_path, 'wb') as f:
            f.write(file.read())
        st.session_state.timetable_image_master = pd.concat((st.session_state.timetable_image_master,pd.DataFrame([[st.session_state.pj_name,st.session_state.event_type,st.session_state.img_event_no,img_type,st.session_state.img_format,1,1]],columns=st.session_state.timetable_image_master.columns)))
        st.session_state.timetable_image_master.drop_duplicates(subset=["project_name","event_no","image_type"],inplace=True,keep="last")
        st.session_state.timetable_image_master.to_csv(os.path.join(DATA_PATH, "master", "timetable_image_master.csv"), index=False)
        st.session_state.pj_timetable_master = st.session_state.timetable_image_master[st.session_state.timetable_image_master["project_name"]==st.session_state.pj_name]
        with col_file_uploader[1]:
            st.success("画像を登録しました")
        st.session_state.crop_tgt_event = "event_"+str(st.session_state.img_event_no)
        st.session_state.crop_tgt_img_type = img_type
        st.session_state.ocr_tgt_event = "event_"+str(st.session_state.img_event_no)
        st.session_state.ocr_tgt_img_type = img_type

def get_event_type_list(event_no):#ライブ、特典会を先頭にしつつイベントごとに存在する画像種別のリストを出力する
    event_type_list = []
    for event_type in ["ライブ","特典会"]:
        if event_type in os.listdir(os.path.join(st.session_state.pj_path, "event_{}".format(event_no))):
            event_type_list.append(event_type)
    for event_type in os.listdir(os.path.join(st.session_state.pj_path, "event_{}".format(event_no))):
        if event_type not in event_type_list:
            event_type_list.append(event_type)
    return event_type_list

def set_crop_image():
    # st.session_state.crop_tgt_event
    # st.session_state.crop_tgt_img_type
    st.session_state.images_eachstage=[]

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

def detect_stageline(image):#ステージ領域を特定する縦線を取得
    img_path = os.path.join(st.session_state.pj_path, st.session_state.crop_tgt_event, st.session_state.crop_tgt_img_type, "raw_cropped.png")
    st.session_state.cropped_image.save(img_path)

    bgr_img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    height, width = gray_img.shape
    minlength = height * st.session_state.x_minlength_rate

    im_edges = cv2.Canny(gray_img, st.session_state.x_edge_threshold_1, st.session_state.x_edge_threshold_2, L2gradient=True)#エッジ検出
    lines = []
    lines = cv2.HoughLinesP(im_edges, rho=1, theta=np.pi/360, threshold=st.session_state.x_hough_threshold, minLineLength=minlength, maxLineGap=st.session_state.x_hough_gap)#ハフ変換による直線抽出

    line_list = []
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2 or abs((y1 - y2)/(x1 - x2)) > np.tan(np.pi/180*85):#傾きが85度より大きいのもののみ検出
            line_list.append([x1, y1, x2, y2, abs(y1-y2)])
            draw.line((x1, y1, x2, y2), fill="red", width=20)

    line_list.sort(key=itemgetter(0, 1, 2, 3))
    line_x_list = pd.DataFrame(line_list,columns=["x1","y1","x2","y2","length"]).groupby("x1").sum()[["length"]]
    x_before=0
    new_line_x_list=[]
    for x,row in line_x_list.iterrows():#近接線の統合
        if x-x_before < st.session_state.x_identify_interval:
            if len(new_line_x_list)>0:
                new_line_x_list[-1][0].append(x)
                new_line_x_list[-1][1]+=row["length"]
            else:
                new_line_x_list.append([[0,x],row["length"]])
        else:
            if len(new_line_x_list)>0:
                new_line_x_list[-1][0] = np.mean(new_line_x_list[-1][0])
            new_line_x_list.append([[x],row["length"]])
        x_before=x
    if len(new_line_x_list)>0:
        new_line_x_list[-1][0] = np.mean(new_line_x_list[-1][0])

    st.session_state.stage_line_list = pd.DataFrame(new_line_x_list,columns=["x","length"])
    # x_mintotallength_rate = 0.03
    # st.session_state.stage_line_list = st.session_state.stage_line_list.query(" length > {}".format(int(height * x_mintotallength_rate)))

    left = 0
    num = len(st.session_state.stage_line_list)+1
    st.session_state.images_eachstage=[]
    for i in range(num):
        if i<num-1:
            right = st.session_state.stage_line_list.iat[i,0]
            # draw.line((right, 0, right, height), fill="red", width=10)
        else:
            right =width
        st.session_state.images_eachstage.append(image.crop((left, 0, right, height)))
        left = right

    with edge_result:
        # st.write(st.session_state.stage_line_list)
        st.image(image_copy,caption="縦線抽出結果")
        #ここで抽出に失敗している時のリカバリー（クリックで分割線を追加）をできるようにする

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

def get_image_eachstage_for_linecroppedimage_byevenly(): #使ってない
    st.session_state.images_eachstage = []
    for i,line_cropped_image in enumerate(st.session_state.line_cropped_images):
        stage_num = st.session_state["stage_num_{}".format(i)]
        if stage_num>0:
            width, height = line_cropped_image.size
            segment_width = width // stage_num
            for j in range(stage_num):
                left = max(0, (j - 0.05) * segment_width)
                right = min((j + 1.05) * segment_width,width)
                segment = line_cropped_image.crop((left, 0, right, height))
                st.session_state.images_eachstage.append(segment)

def get_image_eachstage_for_croppedimage_byevenly():
    img_path = os.path.join(st.session_state.pj_path, st.session_state.crop_tgt_event, st.session_state.crop_tgt_img_type, "raw_cropped.png")
    st.session_state.cropped_image.save(img_path)
    st.session_state.images_eachstage = []
    stage_num = st.session_state.stage_num
    if stage_num>0:
        width, height = st.session_state.cropped_image.size
        segment_width = width // stage_num
        for j in range(stage_num):
            left = max(0, (j - 0.05) * segment_width)
            right = min((j + 1.05) * segment_width,width)
            segment = st.session_state.cropped_image.crop((left, 0, right, height))
            st.session_state.images_eachstage.append(segment)

def determine_image_eachstage():
    #前に保存した画像を削除するべきではある
    img_path = os.path.join(st.session_state.pj_path, st.session_state.crop_tgt_event, st.session_state.crop_tgt_img_type, "raw_cropped.png")
    st.session_state.cropped_image.save(img_path)
    for i, image_eachstag in enumerate(st.session_state.images_eachstage):
        img_path = os.path.join(st.session_state.pj_path, st.session_state.crop_tgt_event, st.session_state.crop_tgt_img_type, "stage_{}.png".format(i))
        image_eachstag.save(img_path)

    st.session_state.timetable_image_master.loc[
        (st.session_state.timetable_image_master["project_name"]==st.session_state.pj_name)
        &(st.session_state.timetable_image_master["event_no"]==int(st.session_state.crop_tgt_event.split("_")[1]))
        &(st.session_state.timetable_image_master["image_type"]==st.session_state.crop_tgt_img_type),"stage_num"]=len(st.session_state.images_eachstage)
    st.session_state.timetable_image_master.to_csv(os.path.join(DATA_PATH, "master", "timetable_image_master.csv"), index=False)
    st.session_state.pj_timetable_master = st.session_state.timetable_image_master[st.session_state.timetable_image_master["project_name"]==st.session_state.pj_name]

def determine_image_eachstage_without_nocheck():
    #前に保存した画像を削除するべきではある
    stage_num = 0
    for i, image_eachstag in enumerate(st.session_state.images_eachstage):
        if st.session_state["each_stage_acept_{}".format(i)]:
            img_path = os.path.join(st.session_state.pj_path, st.session_state.crop_tgt_event, st.session_state.crop_tgt_img_type, "stage_{}.png".format(stage_num))
            image_eachstag.save(img_path)
            stage_num+=1

    st.session_state.timetable_image_master.loc[
        (st.session_state.timetable_image_master["project_name"]==st.session_state.pj_name)
        &(st.session_state.timetable_image_master["event_no"]==int(st.session_state.crop_tgt_event.split("_")[1]))
        &(st.session_state.timetable_image_master["image_type"]==st.session_state.crop_tgt_img_type),"stage_num"]=stage_num
    st.session_state.timetable_image_master.to_csv(os.path.join(DATA_PATH, "master", "timetable_image_master.csv"), index=False)
    st.session_state.pj_timetable_master = st.session_state.timetable_image_master[st.session_state.timetable_image_master["project_name"]==st.session_state.pj_name]

def set_ocr_image():
    # st.session_state.ocr_tgt_event
    # st.session_state.ocr_tgt_img_type
    st.session_state.time_axis_detect = None
    st.session_state.timeline_eachstage=[]

def get_timetabledata_onlyonestage(stage_no,user_prompt):
    img_path = os.path.join(st.session_state.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "stage_{}.png".format(stage_no))
    user_prompt = "この画像のタイムテーブルをJSONデータとして出力して。" + user_prompt
    return_json = gpt_ocr.getocr_fes_timetable(img_path,user_prompt)
    return_json["ステージ名"] = "ステージ{}".format(stage_no)#st.session_state.stage_names[stage_no]
    json_path = os.path.join(st.session_state.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "stage_{}.json".format(stage_no))
    with open(json_path,"w",encoding = "utf8") as f:
        json.dump(return_json, f, indent = 4, ensure_ascii = False)

def get_timetabledata_eachstage(stage_num,user_prompt):
    for i in range(stage_num):
        get_timetabledata_onlyonestage(i,user_prompt)

def get_timetabledata_withtokutenkai_onlyonestage(stage_no,user_prompt):
    img_path = os.path.join(st.session_state.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "stage_{}.png".format(stage_no))
    user_prompt = "この画像のタイムテーブルをJSONデータとして出力して。" + user_prompt
    return_json = gpt_ocr.getocr_fes_withtokutenkai_timetable(img_path,user_prompt)
    return_json["ステージ名"] = "ステージ{}".format(stage_no)#st.session_state.stage_names[stage_no]
    json_path = os.path.join(st.session_state.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "stage_{}.json".format(stage_no))
    with open(json_path,"w",encoding = "utf8") as f:
        json.dump(return_json, f, indent = 4, ensure_ascii = False)

def get_timetabledata_withtokutenkai_eachstage(stage_num,user_prompt):
    for i in range(stage_num):
        get_timetabledata_withtokutenkai_onlyonestage(i,user_prompt)

def detect_timeline_onlyonestage(stage_no):
    if len(st.session_state.timeline_eachstage)==0:
        st.session_state.timeline_eachstage = [None for _ in range(stage_num)]

    img_path = os.path.join(st.session_state.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "stage_{}.png".format(stage_no))
    if os.path.exists(img_path):
        image = Image.open(img_path)
    else:
        raise ValueError

    bgr_img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    height, width = gray_img.shape
    minlength = height * st.session_state.y_minlength_rate

    # if st.session_state.y_binary_threshold>0:
    #     _, gray_img = cv2.threshold(gray_img, st.session_state.y_binary_threshold, 255, cv2.THRESH_BINARY_INV)
    im_edges = cv2.Canny(gray_img, st.session_state.y_edge_threshold_1, st.session_state.y_edge_threshold_2, L2gradient=True)#エッジ検出
    lines = []
    lines = cv2.HoughLinesP(im_edges, rho=1, theta=np.pi/360, threshold=st.session_state.y_hough_threshold, minLineLength=minlength, maxLineGap=st.session_state.y_hough_gap)#ハフ変換による直線抽出

    line_list = []
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 != x2 and abs((y1 - y2)/(x1 - x2)) < np.tan(np.pi/180*5):#傾きが5度より小さいもののみ検出
            line_list.append([x1, y1, x2, y2, abs(x1-x2)])
            draw.line((x1, y1, x2, y2), fill="white", width=10)

    line_list.sort(key=itemgetter(1, 0, 2, 3))
    line_y_list = pd.DataFrame(line_list,columns=["x1","y1","x2","y2","length"]).groupby("y1").sum()[["length"]]
    y_before=0
    new_line_y_list=[]
    for y,row in line_y_list.iterrows():#近接線の統合
        if y-y_before < st.session_state.y_identify_interval:
            if len(new_line_y_list)>0:
                new_line_y_list[-1][0].append(y)
                new_line_y_list[-1][1]+=row["length"]
            else:
                new_line_y_list.append([[0,y],row["length"]])
        else:
            if len(new_line_y_list)>0:
                new_line_y_list[-1][0] = np.mean(new_line_y_list[-1][0])
            new_line_y_list.append([[y],row["length"]])
        y_before=y
    if len(new_line_y_list)>0:
        new_line_y_list[-1][0] = np.mean(new_line_y_list[-1][0])
    st.session_state.timeline_eachstage[stage_no-1] = pd.DataFrame(new_line_y_list,columns=["y","length"])
    # y_mintotallength_rate = 0.03
    # st.session_state.timeline_eachstage[stage_no-1] = st.session_state.timeline_eachstage[stage_no-1].query(" length > {}".format(int(height * y_mintotallength_rate)))
    st.session_state.timeline_eachstage[stage_no-1]["time"] = st.session_state.timeline_eachstage[stage_no-1]["y"].map(pix_to_time)
    # st.session_state.timeline_eachstage[stage_no-1]["time"].to_csv(img_path.replace(".png","_timeline.csv"),index=False)

    font_size = int(st.session_state.total_pix/st.session_state.total_duration*4)
    face = cv2.FONT_HERSHEY_PLAIN
    thickness = 1#パラメータ化する？
    scale = cv2.getFontScaleFromHeight(face, font_size, thickness)
    font_pixel, baseline = cv2.getTextSize("00:00-00:00", face, scale, thickness)
    #時刻情報を画像に追記
    # img_time = bgr_img.copy()
    # for _ ,row in st.session_state.timeline_eachstage[stage_no-1].iterrows():
    #     # cv2.putText(img_time, row["time"].strftime('%H:%M'), (int(width*0.95-font_pixel[0]), int(row["y"])-int(font_size*0.2)), cv2.FONT_HERSHEY_PLAIN, scale, (0,0,0), thickness)
    #     cv2.putText(img_time, row["time"].strftime('%H:%M'), (int(width*0.05), int(row["y"])-int(font_size*0.2)), cv2.FONT_HERSHEY_PLAIN, scale, (0,0,0), thickness)
    #     cv2.putText(img_time, row["time"].strftime('%H:%M'), (int(width*0.05), int(row["y"])+int(font_size*1.2)), cv2.FONT_HERSHEY_PLAIN, scale, (0,0,0), thickness)
    
    #時刻情報を画像右側に拡張して追記
    extension_width = int(font_pixel[0]*1.2)#右側に拡張する幅
    extension_image = np.full((height, width + extension_width, 3), (255, 255, 255), dtype=np.uint8)#拡張した領域を持つ新しい画像
    extension_image[:, :width] = bgr_img#元の画像を新しい画像の左側に貼り付ける
    for j in range(len(st.session_state.timeline_eachstage[stage_no-1])-1):
        time_pix = (st.session_state.timeline_eachstage[stage_no-1].loc[j,"y"]+st.session_state.timeline_eachstage[stage_no-1].loc[j+1,"y"])/2
        time_stamp = st.session_state.timeline_eachstage[stage_no-1].loc[j,"time"].strftime('%H:%M') + "-" + st.session_state.timeline_eachstage[stage_no-1].loc[j+1,"time"].strftime('%H:%M')
        cv2.putText(extension_image, time_stamp, (width+int(width*0.05), int(time_pix)+int(font_size*0.6)), cv2.FONT_HERSHEY_PLAIN, scale, (0,0,0), thickness)
    for _ ,row in st.session_state.timeline_eachstage[stage_no-1].iterrows():
        cv2.line(extension_image, (width, int(row["y"])), (width + extension_width, int(row["y"])), (0,0,0), 1)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmpfile:
        temp_path = tmpfile.name
        cv2.imwrite(temp_path, extension_image)
    shutil.move(temp_path, img_path.replace(".png","_addtime.png"))

    # with tmp_timeline:
    #     img_time = cv2.cvtColor(img_time, cv2.COLOR_BGR2RGB)
    #     st.image(img_time)

    # left = 0
    # num = len(st.session_state.stage_line_list)+1
    # st.session_state.images_eachstage=[]
    # for i in range(num):
    #     if i<num-1:
    #         right = st.session_state.stage_line_list.iat[i,0]
    #         draw.line((right, 0, right, height), fill="red", width=10)
    #     else:
    #         right =width
    #     st.session_state.images_eachstage.append(image.crop((left, 0, right, height)))
    #     left = right

    # with tmp_timeline:
    #     st.write(st.session_state.timeline_eachstage[stage_no-1])
    #     st.image(image_copy)
        #ここで抽出に失敗している時のリカバリー（クリックで分割線を追加）

def detect_timeline_eachstage(stage_num):
    for i in range(stage_num):
        detect_timeline_onlyonestage(i)

def get_timetabledata_onlyonestage_notime(stage_no,user_prompt):
    img_path = os.path.join(st.session_state.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "stage_{}_addtime.png".format(stage_no))
    user_prompt = "この画像のタイムテーブルをJSONデータとして出力して。" + user_prompt
    if not os.path.exists(img_path):
        detect_timeline_onlyonestage(stage_no)#時刻の読み取り
    if st.session_state.ocr_tgt_img_type == "ライブ":
        return_json = gpt_ocr.getocr_fes_timetable_notime(img_path,user_prompt)
    elif st.session_state.ocr_tgt_img_type == "特典会":
        return_json = gpt_ocr.getocr_fes_timetable_notime(img_path,user_prompt,live=False)
    return_json["ステージ名"] = "ステージ{}".format(stage_no)#st.session_state.stage_names[stage_no]
    json_path = os.path.join(st.session_state.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "stage_{}.json".format(stage_no))
    with open(json_path,"w",encoding = "utf8") as f:
        json.dump(return_json, f, indent = 4, ensure_ascii = False)

def get_timetabledata_eachstage_notime(stage_num,user_prompt):
    for i in range(stage_num):
        get_timetabledata_onlyonestage_notime(i,user_prompt)

def pix_to_time(pix):#ピクセル値を時刻に変換する関数
    min = np.round((pix-st.session_state.start_pix)/(st.session_state.total_pix/st.session_state.total_duration*5))*5
    return (datetime(2024,1,1,st.session_state.time_start.hour,st.session_state.time_start.minute)+timedelta(minutes=min)).time()

def idolname_correct_onlyonestage(stage_no):
    json_path = os.path.join(st.session_state.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "stage_{}.json".format(stage_no))
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

def idolname_correct_eachstage(stage_num):#アイドル名の修正
    for i in range(stage_num):
        idolname_correct_onlyonestage(i)


def save_timetable_data_onlyonestage(stage_no):#df_timetable, json_path
    df_timetable = st.session_state.df_timetables[stage_no-1]
    json_path = os.path.join(st.session_state.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "stage_{}.json".format(stage_no))
    json_timetable = timetabledata.df_to_json(df_timetable)
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            json_old = json.load(f)
    else:
        json_old={}
    json_old["タイムテーブル"]=json_timetable
    with open(json_path,"w",encoding = "utf8") as f:
        json.dump(json_old, f, indent = 4, ensure_ascii = False)

def save_timetable_data_eachstage(stage_num):
    for i in range(stage_num):
        save_timetable_data_onlyonestage(i)


def get_all_stage_info():#全ステージ情報の出力 #暫定
    all_stage_df = pd.concat(st.session_state.df_timetables).reset_index(drop=True)
    with all_stage_info:
        st.dataframe(all_stage_df,hide_index=True)


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
    st.markdown("""#### ②タイムテーブル画像の登録""")
    all_files_raw =  st.container(height=200)
    col_file_uploader = st.columns((3,1))
with all_files_raw:
    st.markdown("""###### 登録済みタイムテーブル画像一覧""")
    image_num = 0
    for i in range(1,st.session_state.event_num+1):
        image_num += len(os.listdir(os.path.join(st.session_state.pj_path, "event_{}".format(i))))
    if image_num <1:
        image_num=1
    col_all_files = st.columns(image_num)
    image_idx = 0
    for i in range(1,st.session_state.event_num+1):
        event_type_list = get_event_type_list(i)
        for img_type in event_type_list:
            img_path = os.path.join(st.session_state.pj_path, "event_{}".format(i), img_type, "raw.png")
            if os.path.exists(img_path):
                with col_all_files[image_idx]:
                    image = Image.open(img_path)
                    st.image(image)
            image_idx += 1
with col_file_uploader[0]:
    st.file_uploader("読み取りたいタイムテーブル画像をアップロードしてください。"
                            , type=["jpg", "jpeg", "png", "jfif"]
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
        st.radio("種別", ("ライブ", "特典会", "両方(特典会別添え)", "両方(特典会併記)","その他"), key="img_type", horizontal=True)
        st.text_input("その他の種別", key="img_type_alternative")
        # st.radio("種別", ("ライブ", "特典会"), key="img_type", horizontal=True)
        if st.session_state.img_type != "両方(特典会併記)":
            st.radio("形式", ("通常", "ライムライト式"), key="img_format", horizontal=True)
        else:
            st.radio("形式", ("通常", "ライムライト式"), key="img_format", horizontal=True, disabled=True)
        if st.session_state.img_type == "その他" and (st.session_state.img_type_alternative == "" or st.session_state.img_type_alternative is None):
            st.button(label="画像を登録する", on_click=determine_timetable_image, type="primary", disabled=True)
        else:
            st.button(label="画像を登録する", on_click=determine_timetable_image, type="primary")

timetable_crop = st.container()#タイテ画像の切り取り
with timetable_crop:
    st.markdown("""#### ③タイムテーブル画像の切り取り""")
    st.info(
"""複数ステージが一つの画像に存在している場合、各ステージごとに画像を切り分けます。  
この時、各ステージの幅が概ね均等であり、また間に時間軸なども等しく入っているor入っていない場合は、  
画像を均等に分割することによりステージごとの画像を取得できます。  
しかし、そうでない場合などには縦線を検出して、それに基づいた分割を行います。  
縦線検による分割は、精度が100%出るとは限らないため注意してください。
一方でステージに関係ある情報の部分のみを取得できるというメリットがあります。
画像を見て、どちらのパターンがふさわしいかを判断して作業を行ってください。  
""")
    event_list = os.listdir(st.session_state.pj_path)
    st.selectbox("イベント", event_list,index=0,key="crop_tgt_event",on_change=set_crop_image)
    event_type_list = ["ライブ","特典会"]
    for event_type in os.listdir(os.path.join(st.session_state.pj_path,st.session_state.crop_tgt_event)):
        if event_type not in event_type_list:
            event_type_list.append(event_type)
    st.selectbox("種別", event_type_list,index=0,key="crop_tgt_img_type",on_change=set_crop_image)
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
・このあとのステップで均等割を行う場合には、それに適した領域に切り出してください  
""")
            col_cropimage_first = st.columns(2)
            with col_cropimage_first[0]:
                st.session_state.cropped_image = st_cropper(image)
            with col_cropimage_first[1]:
                st.image(st.session_state.cropped_image,use_column_width=True)

        with st.container():# ステージごとにタイムテーブル領域を分割する
            st.markdown("""###### ステージごとにタイムテーブル領域を分割する""")
            st.info(
"""次に、切り出した領域を複数あるステージごとに分割します。  
分割の方法は2種類あり、「縦線検出による分割」と「均等幅での分割」です。  
分割を行い、採用不採用を決めたら確定ボタンを押してステージごとの画像を確定してください。
  
###### 縦線検出による分割
- エッジ検出とハフ変換という手法によって、縦に長いラインを画像から発見し、そこで画像を分割します。  
- 分割の精度は100%ではありません。アルゴリズムのパラメータ変更である程度対応できるようになります（未実装）。  
- パラメータ変更でも対応できなかったときの場合に、手動で分割併合が出来る機能を実装予定です。  
- 分割された領域の中には、ステージに該当しない余白領域などが存在する可能性がありますので、採用不採用をチェックしてください。  
  
###### 均等幅での分割  
- ステージ数を入力し、横幅が均等になるように画像をステージの数だけ分割します。  
- 均等にステージが表現されている場合はこちらの手法が安定します。
- 分割位置がきれいでない場合、一つ前の工程での領域抽出を修正すると良いかもしれません。
- 前後5%ぶん余裕をとっているので、隣り合う画像に重複する部分が出現します
""")

                    
            split_button = st.columns(2)
            with split_button[0]:
                st.button("縦ラインの自動抽出による分割",on_click=detect_stageline,args=(st.session_state.cropped_image,),type="primary")
                with st.expander("縦ライン抽出のパラメータ"):
                    st.slider('エッジ抽出の閾値（下限）', value=130, min_value=1, max_value=500, step=1, key="x_edge_threshold_1", on_change=detect_stageline,args=(st.session_state.cropped_image,))
                    st.slider('エッジ抽出の閾値（上限）', value=285, min_value=1, max_value=500, step=1, key="x_edge_threshold_2", on_change=detect_stageline,args=(st.session_state.cropped_image,))
                    st.slider('ハフ変換の閾値', value=100, min_value=1, max_value=500, step=1, key="x_hough_threshold", on_change=detect_stageline,args=(st.session_state.cropped_image,))
                    st.slider('ハフ変換で許容する線分の飛び', value=3, min_value=0, max_value=100, step=1, key="x_hough_gap", on_change=detect_stageline,args=(st.session_state.cropped_image,))
                    st.slider('抽出線分の長さ（元画像の縦に対する比率）', value=0.01, min_value=0.0, max_value=1.0, step=0.001, key="x_minlength_rate", on_change=detect_stageline,args=(st.session_state.cropped_image,))
                    st.slider('同一視する線分の許容誤差幅', value=5, min_value=1, max_value=30, step=1, key="x_identify_interval", on_change=detect_stageline,args=(st.session_state.cropped_image,))
            with split_button[1]:
                st.button("画像を均等な横幅で指定数に分割",on_click=get_image_eachstage_for_croppedimage_byevenly,type="primary")
                st.number_input("ステージ数",1,step=1,key="stage_num")
            edge_result = st.container()
            determine_image_eachstage_button_area_1 = st.container()
            each_stage_area = st.container(height=500)
            determine_image_eachstage_button_area_2 = st.container()
            if "images_eachstage" in st.session_state:
                stage_num = len(st.session_state.images_eachstage)
                if stage_num >0:
                    with determine_image_eachstage_button_area_1:
                        st.button("ステージごとの画像を確定",on_click=determine_image_eachstage_without_nocheck,type="primary",key="determine_image_eachstage_button_1")
                    with each_stage_area:
                        col_eachstage = st.columns(stage_num)
                        for i in range(stage_num):
                            with col_eachstage[i]:
                                #画像の幅（最頻値近辺）で採用不採用のデフォルト値を切り替えても良いかも
                                st.checkbox("採用",key="each_stage_acept_{}".format(i),value=True)
                                st.image(st.session_state.images_eachstage[i])
                    with determine_image_eachstage_button_area_2:
                        st.button("ステージごとの画像を確定",on_click=determine_image_eachstage_without_nocheck,key="determine_image_eachstage_button_2")


            # if len(st.session_state.images_eachstage)>0:
            #     col_cropimage_eachstage = st.columns([img.size[0] for img in st.session_state.images_eachstage])
            #     for i,eachstage_image in enumerate(st.session_state.images_eachstage):
            #         with col_cropimage_eachstage[i]:
            #             st.image(eachstage_image)

            #     st.button("ステージごとの画像分割を確定",on_click=determine_image_eachstage, type="primary")

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
#     img_path = os.path.join(st.session_state.pj_path, st.session_state.crop_tgt_event, st.session_state.crop_tgt_img_type, "raw.png")
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
#                 st.image(st.session_state.cropped_image,use_column_width=True)

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

timetable_ocr = st.container()#タイテ画像の読み取り
with timetable_ocr:
    st.markdown("""#### ④タイムテーブル画像の読み取り""")
    event_list = os.listdir(st.session_state.pj_path)
    st.selectbox("イベント", event_list,index=0,key="ocr_tgt_event",on_change=set_ocr_image)#変わった時にst.session_state.timeline_eachstageなどをリセット
    event_type_list = ["ライブ","特典会"]
    for event_type in os.listdir(os.path.join(st.session_state.pj_path,st.session_state.ocr_tgt_event)):
        if event_type not in event_type_list:
            event_type_list.append(event_type)
    st.selectbox("種別", event_type_list,index=0,key="ocr_tgt_img_type",on_change=set_ocr_image)#同上
    img_path = os.path.join(st.session_state.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "raw.png")
    if os.path.exists(img_path):
        image = Image.open(img_path)
        image_info = st.session_state.pj_timetable_master[
            (st.session_state.pj_timetable_master["event_no"]==int(st.session_state.ocr_tgt_event.split("_")[1]))
            & (st.session_state.pj_timetable_master["image_type"]==st.session_state.ocr_tgt_img_type)
        ].iloc[0]
        stage_num = image_info["stage_num"]
        if image_info["image_format"]=="ライムライト式":# 時間軸の設定（ライムライト式の場合のみ実施）
            cropped_img_path = os.path.join(st.session_state.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "raw_cropped.png")
            if os.path.exists(cropped_img_path):
                cropped_image = Image.open(cropped_img_path)
                with st.container():
                    st.markdown("""###### 時間軸の設定（ライムライト式の場合のみ実施）""")
                    st.info(
"""時間軸の基準となる位置を指定してください。  
・画像上でドラッグして長方形を描画します  
・横幅は特に関係ないので、適当で大丈夫です  
・縦位置は出演枠が存在している領域でセットしてください。余白部分は微妙に間隔が違ったりします
・ドラッグした領域の上端と下端に対応する時刻を入力して、画像上のどの位置がどの時刻か対応づけます  
・複数の長方形が描画できしまいますが、一番最初のものが採用されるので、変更したい時には画像下の削除ボタンやundoボタンを使ってください  
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
                            key="time_axis_detect"
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
                                st.session_state.start_pix = resized_start_pix/canvas_rate
                                st.session_state.total_pix = resized_total_pix/canvas_rate
                                st.session_state.total_duration = total_duration 

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

        with st.container():# 各ステージ情報の読み取り
            st.markdown("""###### 各ステージ情報の読み取り""")
            if image_info["image_format"]=="ライムライト式":
                st.info(
"""アーティストごとに時間が書いていないタイムテーブルの場合、横線を検出して時間を推定します。  
その後推定した時刻を画像に書き込み、そこからタイムテーブル情報を生成します。  
画像の色合いによって横線の検出の難易度が変わるので、パラメータをいじってください。  
- 出演枠の線と塗りつぶしの色合いが近い場合、「エッジ抽出の閾値（上限）」を下げてみてください  
- 検出された線が途切れ途切れになったり短かったりする場合、「エッジ抽出の閾値（下限）」を下げてみてください  
- 文字上などの検出したくない線が検出されている場合、「ハフ変換で許容する線分の飛び」を小さく(0に)してみてください  
- その他、うまくいかない時には「エッジ抽出の閾値（上限）」や「ハフ変換の閾値」を下げてみてください  
  
※「全ステージの画像を読み取りを実施」をいきなり押すと、先に横線の時刻の読み取りを裏で実行してからタイムテーブル情報の読み取りを行います  
※「全ステージの横線の時刻の読み取りを実施」を押して、正しく時刻が取得できたことを画像で確認してから、「全ステージの画像を読み取りを実施」を押すことを推奨します
""")
                # st.button("全ステージの画像をそれぞれ読み取り実施",on_click=get_timetabledata_eachstage,args=(stage_num,),type="primary")
                st.button("全ステージの横線の時刻の読み取りを実施",on_click=detect_timeline_eachstage,args=(stage_num,),type="primary")
                st.text_input("画像読み取りの追加指示",key="ocr_user_prompt")
                st.button("全ステージの画像を読み取りを実施",on_click=get_timetabledata_eachstage_notime,args=(stage_num,st.session_state.ocr_user_prompt),type="primary")
                st.number_input("ステージ番号",0,stage_num-1,key="ocr_tgt_stage_no")
                st.button("あるステージのみ横線の時刻の読み取りを実施",on_click=detect_timeline_onlyonestage,args=(st.session_state.ocr_tgt_stage_no,))
                st.button("あるステージのみ読み取りを実施",on_click=get_timetabledata_onlyonestage_notime,args=(st.session_state.ocr_tgt_stage_no,st.session_state.ocr_user_prompt))
                with st.expander("時間ライン抽出のパラメータ"):
                    # st.slider('白黒二値化の閾値（0はグレースケールのまま実施）', value=0, min_value=0, max_value=255, step=1, key="y_binary_threshold")#, on_change=detect_timeline_onlyonestage,args=(st.session_state.ocr_tgt_stage_no,))
                    st.slider('エッジ抽出の閾値（下限）', value=80, min_value=1, max_value=500, step=1, key="y_edge_threshold_1")#, on_change=detect_timeline_onlyonestage,args=(st.session_state.ocr_tgt_stage_no,))
                    st.slider('エッジ抽出の閾値（上限）', value=150, min_value=1, max_value=500, step=1, key="y_edge_threshold_2")#, on_change=detect_timeline_onlyonestage,args=(st.session_state.ocr_tgt_stage_no,))
                    st.slider('ハフ変換の閾値', value=60, min_value=1, max_value=500, step=1, key="y_hough_threshold")#, on_change=detect_timeline_onlyonestage,args=(st.session_state.ocr_tgt_stage_no,))
                    st.slider('ハフ変換で許容する線分の飛び', value=1, min_value=0, max_value=100, step=1, key="y_hough_gap")#, on_change=detect_timeline_onlyonestage,args=(st.session_state.ocr_tgt_stage_no,))
                    st.slider('抽出線分の長さ（元画像の縦に対する比率）', value=0.01, min_value=0.0, max_value=1.0, step=0.001, key="y_minlength_rate")#, on_change=detect_timeline_onlyonestage,args=(st.session_state.ocr_tgt_stage_no,))
                    st.slider('同一視する線分の許容誤差幅', value=5, min_value=1, max_value=30, step=1, key="y_identify_interval")#, on_change=detect_timeline_onlyonestage,args=(st.session_state.ocr_tgt_stage_no,))
                tmp_timeline = st.container()#暫定
            elif image_info["image_format"]=="特典会併記":
                st.text_input("画像読み取りの追加指示",key="ocr_user_prompt")
                st.button("全ステージの画像をそれぞれ読み取り実施",on_click=get_timetabledata_withtokutenkai_eachstage,args=(stage_num,st.session_state.ocr_user_prompt),type="primary")
                st.number_input("ステージ番号",0,stage_num-1,key="ocr_tgt_stage_no")
                st.button("あるステージのみ読み取りを実施",on_click=get_timetabledata_withtokutenkai_onlyonestage,args=(st.session_state.ocr_tgt_stage_no,st.session_state.ocr_user_prompt))
            else:
                st.text_input("画像読み取りの追加指示",key="ocr_user_prompt")
                st.button("全ステージの画像をそれぞれ読み取り実施",on_click=get_timetabledata_eachstage,args=(stage_num,st.session_state.ocr_user_prompt),type="primary")
                st.number_input("ステージ番号",0,stage_num-1,key="ocr_tgt_stage_no")
                st.button("あるステージのみ読み取りを実施",on_click=get_timetabledata_onlyonestage,args=(st.session_state.ocr_tgt_stage_no,st.session_state.ocr_user_prompt))
            stage_tabs = st.tabs(["ステージ"+str(i) for i in range(stage_num)])

            st.session_state.df_timetables = []
            for i in range(stage_num):
                with stage_tabs[i]:
                    if image_info["image_format"]=="ライムライト式":
                        img_path = os.path.join(st.session_state.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "stage_{}_addtime.png".format(i))
                        if not os.path.exists(img_path):
                            img_path = os.path.join(st.session_state.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "stage_{}.png".format(i))
                    else:
                        img_path = os.path.join(st.session_state.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "stage_{}.png".format(i))
                    if os.path.exists(img_path):
                        image = Image.open(img_path)
                    ocr_col = st.columns([min(int(image.size[1]/50),40),100])
                    with ocr_col[0]:
                        st.markdown("""###### タイテ画像""")
                        with st.container(height=500):
                            if os.path.exists(img_path):
                                st.image(image, use_column_width=True)
                    with ocr_col[1]:
                        st.markdown("""###### 読み取り結果""")
                        if image_info["image_format"]=="ライムライト式":
                            st.button("このステージの横線の時刻の読み取りを実施",on_click=detect_timeline_onlyonestage,args=(i,),key="button_ocr_timeline_stage{}".format(i))
                            st.text_input("画像読み取りの追加指示",key="ocr_user_prompt_stage{}".format(i))
                            st.button("このステージの読み取りを実施",on_click=get_timetabledata_onlyonestage_notime,args=(i,st.session_state["ocr_user_prompt_stage{}".format(i)]),key="button_ocr_stage{}".format(i))
                        elif image_info["image_format"]=="特典会併記":
                            st.text_input("画像読み取りの追加指示",key="ocr_user_prompt_stage{}".format(i))
                            st.button("このステージの読み取りを実施",on_click=get_timetabledata_withtokutenkai_onlyonestage,args=(i,st.session_state["ocr_user_prompt_stage{}".format(i)]),key="button_ocr_stage{}".format(i))
                        else:
                            st.text_input("画像読み取りの追加指示",key="ocr_user_prompt_stage{}".format(i))
                            st.button("このステージの読み取りを実施",on_click=get_timetabledata_onlyonestage,args=(i,st.session_state["ocr_user_prompt_stage{}".format(i)]),key="button_ocr_stage{}".format(i))
                        json_path = os.path.join(st.session_state.pj_path, st.session_state.ocr_tgt_event, st.session_state.ocr_tgt_img_type, "stage_{}.json".format(i))
                        if os.path.exists(json_path):
                            with open(json_path, encoding="utf-8") as f:
                                return_json = json.load(f)
                            if image_info["image_format"]=="特典会併記":
                                return_json_df = timetabledata.json_to_df(return_json)
                            else:
                                return_json_df = timetabledata.json_to_df(return_json,tokutenkai=False)
                            # st.dataframe(return_json_df)
                            edited_df = st.data_editor(return_json_df, num_rows="dynamic")
                            
                            edited_df["ステージNo."]=i
                            st.session_state.df_timetables.append(edited_df)

                            st.button("このステージのグループ名をAIで修正",on_click=idolname_correct_onlyonestage,args=(i,),key="button_correct_idolname_stage{}".format(i))
                            st.button("このステージの編集結果を保存",on_click=save_timetable_data_onlyonestage,args=(i,),key="button_save_timetable_stage{}".format(i))

            st.button("全ステージのグループ名をAIで修正",on_click=idolname_correct_eachstage,args=(stage_num,),key="button_correct_idolname")
            st.button("全ステージの編集結果を保存",on_click=save_timetable_data_eachstage,args=(stage_num,),key="button_save_timetable")
            st.button("全ステージのデータを表形式で出力",on_click=get_all_stage_info,type="primary")
            all_stage_info = st.container()

# st.session_state.event_type#フェスか対バンか
# st.session_state.event_num#イベント数
# st.session_state.pj_timetable_master#各イベントのライブ、特典会の情報