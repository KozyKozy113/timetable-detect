import streamlit as st
from streamlit_cropperjs import st_cropperjs

from PIL import Image, ImageDraw
import cv2

import numpy as np
import pandas as pd
import json

import os

st.set_page_config(
    layout="wide"
)
# st.file_uploader("読み取りたいタイムテーブル画像をアップロードしてください。"
#                         , type=["jpg", "jpeg", "png", "jfif"]
#                         , key="uploaded_image")
# if st.session_state.uploaded_image:
#     image = st.session_state.uploaded_image.read()
# else:
#     st.stop()
# st.text(type(image))
# cropped_image = st_cropperjs(pic=image,btn_text="detect", key="foo")
# if cropped_image:
#     st.image(cropped_image, output_format="PNG",use_column_width=True)
#     st.session_state.cropped_image = cropped_image

img_path = "data/projects/atjam2024/event_1/ライブ/raw.png"#os.path.join(st.session_state.pj_path, st.session_state.crop_tgt_event, st.session_state.crop_tgt_img_type, "raw.png")
if os.path.exists(img_path):
    image = Image.open(img_path)

# import base64
# from io import BytesIO
# def image_to_base64(image):
#     buffered = BytesIO()
#     image.save(buffered, format="JPEG")
#     img_byte = buffered.getvalue()
#     return img_byte

# cropped_image = st_cropperjs(pic=image_to_base64(image),btn_text="detect", key="foo")
# if cropped_image:
#     # for zokusei in dir(cropped_image):
#     #     st.text(zokusei + ":")
#     #     try:
#     #         exec("st.text(cropped_image."+zokusei+"())")
#     #     except (TypeError, UnicodeDecodeError):
#     #         exec("st.text(cropped_image."+zokusei+")")
#     st.image(cropped_image, output_format="PNG",use_column_width=True)
#     st.session_state.cropped_image = cropped_image


from streamlit_cropper import st_cropper
box_color = st.color_picker(label="Box Color", value='#0000FF')
stroke_width = st.number_input(label="Box Thickness", value=1, step=1)
cols = st.columns([9,1])
with cols[0]:
    rect = st_cropper(image,
                    box_color=box_color,
                    stroke_width=stroke_width,
                    return_type="box", should_resize_image=True)
left, top, width, height = tuple(map(int, rect.values()))
st.write(f"Cropped Area (x: {left}, y: {top}, width: {width}, height: {height}) in original image.")
# st.image(st.session_state.cropped_image,use_column_width=True)
