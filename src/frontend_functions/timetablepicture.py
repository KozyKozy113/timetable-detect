import sys
import os
from PIL import Image, ImageDraw, ImageFont
import textwrap
import datetime
# import numpy as np

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
# font_path = os.path.abspath(os.path.join(DIR_PATH, "Fonts/BIZ-UDGothicB.ttc"))
font_path = os.path.abspath(os.path.join(DIR_PATH, "Fonts/NotoSansJP-Regular.otf"))

def create_timetable_image(json_data, start_margin=None, time_line_spacing=None, box_color="yellow"):
    if "タイムテーブル" not in json_data.keys() or len(json_data["タイムテーブル"])==0:
        return None
    time_format = "%H:%M"
    json_data_timetable = []
    for live in json_data["タイムテーブル"]:
        try:
            datetime.datetime.strptime(live["ライブステージ"]["from"], time_format)
            datetime.datetime.strptime(live["ライブステージ"]["to"], time_format)
            json_data_timetable.append(live)
        except Exception:
            continue
    json_data["タイムテーブル"] = json_data_timetable
    min_minutes = min(int((datetime.datetime.strptime(live["ライブステージ"]["to"], time_format)
                    -datetime.datetime.strptime(live["ライブステージ"]["from"], time_format)).total_seconds() / 60)
                      for live in json_data["タイムテーブル"])
    # 画像の基本設定
    image_width = 400 #画像横幅
    if start_margin is None:
        start_margin = 10 #画像上部のマージン
    margin = 10 # 周囲のマージン
    box_margin = 20 # 出演枠の四角の左右マージン
    text_margin = 10 # 四角内のテキスト表示マージン
    duplicate_margin = 20 # 時間が被った際のインデント幅
    oneline_flag = False #アーティスト名と時間を一列で表示するか
    if time_line_spacing is None:
        time_line_spacing = 90 # 縦pixel幅/30分ごと
        text_font_size = 20 #フォントサイズ
    else:
        if time_line_spacing*min_minutes/30 >= 60:
            text_font_size = int(time_line_spacing*min_minutes/30/5)
            image_width = int(text_font_size*20)
        else:
            text_font_size = int(time_line_spacing/2)
            image_width = int(text_font_size*30)
            oneline_flag = True
    timeline_text_margin = int(text_font_size*3) # 時間軸（値）表示幅

    text_color = "black"
    background_color = "white"
    line_color = "gray"
    
    # タイムテーブルの時間範囲を計算
    start_time = min(datetime.datetime.strptime(live["ライブステージ"]["from"], time_format) 
                     for live in json_data["タイムテーブル"])
    end_time = max(datetime.datetime.strptime(live["ライブステージ"]["to"], time_format) 
                   for live in json_data["タイムテーブル"])
    
    # タイムテーブル全体の時間範囲を30分刻みに拡張
    start_time = start_time.replace(minute=0)
    end_time = end_time.replace(minute=0) + datetime.timedelta(hours=1)
    total_minutes = int((end_time - start_time).total_seconds() / 60)
    
    # 画像の高さを動的に調整
    image_height = int(start_margin + margin + (total_minutes // 30) * time_line_spacing)
    
    # 画像を生成
    image = Image.new("RGB", (image_width, image_height), background_color)
    draw = ImageDraw.Draw(image)
    
    # フォントの設定（必要に応じて変更）
    try:
        if not os.path.exists(font_path):
            print("フォントファイルが見つかりません:", font_path)
        # font = ImageFont.truetype('\C:\Windows\Fonts\BIZ-UDGothicB.ttc', text_font_size)
        font = ImageFont.truetype(font_path, text_font_size)
        # font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        print("font error")
        font = ImageFont.load_default()

    # 時間軸を描画
    current_y = start_margin
    current_time = start_time
    while current_time <= end_time:
        draw.line([(margin + timeline_text_margin, int(current_y)), (image_width - margin, int(current_y))], fill=line_color, width=1)
        draw.text((10, int(current_y) - margin), current_time.strftime("%H:%M"), fill=text_color, font=font)
        current_y += time_line_spacing
        current_time += datetime.timedelta(minutes=30)
    
    # 各ライブ枠を描画
    duplicate_margin_total = 0
    before_end = datetime.datetime.strptime(json_data["タイムテーブル"][0]["ライブステージ"]["from"], time_format)
    for live in json_data["タイムテーブル"]:
        # 開始時間と終了時間を計算
        start = datetime.datetime.strptime(live["ライブステージ"]["from"], time_format)
        end = datetime.datetime.strptime(live["ライブステージ"]["to"], time_format)
        if start<before_end:
            duplicate_margin_total += duplicate_margin
        else:
            duplicate_margin_total = 0

        minutes = int((end-start).total_seconds() / 60)

        start_y = start_margin + int((start - start_time).total_seconds() / 60 / 30 * time_line_spacing)
        end_y = start_margin + int((end - start_time).total_seconds() / 60 / 30 * time_line_spacing)
        
        # 枠を描画
        draw.rectangle([(margin + timeline_text_margin + box_margin + duplicate_margin_total, start_y), (image_width - margin - box_margin, end_y)], 
                       fill=box_color, outline="black")
        
        # テキストを描画
        try:
            artist_name = live["グループ名_採用"]
            if artist_name=="" or artist_name is None:
                artist_name = live["グループ名"]
        except KeyError:
            artist_name = live["グループ名"]
        time_text = f"{live['ライブステージ']['from']} ～ {live['ライブステージ']['to']} ({minutes})"

        box_height = end_y - start_y
        max_width = image_width - 2 * (margin + box_margin + text_margin) - timeline_text_margin - duplicate_margin_total
        if oneline_flag:
            text_lines = [f"{time_text} {artist_name}"]
        else:
            wrapped_text = textwrap.fill(artist_name, width=int(max_width / font.getbbox("あ")[2]))#折り返し処理 #日本語基準で測っているので英語の場合への対応が必要 #最初と最後が英語とか？
            text_lines = wrapped_text.split('\n') + [time_text]
        text_height = sum(font.getbbox(line)[3] for line in text_lines)        
        text_x = margin + timeline_text_margin + box_margin + text_margin + duplicate_margin_total
        text_start_y = start_y + (box_height - text_height) // 2
        for line in text_lines:
            draw.text((text_x, text_start_y), line, fill=text_color, font=font)
            text_start_y += font.getbbox(line)[3]

        before_end = max(before_end,end)
    # 画像を出力
    return image
    
    # 画像を保存
    # image.save(output_path)
    # print(f"タイムテーブル画像が作成されました: {output_path}")