import os
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    OpenAI.api_key = openai_api_key
client = OpenAI()

import base64
import json

GPT_MODEL_NAME = "gpt-4o"
DIR_PATH = os.path.dirname(__file__)

#画像のエンコーディング
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

#画像の読み解き
def getocr(image_path, prompt_user, prompt_system):
    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model=GPT_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": prompt_system
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_user},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        response_format={"type": "json_object"},
        max_tokens=4096
    )

    return response

def getocr_taiban(image_path, prompt_user = "この画像のタイムテーブルをJSONデータとして出力して"):
    with open(DIR_PATH+"/../prompt_system/taiban.txt", "r", encoding="utf-8") as f:
        prompt_system = f.read()
    response = getocr(image_path, prompt_user, prompt_system)
    return json.loads(response.choices[0].message.content)

def getocr_fes_info(image_path, prompt_user = "このタイムテーブルの情報を教えて"):
    with open(DIR_PATH+"/../prompt_system/fes_info.txt", "r", encoding="utf-8") as f:
        prompt_system = f.read()
    response = getocr(image_path, prompt_user, prompt_system)
    return json.loads(response.choices[0].message.content)

def getocr_fes_timetable(image_path, prompt_user = "この画像のタイムテーブルをJSONデータとして出力して"):
    with open(DIR_PATH+"/../prompt_system/fes_timetable_singlestage.txt", "r", encoding="utf-8") as f:
        prompt_system = f.read()
    response = getocr(image_path, prompt_user, prompt_system)
    return json.loads(response.choices[0].message.content)

def getocr_fes_timetable_notime(image_path, prompt_user = "この画像のタイムテーブルをJSONデータとして出力して"):
    with open(DIR_PATH+"/../prompt_system/fes_timetable_singlestage_notime.txt", "r", encoding="utf-8") as f:
        prompt_system = f.read()
    response = getocr(image_path, prompt_user, prompt_system)
    return json.loads(response.choices[0].message.content)

#度数分布データから、等間隔に出現するピーク値を指定個数分出力する
def get_xpoint(data,n):
    value_counts_dict = data.to_dict()
    value_counts_json = json.dumps(value_counts_dict)

    response = client.chat.completions.create(
        model=GPT_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": """
あなたの役割は、与えられた度数分布表を元に、指示された個数の頻出値を出力することです。
ただし、出力する頻出値は、最頻値の上位から順番に選択するのではなく、
下記の条件に従う必要があります。
・値のわずかなズレは誤差であるため、同じ値として扱う（その中で最も頻度の高い値を代表値とする）
・選択する頻出値の間隔は概ね等しくなるようにする
・出力はJSON形式で行う
・頻出値は小さい順に並べる

#入出力例
##入力
・度数分布表データ
{
    1:3
    15:100
    16:70
    17:20
    25:3
    30:1
    68:3
    124:18
    125:90
    126:60
    129:8
    144:3
    160:3
    177:2
    178:1
    235:20
    236:70
    237:65
    240:10
    245:3
    280:1
    319:2
    320:1
    321:1
    345:2
    346:8
}
・抽出する頻出値の個数
3

##出力
{"xpoint":[15,125,236]}
"""
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": """
# 度数分布表データ
{data_hist}
                     
# 抽出する頻出値の個数
{n}個""".format(data_hist=value_counts_json,n=n)},
                ],
            }
        ],
        response_format={"type": "json_object"},
        max_tokens=4096
    )

    return response