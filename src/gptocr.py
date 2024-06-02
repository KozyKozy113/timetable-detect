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
FILE_PATH_TIMETABLE = DIR_PATH +"/../data/timetable_sample"

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

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

def getocr_taiban_filename(file_name, prompt_user = "この画像のタイムテーブルをJSONデータとして出力して"):
    with open(DIR_PATH+"/prompt_system/taiban.txt", "r", encoding="utf-8") as f:
        prompt_system = f.read()
    image_path = os.path.join(FILE_PATH_TIMETABLE, file_name)
    response = getocr(image_path, prompt_user, prompt_system)
    return json.loads(response.choices[0].message.content)

def getocr_taiban(image_path, prompt_user = "この画像のタイムテーブルをJSONデータとして出力して"):
    with open(DIR_PATH+"/prompt_system/taiban.txt", "r", encoding="utf-8") as f:
        prompt_system = f.read()
    response = getocr(image_path, prompt_user, prompt_system)
    return json.loads(response.choices[0].message.content)

def getocr_fes_info(image_path, prompt_user = "このタイムテーブルの情報を教えて"):
    with open(DIR_PATH+"/prompt_system/fes_info.txt", "r", encoding="utf-8") as f:
        prompt_system = f.read()
    response = getocr(image_path, prompt_user, prompt_system)
    return json.loads(response.choices[0].message.content)

def getocr_fes_timetable(image_path, prompt_user = "この画像のタイムテーブルをJSONデータとして出力して"):
    with open(DIR_PATH+"/prompt_system/fes_timetable_singlestage.txt", "r", encoding="utf-8") as f:
        prompt_system = f.read()
    response = getocr(image_path, prompt_user, prompt_system)
    return json.loads(response.choices[0].message.content)