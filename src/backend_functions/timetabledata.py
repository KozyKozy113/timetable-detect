import os
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
import faiss

openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    OpenAI.api_key = openai_api_key
client = OpenAI()

import pandas as pd
import numpy as np

EMBEDDING_MODEL_NAME = "text-embedding-3-small"
DIR_PATH = os.path.dirname(__file__)
DATA_PATH = DIR_PATH +"/../../data"

def json_to_df(json_data, tokutenkai=True):
    df_timetable = []
    for item in json_data["タイムテーブル"]:
        # ライブステージの時間を取得
        try:
            live_stage_from = item['ライブステージ']['from']            
        except KeyError:
            live_stage_from = ""
        try:
            live_stage_to = item['ライブステージ']['to']
        except KeyError:
            live_stage_to = ""
        try:
            group_name_corrected = item['グループ名_修正候補']         
        except KeyError:
            group_name_corrected = ""
        
        # 特典会の情報を処理
        if tokutenkai:
            if type(item['特典会'])==list:
                for meeting in item['特典会']:
                    try:
                        meeting_from = meeting['from']
                    except KeyError:
                        meeting_from = ""
                    try:
                        meeting_to = meeting['to']
                    except KeyError:
                        meeting_to = ""
                    try:
                        booth = meeting['ブース']
                    except KeyError:
                        booth = ""
                
                    # DataFrameに行を追加
                    df_timetable.append({
                        'グループ名': item['グループ名'],
                        'グループ名_修正候補': group_name_corrected,
                        'ライブ_from': live_stage_from,
                        'ライブ_to': live_stage_to,
                        '特典会_from': meeting_from,
                        '特典会_to': meeting_to,
                        'ブース': booth
                    })
                if len(item['特典会'])==0:
                    df_timetable.append({
                        'グループ名': item['グループ名'],
                        'グループ名_修正候補': group_name_corrected,
                        'ライブ_from': live_stage_from,
                        'ライブ_to': live_stage_to,
                        '特典会_from': "",
                        '特典会_to': "",
                        'ブース': ""
                    })
            else:
                try:
                    meeting_from = item['特典会']['from']
                except KeyError:
                    meeting_from = ""
                try:
                    meeting_to = item['特典会']['to']
                except KeyError:
                    meeting_to = ""
                try:
                    booth = item['特典会']['ブース']
                except KeyError:
                    booth = ""
            
                # DataFrameに行を追加
                df_timetable.append({
                    'グループ名': item['グループ名'],
                    'グループ名_修正候補': group_name_corrected,
                    'ライブ_from': live_stage_from,
                    'ライブ_to': live_stage_to,
                    '特典会_from': meeting_from,
                    '特典会_to': meeting_to,
                    'ブース': booth
                })

        else:
            # DataFrameに行を追加
            df_timetable.append({
                'グループ名': item['グループ名'],
                'グループ名_修正候補': group_name_corrected,
                'ライブ_from': live_stage_from,
                'ライブ_to': live_stage_to,
            })

    df_timetable = pd.DataFrame(df_timetable,columns=['グループ名', 'グループ名_修正候補', 'ライブ_from', 'ライブ_to', '特典会_from', '特典会_to', 'ブース'])
    try:
        df_timetable['ライブ_from'] = pd.to_datetime(df_timetable['ライブ_from'], format='%H:%M')
        df_timetable['ライブ_to'] = pd.to_datetime(df_timetable['ライブ_to'], format='%H:%M')
        df_timetable.sort_values(by="ライブ_from",inplace=True)
        df_timetable["ライブ_長さ(分)"] = ((df_timetable['ライブ_to'] - df_timetable['ライブ_from']).dt.total_seconds() / 60).astype(int)
        df_timetable['ライブ_from'] = df_timetable['ライブ_from'].dt.strftime('%H:%M')
        df_timetable['ライブ_to'] = df_timetable['ライブ_to'].dt.strftime('%H:%M')
    except ValueError:
        df_timetable["ライブ_長さ(分)"] = ""

    if tokutenkai:
        return df_timetable[['グループ名', 'グループ名_修正候補', 'ライブ_from', 'ライブ_to', 'ライブ_長さ(分)', '特典会_from', '特典会_to', 'ブース']]
    else:
        return df_timetable[['グループ名', 'グループ名_修正候補', 'ライブ_from', 'ライブ_to', 'ライブ_長さ(分)']]

def df_to_json(df_timetable):
    dict_timetable = df_timetable.to_dict(orient='records')
    json_timetable = []
    for item in dict_timetable:
        json_item = {}
        for col, v in item.items():
            if col in ['グループ名', 'グループ名_修正候補']:
                json_item[col]=v
            if col == "ライブ_from":
                json_item["ライブステージ"] = {"from":v}
            elif col == "ライブ_to":
                json_item["ライブステージ"]["to"]=v
            if col == "特典会_from":
                json_item["特典会"] = [{"from":v}]
            elif col == "特典会_to":
                json_item["特典会"][0]["to"]=v
            elif col == "ブース":
                json_item["特典会"][0]["ブース"]=v
        json_timetable.append(json_item)
    return json_timetable

#ベクトル化する関数
def get_embedding(text, model=EMBEDDING_MODEL_NAME, dim=100):
    response = client.embeddings.create(input=text, model=model, dimensions=dim)
    return response.data[0].embedding

#ベクトルデータベースの作成
data = pd.read_csv(os.path.join(DATA_PATH, "master/idolname_embedding_data.csv"))
embeddings = data.drop("idol_group_name",axis=1).values
d = len(embeddings[0])  # 次元数
index = faiss.IndexFlatL2(d)
index.add(embeddings)

#類似するデータの検索関数
def find_similar(text, k=3):
    embedding = np.array([get_embedding(text)]).astype('float32')
    distances, indices = index.search(embedding, k)
    return indices[0], distances[0]

#候補を適切な数出力する関数
def get_name_list(text, search_num=1):#return (bool(完全一致があったか), 名前候補(リスト))
    if search_num == 1:
        indices, distances = find_similar(text, 1)
        if distances[0]==0:
            return (True, data.iloc[indices[0]]['idol_group_name'])
        else:
            return (False, data.iloc[indices[0]]['idol_group_name'])
    else:
        indices, distances = find_similar(text, search_num)
        dist_before = 0
        name_list = []
        for i, dist in zip(indices, distances):
            if dist==0:
                return (True, [data.iloc[i]['idol_group_name']])
            elif dist - dist_before > 0.5:
                break
            else:
                name_list.append(data.iloc[i]['idol_group_name'])
                dist_before = dist
        return (False, name_list)
