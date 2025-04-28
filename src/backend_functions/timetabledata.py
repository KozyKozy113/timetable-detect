import os
# from dotenv import load_dotenv
# load_dotenv()
# from openai import OpenAI
# import faiss

# openai_api_key = os.getenv('OPENAI_API_KEY')
# if openai_api_key:
#     OpenAI.api_key = openai_api_key
# client = OpenAI()

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

EMBEDDING_MODEL_NAME = "text-embedding-3-small"
DIR_PATH = os.path.dirname(__file__)
DATA_PATH = DIR_PATH +"/../../data"

def calculate_duration(row, event_type):
    try:
        # event_type（'特典会' or 'ライブ'）に応じてカラム名を決定
        from_col = f'{event_type}_from'
        to_col = f'{event_type}_to'
        
        # from_col と to_col の時刻を変換
        from_time = pd.to_datetime(row[from_col], format='%H:%M')
        to_time = pd.to_datetime(row[to_col], format='%H:%M')
        
        # 時刻の差分を計算し、分に変換
        duration = (to_time - from_time).total_seconds() / 60
        return int(duration)  # 分単位で返す
    except Exception:
        return ""

def add_minutes_to_time(time_str: str, add_minutes: int) -> str:
    dt = datetime.strptime(time_str, "%H:%M")
    dt += timedelta(minutes=add_minutes)
    return dt.strftime("%H:%M")

def todatetime_strftime(row, col):
    try:
        return pd.to_datetime(row[col], format='%H:%M').dt.strftime('%H:%M')
    except Exception:
        return row[col]
        

def json_to_df(json_data, tokutenkai=True):
    df_timetable = []
    for item in json_data["タイムテーブル"]:
        try:
            group_name_corrected = item['グループ名_採用']         
        except KeyError:
            group_name_corrected = ""
        try:
            live_stage_from = item['ライブステージ']['from']            
        except KeyError:
            live_stage_from = ""
        try:
            live_stage_to = item['ライブステージ']['to']
        except KeyError:
            live_stage_to = ""
        try:
            artist_id = int(item['グループID'])
        except (KeyError, ValueError):
            artist_id = None
        try:
            turn_id = int(item['出番ID'])
        except (KeyError, ValueError):
            turn_id = None
        try:
            stage_id = int(item['ステージID'])
        except (KeyError, ValueError):
            stage_id = None
        try:
            remarks = item['備考']         
        except KeyError:
            remarks = ""
        
        # 特典会の情報を処理
        if tokutenkai:
            if type(item['特典会'])==list:
                for i, meeting in enumerate(item['特典会']):
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
                    try:
                        meeting_turn_id = int(meeting['出番ID'])
                    except (KeyError, ValueError):
                        meeting_turn_id = None
                    try:
                        meeting_stage_id = int(meeting['ステージID'])
                    except (KeyError, ValueError):
                        meeting_stage_id = None
                
                    # DataFrameに行を追加
                    if i==0:
                        df_timetable.append({
                            'グループ名': item['グループ名']
                            ,'グループ名_採用': group_name_corrected
                            ,'グループID': artist_id
                            ,'出番ID': turn_id
                            ,'ステージID': stage_id
                            ,'ライブ_from': live_stage_from
                            ,'ライブ_to': live_stage_to
                            ,'特典会_出番ID': meeting_turn_id
                            ,'特典会_ステージID': meeting_stage_id
                            ,'特典会_from': meeting_from
                            ,'特典会_to': meeting_to
                            ,'ブース': booth
                            ,'備考':remarks
                        })
                    else:
                        df_timetable.append({
                            'グループ名': item['グループ名']
                            ,'グループ名_採用': group_name_corrected
                            ,'グループID': artist_id
                            ,'出番ID': None
                            ,'ステージID': None
                            ,'ライブ_from': ""
                            ,'ライブ_to': ""
                            ,'特典会_出番ID': meeting_turn_id
                            ,'特典会_ステージID': meeting_stage_id
                            ,'特典会_from': meeting_from
                            ,'特典会_to': meeting_to
                            ,'ブース': booth
                            ,'備考':remarks
                        })
                if len(item['特典会'])==0:
                    df_timetable.append({
                        'グループ名': item['グループ名']
                        ,'グループ名_採用': group_name_corrected
                        ,'グループID': artist_id
                        ,'出番ID': turn_id
                        ,'ステージID': stage_id
                        ,'ライブ_from': live_stage_from
                        ,'ライブ_to': live_stage_to
                        ,'特典会_出番ID': None
                        ,'特典会_ステージID': None
                        ,'特典会_from': ""
                        ,'特典会_to': ""
                        ,'ブース': ""
                        ,'備考':remarks
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
                try:
                    meeting_turn_id = int(meeting['出番ID'])
                except (KeyError, ValueError):
                    meeting_turn_id = None
                try:
                    meeting_stage_id = int(meeting['ステージID'])
                except (KeyError, ValueError):
                    meeting_stage_id = None
            
                # DataFrameに行を追加
                df_timetable.append({
                    'グループ名': item['グループ名']
                    ,'グループ名_採用': group_name_corrected
                    ,'グループID': artist_id
                    ,'出番ID': turn_id
                    ,'ステージID': stage_id
                    ,'ライブ_from': live_stage_from
                    ,'ライブ_to': live_stage_to
                    ,'特典会_from': meeting_from
                    ,'特典会_to': meeting_to
                    ,'特典会_出番ID': meeting_turn_id
                    ,'特典会_ステージID': meeting_stage_id
                    ,'ブース': booth
                    ,'備考':remarks
                })

        else:
            # DataFrameに行を追加
            df_timetable.append({
                'グループ名': item['グループ名']
                ,'グループ名_採用': group_name_corrected
                ,'グループID': artist_id
                ,'出番ID': turn_id
                ,'ステージID': stage_id
                ,'ライブ_from': live_stage_from
                ,'ライブ_to': live_stage_to
                ,'備考':remarks
            })

    df_timetable = pd.DataFrame(df_timetable,columns=['グループ名', 'グループ名_採用', 'グループID', '出番ID', 'ステージID','ライブ_from', 'ライブ_to', '特典会_from', '特典会_to', 'ブース', '特典会_出番ID', '特典会_ステージID', '備考'])
    try:
        df_timetable["ライブ_長さ(分)"] = df_timetable.apply(calculate_duration, axis=1, event_type='ライブ')
        df_timetable["ライブ_from"] = df_timetable.apply(todatetime_strftime, axis=1, col='ライブ_from')
        df_timetable["ライブ_to"] = df_timetable.apply(todatetime_strftime, axis=1, col='ライブ_to')
        # df_timetable['ライブ_from'] = pd.to_datetime(df_timetable['ライブ_from'], format='%H:%M')
        # df_timetable['ライブ_to'] = pd.to_datetime(df_timetable['ライブ_to'], format='%H:%M')
        df_timetable.sort_values(by="ライブ_from",inplace=True)
        # df_timetable["ライブ_長さ(分)"] = ((df_timetable['ライブ_to'] - df_timetable['ライブ_from']).dt.total_seconds() / 60).astype(int)
        # df_timetable['ライブ_from'] = df_timetable['ライブ_from'].dt.strftime('%H:%M')
        # df_timetable['ライブ_to'] = df_timetable['ライブ_to'].dt.strftime('%H:%M')
    except ValueError:
        df_timetable["ライブ_長さ(分)"] = ""

    if tokutenkai:
        try:
            df_timetable["特典会_長さ(分)"] = df_timetable.apply(calculate_duration, axis=1, event_type='特典会')
            df_timetable["特典会_from"] = df_timetable.apply(todatetime_strftime, axis=1, col='特典会_from')
            df_timetable["特典会_to"] = df_timetable.apply(todatetime_strftime, axis=1, col='特典会_to')
            # df_timetable['特典会_from'] = pd.to_datetime(df_timetable['特典会_from'], format='%H:%M')
            # df_timetable['特典会_to'] = pd.to_datetime(df_timetable['特典会_to'], format='%H:%M')
            # df_timetable["特典会_長さ(分)"] = ((df_timetable['特典会_to'] - df_timetable['特典会_from']).dt.total_seconds() / 60).astype(int)
            # df_timetable['特典会_from'] = df_timetable['特典会_from'].dt.strftime('%H:%M')
            # df_timetable['特典会_to'] = df_timetable['特典会_to'].dt.strftime('%H:%M')
        except ValueError:
            df_timetable["特典会_長さ(分)"] = ""
        df_timetable = df_timetable[['グループ名', 'グループ名_採用', 'ライブ_from', 'ライブ_to', 'ライブ_長さ(分)', '特典会_from', '特典会_to', '特典会_長さ(分)', 'ブース', '出番ID', 'グループID', 'ステージID', '特典会_出番ID', '特典会_ステージID', '備考']]
        for col in ['特典会_出番ID', '特典会_ステージID']:
            if df_timetable[col].isna().all():
                df_timetable = df_timetable.drop(columns=[col])
    else:
        df_timetable = df_timetable[['グループ名', 'グループ名_採用', 'ライブ_from', 'ライブ_to', 'ライブ_長さ(分)', '出番ID', 'グループID', 'ステージID', '備考']]
    for col in ['グループID', '出番ID', 'ステージID']:
        if df_timetable[col].isna().all():
            df_timetable = df_timetable.drop(columns=[col])
    return df_timetable

def df_to_json(df_timetable):
    #特典会が2つ以上紐づいているものを1つに集約する処理を行っていない
    dict_timetable = df_timetable.to_dict(orient='records')
    json_timetable = []
    for item in dict_timetable:
        json_item = {}
        for col, v in item.items():
            if col in ['グループ名', 'グループ名_採用','備考']:
                json_item[col]=v
            elif col in [ 'グループID', '出番ID', 'ステージID']:
                try:
                    json_item[col]=int(v)
                except ValueError:
                    continue
            if col == "ライブ_from":
                json_item["ライブステージ"] = {"from":v}
            elif col == "ライブ_to":
                if isinstance(v, str):
                    try:
                        datetime.strptime(v, "%H:%M")
                        json_item["ライブステージ"]["to"]=v
                    except ValueError:
                        json_item["ライブステージ"]["to"] = add_minutes_to_time(json_item["ライブステージ"]["from"], item['ライブ_長さ(分)'])
                else:
                    json_item["ライブステージ"]["to"] = add_minutes_to_time(json_item["ライブステージ"]["from"], item['ライブ_長さ(分)'])
            if col == "特典会_from":
                json_item["特典会"] = [{"from":v}]
            elif col == "特典会_to":
                if isinstance(v, str):
                    try:
                        datetime.strptime(v, "%H:%M")
                        json_item["特典会"][0]["to"]=v
                    except ValueError:
                        json_item["特典会"][0]["to"] = add_minutes_to_time(json_item["特典会"][0]["from"], item['特典会_長さ(分)'])
                else:
                    json_item["特典会"][0]["to"] = add_minutes_to_time(json_item["特典会"][0]["from"], item['特典会_長さ(分)'])
            elif col == "ブース":
                json_item["特典会"][0]["ブース"]=v
            elif col in ["特典会_出番ID", "特典会_ステージID"]:
                json_item["特典会"][0][col[4:]]=v
        json_timetable.append(json_item)
    return json_timetable

def devide_df_live_tokutenkai(df_timetable):
    if 'グループID' in df_timetable.columns:
        df_live = df_timetable[['グループ名', 'グループ名_採用', 'グループID', '出番ID', 'ステージID', 'ライブ_from', 'ライブ_to', 'ライブ_長さ(分)', '備考']]
        df_tokutenkai = df_timetable[['グループ名', 'グループ名_採用', 'グループID', '特典会_出番ID', '特典会_ステージID', '特典会_from', '特典会_to', '特典会_長さ(分)', '備考', 'ブース']].rename(columns={'特典会_from':'ライブ_from', '特典会_to':'ライブ_to', '特典会_長さ(分)':'ライブ_長さ(分)','ブース':'ステージ名','特典会_出番ID':'出番ID', '特典会_ステージID':'ステージID'})
    else:
        df_live = df_timetable[['グループ名', 'グループ名_採用', 'ライブ_from', 'ライブ_to', 'ライブ_長さ(分)', '備考']]
        df_tokutenkai = df_timetable[['グループ名', 'グループ名_採用', '特典会_from', '特典会_to', '特典会_長さ(分)', '備考', 'ブース']].rename(columns={'特典会_from':'ライブ_from', '特典会_to':'ライブ_to', '特典会_長さ(分)':'ライブ_長さ(分)','ブース':'ステージ名'})
    return df_live, df_tokutenkai

def id_apply_to_json(json_data, turn_id_data, stage_name, with_tokutenkai):
    turn_id_data = turn_id_data.reset_index()
    stage_turn_id_data = turn_id_data[turn_id_data["ステージ名"]==stage_name]
    for i, turn_data in enumerate(json_data["タイムテーブル"]):
        try:
            tgt_turn_id_data = stage_turn_id_data[((stage_turn_id_data["グループ名"]==turn_data["グループ名_採用"])
            &(stage_turn_id_data["ライブ_from"]==turn_data["ライブステージ"]["from"]))].iloc[0]
        except KeyError:
            tgt_turn_id_data = stage_turn_id_data[((stage_turn_id_data["グループ名_raw"]==turn_data["グループ名"])
            &(stage_turn_id_data["ライブ_from"]==turn_data["ライブステージ"]["from"]))].iloc[0]
        json_data["タイムテーブル"][i]["出番ID"] = int(tgt_turn_id_data["出番ID"])
        json_data["タイムテーブル"][i]["グループID"] = int(tgt_turn_id_data["グループID"])
        json_data["タイムテーブル"][i]["ステージID"] = int(tgt_turn_id_data["ステージID"])
        if with_tokutenkai:#特典会併記形式の場合
            for j, turn_data_tokutenkai in enumerate(turn_data["特典会"]):
                try:
                    tgt_turn_id_data = turn_id_data[((turn_id_data["ステージ名"]==turn_data_tokutenkai["ブース"])
                    &(turn_id_data["グループ名"]==turn_data["グループ名_採用"])
                    &(turn_id_data["ライブ_from"]==turn_data_tokutenkai["from"]))].iloc[0]
                except KeyError:
                    tgt_turn_id_data = turn_id_data[((turn_id_data["ステージ名"]==turn_data_tokutenkai["ブース"])
                    &(turn_id_data["グループ名_raw"]==turn_data["グループ名"])
                    &(turn_id_data["ライブ_from"]==turn_data_tokutenkai["from"]))].iloc[0]
                json_data["タイムテーブル"][i]["特典会"][j]["出番ID"] = int(tgt_turn_id_data["出番ID"])
                json_data["タイムテーブル"][i]["特典会"][j]["ステージID"] = int(tgt_turn_id_data["ステージID"])
    return json_data