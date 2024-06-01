import pandas as pd
def json_to_df(json_data):
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
        
        # 特典会の情報を処理
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
                'ライブステージ_from': live_stage_from,
                'ライブステージ_to': live_stage_to,
                '特典会_from': meeting_from,
                '特典会_to': meeting_to,
                'ブース': booth
            })

    df_timetable = pd.DataFrame(df_timetable,columns=['グループ名', 'ライブステージ_from', 'ライブステージ_to', '特典会_from', '特典会_to', 'ブース'])

    return df_timetable