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

# ---------------------------------------------------------------------------
# タイムテーブル DataFrame の正規カラム構成
#   json_to_df の出力 (ID列ドロップ前) と空タイテ用 empty_timetable_df を
#   同一定義から構築し、ドリフトを防ぐ。
# ---------------------------------------------------------------------------

# 特典会併記 (tokutenkai=True) 時のフルカラム (ID列ドロップ前)
_COLS_TOKUTENKAI_FULL = [
    'グループ名', 'グループ名_採用', 'ライブ_from', 'ライブ_to', 'ライブ_長さ(分)',
    '特典会_from', '特典会_to', '特典会_長さ(分)', 'ブース',
    '出番ID', 'グループID', '特典会_出番ID', '特典会_ステージID',
    '備考', 'コラボグループID', 'コラボタイトル',
]
# ライブのみ (tokutenkai=False) 時のフルカラム (ID列ドロップ前)
_COLS_LIVE_FULL = [
    'グループ名', 'グループ名_採用', 'ライブ_from', 'ライブ_to', 'ライブ_長さ(分)',
    '出番ID', 'グループID', '備考', 'コラボグループID', 'コラボタイトル',
]
# ID未採番時に落とすカラム
_ID_COLS_DROP_TOKUTENKAI = ['出番ID', 'グループID', '特典会_出番ID', '特典会_ステージID']
_ID_COLS_DROP_LIVE = ['出番ID', 'グループID']

# Int64 (nullable整数) として扱うカラム。それ以外は文字列(object)。
_INT_COLS = {
    'ライブ_長さ(分)', '特典会_長さ(分)',
    '出番ID', 'グループID', '特典会_出番ID', '特典会_ステージID', 'コラボグループID',
}


def _empty_col_dtype(col: str):
    return 'Int64' if col in _INT_COLS else 'object'


def empty_timetable_df(tokutenkai: bool, id_assigned: bool) -> pd.DataFrame:
    """空タイムテーブル用の 0 行 DataFrame を返す。

    カラム構成は json_to_df の通常パス (ID列ドロップ後) と一致させる。

    Args:
        tokutenkai: 特典会併記カラムを含めるか (kind=="live_tokutenkai_heiki")。
        id_assigned: ID採番済か。False のとき出番ID/グループID/特典会_出番ID/
            特典会_ステージID を落とす (通常パスの isna().all() ドロップ相当)。
    """
    cols = list(_COLS_TOKUTENKAI_FULL if tokutenkai else _COLS_LIVE_FULL)
    if not id_assigned:
        drop = _ID_COLS_DROP_TOKUTENKAI if tokutenkai else _ID_COLS_DROP_LIVE
        cols = [c for c in cols if c not in drop]
    return pd.DataFrame(
        {c: pd.Series(dtype=_empty_col_dtype(c)) for c in cols},
        columns=cols,
    )

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
        

def _resolve_id_columns(df, cols, id_assigned):
    """ID系カラムの表示/非表示を解決する。

    id_assigned is None : 従来どおりデータ駆動 (全NaNなら落とす)。
    id_assigned is True : 残す (採番済の種別)。
    id_assigned is False: 落とす (未採番の種別)。
    """
    for col in cols:
        if col not in df.columns:
            continue
        drop = (df[col].isna().all() if id_assigned is None else not id_assigned)
        if drop:
            df = df.drop(columns=[col])
    return df


def json_to_df(json_data, tokutenkai=True, id_assigned=None):
    """タイムテーブル JSON を DataFrame 化する。

    id_assigned: ID系カラム (出番ID/グループID/特典会_出番ID/特典会_ステージID) の
        表示制御。None=データ駆動 (ビルド系の既存呼び出し), True/False=種別単位の
        採番状態で明示制御 (④画面)。
    """
    if "タイムテーブル" not in json_data.keys() or len(json_data["タイムテーブル"])==0:
        return pd.DataFrame()
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
        # ステージID はトップレベル (json_data["ステージID"]) に保持されるため
        # 出番粒度からは取得しない。build_event_output 側で渡される。
        try:
            remarks = item['備考']
        except KeyError:
            remarks = ""
        # コラボ情報 (Phase 1)
        try:
            collab_group_id = item['コラボグループID']
            if collab_group_id is not None and collab_group_id != "":
                collab_group_id = int(collab_group_id)
            else:
                collab_group_id = None
        except (KeyError, ValueError, TypeError):
            collab_group_id = None
        try:
            collab_title = item.get('コラボタイトル', "") or ""
        except AttributeError:
            collab_title = ""

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
                            ,'ライブ_from': live_stage_from
                            ,'ライブ_to': live_stage_to
                            ,'特典会_出番ID': meeting_turn_id
                            ,'特典会_ステージID': meeting_stage_id
                            ,'特典会_from': meeting_from
                            ,'特典会_to': meeting_to
                            ,'ブース': booth
                            ,'備考':remarks
                            ,'コラボグループID': collab_group_id
                            ,'コラボタイトル': collab_title
                        })
                    else:
                        df_timetable.append({
                            'グループ名': item['グループ名']
                            ,'グループ名_採用': group_name_corrected
                            ,'グループID': artist_id
                            ,'出番ID': None
                            ,'ライブ_from': ""
                            ,'ライブ_to': ""
                            ,'特典会_出番ID': meeting_turn_id
                            ,'特典会_ステージID': meeting_stage_id
                            ,'特典会_from': meeting_from
                            ,'特典会_to': meeting_to
                            ,'ブース': booth
                            ,'備考':remarks
                            ,'コラボグループID': collab_group_id
                            ,'コラボタイトル': collab_title
                        })
                if len(item['特典会'])==0:
                    df_timetable.append({
                        'グループ名': item['グループ名']
                        ,'グループ名_採用': group_name_corrected
                        ,'グループID': artist_id
                        ,'出番ID': turn_id
                        ,'ライブ_from': live_stage_from
                        ,'ライブ_to': live_stage_to
                        ,'特典会_出番ID': None
                        ,'特典会_ステージID': None
                        ,'特典会_from': ""
                        ,'特典会_to': ""
                        ,'ブース': ""
                        ,'備考':remarks
                        ,'コラボグループID': collab_group_id
                        ,'コラボタイトル': collab_title
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
                    ,'ライブ_from': live_stage_from
                    ,'ライブ_to': live_stage_to
                    ,'特典会_from': meeting_from
                    ,'特典会_to': meeting_to
                    ,'特典会_出番ID': meeting_turn_id
                    ,'特典会_ステージID': meeting_stage_id
                    ,'ブース': booth
                    ,'備考':remarks
                    ,'コラボグループID': collab_group_id
                    ,'コラボタイトル': collab_title
                })

        else:
            # DataFrameに行を追加
            df_timetable.append({
                'グループ名': item['グループ名']
                ,'グループ名_採用': group_name_corrected
                ,'グループID': artist_id
                ,'出番ID': turn_id
                ,'ライブ_from': live_stage_from
                ,'ライブ_to': live_stage_to
                ,'備考':remarks
                ,'コラボグループID': collab_group_id
                ,'コラボタイトル': collab_title
            })

    df_timetable = pd.DataFrame(df_timetable,columns=['グループ名', 'グループ名_採用', 'グループID', '出番ID','ライブ_from', 'ライブ_to', '特典会_from', '特典会_to', 'ブース', '特典会_出番ID', '特典会_ステージID', '備考', 'コラボグループID', 'コラボタイトル'])
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
        df_timetable["ライブ_長さ(分)"] = None
        dtype_map = {
            'グループ名': str,
            'グループ名_採用': str,
            'ライブ_from': str,
            'ライブ_to': str,
            'ライブ_長さ(分)': 'Int64',
            'ブース': str,
            '備考': str,
        }
        df_timetable = df_timetable.astype(dtype_map)

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
            df_timetable["特典会_長さ(分)"] = None
            dtype_map = {
                '特典会_from': str,
                '特典会_to': str,
                '特典会_長さ(分)': 'Int64'
            }
            df_timetable = df_timetable.astype(dtype_map)
        df_timetable = df_timetable[_COLS_TOKUTENKAI_FULL]
        df_timetable = _resolve_id_columns(
            df_timetable, ['特典会_出番ID', '特典会_ステージID'], id_assigned,
        )
    else:
        df_timetable = df_timetable[_COLS_LIVE_FULL]
    df_timetable = _resolve_id_columns(df_timetable, ['グループID', '出番ID'], id_assigned)
    # ④画面 (id_assigned 明示) で残したID列は、空df (empty_timetable_df) と揃えて
    # Int64 で表示する (全NaNの採番済種別でも数値カラムとして編集できるように)。
    if id_assigned is not None:
        for col in ('出番ID', 'グループID', '特典会_出番ID', '特典会_ステージID'):
            if col in df_timetable.columns:
                df_timetable[col] = df_timetable[col].astype('Int64')
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
            elif col == 'コラボグループID':
                # 欠損 / NaN は null として出力 (Phase 1 後方互換)
                if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
                    json_item['コラボグループID'] = None
                else:
                    try:
                        json_item['コラボグループID'] = int(v)
                    except (ValueError, TypeError):
                        json_item['コラボグループID'] = None
                continue
            elif col == 'コラボタイトル':
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    json_item['コラボタイトル'] = ""
                else:
                    json_item['コラボタイトル'] = str(v)
                continue
            elif col in ['グループID', '出番ID']:
                try:
                    json_item[col]=int(v)
                except ValueError:
                    continue
            elif col == 'ステージID':
                # ステージID は JSON のトップレベルに保持する設計に変更されたため、
                # 出番粒度では書き出さない（後方互換のため受け取っても無視）。
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
                        try:
                            json_item["特典会"][0]["to"] = add_minutes_to_time(json_item["特典会"][0]["from"], item['特典会_長さ(分)'])
                        except ValueError:
                            json_item["特典会"][0]["to"] = ""
                else:
                    json_item["特典会"][0]["to"] = add_minutes_to_time(json_item["特典会"][0]["from"], item['特典会_長さ(分)'])
            elif col == "ブース":
                json_item["特典会"][0]["ブース"]=v
            elif col in ["特典会_出番ID", "特典会_ステージID"]:
                json_item["特典会"][0][col[4:]]=v
        json_timetable.append(json_item)
    return json_timetable

def detect_cross_stage_turn_id_collision(
    pj_path: str,
    event_name: str,
    target_img_type: str,
    target_stage_no: int,
    edited_df: pd.DataFrame,
    project_info_json: dict,
) -> list[dict]:
    """編集中のDFが持つ 出番ID が、**他ステージ**の stage_*.json と衝突するかを検出。

    コラボはステージを跨がない設計のため、同じ 出番ID が異なる stage_*.json に
    存在するのは整合性違反 (=データ破損リスク)。④保存ボタン押下時にこの関数で
    事前検証し、衝突があれば保存を中止する。

    検査対象の 出番ID:
        - 編集中DFの `出番ID` 列 (親エントリ)
        - 編集中DFの `特典会_出番ID` 列 (heiki 形式の booth 子ID)

    比較対象の 出番ID:
        - 同 event の全 img_type 配下の `stage_*.json` (自分自身を除く)
        - 各 JSON の タイムテーブル[].出番ID および タイムテーブル[].特典会[].出番ID

    Returns:
        衝突情報のリスト。空ならOK。各要素は
            {"出番ID": int, "他種別": str, "他ステージNo": int, "場所": "親" | "特典会"}
    """
    import json as _json
    # backend_functions の循環 import を避けるため遅延 import
    from backend_functions import project_repository as _repo

    edited_ids: set[int] = set()
    for col in ("出番ID", "特典会_出番ID"):
        if col not in edited_df.columns:
            continue
        for v in edited_df[col].dropna():
            try:
                edited_ids.add(int(v))
            except (ValueError, TypeError):
                continue
    if not edited_ids:
        return []

    event_no = _repo.get_event_no_by_event_name(project_info_json, event_name)
    if event_no is None:
        return []

    collisions: list[dict] = []
    for img_type in _repo.get_event_type_list(project_info_json, event_no):
        entry = _repo.get_image_entry_by_dir_name(project_info_json, event_no, img_type)
        if entry is None:
            continue
        for stage_no in range(entry.get("stage_num", 0)):
            # 自分自身はスキップ (同じファイル内の重複は許容: コラボ)
            if img_type == target_img_type and stage_no == target_stage_no:
                continue
            json_path = os.path.join(pj_path, event_name, img_type, f"stage_{stage_no}.json")
            if not os.path.exists(json_path):
                continue
            try:
                with open(json_path, encoding="utf-8") as f:
                    data = _json.load(f)
            except (OSError, ValueError):
                continue
            for turn in data.get("タイムテーブル", []) or []:
                tid = turn.get("出番ID")
                if tid is not None:
                    try:
                        tid_int = int(tid)
                    except (ValueError, TypeError):
                        tid_int = None
                    if tid_int is not None and tid_int in edited_ids:
                        collisions.append({
                            "出番ID": tid_int,
                            "他種別": img_type,
                            "他ステージNo": stage_no,
                            "場所": "親",
                        })
                for tk in turn.get("特典会", []) or []:
                    tk_id = tk.get("出番ID")
                    if tk_id is None:
                        continue
                    try:
                        tk_id_int = int(tk_id)
                    except (ValueError, TypeError):
                        continue
                    if tk_id_int in edited_ids:
                        collisions.append({
                            "出番ID": tk_id_int,
                            "他種別": img_type,
                            "他ステージNo": stage_no,
                            "場所": "特典会",
                        })
    return collisions


def _format_collab_artist_names(
    rows: list[dict],
    join_separator: str = "・",
    max_names: int = 4,
) -> str:
    """コラボ各行から表示用のアーティスト名連結文字列を作る。

    - 重複排除後のグループ数 <= max_names: 全件連結 ("A・B・C")
    - 超える場合: 先頭 max_names 件 + "ほか計{N}グループ"
    """
    names: list[str] = []
    seen: set[str] = set()
    for r in rows:
        n = r.get("グループ名_採用") or r.get("グループ名") or ""
        if isinstance(n, str) and n != "" and n not in seen:
            seen.add(n)
            names.append(n)
    total = len(names)
    if total <= max_names:
        return join_separator.join(names)
    return join_separator.join(names[:max_names]) + f"ほか計{total}グループ"


def consolidate_collab_entries(
    timetable: list[dict],
    join_separator: str = "・",
    max_names: int = 4,
) -> list[dict]:
    """`タイムテーブル[]` の dict 群をコラボキーで統合して新しいリストを返す。

    タイテ画像生成のプレプロセス用 (Phase 1 画像対応)。
    元の dict はミューテートせず、統合エントリは新規 dict として返す。

    キー優先順位 (output_builder の 出番ID 採番ロジックと一致):
        1. 出番ID が同じ複数行 → 1 つのコラボエントリ
        2. 出番ID NULL かつ コラボグループID が同じ複数行 → 1 つのコラボエントリ
        3. それ以外 → 単独 (元エントリそのまま)

    コラボエントリの表示名:
        - `コラボタイトル` 非空: `"タイトル(A・B・C)"` 形式
        - 空: `"A・B・C"` 形式
        いずれもアーティスト名連結は `max_names` 件まで、超過時は
        `"A・B・C・Dほか計{N}グループ"` で省略する。
    その他フィールド (時刻 / 備考等) は先頭行から継承する。
    """
    def _key_for(row: dict, idx: int):
        tid = row.get("出番ID")
        if tid is not None and not (isinstance(tid, float) and pd.isna(tid)):
            try:
                return ("turn", int(tid))
            except (ValueError, TypeError):
                pass
        cgid = row.get("コラボグループID")
        if cgid is not None and not (isinstance(cgid, float) and pd.isna(cgid)):
            try:
                return ("cgid", int(cgid))
            except (ValueError, TypeError):
                pass
        # 単独: ユニークキーで自分自身しかグループに入らないように
        return ("solo", idx)

    if not timetable:
        return []

    grouped: dict = {}
    order: list = []
    for i, row in enumerate(timetable):
        key = _key_for(row, i)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(row)

    result: list[dict] = []
    for key in order:
        rows = grouped[key]
        if len(rows) == 1:
            result.append(rows[0])
            continue
        merged = dict(rows[0])  # 浅いコピー
        # 連結 (省略対応) 文字列を構築
        joined = _format_collab_artist_names(
            rows, join_separator=join_separator, max_names=max_names,
        )
        # コラボタイトル (最初の非空) を取得
        title = ""
        for r in rows:
            t = r.get("コラボタイトル")
            if isinstance(t, str) and t != "":
                title = t
                break
        if title:
            display_name = f"{title}({joined})" if joined else title
        else:
            display_name = joined
        merged["グループ名_採用"] = display_name
        merged["グループ名"] = display_name
        result.append(merged)
    return result


def autodetect_collab_groups(df_timetable: pd.DataFrame, clear_turn_id: bool = False) -> pd.DataFrame:
    """ステージ内で同じ `ライブ_from` を持つ行群に同一の `コラボグループID` を採番する。

    Phase 1-2: 「コラボ出番を自動検出」ボタンのロジック。
    - 対象は `コラボグループID` が NULL (NaN / None / 空) の行のみ
    - 既にIDが入っている行 (手動設定 / 前回検出済) は触らない (冪等)
    - 採番は **同一 stage_*.json (=DataFrame全体)** スコープで、既存IDの最大値+1から
    - 同じ from を持つ行が2件以上ある場合のみグループ化
    - `clear_turn_id=True` の場合、今回の実行で新たに `コラボグループID` を採番した行の `出番ID` を None にクリアする

    Returns:
        変更を加えた DataFrame (引数を変更しない新オブジェクト)
    """
    if df_timetable is None or len(df_timetable) == 0:
        return df_timetable
    df = df_timetable.copy()
    if 'コラボグループID' not in df.columns:
        df['コラボグループID'] = None
    if 'コラボタイトル' not in df.columns:
        df['コラボタイトル'] = ""

    existing = pd.to_numeric(df['コラボグループID'], errors='coerce')
    next_id = int(existing.max()) + 1 if existing.notna().any() else 1

    null_mask = existing.isna()
    target_rows = df[null_mask]
    if len(target_rows) == 0:
        return df

    for from_val, group in target_rows.groupby('ライブ_from'):
        if not from_val or pd.isna(from_val):
            continue
        if len(group) < 2:
            continue
        df.loc[group.index, 'コラボグループID'] = next_id
        if clear_turn_id and '出番ID' in df.columns:
            df.loc[group.index, '出番ID'] = None
        next_id += 1
    return df


def devide_df_live_tokutenkai(df_timetable):
    # ライブ側の ステージID は JSON トップレベル管理のため出番粒度に持たない。
    # 特典会側の 特典会_ステージID (ブース別子ID) は出番粒度を維持する。
    # コラボ列 (コラボグループID / コラボタイトル) はライブ側のみに付与する。
    has_collab = (
        'コラボグループID' in df_timetable.columns
        and 'コラボタイトル' in df_timetable.columns
    )
    if 'グループID' in df_timetable.columns:
        live_cols = ['グループ名', 'グループ名_採用', 'グループID', '出番ID', 'ライブ_from', 'ライブ_to', 'ライブ_長さ(分)', '備考']
        if has_collab:
            live_cols += ['コラボグループID', 'コラボタイトル']
        df_live = df_timetable[live_cols]
        df_tokutenkai = df_timetable[['グループ名', 'グループ名_採用', 'グループID', '特典会_出番ID', '特典会_ステージID', '特典会_from', '特典会_to', '特典会_長さ(分)', '備考', 'ブース']].rename(columns={'特典会_from':'ライブ_from', '特典会_to':'ライブ_to', '特典会_長さ(分)':'ライブ_長さ(分)','ブース':'ステージ名','特典会_出番ID':'出番ID', '特典会_ステージID':'ステージID'})
    else:
        live_cols = ['グループ名', 'グループ名_採用', 'ライブ_from', 'ライブ_to', 'ライブ_長さ(分)', '備考']
        if has_collab:
            live_cols += ['コラボグループID', 'コラボタイトル']
        df_live = df_timetable[live_cols]
        df_tokutenkai = df_timetable[['グループ名', 'グループ名_採用', '特典会_from', '特典会_to', '特典会_長さ(分)', '備考', 'ブース']].rename(columns={'特典会_from':'ライブ_from', '特典会_to':'ライブ_to', '特典会_長さ(分)':'ライブ_長さ(分)','ブース':'ステージ名'})
    return df_live, df_tokutenkai

def id_apply_to_json(json_data, turn_id_data, stage_name, with_tokutenkai):
    """出番ID / グループID / ステージID を JSON に書き戻す。

    ステージID はトップレベル (`json_data["ステージID"]`) に書き込み、
    出番粒度の `タイムテーブル[i]["ステージID"]` は書き込まない。
    特典会併記形式では子ブース別IDのみ `特典会[j]["ステージID"]` に維持する。
    """
    turn_id_data = turn_id_data.reset_index()
    stage_turn_id_data = turn_id_data[turn_id_data["ステージ名"]==stage_name]

    def _first_match(df, base_mask, turn_data):
        """グループ名(採用)優先・グループ名_raw劣後で先頭一致行を返す。

        from/to 未入力などで 長さ(分) が空となり turn_id_data から
        除外されたエントリは一致行0件となるため、その場合は None を返して
        呼び出し側でスキップさせる (IndexError で ⑥ 全体が落ちるのを防ぐ)。
        """
        if "グループ名" in df.columns and "グループ名_採用" in turn_data:
            sel = df[base_mask & (df["グループ名"]==turn_data["グループ名_採用"])]
            if len(sel) > 0:
                return sel.iloc[0]
        if "グループ名_raw" in df.columns and "グループ名" in turn_data:
            sel = df[base_mask & (df["グループ名_raw"]==turn_data["グループ名"])]
            if len(sel) > 0:
                return sel.iloc[0]
        return None

    # ステージID をトップレベルに書き込む(ライブ親ステージのID)
    if len(stage_turn_id_data) > 0 and "ステージID" in stage_turn_id_data.columns:
        try:
            json_data["ステージID"] = int(stage_turn_id_data["ステージID"].iloc[0])
        except (ValueError, TypeError):
            pass
    for i, turn_data in enumerate(json_data["タイムテーブル"]):
        live_from = turn_data.get("ライブステージ", {}).get("from")
        tgt_turn_id_data = _first_match(
            stage_turn_id_data,
            stage_turn_id_data["ライブ_from"]==live_from,
            turn_data,
        )
        # 一致行なし (長さ(分)空などで turn_id_data 除外) はID書き戻しをスキップ
        if tgt_turn_id_data is not None:
            json_data["タイムテーブル"][i]["出番ID"] = int(tgt_turn_id_data["出番ID"])
            json_data["タイムテーブル"][i]["グループID"] = int(tgt_turn_id_data["グループID"])
        # 出番粒度の ステージID は廃止(トップレベルに集約)。
        # 旧フォーマット互換のため、古いキーが残っていれば削除する。
        if "ステージID" in json_data["タイムテーブル"][i]:
            del json_data["タイムテーブル"][i]["ステージID"]
        if with_tokutenkai:#特典会併記形式の場合
            parent_turn_id = json_data["タイムテーブル"][i].get("出番ID")
            for j, turn_data_tokutenkai in enumerate(turn_data["特典会"]):
                tgt_turn_id_data = _first_match(
                    turn_id_data,
                    (turn_id_data["ステージ名"]==turn_data_tokutenkai.get("ブース"))
                    &(turn_id_data["ライブ_from"]==turn_data_tokutenkai.get("from")),
                    turn_data,
                )
                # 一致行なし (長さ(分)空などで turn_id_data 除外) はスキップ
                if tgt_turn_id_data is None:
                    continue
                json_data["タイムテーブル"][i]["特典会"][j]["出番ID"] = int(tgt_turn_id_data["出番ID"])
                json_data["タイムテーブル"][i]["特典会"][j]["ステージID"] = int(tgt_turn_id_data["ステージID"])
                # 対応出番ID: 親エントリの出番IDをコピーして、編集モードでの書き戻しキーにする
                if parent_turn_id is not None:
                    json_data["タイムテーブル"][i]["特典会"][j]["対応出番ID"] = int(parent_turn_id)
    return json_data