"""
出力データの構築・エクスポート。

ステージマスタ・アーティストマスタ・出番データの組み立て、
ID確定、Excelエクスポート、グループ名マスタ更新を行う。
Streamlitに依存しない。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from openpyxl import Workbook

from backend_functions import timetabledata, idolname, s3access
from backend_functions import project_repository as repo


# ---------------------------------------------------------------------------
# カラープリセット (Stella stageList.colorName)
# ---------------------------------------------------------------------------

_PRESET_COLOR_NAMES: list[str] = [
    "stage-red", "stage-pink", "stage-purple", "stage-deep-purple",
    "stage-indigo", "stage-blue", "stage-blue2", "stage-lightBlue",
    "stage-cyan", "stage-teal", "stage-green", "stage-light-green",
    "stage-light-green2", "stage-lime", "stage-yellow", "stage-amber",
    "stage-orange", "stage-deepOrange", "stage-brown", "stage-brown2",
    "stage-blueGrey", "stage-grey", "stage-black", "stage-white",
    "stage-redGrey", "stage-greenGrey", "stage-yellowGrey",
]


def _default_color_name(stage_order: int, is_tokutenkai: bool, kind_hint: str | None = None) -> str:
    """ステージ種別に応じてデフォルト カラー名 を返す (Phase 2-2)。

    - 縁日ステージ (kind_hint="ennichi"): stage-blueGrey
    - 特典会ステージ: stage-grey
    - ライブステージ: プリセット27色を `stage_order` で循環
    """
    if kind_hint == "ennichi":
        return "stage-blueGrey"
    if is_tokutenkai:
        return "stage-grey"
    return _PRESET_COLOR_NAMES[int(stage_order) % len(_PRESET_COLOR_NAMES)]


# ---------------------------------------------------------------------------
# ステージマスタ・出力データ組み立て
# ---------------------------------------------------------------------------

def _stage_master_get(stage_master: dict, stage_id: int) -> dict | None:
    """stage_master の key 型ゆれ (int / str) を吸収して引き当てる。"""
    if stage_id in stage_master:
        return stage_master[stage_id]
    if str(stage_id) in stage_master:
        return stage_master[str(stage_id)]
    return None


def _stage_master_keys_as_int(stage_master: dict) -> list[int]:
    """stage_master の key を int に正規化して返す。"""
    return [int(k) for k in stage_master.keys()]


def find_or_create_stage_id(
    stage_master: dict,
    stage_name: str,
    is_tokutenkai: bool,
    next_id: int,
    existing_stage_id: int | None = None,
) -> tuple[int, int]:
    """既存マスタからステージIDを探し、なければnext_idで新規登録する。

    引き当て優先順位:
        1. existing_stage_id 指定があり、マスタにそのIDがあれば使用。
           ステージ名が異なれば**マスタ側を更新**(編集モードの反映)。
        2. existing_stage_id 指定がない/マスタ未登録のとき、ステージ名一致で検索。
        3. いずれにも当たらなければ next_id で新規登録。

    新規登録時は引数 stage_master が副作用で更新される(参照渡し)。
    新規登録時に既存ステージマスタの `表示順` 最大値 + 1 を表示順として採番する。
    `非活性化フラグ` は False で初期化する。

    Returns:
        (assigned_id, updated_next_id):
            既存ステージ → (既存ID, next_id)
            新規ステージ → (next_id, next_id + 1)
    """
    if existing_stage_id is not None:
        existing_entry = _stage_master_get(stage_master, existing_stage_id)
        if existing_entry is not None:
            if existing_entry.get("ステージ名") != stage_name:
                existing_entry["ステージ名"] = stage_name
            return int(existing_stage_id), next_id

    for k, v in stage_master.items():
        if v["ステージ名"] == stage_name:
            return int(k), next_id

    # 新規登録: 表示順 = 既存最大値 + 1
    existing_orders = [
        v.get("表示順")
        for v in stage_master.values()
        if v.get("表示順") is not None
    ]
    next_order = max(existing_orders) + 1 if existing_orders else 0
    stage_master[next_id] = {
        "ステージ名": stage_name,
        "特典会フラグ": is_tokutenkai,
        "表示順": next_order,
        "非活性化フラグ": False,
        # Phase 2: Stella stageList 拡張フィールド
        "ステージ名_短縮": stage_name,
        "カラー名": _default_color_name(next_order, is_tokutenkai),
    }
    return next_id, next_id + 1


def load_existing_masters(
    output_path: str,
) -> tuple[dict, int, pd.DataFrame, int]:
    """master_stage.csv / master_idolname.csv が存在すれば読み込み、なければ空の初期値を返す。

    「IDマスタ確定」前後の挙動差をここで吸収する。

    Returns:
        (stage_master, next_stage_id, idolname_master_df, next_artist_id)
    """
    stage_csv = os.path.join(output_path, "master_stage.csv")
    if os.path.exists(stage_csv):
        stage_master_df = pd.read_csv(stage_csv, index_col=0)
        # 後方互換: 表示順 / 非活性化フラグ カラムが無ければ補完
        if "表示順" not in stage_master_df.columns:
            stage_master_df["表示順"] = range(len(stage_master_df))
        if "非活性化フラグ" not in stage_master_df.columns:
            stage_master_df["非活性化フラグ"] = False
        stage_master_df["非活性化フラグ"] = (
            stage_master_df["非活性化フラグ"].fillna(False).astype(bool)
        )
        # Phase 2: ステージ名_短縮 / カラー名 の補完
        if "ステージ名_短縮" not in stage_master_df.columns:
            stage_master_df["ステージ名_短縮"] = stage_master_df["ステージ名"]
        stage_master_df["ステージ名_短縮"] = (
            stage_master_df["ステージ名_短縮"]
            .fillna(stage_master_df["ステージ名"]).astype(str)
        )
        if "カラー名" not in stage_master_df.columns:
            stage_master_df["カラー名"] = None
        # 欠損行を デフォルト割当ルール で埋める
        for sid in stage_master_df.index:
            cur = stage_master_df.at[sid, "カラー名"]
            if cur is None or (isinstance(cur, float) and pd.isna(cur)) or cur == "":
                order = int(stage_master_df.at[sid, "表示順"])
                is_tk = bool(stage_master_df.at[sid, "特典会フラグ"])
                stage_master_df.at[sid, "カラー名"] = _default_color_name(order, is_tk)
        stage_master = json.loads(stage_master_df.T.to_json())
        next_stage_id = int(max(stage_master_df.index)) + 1
    else:
        stage_master = {}
        next_stage_id = 0

    idolname_csv = os.path.join(output_path, "master_idolname.csv")
    if os.path.exists(idolname_csv):
        idolname_master_df = pd.read_csv(idolname_csv, index_col=0).rename(
            columns={"グループ名": "グループ名_採用"},
        )
        next_artist_id = int(max(idolname_master_df.index)) + 1
    else:
        idolname_master_df = pd.DataFrame(
            columns=["グループID", "グループ名_採用"],
        ).set_index("グループID")
        next_artist_id = 0

    return stage_master, next_stage_id, idolname_master_df, next_artist_id


_LIVE_OUTPUT_COLUMNS = [
    "ライブ_from", "ライブ_to", "ライブ_長さ(分)", "グループID", "ステージID",
    "グループ名_raw", "グループ名", "ステージ名", "備考",
    "コラボタイトル",
]
# Excel 出力時に省略する列 (UI 表示と Stella用算出のために output_df には含めるが、
# 既存 Excel 出力フォーマットを維持するため書き出さない)
_LIVE_EXCEL_DROP_COLUMNS = ["ライブ_to"]

_DURATION_COL_LIVE = "ライブステージ"
_DURATION_COL_TOKUTENKAI = "特典会ステージ"
_DURATION_INDEX_NAME = "長さ(分)"

_GROUP_COUNT_COL_NAME = "グループ名"
_GROUP_COUNT_COL_LIVE = "ライブ出演回数"
_GROUP_COUNT_COL_TOKUTENKAI = "特典会出演回数"
_GROUP_COUNT_COL_TOTAL = "合計"

_OVERLAP_COLUMNS = [
    "グループID", "グループ名",
    "出番ID_1", "ステージID_1", "ステージ名_1", "開始_1", "終了_1",
    "出番ID_2", "ステージID_2", "ステージ名_2", "開始_2", "終了_2",
    "重複(分)",
]
_APPEARANCE_COLUMNS = [
    "グループID", "グループ名", "ステージID", "ステージ名",
    "ライブ_from", "ライブ_to", "ライブ_長さ(分)", "備考",
]


def _parse_hhmm(value) -> "datetime | None":
    """'HH:MM' 文字列を datetime に変換。失敗時 None。"""
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return None
    try:
        return datetime.strptime(str(value), "%H:%M")
    except (ValueError, TypeError):
        return None


def build_duration_distribution(
    df_live: pd.DataFrame,
    df_stage: pd.DataFrame,
) -> pd.DataFrame:
    """出演枠の長さ(分)ごとの出現回数を、ライブ／特典会ステージ別にカウントする。

    Returns:
        index : 長さ(分) 昇順
        cols  : [ライブステージ, 特典会ステージ]  (int, 0埋め)
    """
    empty = pd.DataFrame(
        {_DURATION_COL_LIVE: pd.Series(dtype="int64"),
         _DURATION_COL_TOKUTENKAI: pd.Series(dtype="int64")},
    )
    empty.index.name = _DURATION_INDEX_NAME

    if df_live is None or len(df_live) == 0:
        return empty

    flag_map = df_stage["特典会フラグ"].to_dict()
    df = df_live[["ステージID", "ライブ_長さ(分)"]].copy()
    df = df[df["ライブ_長さ(分)"].notna() & (df["ライブ_長さ(分)"] != "")]
    if len(df) == 0:
        return empty
    df["特典会フラグ"] = df["ステージID"].map(flag_map).fillna(False).astype(bool)

    grouped = (
        df.groupby(["ライブ_長さ(分)", "特典会フラグ"])
        .size()
        .unstack("特典会フラグ", fill_value=0)
    )
    if False not in grouped.columns:
        grouped[False] = 0
    if True not in grouped.columns:
        grouped[True] = 0
    result = grouped.rename(
        columns={False: _DURATION_COL_LIVE, True: _DURATION_COL_TOKUTENKAI},
    )[[_DURATION_COL_LIVE, _DURATION_COL_TOKUTENKAI]]
    result = result.sort_index()
    result.index.name = _DURATION_INDEX_NAME
    return result.astype("int64")


def build_group_appearance_count(
    df_live: pd.DataFrame,
    df_stage: pd.DataFrame,
) -> pd.DataFrame:
    """グループごとの出演回数を、ライブ／特典会ステージ別にカウントする。

    Returns:
        index : グループID 昇順
        cols  : [グループ名, ライブ出演回数, 特典会出演回数, 合計]
    """
    empty = pd.DataFrame(
        {_GROUP_COUNT_COL_NAME: pd.Series(dtype="object"),
         _GROUP_COUNT_COL_LIVE: pd.Series(dtype="int64"),
         _GROUP_COUNT_COL_TOKUTENKAI: pd.Series(dtype="int64"),
         _GROUP_COUNT_COL_TOTAL: pd.Series(dtype="int64")},
    )
    empty.index.name = "グループID"

    if df_live is None or len(df_live) == 0:
        return empty

    flag_map = df_stage["特典会フラグ"].to_dict()
    df = df_live[["グループID", "グループ名", "ステージID"]].copy()
    df = df[df["グループID"].notna()]
    if len(df) == 0:
        return empty
    df["特典会フラグ"] = df["ステージID"].map(flag_map).fillna(False).astype(bool)

    grouped = (
        df.groupby(["グループID", "グループ名", "特典会フラグ"])
        .size()
        .unstack("特典会フラグ", fill_value=0)
    )
    if False not in grouped.columns:
        grouped[False] = 0
    if True not in grouped.columns:
        grouped[True] = 0
    grouped = grouped.rename(
        columns={False: _GROUP_COUNT_COL_LIVE, True: _GROUP_COUNT_COL_TOKUTENKAI},
    )
    grouped[_GROUP_COUNT_COL_TOTAL] = (
        grouped[_GROUP_COUNT_COL_LIVE] + grouped[_GROUP_COUNT_COL_TOKUTENKAI]
    )

    result = grouped.reset_index(level="グループ名")
    result = result[[
        _GROUP_COUNT_COL_NAME,
        _GROUP_COUNT_COL_LIVE,
        _GROUP_COUNT_COL_TOKUTENKAI,
        _GROUP_COUNT_COL_TOTAL,
    ]].sort_index()
    result.index = result.index.astype("int64")
    result.index.name = "グループID"
    for col in (_GROUP_COUNT_COL_LIVE, _GROUP_COUNT_COL_TOKUTENKAI, _GROUP_COUNT_COL_TOTAL):
        result[col] = result[col].astype("int64")
    return result


def build_overlap_alerts(
    df_live: pd.DataFrame,
    df_stage: pd.DataFrame,
) -> pd.DataFrame:
    """同一グループ内で出演時間が重なっているペアを検出する。

    ライブ／特典会ステージを跨いで検出する。

    Returns:
        cols : [グループID, グループ名,
                出番ID_1, ステージID_1, ステージ名_1, 開始_1, 終了_1,
                出番ID_2, ステージID_2, ステージ名_2, 開始_2, 終了_2,
                重複(分)]
        重複が無い場合は空 DataFrame を返す。
    """
    empty = pd.DataFrame(columns=_OVERLAP_COLUMNS)

    if df_live is None or len(df_live) == 0:
        return empty
    required = {"グループID", "グループ名", "ステージID", "ライブ_from", "ライブ_to"}
    if not required.issubset(df_live.columns):
        return empty

    df = df_live.copy()
    df["_from_dt"] = df["ライブ_from"].apply(_parse_hhmm)
    df["_to_dt"] = df["ライブ_to"].apply(_parse_hhmm)
    df = df[df["_from_dt"].notna() & df["_to_dt"].notna()]
    df = df[df["グループID"].notna()]
    if len(df) == 0:
        return empty

    rows = []
    for group_id, group_df in df.groupby("グループID"):
        appearances = group_df.sort_values("_from_dt").reset_index()
        n = len(appearances)
        if n < 2:
            continue
        group_name = appearances.iloc[0]["グループ名"]
        for i in range(n):
            for j in range(i + 1, n):
                a = appearances.iloc[i]
                b = appearances.iloc[j]
                overlap_start = max(a["_from_dt"], b["_from_dt"])
                overlap_end = min(a["_to_dt"], b["_to_dt"])
                if overlap_start < overlap_end:
                    overlap_min = int(
                        (overlap_end - overlap_start).total_seconds() // 60
                    )
                    rows.append({
                        "グループID": int(group_id),
                        "グループ名": group_name,
                        "出番ID_1": a["出番ID"] if "出番ID" in a.index else a["index"],
                        "ステージID_1": int(a["ステージID"]),
                        "ステージ名_1": a.get("ステージ名", ""),
                        "開始_1": a["ライブ_from"],
                        "終了_1": a["ライブ_to"],
                        "出番ID_2": b["出番ID"] if "出番ID" in b.index else b["index"],
                        "ステージID_2": int(b["ステージID"]),
                        "ステージ名_2": b.get("ステージ名", ""),
                        "開始_2": b["ライブ_from"],
                        "終了_2": b["ライブ_to"],
                        "重複(分)": overlap_min,
                    })

    if not rows:
        return empty
    return pd.DataFrame(rows, columns=_OVERLAP_COLUMNS).reset_index(drop=True)


def build_group_appearances(
    df_live: pd.DataFrame,
    df_stage: pd.DataFrame,
) -> pd.DataFrame:
    """全グループの出番一覧を返す。UI側でグループIDで絞り込んで使う。

    Returns:
        index : 出番ID
        cols  : [グループID, グループ名, ステージID, ステージ名,
                 ライブ_from, ライブ_to, ライブ_長さ(分), 備考]
    """
    empty = pd.DataFrame(columns=_APPEARANCE_COLUMNS)
    empty.index.name = "出番ID"

    if df_live is None or len(df_live) == 0:
        return empty
    required = {"グループID", "グループ名", "ステージID", "ステージ名",
                "ライブ_from", "ライブ_to", "ライブ_長さ(分)", "備考"}
    if not required.issubset(df_live.columns):
        return empty

    df = df_live[df_live["グループID"].notna()].copy()
    if len(df) == 0:
        return empty

    if "出番ID" in df.columns:
        df = df.set_index("出番ID")
    df = df[_APPEARANCE_COLUMNS]
    df.index.name = "出番ID"
    return df.sort_values(["グループID", "ライブ_from"])


def build_event_output(
    pj_path: str,
    event_name: str,
    event_no: int,
    project_info_json: dict,
) -> dict[str, pd.DataFrame] | None:
    """1イベント分の出力データ(stage / idolname / live)を組み立てて返す。

    データが1件も存在しない場合は None を返す。

    処理フロー:
        1. load_existing_masters() で既存マスタを読み込み(IDマスタ確定前後の差を吸収)
        2. event_type × stage を走査してステージマスタとライブデータを構築
        3. 特典会併記タイテの場合、特典会データをブース別にステージマスタへ統合
        4. アーティストマスタを構築
        5. 出番データを構築(既存出番IDの維持と新規ID採番)

    Returns:
        {"stage": df_stage, "idolname": df_idolname, "live": df_live} or None
    """
    output_path = os.path.join(pj_path, event_name)
    stage_master, stage_id, idolname_master_df, artist_id = load_existing_masters(
        output_path,
    )

    event_type_list = repo.get_event_type_list(project_info_json, event_no)
    tokutenkai_timetable = []
    event_timetable_all = []
    # 特典会併記形式のブース名 -> 既存子ステージID (引き当てヒント)
    booth_existing_id_map_all: dict[str, int] = {}

    for event_type in event_type_list:
        tgt_event_type_info = repo.get_image_entry_by_dir_name(
            project_info_json, event_no, event_type,
        )
        stage_name_list = repo.get_stage_name_list(project_info_json, event_no, event_type)
        kind = tgt_event_type_info["kind"]
        tokutenkai_flg = kind == "tokutenkai"
        is_heiki = kind == "live_tokutenkai_heiki"

        for stage_no in range(tgt_event_type_info["stage_num"]):
            json_path = os.path.join(output_path, event_type, f"stage_{stage_no}.json")
            if not os.path.exists(json_path):
                continue

            try:
                with open(json_path, encoding="utf-8") as f:
                    edit_tgt_json = json.load(f)

                # トップレベル ステージID (親=ライブステージID) を取得。
                # 未確定時は None。
                existing_top_stage_id = edit_tgt_json.get("ステージID")
                try:
                    existing_top_stage_id = (
                        int(existing_top_stage_id)
                        if existing_top_stage_id is not None else None
                    )
                except (ValueError, TypeError):
                    existing_top_stage_id = None

                # 特典会併記形式の場合、特典会[].ステージID (子=ブース別ID) を
                # ブース名 -> 子ID の辞書として収集
                booth_existing_id_map: dict[str, int] = {}
                if is_heiki:
                    for turn in edit_tgt_json.get("タイムテーブル", []):
                        for tk in turn.get("特典会", []) or []:
                            booth_name = tk.get("ブース")
                            tk_stage_id = tk.get("ステージID")
                            if booth_name and tk_stage_id is not None:
                                try:
                                    booth_existing_id_map[booth_name] = int(tk_stage_id)
                                except (ValueError, TypeError):
                                    pass

                if is_heiki:
                    df_edit_tgt = timetabledata.json_to_df(edit_tgt_json, tokutenkai=True)
                    df_edit_live, df_edit_tokutenkai = \
                        timetabledata.devide_df_live_tokutenkai(df_edit_tgt)
                    df_edit_tokutenkai = df_edit_tokutenkai[
                        df_edit_tokutenkai["ライブ_長さ(分)"].notnull()
                        & (df_edit_tokutenkai["ライブ_長さ(分)"] != "")
                    ]
                    tokutenkai_timetable.append(df_edit_tokutenkai)
                else:
                    df_edit_live = timetabledata.json_to_df(edit_tgt_json, tokutenkai=False)

                df_edit_live = df_edit_live[
                    df_edit_live["ライブ_長さ(分)"].notnull()
                    & (df_edit_live["ライブ_長さ(分)"] != "")
                ].copy()

                this_stage_id, stage_id = find_or_create_stage_id(
                    stage_master, stage_name_list[stage_no], tokutenkai_flg, stage_id,
                    existing_stage_id=existing_top_stage_id,
                )
                if "ステージID" not in df_edit_live.columns:
                    df_edit_live["ステージID"] = None
                    df_edit_live["ステージ名"] = None
                df_edit_live.loc[:, "ステージID"] = this_stage_id
                df_edit_live.loc[:, "ステージ名"] = stage_name_list[stage_no]
                event_timetable_all.append(df_edit_live)
                # 子ブースIDの引き当てヒントを後続処理に引き渡す
                booth_existing_id_map_all.update(booth_existing_id_map)
            except KeyError:
                pass

    stage_master_tokutenkai = {}
    df_tokutenkai = None
    if len(tokutenkai_timetable) > 0:
        df_tokutenkai = pd.concat(tokutenkai_timetable).reset_index(drop=True)
        df_tokutenkai = df_tokutenkai.drop(columns=["ステージID"], errors="ignore")
        booth_name_list = df_tokutenkai["ステージ名"].drop_duplicates().tolist()
        for booth_name in booth_name_list:
            existing_booth_id = booth_existing_id_map_all.get(booth_name)
            this_stage_id, stage_id = find_or_create_stage_id(
                stage_master, booth_name, True, stage_id,
                existing_stage_id=existing_booth_id,
            )
            stage_master_tokutenkai[this_stage_id] = {
                "ステージ名": booth_name, "特典会フラグ": True,
            }

    if len(event_timetable_all) == 0:
        return None

    df_stage = pd.DataFrame.from_dict(stage_master, orient="index")
    df_stage.index.name = "ステージID"
    df_stage.index = df_stage.index.astype(int)
    # 後方互換: 表示順 / 非活性化フラグ が無い (新規作成パス) 場合は補完
    if "表示順" not in df_stage.columns:
        df_stage["表示順"] = range(len(df_stage))
    else:
        # 欠損行 (find_or_create_stage_id が古い経路で作られた等) を index 順で埋める
        missing_mask = df_stage["表示順"].isna()
        if missing_mask.any():
            df_stage.loc[missing_mask, "表示順"] = list(range(len(df_stage)))[: int(missing_mask.sum())]
    if "非活性化フラグ" not in df_stage.columns:
        df_stage["非活性化フラグ"] = False
    df_stage["非活性化フラグ"] = df_stage["非活性化フラグ"].fillna(False).astype(bool)
    df_stage["表示順"] = df_stage["表示順"].astype(int)
    # Phase 2: ステージ名_短縮 / カラー名 補完
    if "ステージ名_短縮" not in df_stage.columns:
        df_stage["ステージ名_短縮"] = df_stage["ステージ名"]
    df_stage["ステージ名_短縮"] = (
        df_stage["ステージ名_短縮"].fillna(df_stage["ステージ名"]).astype(str)
    )
    if "カラー名" not in df_stage.columns:
        df_stage["カラー名"] = None
    for sid in df_stage.index:
        cur = df_stage.at[sid, "カラー名"]
        if cur is None or (isinstance(cur, float) and pd.isna(cur)) or cur == "":
            order = int(df_stage.at[sid, "表示順"])
            is_tk = bool(df_stage.at[sid, "特典会フラグ"])
            df_stage.at[sid, "カラー名"] = _default_color_name(order, is_tk)
    df_stage = df_stage.sort_values("表示順")

    df_live = pd.concat(event_timetable_all).reset_index(drop=True)
    if df_tokutenkai is not None:
        df_stage_tokutenkai = pd.DataFrame.from_dict(stage_master_tokutenkai, orient="index")
        df_stage_tokutenkai.index.name = "ステージID"
        df_tokutenkai = pd.merge(
            df_tokutenkai,
            df_stage_tokutenkai.reset_index().drop("特典会フラグ", axis=1),
            on="ステージ名", how="left",
        )
        df_live = pd.concat([df_live, df_tokutenkai]).reset_index(drop=True)

    # アーティストマスタ構築
    df_idolname = pd.DataFrame(
        df_live["グループ名_採用"].drop_duplicates().sort_values().reset_index(drop=True)
    )
    df_idolname = df_idolname[
        ~df_idolname["グループ名_採用"].isin(idolname_master_df["グループ名_採用"])
    ].reset_index(drop=True)
    df_idolname.index = df_idolname.index + artist_id
    df_idolname.index.name = "グループID"
    df_idolname = pd.concat([idolname_master_df, df_idolname])

    # 出番データ構築
    df_live = df_live.drop(columns=["グループID"], errors="ignore")
    df_live = pd.merge(
        df_live, df_idolname.reset_index(), on="グループ名_採用", how="left",
    ).rename(columns={"グループ名": "グループ名_raw", "グループ名_採用": "グループ名"})

    # Phase 1: コラボグループ統合
    # 方針: 出番ID 優先・コラボグループID 劣後。
    # - 既に 出番ID が振られた行はそのまま (コラボグループID は完全に無視)
    # - 出番ID が NULL かつ 同一 (ステージID, コラボグループID) を持つ行群には
    #   同一の新規 出番ID を採番
    # - 出番ID が NULL かつ コラボグループID も NULL の行は通常通り個別採番
    # - コラボタイトル は最終的な 出番ID 単位で正規化 (最初の非空値を全行に伝播)
    if "コラボタイトル" not in df_live.columns:
        df_live["コラボタイトル"] = None
    if "コラボグループID" not in df_live.columns:
        df_live["コラボグループID"] = None
    if "出番ID" not in df_live.columns:
        df_live["出番ID"] = None

    turn_id_series = pd.to_numeric(df_live["出番ID"], errors="coerce")
    cgid_series = pd.to_numeric(df_live["コラボグループID"], errors="coerce")

    existing_ids = turn_id_series.dropna()
    next_turn_id = int(existing_ids.max()) + 1 if len(existing_ids) > 0 else 0

    # 出番ID 未採番 かつ コラボグループID あり の行群を (ステージID, cgid) でまとめて採番
    unassigned_cgid_mask = turn_id_series.isna() & cgid_series.notna()
    if unassigned_cgid_mask.any():
        df_live.loc[unassigned_cgid_mask, "コラボグループID"] = (
            cgid_series[unassigned_cgid_mask].astype(int)
        )
        for (_sid, _cgid), group in df_live[unassigned_cgid_mask].groupby(
            ["ステージID", "コラボグループID"], dropna=False,
        ):
            df_live.loc[group.index, "出番ID"] = next_turn_id
            next_turn_id += 1

    # 残り (出番ID NULL かつ コラボグループID も NULL) は 1 行ずつ個別採番
    missing_mask = df_live["出番ID"].isna()
    new_ids = list(range(next_turn_id, next_turn_id + int(missing_mask.sum())))
    df_live.loc[missing_mask, "出番ID"] = new_ids
    df_live["出番ID"] = df_live["出番ID"].astype(int)

    # コラボタイトル正規化: 確定後の 出番ID 単位で最初の非空値を伝播
    # (cgid ベースではなく 出番ID ベース。出番ID が同じなら cgid が違っても同一タイトル)
    for _turn_id, group in df_live.groupby("出番ID"):
        if len(group) < 2:
            continue
        non_empty = [
            v for v in group["コラボタイトル"].tolist()
            if isinstance(v, str) and v != ""
        ]
        if non_empty:
            df_live.loc[group.index, "コラボタイトル"] = non_empty[0]

    df_live = df_live.set_index("出番ID")

    df_live_out = df_live[_LIVE_OUTPUT_COLUMNS]

    # 非活性化ステージとそれに紐づく出番を出力・集計から除外する
    # (実データ master_stage.csv / turn_id_data.csv 等への保存時は除外しないため、
    #  非活性化ステージIDのフィルタは戻り値ベースで実施する)
    disabled_stage_ids = set(df_stage[df_stage["非活性化フラグ"]].index.tolist())
    if disabled_stage_ids:
        df_stage_visible = df_stage[~df_stage["非活性化フラグ"]]
        df_live_visible = df_live[~df_live["ステージID"].isin(disabled_stage_ids)]
        df_live_out_visible = df_live_out[~df_live_out["ステージID"].isin(disabled_stage_ids)]
    else:
        df_stage_visible = df_stage
        df_live_visible = df_live
        df_live_out_visible = df_live_out

    return {
        "stage": df_stage_visible,
        "idolname": df_idolname,
        "live": df_live_out_visible,
        "duration_distribution": build_duration_distribution(df_live_out_visible, df_stage_visible),
        "group_count": build_group_appearance_count(df_live_out_visible, df_stage_visible),
        "overlap_alerts": build_overlap_alerts(df_live_visible, df_stage_visible),
        "group_appearances": build_group_appearances(df_live_visible, df_stage_visible),
    }


def build_all_event_outputs(
    pj_path: str,
    project_info_json: dict,
) -> dict[str, dict[str, pd.DataFrame]]:
    """全イベントの出力データを組み立てて返す。

    データが存在しないイベントは空dictで保持する(キーは保つ)。
    後段の determine_id_master / export_excel / listup_new_idolname は
    空dictを検知してスキップする。
    """
    event_list = repo.get_event_name_list(project_info_json)
    result: dict[str, dict[str, pd.DataFrame]] = {}
    for event_name in event_list:
        event_no = repo.get_event_no_by_event_name(project_info_json, event_name)
        data = build_event_output(pj_path, event_name, event_no, project_info_json)
        result[event_name] = data if data is not None else {}
    return result


# ---------------------------------------------------------------------------
# ID確定
# ---------------------------------------------------------------------------

def determine_id_master(
    output_df: dict[str, dict[str, pd.DataFrame]],
    pj_path: str,
    project_info_json: dict,
) -> None:
    """ステージマスタ・グループマスタ・出番マスタのIDを確定させ、CSV/JSONに保存する。

    ステージID は各 stage_*.json のトップレベルと project_info.json の
    stage_list[i].stage_id に書き戻される。
    """
    event_list = repo.get_event_name_list(project_info_json)
    for event_name in event_list:
        if not output_df.get(event_name):
            continue
        output_path = os.path.join(pj_path, event_name)
        # build_event_output で 非活性化ステージが除外されているため、
        # ここで保存される master_stage.csv は「表示用」のみ。
        # ただし非活性化ステージも実体保持が必要なため、
        # 既存 master_stage.csv (除外前) があれば次回 load で残る。
        # determine_id_master は初回確定時のみ呼ばれる前提なので、
        # 初回時点では非活性化ステージは存在しない。
        output_df[event_name]["stage"].to_csv(os.path.join(output_path, "master_stage.csv"))
        output_df[event_name]["idolname"].to_csv(os.path.join(output_path, "master_idolname.csv"))
        turn_id_data = output_df[event_name]["live"]
        turn_id_data.to_csv(os.path.join(output_path, "turn_id_data.csv"))

        event_no = repo.get_event_no_by_event_name(project_info_json, event_name)
        event_type_list = repo.get_event_type_list(project_info_json, event_no)
        for event_type in event_type_list:
            tgt_event_type_info = repo.get_image_entry_by_dir_name(
                project_info_json, event_no, event_type,
            )
            for stage_no in range(tgt_event_type_info["stage_num"]):
                stage_name = repo.get_stage_name(project_info_json, event_no, event_type, stage_no)
                json_path = os.path.join(output_path, event_type, f"stage_{stage_no}.json")
                if os.path.exists(json_path):
                    with open(json_path, encoding="utf-8") as f:
                        json_data = json.load(f)
                    json_data = timetabledata.id_apply_to_json(
                        json_data, turn_id_data, stage_name,
                        tgt_event_type_info["kind"] == "live_tokutenkai_heiki",
                    )
                    with open(json_path, "w", encoding="utf8") as f:
                        json.dump(json_data, f, indent=4, ensure_ascii=False)

                    # project_info.stage_list[stage_no].stage_id にも同期書き込み
                    top_stage_id = json_data.get("ステージID")
                    if top_stage_id is not None:
                        try:
                            repo.set_stage_id(
                                project_info_json, event_no, event_type, stage_no,
                                int(top_stage_id),
                            )
                        except (KeyError, ValueError):
                            pass
    # project_info.json を永続化(stage_id の書き込みを反映)
    repo.save_project_json(pj_path, project_info_json)


# ---------------------------------------------------------------------------
# S3保存
# ---------------------------------------------------------------------------

def save_to_s3(pj_name: str) -> None:
    """プロジェクトデータをS3にアップロードする。"""
    s3access.put_project_data(pj_name)


# ---------------------------------------------------------------------------
# Excel出力
# ---------------------------------------------------------------------------

def export_excel(
    output_df: dict[str, dict[str, pd.DataFrame]],
    pj_path: str,
    event_list: list[str],
) -> str:
    """Excel形式でデータを出力し、出力パスを返す。"""
    output_path = os.path.join(pj_path, "output.xlsx")
    wb = Workbook()
    for event_name in event_list:
        if not output_df.get(event_name):
            continue
        for df_type, position in zip(["stage", "idolname", "live"], [(1, 1), (5, 1), (8, 1)]):
            df_to_write = output_df[event_name][df_type]
            if df_type == "live":
                drop_cols = [c for c in _LIVE_EXCEL_DROP_COLUMNS if c in df_to_write.columns]
                if drop_cols:
                    df_to_write = df_to_write.drop(columns=drop_cols)
                df_to_write = df_to_write.rename(columns={"グループ名_raw": "グループ名(OCR)"})
            elif df_type == "idolname":
                df_to_write = df_to_write.rename(columns={"グループ名_採用": "グループ名"})
            save_dataframe_to_excel(wb, event_name, df_to_write, position)
    default_sheet = wb["Sheet"]
    wb.remove(default_sheet)
    wb.save(output_path)
    return output_path


def save_dataframe_to_excel(
    wb: Workbook,
    sheet_name: str,
    df: pd.DataFrame,
    position: tuple[int, int],
) -> None:
    """DataFrameをExcelワークブックの指定位置に書き込む。"""
    existing_sheets = wb.sheetnames
    if sheet_name not in existing_sheets:
        ws = wb.create_sheet(title=sheet_name)
    else:
        ws = wb[sheet_name]
    for i, row in enumerate(df.itertuples(), start=position[1]):
        for j, value in enumerate(row, start=position[0]):
            ws.cell(row=i + 1, column=j, value=value)
    for j, header in enumerate(df.columns, start=position[0]):
        ws.cell(row=position[1], column=j + 1, value=header)
    ws.cell(row=position[1], column=position[0], value=df.index.name)


# ---------------------------------------------------------------------------
# グループ名マスタ
# ---------------------------------------------------------------------------

def listup_new_idolname(
    output_df: dict[str, dict[str, pd.DataFrame]],
    event_list: list[str],
) -> pd.DataFrame:
    """新しく出現したグループ名をリストアップする。"""
    idol_name_all = []
    for event_name in event_list:
        if not output_df.get(event_name):
            continue
        idol_name_all.extend(list(output_df[event_name]["idolname"]["グループ名_採用"]))
    new_idol_name = idolname.detect_new_data(list(set(idol_name_all)))
    return pd.DataFrame({
        "追加": [True for _ in range(len(new_idol_name))],
        "グループ名": new_idol_name,
    }).sort_values(by="グループ名").reset_index(drop=True)


def update_master_idolname(
    df_new_idolname: pd.DataFrame,
    data_path: str,
) -> None:
    """新しく出現したグループ名をマスタに追加し、S3にアップロードする。"""
    new_idolname = list(df_new_idolname[df_new_idolname["追加"]]["グループ名"])
    idolname.add_new_data_file(new_idolname)

    json_path = os.path.join(data_path, "master/master_version_s3.json")
    with open(json_path, 'r', encoding='utf-8') as f:
        master_version_s3 = json.load(f)
    jst = ZoneInfo("Asia/Tokyo")
    now_jst = datetime.now(jst)
    updated_at = now_jst.strftime('%Y/%m/%d %H:%M:%S.%f')
    master_version_s3["idolname_embedding_data.csv"] = updated_at
    master_version_s3["idolname_latest.csv"] = updated_at
    with open(json_path, "w", encoding="utf8") as f:
        json.dump(master_version_s3, f, indent=4, ensure_ascii=False)

    s3_prefix = "master"
    s3access.upload_s3_file(s3_prefix, "master_version_s3.json", os.path.join(data_path, "master/master_version_s3.json"))
    s3access.upload_s3_file(s3_prefix, "idolname_embedding_data.csv", os.path.join(data_path, "master/idolname_embedding_data.csv"))
    s3access.upload_s3_file(s3_prefix, "idolname_latest.csv", os.path.join(data_path, "master/idolname_latest.csv"))
