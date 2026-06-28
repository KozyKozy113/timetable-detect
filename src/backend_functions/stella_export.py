"""Stella JSON (live{id}.json / liveList.json) のエクスポート。

⑥出力データ (`build_event_output()` の戻り値) を Stella クライアント向け
JSON 形式に変換する。GitHub 連携は別モジュール (Phase 6) に分離する。
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import pandas as pd


# ---------------------------------------------------------------------------
# 内部ヘルパ
# ---------------------------------------------------------------------------

def _to_int_or_none(v) -> int | None:
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def _build_art_list(df_idolname: pd.DataFrame) -> list[dict]:
    """idolname DataFrame → `[{"artId": ..., "name": ...}, ...]`。"""
    result: list[dict] = []
    for art_id, row in df_idolname.iterrows():
        try:
            result.append({"artId": int(art_id), "name": str(row["グループ名_採用"])})
        except (ValueError, TypeError):
            continue
    return result


def _build_stage_list(df_stage: pd.DataFrame) -> list[dict]:
    """stage DataFrame → `[{stageId, stageName, stageNameShort, colorName, stageOrder}, ...]`。"""
    result: list[dict] = []
    for sid, row in df_stage.iterrows():
        order_val = row.get("表示順") if "表示順" in df_stage.columns else None
        stage_order = _to_int_or_none(order_val)
        if stage_order is None:
            stage_order = int(sid)
        name = str(row["ステージ名"])
        name_short = (
            str(row["ステージ名_短縮"])
            if "ステージ名_短縮" in df_stage.columns
            and isinstance(row["ステージ名_短縮"], str)
            and row["ステージ名_短縮"] != ""
            else name
        )
        color_name = (
            str(row["カラー名"])
            if "カラー名" in df_stage.columns
            and isinstance(row["カラー名"], str)
            and row["カラー名"] != ""
            else ""
        )
        result.append({
            "stageId": int(sid),
            "stageName": name,
            "stageNameShort": name_short,
            "colorName": color_name,
            "stageOrder": stage_order,
        })
    result.sort(key=lambda x: (x["stageOrder"], x["stageId"]))
    return result


def _calc_minutes(from_str: str, length_min) -> int:
    """`min` (出演分数) を返す。`ライブ_長さ(分)` 優先、欠損時は 0。"""
    try:
        return int(length_min)
    except (ValueError, TypeError):
        return 0


def _make_filler_turn(turn_id: int) -> dict:
    """欠番補完用の min=0 レコード。値は Stella JSON の型に合わせる
    (文字列フィールドは空文字、整数フィールドは 0)。"""
    return {
        "turnId": turn_id,
        "startTime": "",
        "min": 0,
        "artId": 0,
        "stageId": 0,
    }


def _build_turn_list(df_live: pd.DataFrame) -> list[dict]:
    """live DataFrame → `[{turnId, startTime, min, artId, stageId, ...}, ...]`。

    同一 `出番ID` の複数行はコラボ出番として 1 件にまとめる:
      - `artId`: 先頭行のグループID
      - `collabArtList`: 当該グループの全 グループID (先頭含む)
      - `title`: `コラボタイトル` が非空ならそのまま設定 (空ならフィールド省略)

    出力は `turnId` 昇順で並べ、欠番を許容しない (0..max の連番に揃える)。
    欠番箇所には `min=0` の補完レコード (`_make_filler_turn`) を挿入する。
    """
    if df_live is None or len(df_live) == 0:
        return []
    df = df_live.copy()
    if df.index.name == "出番ID":
        df = df.reset_index()

    result: list[dict] = []
    for turn_id, group in df.groupby("出番ID", sort=True):
        head = group.iloc[0]
        art_ids: list[int] = []
        for gid in group["グループID"]:
            gid_int = _to_int_or_none(gid)
            if gid_int is not None and gid_int not in art_ids:
                art_ids.append(gid_int)
        if not art_ids:
            continue
        stage_id = _to_int_or_none(head.get("ステージID"))
        if stage_id is None:
            continue
        entry: dict = {
            "turnId": int(turn_id),
            "startTime": str(head["ライブ_from"]),
            "min": _calc_minutes(head["ライブ_from"], head.get("ライブ_長さ(分)")),
            "artId": art_ids[0],
            "stageId": stage_id,
        }
        if len(art_ids) > 1:
            entry["collabArtList"] = art_ids
        title_raw = head.get("コラボタイトル") if "コラボタイトル" in group.columns else None
        if isinstance(title_raw, str) and title_raw != "":
            entry["title"] = title_raw
        result.append(entry)

    if not result:
        return result
    # turnId 昇順に整列し、欠番を min=0 の補完レコードで埋めて 0..max の連番にする
    by_id = {entry["turnId"]: entry for entry in result}
    max_id = max(by_id)
    return [
        by_id[tid] if tid in by_id else _make_filler_turn(tid)
        for tid in range(max_id + 1)
    ]


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------

def compute_open_close_default(
    output_df: dict[str, pd.DataFrame],
) -> tuple[str, str]:
    """全出番の最早開始 / 最遅終了から openTime / closeTime を算出する。

    - openTime: min("ライブ_from") の H:MM について、MM==0 なら H-1、MM>=1 なら H
                (例 11:01～12:00 → 11、12:01～13:00 → 12)
    - closeTime: max("ライブ_to")＋1時間 を基準に、その H について常に H+1
                (例 終了20:25 → 21:25基準 → 22、終了21:00 → 22:00基準 → 23)
    算出不能な場合は ("", "") を返す。非活性化ステージの出番は対象外。
    """
    live = output_df.get("live")
    if live is None or len(live) == 0:
        return ("", "")
    df_stage_master = output_df.get("stage")
    if df_stage_master is not None and "非活性化フラグ" in df_stage_master.columns:
        disabled_stage_ids = set(
            df_stage_master[df_stage_master["非活性化フラグ"]].index.tolist()
        )
        if disabled_stage_ids and "ステージID" in live.columns:
            live = live[~live["ステージID"].isin(disabled_stage_ids)]
            if len(live) == 0:
                return ("", "")

    def _to_minutes(s) -> int | None:
        if s is None or (isinstance(s, float) and pd.isna(s)) or s == "":
            return None
        try:
            h, m = str(s).split(":")
            return int(h) * 60 + int(m)
        except (ValueError, AttributeError):
            return None

    from_series = live["ライブ_from"] if "ライブ_from" in live.columns else []
    to_series = live["ライブ_to"] if "ライブ_to" in live.columns else []
    from_min = [m for m in (_to_minutes(v) for v in from_series) if m is not None]
    end_min = [m for m in (_to_minutes(v) for v in to_series) if m is not None]
    if not from_min or not end_min:
        return ("", "")

    fh, fm = divmod(min(from_min), 60)
    # closeTime は「最遅終了時間＋1時間」を基準に H+1 で算出する
    th, _ = divmod(max(end_min) + 60, 60)
    open_h = fh - 1 if fm == 0 else fh
    close_h = th + 1
    return (str(open_h), str(close_h))


def build_stella_json(
    output_df: dict[str, pd.DataFrame],
    stella_metadata: dict,
) -> dict:
    """⑥のDataFrameからStella JSON形式のdictを構築する。

    output_df は `build_event_output()` の戻り値構造を想定:
      - "stage" / "idolname" / "live" (Stella JSON生成に使用) — master 全行 (非活性化含む)
      - その他 (duration_distribution 等) は未使用
    Stella JSON では非活性化ステージとその出番を出力対象から除外する。
    """
    df_stage_master = output_df["stage"]
    disabled_stage_ids: set = set()
    if "非活性化フラグ" in df_stage_master.columns:
        disabled_stage_ids = set(
            df_stage_master[df_stage_master["非活性化フラグ"]].index.tolist()
        )
        df_stage_for_export = df_stage_master[~df_stage_master["非活性化フラグ"]]
    else:
        df_stage_for_export = df_stage_master
    df_live_for_export = output_df["live"]
    if disabled_stage_ids and "ステージID" in df_live_for_export.columns:
        df_live_for_export = df_live_for_export[
            ~df_live_for_export["ステージID"].isin(disabled_stage_ids)
        ]

    art_list = _build_art_list(output_df["idolname"])
    stage_list = _build_stage_list(df_stage_for_export)
    turn_list = _build_turn_list(df_live_for_export)

    result: dict = {}
    # liveId / jsonVersion はメタデータ側からコピー (欠損キーは省略)
    for k in ("liveId", "jsonVersion"):
        if k in stella_metadata and stella_metadata[k] is not None:
            result[k] = stella_metadata[k]
    for k in ("notificationVersion", "notification"):
        result[k] = stella_metadata.get(k, "")
    # openTime / closeTime が未入力なら、メタデータ編集フォームと同じロジックで
    # デフォルト値 (全出番の最早開始 / 最遅終了から算出) を補完する
    open_default, close_default = compute_open_close_default(output_df)
    open_time = str(stella_metadata.get("openTime", "") or "")
    close_time = str(stella_metadata.get("closeTime", "") or "")
    result["openTime"] = open_time if open_time else open_default
    result["closeTime"] = close_time if close_time else close_default
    result["artList"] = art_list
    result["stageList"] = stage_list
    result["turnList"] = turn_list
    return result


def write_stella_json(
    stella_json: dict,
    output_dir: str,
    live_id: int | None = None,
) -> str:
    """Stella JSON を整形 (インデント) 形式で `live{liveId}.json` として書き出す。

    可読性のため 2 スペースインデントで出力し、末尾に改行を付与する。
    `live_id` 指定が無ければ `stella_json["liveId"]` を採用。
    どちらも無い場合は `live_unassigned.json`。
    """
    os.makedirs(output_dir, exist_ok=True)
    if live_id is None:
        live_id = stella_json.get("liveId")
    fname = f"live{int(live_id)}.json" if live_id is not None else "live_unassigned.json"
    path = os.path.join(output_dir, fname)
    # Stella 運用リポは全 JSON が UTF-8 BOM 付きで配置されているため合わせる
    with open(path, "w", encoding="utf-8-sig") as f:
        json.dump(stella_json, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return path


def update_live_list(
    live_list_path: str,
    new_entries: list[dict],
) -> None:
    """`liveList.json` に新規エントリを追加/更新する (liveId 一致なら上書き)。

    入出力は UTF-8 BOM 付き (Stella 運用リポの既存仕様)。
    Phase 4 で詳細セマンティクスを再検討するが、最低限のヘルパは本Phaseで用意。
    """
    if os.path.exists(live_list_path):
        with open(live_list_path, encoding="utf-8-sig") as f:
            data = json.load(f)
        live_list = data.get("liveList", [])
    else:
        live_list = []

    by_id = {e["liveId"]: e for e in live_list if "liveId" in e}
    for entry in new_entries:
        if "liveId" not in entry:
            continue
        by_id[entry["liveId"]] = entry
    merged = sorted(by_id.values(), key=lambda x: x.get("liveId", 0))

    with open(live_list_path, "w", encoding="utf-8-sig") as f:
        json.dump({"liveList": merged}, f, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# notificationData4.json (アプリトップのお知らせ) 生成・追記
# ---------------------------------------------------------------------------

NOTIFICATION_FILENAME = "notificationData4.json"
NOTIFICATION_ICON = "mdi-alarm-plus"

# genre code -> (JP ラベル, EN ラベル)。お知らせ文の表示語に使う。
# 既定 (未知 genre) は アイドル / IDLE。
GENRE_NOTIFICATION_LABELS: dict[int, tuple[str, str]] = {
    1: ("バンド", "BAND"),
    2: ("アイドル", "IDLE"),
}


def _parse_yyyymmdd(value) -> "datetime | None":
    try:
        return datetime.strptime(str(value), "%Y%m%d")
    except (ValueError, TypeError):
        return None


def build_notification_date_range(dates: list) -> str:
    """`YYYYMMDD` 文字列の list からお知らせ文用の日付レンジ表記を作る。

    - 単日           → `M/D`        (例 6/27)
    - 同月の複数日   → `M/D-D`      (例 6/27-28)
    - 月跨ぎの複数日 → `M/D-M/D`    (例 6/27-7/1)
    解析できる日付が無ければ空文字。ゼロ埋めはしない。
    """
    parsed = sorted(d for d in (_parse_yyyymmdd(x) for x in dates) if d is not None)
    if not parsed:
        return ""
    lo, hi = parsed[0], parsed[-1]
    if lo == hi:
        return f"{lo.month}/{lo.day}"
    if lo.month == hi.month:
        return f"{lo.month}/{lo.day}-{hi.day}"
    return f"{lo.month}/{lo.day}-{hi.month}/{hi.day}"


def build_notification_messages(
    live_name: str, date_range: str, area: str, genre,
) -> tuple[str, str]:
    """お知らせ文 (日本語 / 英語) を自動生成する。

    - JP: `[liveName] ([日付] @[エリア] [ジャンルJP]) に対応しました。`
          エリアが空のときは ` @[エリア]` を省く。
    - EN: `[liveName] ([日付] [ジャンルEN]) is supported.`
    """
    genre_int = _to_int_or_none(genre)
    genre_jp, genre_en = GENRE_NOTIFICATION_LABELS.get(
        genre_int if genre_int is not None else 2, GENRE_NOTIFICATION_LABELS[2],
    )
    name = str(live_name or "").strip()
    area = str(area or "").strip()
    jp_inner = f"{date_range} @{area} {genre_jp}" if area else f"{date_range} {genre_jp}"
    message = f"{name} ({jp_inner}) に対応しました。"
    message_en = f"{name} ({date_range} {genre_en}) is supported."
    return message, message_en


def build_notification_entry(
    live_ids: list, date: str, message: str, message_en: str,
) -> dict:
    """notificationList に追加する 1 エントリを構築する (フィールド順は運用例に合わせる)。"""
    return {
        "icon": NOTIFICATION_ICON,
        "liveId": [int(i) for i in live_ids],
        "date": str(date),
        "message": str(message),
        "message_en": str(message_en),
    }


def find_duplicate_notifications(notification_list: list, live_ids: list) -> list[int]:
    """`notificationList` 中、`liveId` が指定 live_ids と 1 つでも交差するエントリの
    インデックス list を返す。`liveId` を持たない旧エントリ (手書き運用) は対象外。
    """
    target = {int(i) for i in live_ids}
    result: list[int] = []
    for idx, entry in enumerate(notification_list):
        ids = entry.get("liveId") if isinstance(entry, dict) else None
        if not isinstance(ids, list):
            continue
        existing = {iv for iv in (_to_int_or_none(v) for v in ids) if iv is not None}
        if existing & target:
            result.append(idx)
    return result


def read_notification_data(path: str) -> dict:
    """notificationData JSON を読み込む。未存在なら空の `notificationList` を返す。"""
    if not os.path.exists(path):
        return {"notificationList": []}
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def prepend_notification(
    data: dict, entry: dict, *, remove_indices: "list[int] | tuple" = (),
) -> dict:
    """`notificationList` の **先頭**に entry を追加する (data を破壊的に更新)。

    `remove_indices` 指定時は、まず該当インデックスのエントリを除去してから先頭追加する
    (重複の「過去メッセージを削除したうえで追加」用)。インデックスは除去前の位置基準。
    """
    nlist = list(data.get("notificationList", []))
    if remove_indices:
        drop = set(remove_indices)
        nlist = [e for i, e in enumerate(nlist) if i not in drop]
    nlist.insert(0, entry)
    data["notificationList"] = nlist
    return data


def write_notification_data(path: str, data: dict) -> str:
    """notificationData JSON を書き出す (utf-8-sig / indent=2)。

    既存ファイルの手書き桁揃えは保持せず全面再整形する (実装の単純さ・堅牢性を優先)。
    """
    with open(path, "w", encoding="utf-8-sig") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return path


def increment_versions_on_push(
    stella_metadata: dict,
) -> dict:
    """Push 直前にバージョン番号を更新する (Phase 3-1)。

    - `jsonVersion`: +1 (初回は 1)
    - `notification` が非空かつ直近Push時から変わっていれば `notificationVersion` +1
      (未入力 "" / None は対象外。`_last_pushed_notification` の初期値 None と
       既定の notification "" を別物と見なして初回Pushで誤って +1 しないよう正規化する)
    - `_last_pushed_notification` を現在の `notification` に同期

    新しい metadata を返す (元 dict は破壊しない)。
    """
    new_meta = dict(stella_metadata)
    prev_json_version = _to_int_or_none(stella_metadata.get("jsonVersion")) or 0
    new_meta["jsonVersion"] = prev_json_version + 1

    last_pushed = str(stella_metadata.get("_last_pushed_notification") or "").strip()
    current_notif = str(stella_metadata.get("notification") or "").strip()
    if current_notif and current_notif != last_pushed:
        prev_nv = _to_int_or_none(stella_metadata.get("notificationVersion")) or 0
        new_meta["notificationVersion"] = str(prev_nv + 1)
    new_meta["_last_pushed_notification"] = current_notif
    return new_meta
