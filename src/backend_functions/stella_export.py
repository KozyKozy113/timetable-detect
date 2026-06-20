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


def _build_turn_list(df_live: pd.DataFrame) -> list[dict]:
    """live DataFrame → `[{turnId, startTime, min, artId, stageId, ...}, ...]`。

    同一 `出番ID` の複数行はコラボ出番として 1 件にまとめる:
      - `artId`: 先頭行のグループID
      - `collabArtList`: 当該グループの全 グループID (先頭含む)
      - `title`: `コラボタイトル` が非空ならそのまま設定 (空ならフィールド省略)
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

    result.sort(key=lambda x: (x["startTime"], x["turnId"]))
    return result


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------

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
    for k in ("openTime", "closeTime", "notificationVersion", "notification"):
        result[k] = stella_metadata.get(k, "")
    result["artList"] = art_list
    result["stageList"] = stage_list
    result["turnList"] = turn_list
    return result


def write_stella_json(
    stella_json: dict,
    output_dir: str,
    live_id: int | None = None,
) -> str:
    """Stella JSON を 1 行 minify 形式で `live{liveId}.json` として書き出す。

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
        json.dump(stella_json, f, ensure_ascii=False, separators=(",", ":"))
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


def increment_versions_on_push(
    stella_metadata: dict,
) -> dict:
    """Push 直前にバージョン番号を更新する (Phase 3-1)。

    - `jsonVersion`: +1 (初回は 1)
    - `notification` が直近Push時から変わっていれば `notificationVersion` +1
    - `_last_pushed_notification` を現在の `notification` に同期

    新しい metadata を返す (元 dict は破壊しない)。
    """
    new_meta = dict(stella_metadata)
    prev_json_version = _to_int_or_none(stella_metadata.get("jsonVersion")) or 0
    new_meta["jsonVersion"] = prev_json_version + 1

    last_pushed = stella_metadata.get("_last_pushed_notification")
    current_notif = stella_metadata.get("notification", "")
    if last_pushed != current_notif:
        prev_nv = _to_int_or_none(stella_metadata.get("notificationVersion")) or 0
        new_meta["notificationVersion"] = str(prev_nv + 1)
    new_meta["_last_pushed_notification"] = current_notif
    return new_meta
