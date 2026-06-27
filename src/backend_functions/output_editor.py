"""
⑥出力確認・編集タブの編集モードで作られた編集結果を、
master_*.csv / stage_*.json / project_info.json に書き戻す。

Streamlit に依存しない。
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from typing import Iterator

import pandas as pd

from backend_functions import project_repository as repo


# ---------------------------------------------------------------------------
# 内部ヘルパ
# ---------------------------------------------------------------------------

def _iter_stage_jsons(
    pj_path: str,
    event_name: str,
    event_no: int,
    project_info_json: dict,
) -> Iterator[tuple[str, dict, str, str, int]]:
    """イベント配下の全 stage_*.json を yield する。

    Yields:
        (json_path, data, event_type, kind, stage_no)
    """
    output_path = os.path.join(pj_path, event_name)
    event_type_list = repo.get_event_type_list(project_info_json, event_no)
    for event_type in event_type_list:
        entry = repo.get_image_entry_by_dir_name(
            project_info_json, event_no, event_type,
        )
        if entry is None:
            continue
        kind = entry.get("kind", "")
        for stage_no in range(entry.get("stage_num", 0)):
            json_path = os.path.join(
                output_path, event_type, f"stage_{stage_no}.json",
            )
            if not os.path.exists(json_path):
                continue
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            yield json_path, data, event_type, kind, stage_no


def build_stage_kind_map(
    pj_path: str,
    event_name: str,
    event_no: int,
    project_info_json: dict,
) -> dict[int, str]:
    """各 stage_id がどの event_type (dir_name) に属するかのマップを返す。

    heiki kind では トップレベル ステージID も 特典会[].ステージID も
    同じ event_type に紐付ける。
    ⑥編集モードの D&D ラベルに種別名を併記するために使う。
    """
    result: dict[int, str] = {}
    for _path, data, event_type, kind, _sn in _iter_stage_jsons(
        pj_path, event_name, event_no, project_info_json,
    ):
        top_sid = data.get("ステージID")
        if top_sid is not None:
            try:
                result[int(top_sid)] = event_type
            except (ValueError, TypeError):
                pass
        if kind == "live_tokutenkai_heiki":
            for turn in data.get("タイムテーブル", []) or []:
                for tk in turn.get("特典会", []) or []:
                    tk_sid = tk.get("ステージID")
                    if tk_sid is None:
                        continue
                    try:
                        result[int(tk_sid)] = event_type
                    except (ValueError, TypeError):
                        pass
    return result


def _write_json(json_path: str, data: dict) -> None:
    """プロジェクト共通の JSON 書き出しスタイル (indent=4 / utf-8 / 日本語そのまま)。"""
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def _add_minutes(time_str: str, minutes: int) -> str:
    """'HH:MM' に 分を加算して 'HH:MM' を返す。24h 跨ぎはそのまま %H:%M で丸める。"""
    dt = datetime.strptime(str(time_str), "%H:%M")
    dt = dt + timedelta(minutes=int(minutes))
    return dt.strftime("%H:%M")


_HHMM_RE = re.compile(r"^\d{1,2}:\d{2}$")


def _stage_id_to_name_map(df_stage: pd.DataFrame) -> dict[int, str]:
    """df_stage (index=ステージID) を ID -> 新ステージ名 の辞書に変換する。"""
    result: dict[int, str] = {}
    for idx, row in df_stage.iterrows():
        try:
            result[int(idx)] = str(row["ステージ名"])
        except (ValueError, TypeError):
            continue
    return result


def _idolname_id_to_name_map(df_idolname: pd.DataFrame) -> dict[int, str]:
    """df_idolname (index=グループID) を ID -> 採用名 の辞書に変換する。"""
    result: dict[int, str] = {}
    for idx, row in df_idolname.iterrows():
        try:
            result[int(idx)] = str(row["グループ名_採用"])
        except (ValueError, TypeError):
            continue
    return result


# ---------------------------------------------------------------------------
# stage_*.json への伝播
# ---------------------------------------------------------------------------

def _propagate_stage_master_to_project(
    pj_path: str,
    event_name: str,
    event_no: int,
    project_info_json: dict,
    df_stage: pd.DataFrame,
) -> None:
    """編集後のステージマスタを stage_*.json と project_info に伝播する。

    stage_*.json のトップレベル `ステージ名` と project_info.stage_list[i].stage_name を、
    `ステージID` を引き当てキーにして同期更新する。
    特典会併記形式の子ブース別ID については `特典会[].ステージID` 経由で
    `特典会[].ブース` 名も更新する。
    """
    id_to_name = _stage_id_to_name_map(df_stage)
    if not id_to_name:
        return

    for json_path, data, event_type, kind, stage_no in _iter_stage_jsons(
        pj_path, event_name, event_no, project_info_json,
    ):
        is_heiki = kind == "live_tokutenkai_heiki"
        top_stage_id = data.get("ステージID")
        changed = False
        if top_stage_id is not None:
            try:
                new_name = id_to_name.get(int(top_stage_id))
            except (ValueError, TypeError):
                new_name = None
            if new_name is not None and data.get("ステージ名") != new_name:
                data["ステージ名"] = new_name
                changed = True
                repo.set_stage_name(
                    project_info_json, event_no, event_type, stage_no, new_name,
                )

        if is_heiki:
            for turn in data.get("タイムテーブル", []):
                for tk in turn.get("特典会", []) or []:
                    tk_id = tk.get("ステージID")
                    if tk_id is None:
                        continue
                    try:
                        new_booth_name = id_to_name.get(int(tk_id))
                    except (ValueError, TypeError):
                        new_booth_name = None
                    if new_booth_name is not None and tk.get("ブース") != new_booth_name:
                        tk["ブース"] = new_booth_name
                        changed = True

        if changed:
            _write_json(json_path, data)


def _propagate_idolname_master_to_json(
    pj_path: str,
    event_name: str,
    event_no: int,
    project_info_json: dict,
    df_idolname: pd.DataFrame,
) -> None:
    """master_idolname の グループ名_採用 を全 stage_*.json タイムテーブル[].グループ名_採用 に反映。

    引き当てキーは グループID。
    """
    gid_to_name = _idolname_id_to_name_map(df_idolname)
    if not gid_to_name:
        return

    for json_path, data, _et, _kind, _sn in _iter_stage_jsons(
        pj_path, event_name, event_no, project_info_json,
    ):
        changed = False
        for turn in data.get("タイムテーブル", []):
            gid = turn.get("グループID")
            if gid is None:
                continue
            try:
                new_name = gid_to_name.get(int(gid))
            except (ValueError, TypeError):
                new_name = None
            if new_name is None:
                continue
            if turn.get("グループ名_採用") != new_name:
                turn["グループ名_採用"] = new_name
                changed = True
        if changed:
            _write_json(json_path, data)


def build_corresponding_turn_id_map(
    pj_path: str,
    event_name: str,
    event_no: int,
    project_info_json: dict,
) -> dict[int, int]:
    """`live_tokutenkai_heiki` 形式の全 stage_*.json をスキャンし、
    booth 自身の `出番ID` → 親(ライブ)の `出番ID` (= 対応出番ID) のマップを返す。

    編集モード突入時、特典会行に 対応出番ID 列を付与するのに使う。
    """
    result: dict[int, int] = {}
    for _, data, _et, kind, _sn in _iter_stage_jsons(
        pj_path, event_name, event_no, project_info_json,
    ):
        if kind != "live_tokutenkai_heiki":
            continue
        for turn in data.get("タイムテーブル", []):
            parent_id = turn.get("出番ID")
            if parent_id is None:
                continue
            try:
                parent_id_int = int(parent_id)
            except (ValueError, TypeError):
                continue
            for tk in turn.get("特典会", []) or []:
                booth_id = tk.get("出番ID")
                if booth_id is None:
                    continue
                try:
                    result[int(booth_id)] = parent_id_int
                except (ValueError, TypeError):
                    continue
    return result


def _update_live_entry_fields(turn: dict, row: pd.Series) -> bool:
    """ライブ親エントリに row の編集値を反映する。変更があれば True。

    `ライブ_to` は `save_event_edits` 側で `ライブ_from + ライブ_長さ(分)` から
    一括再計算済みである前提で、ここでは row の値をそのまま使う。
    """
    changed = False
    new_from = str(row["ライブ_from"])
    new_to = str(row["ライブ_to"])
    inner = turn.setdefault("ライブステージ", {})
    if inner.get("from") != new_from:
        inner["from"] = new_from
        changed = True
    if inner.get("to") != new_to:
        inner["to"] = new_to
        changed = True
    gid_val = row.get("グループID")
    if gid_val is not None and not (isinstance(gid_val, float) and pd.isna(gid_val)):
        try:
            new_gid = int(gid_val)
            if turn.get("グループID") != new_gid:
                turn["グループID"] = new_gid
                changed = True
        except (ValueError, TypeError):
            pass
    new_adopted = row.get("グループ名")
    if isinstance(new_adopted, str) and turn.get("グループ名_採用") != new_adopted:
        turn["グループ名_採用"] = new_adopted
        changed = True
    new_remarks = row.get("備考")
    if isinstance(new_remarks, str) and turn.get("備考", "") != new_remarks:
        turn["備考"] = new_remarks
        changed = True
    return changed


def _update_tk_fields(tk: dict, row: pd.Series) -> bool:
    """特典会(booth)要素に row の編集値を反映する。変更があれば True。

    対象: from/to + ブース別ステージID (Phase 5)
    対応出番ID の変更は呼び出し側 (`_propagate_live_edits_to_json`) で
    親エントリ移動として処理済。

    `ライブ_to` は `save_event_edits` 側で一括再計算済みである前提。
    """
    changed = False
    new_from = str(row["ライブ_from"])
    new_to = str(row["ライブ_to"])
    if tk.get("from") != new_from:
        tk["from"] = new_from
        changed = True
    if tk.get("to") != new_to:
        tk["to"] = new_to
        changed = True
    new_stage_id_raw = row.get("ステージID") if "ステージID" in row.index else None
    if new_stage_id_raw is not None and not (
        isinstance(new_stage_id_raw, float) and pd.isna(new_stage_id_raw)
    ):
        try:
            new_stage_id_int = int(new_stage_id_raw)
            if tk.get("ステージID") != new_stage_id_int:
                tk["ステージID"] = new_stage_id_int
                changed = True
        except (ValueError, TypeError):
            pass
    return changed


def _shift_booth_index_after_removal(
    booth_index: dict,
    file_idx: int,
    turn_idx: int,
    removed_tk_idx: int,
) -> None:
    """同一 (file_idx, turn_idx) の booth_index エントリのうち、removed_tk_idx より後のものを 1 つ前に詰める。"""
    for bid, (fi, ti, tki) in list(booth_index.items()):
        if fi == file_idx and ti == turn_idx and tki > removed_tk_idx:
            booth_index[bid] = (fi, ti, tki - 1)


def _build_live_indices(
    loaded: list,
) -> tuple[dict[int, list[tuple[int, int]]], dict[int, tuple[int, int, int]]]:
    """loaded から parent_index / booth_index を構築する。

    Returns:
        (parent_index, booth_index):
            parent_index : parent_出番ID -> [(file_idx, turn_idx), ...]
                同一 出番ID を複数エントリが共有するコラボステージに対応するため、
                JSON の タイムテーブル[] 出現順を保ったリストで保持する。
            booth_index  : booth_出番ID  -> (file_idx, turn_idx, tk_idx)  (heiki のみ)
                特典会要素の 出番ID はエントリ毎にユニークなため単一値で保持する。
    """
    parent_index: dict[int, list[tuple[int, int]]] = {}
    booth_index: dict[int, tuple[int, int, int]] = {}
    for file_idx, (_jp, data, kind) in enumerate(loaded):
        for turn_idx, turn in enumerate(data.get("タイムテーブル", [])):
            pid = turn.get("出番ID")
            if pid is not None:
                try:
                    parent_index.setdefault(int(pid), []).append((file_idx, turn_idx))
                except (ValueError, TypeError):
                    pass
            if kind == "live_tokutenkai_heiki":
                for tk_idx, tk in enumerate(turn.get("特典会", []) or []):
                    bid = tk.get("出番ID")
                    if bid is None:
                        continue
                    try:
                        booth_index[int(bid)] = (file_idx, turn_idx, tk_idx)
                    except (ValueError, TypeError):
                        pass
    return parent_index, booth_index


def _locate_turn(loaded: list, turn_obj: dict) -> tuple[int, int] | None:
    """loaded 内で turn_obj (タイムテーブルエントリの実体) の現在位置を同一性で探す。

    Phase 5 の移動 (pop/append) で turn_idx がずれても、オブジェクト参照から
    現在の (file_idx, turn_idx) を引き直すために使う。
    """
    for file_idx, (_jp, data, _kind) in enumerate(loaded):
        for turn_idx, turn in enumerate(data.get("タイムテーブル", [])):
            if turn is turn_obj:
                return (file_idx, turn_idx)
    return None


def _build_file_by_top_stage_id(loaded: list) -> dict[int, tuple[int, str]]:
    """top-level `ステージID` -> (file_idx, kind) のマップを返す。"""
    result: dict[int, tuple[int, str]] = {}
    for file_idx, (_jp, data, kind) in enumerate(loaded):
        tid = data.get("ステージID")
        if tid is None:
            continue
        try:
            result[int(tid)] = (file_idx, kind)
        except (ValueError, TypeError):
            pass
    return result


def _propagate_live_edits_to_json(
    pj_path: str,
    event_name: str,
    event_no: int,
    project_info_json: dict,
    df_live: pd.DataFrame,
) -> None:
    """編集後の出番マスタ各行を stage_*.json に書き戻す。

    引き当てキー:
        - ライブ行 / 非heiki特典会行: 親エントリの `出番ID` + 同一出番ID内の出現順 (parent_index)
              コラボステージは同一出番IDを複数エントリが共有するため、df 行とJSONエントリを
              出現順で位置対応させ、turn オブジェクト参照を掴んで更新する。
        - heiki 特典会行:             `特典会[j].出番ID` 一致 (booth_index)

    変更カテゴリ:
        - Phase 5 ライブエントリ移動: parent_index の行で `ステージID` 変更 →
              タイムテーブル[i] 全体 (含 特典会[]) を別 stage_*.json (同 kind / 同 top-level ステージID) に移動
        - Phase 4 特典会再親付け: booth_index の行で `対応出番ID` 変更 →
              特典会[j] 要素を別親 (場合により別 stage_*.json) へ移植
        - Phase 5 booth別ステージID 変更: heiki 特典会行で `ステージID` 変更 →
              `特典会[j].ステージID` を in-place で更新 (`_update_tk_fields` 経由)
        - 値更新: from/to / グループID / グループ名_採用 / 備考
    """
    if df_live is None or len(df_live) == 0:
        return
    if "特典会フラグ" not in df_live.columns:
        return

    df = df_live.copy()
    if df.index.name == "出番ID":
        df = df.reset_index()

    # 1. 全 stage_*.json をメモリにロード
    loaded: list[list] = []   # [json_path, data, kind]
    for json_path, data, _et, kind, _sn in _iter_stage_jsons(
        pj_path, event_name, event_no, project_info_json,
    ):
        loaded.append([json_path, data, kind])
    if not loaded:
        return

    # 2. インデックス構築
    parent_index, booth_index = _build_live_indices(loaded)
    file_by_top_stage_id = _build_file_by_top_stage_id(loaded)

    changed_file_idxs: set[int] = set()

    # 3. 編集行を分類:
    live_value_updates: list[tuple[dict, pd.Series]] = []         # 親エントリ値更新 (turn_obj, row)
    live_moves: list[tuple[dict, pd.Series, int]] = []            # 親エントリ ステージID移動 (turn_obj, row, new_stage_id)
    tk_value_updates: list[tuple[int, pd.Series]] = []            # heiki 特典会値更新 (対応出番ID変更なし)
    tk_moves: list[tuple[int, pd.Series, int]] = []               # heiki 特典会 対応出番ID 移動 (booth_id, row, new_parent_id)

    # 同一出番ID内の出現順カウンタ (コラボステージのエントリ取り違え防止)
    parent_occurrence: dict[int, int] = {}
    for _, row in df.iterrows():
        turn_id = row.get("出番ID")
        if turn_id is None or (isinstance(turn_id, float) and pd.isna(turn_id)):
            continue
        try:
            turn_id_int = int(turn_id)
        except (ValueError, TypeError):
            continue

        # 親エントリ (非heiki ライブ / 非heiki 特典会 / heiki ライブ親)
        if turn_id_int in parent_index:
            # df 行を 同一出番ID内の出現順で JSON エントリへ位置対応させる。
            occ = parent_occurrence.get(turn_id_int, 0)
            parent_occurrence[turn_id_int] = occ + 1
            locs = parent_index[turn_id_int]
            if occ >= len(locs):
                continue  # df 行数 > JSON エントリ数 (通常起こらない) は防御的にスキップ
            file_idx, turn_idx = locs[occ]
            turn_obj = loaded[file_idx][1]["タイムテーブル"][turn_idx]
            current_top_sid = loaded[file_idx][1].get("ステージID")
            try:
                current_top_sid_int = (
                    int(current_top_sid) if current_top_sid is not None else None
                )
            except (ValueError, TypeError):
                current_top_sid_int = None
            new_sid_raw = row.get("ステージID") if "ステージID" in row.index else None
            new_sid_int: int | None = None
            if new_sid_raw is not None and not (
                isinstance(new_sid_raw, float) and pd.isna(new_sid_raw)
            ):
                try:
                    new_sid_int = int(new_sid_raw)
                except (ValueError, TypeError):
                    new_sid_int = None
            if (
                new_sid_int is not None
                and current_top_sid_int is not None
                and new_sid_int != current_top_sid_int
            ):
                live_moves.append((turn_obj, row, new_sid_int))
            else:
                live_value_updates.append((turn_obj, row))
            continue

        # heiki 特典会行
        if turn_id_int in booth_index:
            loc = booth_index[turn_id_int]
            tk_now = loaded[loc[0]][1]["タイムテーブル"][loc[1]]["特典会"][loc[2]]
            current_corresp = tk_now.get("対応出番ID")
            new_corresp_raw = row.get("対応出番ID") if "対応出番ID" in row.index else None
            new_corresp_int: int | None = None
            if new_corresp_raw is not None and not (
                isinstance(new_corresp_raw, float) and pd.isna(new_corresp_raw)
            ):
                try:
                    new_corresp_int = int(new_corresp_raw)
                except (ValueError, TypeError):
                    new_corresp_int = None
            if new_corresp_int is not None and new_corresp_int != current_corresp:
                tk_moves.append((turn_id_int, row, new_corresp_int))
            else:
                tk_value_updates.append((turn_id_int, row))

    # 4. Phase 5: 親エントリのファイル間移動 (ステージID変更)
    if live_moves:
        for turn_obj, _row, new_sid in live_moves:
            old_loc = _locate_turn(loaded, turn_obj)
            if old_loc is None:
                continue
            old_file_idx, old_turn_idx = old_loc
            target = file_by_top_stage_id.get(new_sid)
            if target is None:
                continue  # 対応 stage_*.json なし (validation で弾く想定だが防御)
            new_file_idx, new_kind = target
            old_kind = loaded[old_file_idx][2]
            if new_kind != old_kind:
                continue  # kind 不整合
            if new_file_idx == old_file_idx:
                continue  # 同ファイル → 移動不要

            entry = loaded[old_file_idx][1]["タイムテーブル"].pop(old_turn_idx)
            loaded[new_file_idx][1]["タイムテーブル"].append(entry)
            changed_file_idxs.add(old_file_idx)
            changed_file_idxs.add(new_file_idx)

        # 親移動後は turn_idx がずれるため booth_index / parent_index を再構築。
        # (live 側の値更新は turn_obj 参照で行うため parent_index は
        #  tk_moves の新親引き当てにのみ使用する)
        parent_index, booth_index = _build_live_indices(loaded)

    # 5. Phase 4: 特典会要素の親付け替え (対応出番ID 変更)
    for booth_id, _row, new_parent_id in tk_moves:
        old_loc = booth_index.get(booth_id)
        new_parent_locs = parent_index.get(new_parent_id)
        if old_loc is None or not new_parent_locs:
            continue
        old_file_idx, old_turn_idx, old_tk_idx = old_loc
        # コラボ親 (同一出番ID 複数エントリ) の場合は先頭エントリへ付け替える
        new_file_idx, new_turn_idx = new_parent_locs[0]

        old_turn = loaded[old_file_idx][1]["タイムテーブル"][old_turn_idx]
        old_tk_list = old_turn.get("特典会", [])
        if old_tk_idx >= len(old_tk_list):
            continue
        tk_element = old_tk_list.pop(old_tk_idx)
        changed_file_idxs.add(old_file_idx)
        _shift_booth_index_after_removal(
            booth_index, old_file_idx, old_turn_idx, old_tk_idx,
        )

        tk_element["対応出番ID"] = int(new_parent_id)

        new_turn = loaded[new_file_idx][1]["タイムテーブル"][new_turn_idx]
        new_tk_list = new_turn.setdefault("特典会", [])
        new_tk_list.append(tk_element)
        new_tk_idx = len(new_tk_list) - 1
        booth_index[booth_id] = (new_file_idx, new_turn_idx, new_tk_idx)
        changed_file_idxs.add(new_file_idx)

    # 6. ライブ親エントリの値更新 (移動済を含む)
    #    turn_obj 参照に直接書き込むため、移動で turn_idx がずれても安全。
    #    変更ファイルは移動後の現在位置 (id->file マップ) から判定する。
    turn_file_map: dict[int, int] = {}
    for fi, (_jp, data, _kind) in enumerate(loaded):
        for turn in data.get("タイムテーブル", []):
            turn_file_map[id(turn)] = fi

    for turn_obj, row in live_value_updates:
        if _update_live_entry_fields(turn_obj, row):
            fi = turn_file_map.get(id(turn_obj))
            if fi is not None:
                changed_file_idxs.add(fi)
    for turn_obj, row, _new_sid in live_moves:
        if _update_live_entry_fields(turn_obj, row):
            fi = turn_file_map.get(id(turn_obj))
            if fi is not None:
                changed_file_idxs.add(fi)

    # 7. 特典会要素の値更新 (booth 別ステージID 含む) - 移動済要素も
    for booth_id, row in tk_value_updates:
        loc = booth_index.get(booth_id)
        if loc is None:
            continue
        file_idx, turn_idx, tk_idx = loc
        tk = loaded[file_idx][1]["タイムテーブル"][turn_idx]["特典会"][tk_idx]
        if _update_tk_fields(tk, row):
            changed_file_idxs.add(file_idx)
    for booth_id, row, _new_parent in tk_moves:
        loc = booth_index.get(booth_id)
        if loc is None:
            continue
        file_idx, turn_idx, tk_idx = loc
        tk = loaded[file_idx][1]["タイムテーブル"][turn_idx]["特典会"][tk_idx]
        if _update_tk_fields(tk, row):
            changed_file_idxs.add(file_idx)

    # 8. 変更ファイルだけ書き戻し
    for file_idx in changed_file_idxs:
        _write_json(loaded[file_idx][0], loaded[file_idx][1])


# ---------------------------------------------------------------------------
# バリデーション
# ---------------------------------------------------------------------------

def validate_stage_master_edits(
    df_stage: pd.DataFrame,
) -> list[str]:
    """ステージマスタ編集結果のバリデーション。エラーメッセージのリストを返す(空ならOK)。"""
    errors: list[str] = []
    names = df_stage["ステージ名"].fillna("").astype(str).str.strip()
    if (names == "").any():
        errors.append("ステージ名が空のステージがあります。")
    if "非活性化フラグ" in df_stage.columns:
        active = df_stage[~df_stage["非活性化フラグ"].fillna(False).astype(bool)]
    else:
        active = df_stage
    active_names = active["ステージ名"].fillna("").astype(str).str.strip()
    dups = active_names[active_names.duplicated() & (active_names != "")]
    if len(dups) > 0:
        errors.append(
            "ステージ名が重複しています: " + ", ".join(sorted(set(dups.tolist())))
        )
    return errors


def validate_idolname_master_edits(
    df_idolname: pd.DataFrame,
) -> list[str]:
    """グループマスタ編集結果のバリデーション。"""
    errors: list[str] = []
    names = df_idolname["グループ名_採用"].fillna("").astype(str).str.strip()
    if (names == "").any():
        errors.append("グループ名_採用 が空のグループがあります。")
    nonempty = names[names != ""]
    dups = nonempty[nonempty.duplicated()]
    if len(dups) > 0:
        errors.append(
            "グループ名_採用 が重複しています: " + ", ".join(sorted(set(dups.tolist())))
        )
    return errors


def validate_live_master_edits(
    df_live: pd.DataFrame,
    df_idolname: pd.DataFrame,
    df_live_original: pd.DataFrame | None = None,
    df_stage: pd.DataFrame | None = None,
) -> list[str]:
    """出番マスタ編集結果のバリデーション。

    Phase 5: ステージID の変更を許容。新しいステージIDが
        - ステージマスタに存在し、かつ
        - 行の `特典会フラグ` と一致する `特典会フラグ` を持つこと
        を要求する。ファイル間移動の実体検証は `_propagate_live_edits_to_json`
        側で kind 整合まで含めて行う。
    """
    errors: list[str] = []
    if df_live is None or len(df_live) == 0:
        return errors

    # HH:MM 形式
    from_series = df_live["ライブ_from"].fillna("").astype(str)
    bad_from_mask = ~from_series.str.match(_HHMM_RE.pattern)
    if bad_from_mask.any():
        bad_ids = df_live.index[bad_from_mask].tolist()
        errors.append(f"ライブ_from が HH:MM 形式でない行: 出番ID={bad_ids}")

    # 正の整数
    dur = pd.to_numeric(df_live["ライブ_長さ(分)"], errors="coerce")
    invalid_dur = dur.isna() | (dur <= 0)
    if invalid_dur.any():
        bad_ids = df_live.index[invalid_dur].tolist()
        errors.append(f"ライブ_長さ(分) は正の整数で入力してください: 出番ID={bad_ids}")

    # グループID がマスタに存在
    valid_gids = set(int(i) for i in df_idolname.index)
    gid_series = pd.to_numeric(df_live["グループID"], errors="coerce")
    gid_invalid_mask = gid_series.isna() | ~gid_series.astype("Int64").isin(valid_gids)
    if gid_invalid_mask.any():
        bad_ids = df_live.index[gid_invalid_mask].tolist()
        errors.append(f"未登録の グループID が選択されています: 出番ID={bad_ids}")

    # 対応出番ID (特典会行のみ): 同イベント内の存在するライブ行 出番ID を指すこと
    # NULL は非heiki tokutenkai 行 (親なし) を想定しスキップする。
    if "対応出番ID" in df_live.columns and "特典会フラグ" in df_live.columns:
        tk_mask = df_live["特典会フラグ"].fillna(False).astype(bool)
        live_ids = set(int(i) for i in df_live.index[~tk_mask])
        tk_view = df_live[tk_mask]
        corresp = pd.to_numeric(tk_view["対応出番ID"], errors="coerce")
        non_null = corresp.notna()
        corresp_invalid = non_null & ~corresp.astype("Int64").isin(live_ids)
        if corresp_invalid.any():
            bad_ids = tk_view.index[corresp_invalid].tolist()
            errors.append(
                "対応出番ID が無効です (同一イベントの ライブ行 出番ID を指定): "
                f"出番ID={bad_ids}"
            )

    # ステージID: マスタに存在 + 特典会フラグ整合
    if df_stage is not None and "ステージID" in df_live.columns:
        valid_sids = set(int(i) for i in df_stage.index)
        sid_series = pd.to_numeric(df_live["ステージID"], errors="coerce")
        sid_invalid_mask = sid_series.isna() | ~sid_series.astype("Int64").isin(valid_sids)
        if sid_invalid_mask.any():
            bad_ids = df_live.index[sid_invalid_mask].tolist()
            errors.append(f"未登録の ステージID が選択されています: 出番ID={bad_ids}")

        if "特典会フラグ" in df_live.columns and "特典会フラグ" in df_stage.columns:
            stage_tk_flag = df_stage["特典会フラグ"].fillna(False).astype(bool).to_dict()
            row_tk_flag = df_live["特典会フラグ"].fillna(False).astype(bool)
            # df_live.index (=出番ID) は NaN / 重複の可能性があるためラベル参照を避け、
            # 位置インデックスで走査する。
            mismatch_positions: list[int] = []
            for pos, (sid, tk_flag) in enumerate(zip(sid_series, row_tk_flag)):
                if pd.isna(sid):
                    continue
                sid_int = int(sid)
                if sid_int not in stage_tk_flag:
                    continue
                if bool(stage_tk_flag[sid_int]) != bool(tk_flag):
                    mismatch_positions.append(pos)
            if mismatch_positions:
                bad_ids = df_live.index[mismatch_positions].tolist()
                errors.append(
                    "ステージID の 特典会フラグ がライブ/特典会の区分と一致しません: "
                    f"出番ID={bad_ids}"
                )

    return errors


# ---------------------------------------------------------------------------
# 保存エントリポイント
# ---------------------------------------------------------------------------

def save_event_edits(
    pj_path: str,
    event_name: str,
    event_no: int,
    project_info_json: dict,
    edits: dict[str, pd.DataFrame],
) -> None:
    """編集結果を master_*.csv と stage_*.json に書き戻す。

    `edits` は以下のキーを持ち得る:
        - "stage"    : ステージマスタ (Phase 2)
        - "idolname" : グループマスタ (Phase 3)
        - "live"     : 出番マスタ     (Phase 4)
    どれも optional だが、最低 1 つは必要。
    """
    output_path = os.path.join(pj_path, event_name)
    os.makedirs(output_path, exist_ok=True)

    # --- ステージマスタ ---
    if "stage" in edits and edits["stage"] is not None:
        df_stage = edits["stage"].copy()
        if "表示順" in df_stage.columns:
            df_stage = df_stage.sort_values("表示順")
            df_stage["表示順"] = range(len(df_stage))
        df_stage.to_csv(os.path.join(output_path, "master_stage.csv"))
        _propagate_stage_master_to_project(
            pj_path, event_name, event_no, project_info_json, df_stage,
        )

    # --- グループマスタ ---
    if "idolname" in edits and edits["idolname"] is not None:
        df_idolname = edits["idolname"].copy()
        df_idolname.to_csv(os.path.join(output_path, "master_idolname.csv"))
        _propagate_idolname_master_to_json(
            pj_path, event_name, event_no, project_info_json, df_idolname,
        )

    # --- 出番マスタ ---
    if "live" in edits and edits["live"] is not None:
        df_live = edits["live"].copy()
        # ライブ_to を ライブ_from + ライブ_長さ(分) から一括再計算
        # (UI 上は read-only だが、編集後の CSV と JSON 双方で同じ値を持たせるため
        # ここで 1 回だけ計算し、以降のラインで使い回す)
        df_live["ライブ_to"] = [
            _add_minutes(f, d)
            for f, d in zip(df_live["ライブ_from"], df_live["ライブ_長さ(分)"])
        ]
        # グループID 変更行は グループ名_採用 (= "グループ名" カラム) を idolname から再導出
        if "idolname" in edits and edits["idolname"] is not None \
                and "グループ名" in df_live.columns and "グループID" in df_live.columns:
            gid_to_name = _idolname_id_to_name_map(edits["idolname"])

            def _resolve_adopted(gid, current_name):
                if gid is None or (isinstance(gid, float) and pd.isna(gid)):
                    return current_name
                try:
                    return gid_to_name.get(int(gid), current_name)
                except (ValueError, TypeError):
                    return current_name

            df_live["グループ名"] = [
                _resolve_adopted(g, cur)
                for g, cur in zip(df_live["グループID"], df_live["グループ名"])
            ]
        # ステージID 変更行は ステージ名 を stage マスタから再導出 (Phase 5)
        if "stage" in edits and edits["stage"] is not None \
                and "ステージ名" in df_live.columns and "ステージID" in df_live.columns:
            sid_to_name = _stage_id_to_name_map(edits["stage"])

            def _resolve_stage_name(sid, current_name):
                if sid is None or (isinstance(sid, float) and pd.isna(sid)):
                    return current_name
                try:
                    return sid_to_name.get(int(sid), current_name)
                except (ValueError, TypeError):
                    return current_name

            df_live["ステージ名"] = [
                _resolve_stage_name(s, cur)
                for s, cur in zip(df_live["ステージID"], df_live["ステージ名"])
            ]
        df_live.to_csv(os.path.join(output_path, "turn_id_data.csv"))
        _propagate_live_edits_to_json(
            pj_path, event_name, event_no, project_info_json, df_live,
        )

    # project_info を永続化 (ステージ名同期等を反映)
    repo.save_project_json(pj_path, project_info_json)
