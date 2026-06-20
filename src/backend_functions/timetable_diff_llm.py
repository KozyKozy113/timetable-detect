"""⑤変更比較：LLMによるタイムテーブル変更案の生成・反映。

- build_llm_timetable_input: stage_n.json をLLM入力用に整形（ID系除外・連番付与）
- detect_with_tokutenkai: 特典会併記型か否かの判定
- propose_changes: 既存/新規の切り出し画像2枚＋既存タイテ＋既知グループ名から変更案を生成
- apply_proposal_to_stage_json: 人手編集後の解決済み操作を stage_n.json に反映（純粋関数）

画像差し替え／CSV更新／可視化再生成などのファイル副作用は workflow 側で行う。
本モジュールの apply_* は I/O を持たない純粋関数として単体テスト可能に保つ。
"""
import os
import json
import copy

from backend_functions import gpt_ocr

DIR_PATH = os.path.dirname(__file__)
_PROMPT_DIR = os.path.join(DIR_PATH, "..", "prompt_system")

# IDポリシー用センチネル（現状維持）
KEEP = "keep"

# LLM入力から除外するID系キー
_ID_KEYS = ("出番ID", "グループID", "ステージID", "対応出番ID", "コラボステージID")


# ---------------------------------------------------------------------------
# 入力整形・形式判定
# ---------------------------------------------------------------------------

def build_llm_timetable_input(stage_json: dict) -> dict:
    """stage_json の "タイムテーブル" 配列をLLM入力用の軽量JSONに整形する。

    - 各レコードに「連番」（配列index, 0始まり）を付与
    - グループ名は「グループ名_採用」があればそれのみ、無ければ「グループ名」
    - ライブステージ {from,to} / 特典会 [{連番,from,to,ブース}] を残す
    - ID系キー（出番ID/グループID/ステージID/対応出番ID/コラボステージID）は除外
    """
    timetable = stage_json.get("タイムテーブル", []) or []
    result = []
    for idx, rec in enumerate(timetable):
        item = {"連番": idx}

        adopted = rec.get("グループ名_採用")
        if adopted not in (None, ""):
            item["グループ名"] = adopted
        elif "グループ名" in rec:
            item["グループ名"] = rec.get("グループ名")

        live = rec.get("ライブステージ")
        if live:
            item["ライブステージ"] = {"from": live.get("from"), "to": live.get("to")}

        if "特典会" in rec:
            tk_list = []
            for j, tk in enumerate(rec.get("特典会") or []):
                tk_list.append({
                    "連番": j,
                    "from": tk.get("from"),
                    "to": tk.get("to"),
                    "ブース": tk.get("ブース"),
                })
            item["特典会"] = tk_list

        result.append(item)
    return {"タイムテーブル": result}


def detect_with_tokutenkai(stage_json: dict, stage_entry: dict = None) -> bool:
    """特典会併記型か否かを判定する。

    stage_entry（project_info の stage_list 要素）の kind を第一に、
    無ければレコードに「特典会」キーが存在するかで判定する。
    """
    if stage_entry is not None:
        kind = stage_entry.get("kind")
        if kind == "live_tokutenkai_heiki":
            return True
        if kind in ("live", "tokutenkai"):
            return False
    for rec in stage_json.get("タイムテーブル", []) or []:
        if "特典会" in rec:
            return True
    return False


# ---------------------------------------------------------------------------
# 変更案の生成（LLM）
# ---------------------------------------------------------------------------

def _load_prompt(with_tokutenkai: bool) -> str:
    with open(os.path.join(_PROMPT_DIR, "fes_timetable_diff_propose_common.txt"), "r", encoding="utf-8") as f:
        prompt = f.read()
    if with_tokutenkai:
        with open(os.path.join(_PROMPT_DIR, "fes_timetable_diff_propose_tokutenkai.txt"), "r", encoding="utf-8") as f:
            prompt += "\n" + f.read()
    return prompt


def build_proposal_user_prompt(stage_json: dict, known_groups) -> str:
    """LLMへ渡すユーザープロンプト（既存タイテJSON＋既知グループ名一覧）を組み立てる。"""
    llm_input = build_llm_timetable_input(stage_json)
    known_groups_text = (
        "\n".join(f"- {g}" for g in known_groups) if known_groups else "（情報なし）"
    )
    return (
        "以下は【既存】画像から作成済みの現在のタイムテーブルデータ（JSON）です。\n"
        + json.dumps(llm_input, ensure_ascii=False, indent=2)
        + "\n\n#既知グループ名一覧\n"
        + known_groups_text
        + "\n\n上記をふまえ、【既存】を【新規】画像に一致させるための変更案を、"
        + "指定のスキーマに従って出力してください。変更が無ければ操作は空配列で構いません。"
    )


def propose_changes(old_crop, new_crop, stage_json: dict, known_groups, with_tokutenkai: bool) -> dict:
    """LLMに変更案を生成させ、structured output（dict）を返す。"""
    prompt_system = _load_prompt(with_tokutenkai)
    prompt_user = build_proposal_user_prompt(stage_json, known_groups)
    json_format = gpt_ocr.build_change_proposal_schema(with_tokutenkai)
    return gpt_ocr.get_change_proposal(old_crop, new_crop, prompt_user, prompt_system, json_format)


# ---------------------------------------------------------------------------
# 変更案の反映（純粋関数・I/Oなし）
# ---------------------------------------------------------------------------
#
# apply_proposal_to_stage_json が受け取る「解決済み操作（resolved_ops）」は、
# UI（app.py）でID方針・マスタ照合・パターン別デフォルトを解決した後の形。
# 各操作 dict の構造:
#   共通:
#     種別           : "変更" | "削除" | "追加"
#     対象連番       : int（変更・削除時。追加は -1）
#   変更:
#     set_グループ名_raw  : str | None    （生グループ名の更新。None/空 は更新なし）
#     set_グループ名_採用 : str | None    （None/空 は変更なし。needs_correction時は無視）
#     needs_correction    : bool          （Trueなら採用名を空にし、後段の補正ステップで生成）
#     set_ライブ時間      : {"from","to"} | None
#     set_出番ID          : KEEP | None | int  （KEEP=現状維持, None=NULL化, int=手入力）
#     set_グループID      : KEEP | None | int
#     特典会操作          : [ ... ]（併記型のみ）
#   削除:
#     削除種別       : "全体" | "特典会のみ"
#     特典会連番     : int（削除種別="特典会のみ"時）
#   追加:
#     set_グループ名_raw  : str           （生グループ名）
#     set_グループ名_採用 : str | None     （マスタ一致時の採用名。needs_correction時はNone）
#     needs_correction    : bool
#     set_ライブ時間      : {"from","to"}
#     set_出番ID / set_グループID : None | int
#     特典会操作          : [{"操作種別":"追加", ...}]（併記型のみ）
#
#   特典会操作 item:
#     操作種別       : "変更" | "追加" | "削除"
#     対象特典会連番 : int（変更・削除時。追加は -1）
#     ブース / from / to : 値
# ---------------------------------------------------------------------------

def apply_proposal_to_stage_json(stage_json: dict, resolved_ops: list, with_tokutenkai: bool) -> dict:
    """解決済み操作を stage_json に反映した新しい dict を返す（元 dict は変更しない）。"""
    new_json = copy.deepcopy(stage_json)
    timetable = new_json.get("タイムテーブル", []) or []

    delete_whole = set()
    tokutenkai_only_deletes = {}  # idx -> [特典会連番, ...]
    additions = []

    for op in resolved_ops or []:
        kind = op.get("種別")
        if kind == "変更":
            idx = op.get("対象連番")
            if idx is None or not (0 <= idx < len(timetable)):
                continue
            _apply_change_to_record(timetable[idx], op, with_tokutenkai)
        elif kind == "削除":
            idx = op.get("対象連番")
            if idx is None or not (0 <= idx < len(timetable)):
                continue
            if op.get("削除種別") == "特典会のみ":
                tokutenkai_only_deletes.setdefault(idx, []).append(op.get("特典会連番"))
            else:
                delete_whole.add(idx)
        elif kind == "追加":
            additions.append(_build_added_record(op, with_tokutenkai))

    # 特典会のみ削除（同一レコード内は降順で処理しindexずれを防ぐ）
    for idx, tk_indices in tokutenkai_only_deletes.items():
        rec = timetable[idx]
        tk_list = rec.get("特典会") or []
        for j in sorted([t for t in tk_indices if t is not None], reverse=True):
            if 0 <= j < len(tk_list):
                tk_list.pop(j)
        rec["特典会"] = tk_list

    # 全体削除を除外して再構築 ＋ 追加レコード
    new_timetable = [rec for i, rec in enumerate(timetable) if i not in delete_whole]
    new_timetable.extend(additions)
    new_json["タイムテーブル"] = new_timetable
    return new_json


def _apply_id_policy(rec: dict, key: str, policy) -> None:
    """IDポリシーを適用。KEEP=現状維持、None=NULL化、int=指定値。"""
    if policy == KEEP:
        return
    rec[key] = policy  # None または int


def _apply_change_to_record(rec: dict, op: dict, with_tokutenkai: bool) -> None:
    if op.get("set_グループ名_raw") not in (None, ""):
        rec["グループ名"] = op["set_グループ名_raw"]
    if op.get("needs_correction"):
        # 採用名は後段の補正ステップ（fill_empty_adopted_names）で生成するため空にする
        rec["グループ名_採用"] = ""
    elif op.get("set_グループ名_採用") not in (None, ""):
        rec["グループ名_採用"] = op["set_グループ名_採用"]

    lt = op.get("set_ライブ時間")
    if lt and (lt.get("from") or lt.get("to")):
        live = rec.get("ライブステージ") or {}
        if lt.get("from"):
            live["from"] = lt["from"]
        if lt.get("to"):
            live["to"] = lt["to"]
        rec["ライブステージ"] = live

    set_turn = op.get("set_出番ID", KEEP)
    _apply_id_policy(rec, "出番ID", set_turn)
    _apply_id_policy(rec, "グループID", op.get("set_グループID", KEEP))

    # 出番ID再採番（NULL化）時は併記の子特典会IDもNULL化
    if with_tokutenkai and set_turn is None:
        for tk in rec.get("特典会") or []:
            tk["出番ID"] = None
            tk["ステージID"] = None

    if with_tokutenkai:
        for tkop in op.get("特典会操作") or []:
            _apply_tokutenkai_op(rec, tkop)


def _apply_tokutenkai_op(rec: dict, tkop: dict) -> None:
    tk_list = rec.get("特典会")
    if tk_list is None:
        tk_list = []
        rec["特典会"] = tk_list
    sub = tkop.get("操作種別")
    if sub == "変更":
        j = tkop.get("対象特典会連番")
        if j is not None and 0 <= j < len(tk_list):
            tk = tk_list[j]
            if tkop.get("ブース") not in (None, ""):
                tk["ブース"] = tkop["ブース"]
            if tkop.get("from"):
                tk["from"] = tkop["from"]
            if tkop.get("to"):
                tk["to"] = tkop["to"]
            # 内容が変わるため子IDはNULL化（⑥で再採番）
            tk["出番ID"] = None
            tk["ステージID"] = None
    elif sub == "追加":
        tk_list.append({
            "from": tkop.get("from"),
            "to": tkop.get("to"),
            "ブース": tkop.get("ブース"),
            "出番ID": None,
            "ステージID": None,
            "対応出番ID": None,
        })
    elif sub == "削除":
        j = tkop.get("対象特典会連番")
        if j is not None and 0 <= j < len(tk_list):
            tk_list.pop(j)


def _build_added_record(op: dict, with_tokutenkai: bool) -> dict:
    raw = op.get("set_グループ名_raw") or op.get("set_グループ名_採用") or op.get("グループ名") or ""
    if op.get("needs_correction"):
        adopted = ""   # 後段の補正ステップで生成
    else:
        adopted = op.get("set_グループ名_採用") or raw
    lt = op.get("set_ライブ時間") or {}
    turn = op.get("set_出番ID")
    group = op.get("set_グループID")

    rec = {
        "グループ名": raw,
        "ライブステージ": {"from": lt.get("from"), "to": lt.get("to")},
    }
    if with_tokutenkai:
        tk_list = []
        for tkop in op.get("特典会操作") or []:
            if tkop.get("操作種別") == "追加":
                tk_list.append({
                    "from": tkop.get("from"),
                    "to": tkop.get("to"),
                    "ブース": tkop.get("ブース"),
                    "出番ID": None,
                    "ステージID": None,
                    "対応出番ID": None,
                })
        rec["特典会"] = tk_list

    rec["グループ名_採用"] = adopted
    rec["出番ID"] = turn if isinstance(turn, int) else None
    rec["グループID"] = group if isinstance(group, int) else None
    return rec
