"""⑥編集モードの ステージマスタ D&D 並び替え用の純関数ヘルパ。

Streamlit / アプリ状態に依存しない。app.py から薄く呼び出して使う。
"""

from __future__ import annotations

import pandas as pd


def make_stage_dnd_label(
    stage_id, row: pd.Series, kind_map: dict[int, str],
) -> str:
    """D&D アイテムのラベル文字列を返す。

    フォーマット: `[{種別名}] ID:{stage_id}  {ステージ名}`
    - 種別名は kind_map から (無ければ "不明")
    - 非活性化ステージは `~~ラベル~~` で打ち消し線
    - stage_id をユニーク識別に含めるため、同名ステージでも区別可能
    """
    try:
        sid_int = int(stage_id)
    except (ValueError, TypeError):
        sid_int = stage_id
    kind = kind_map.get(sid_int, "不明")
    base = f"[{kind}] ID:{stage_id}  {row['ステージ名']}"
    if bool(row.get("非活性化フラグ", False)):
        base = f"~~{base}~~"
    return base


def apply_stage_reorder(
    df_stage: pd.DataFrame, new_order_stage_ids: list,
) -> None:
    """新しい stage_id 順に従って `表示順` 列を 0..N-1 で振り直す (in-place)。

    new_order_stage_ids に含まれない既存 ID は変更されないため、
    呼び出し側で全 stage_id を渡すこと。
    """
    for new_pos, sid in enumerate(new_order_stage_ids):
        df_stage.at[sid, "表示順"] = int(new_pos)
