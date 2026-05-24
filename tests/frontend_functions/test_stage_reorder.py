"""stage_reorder.py: ⑥編集モードの ステージマスタ D&D ヘルパのテスト"""

import pandas as pd

from frontend_functions import stage_reorder


def _stage_df(rows):
    """rows = [(id, name, tok, order, disabled), ...]"""
    return pd.DataFrame(
        {
            "ステージ名": [r[1] for r in rows],
            "特典会フラグ": [r[2] for r in rows],
            "表示順": [r[3] for r in rows],
            "非活性化フラグ": [r[4] for r in rows],
        },
        index=pd.Index([r[0] for r in rows], name="ステージID"),
    )


# ---------------------------------------------------------------------------
# make_stage_dnd_label
# ---------------------------------------------------------------------------

def test_make_stage_dnd_label_active():
    df = _stage_df([(0, "きら", False, 0, False)])
    label = stage_reorder.make_stage_dnd_label(
        0, df.loc[0], {0: "ライブ"},
    )
    assert label == "[ライブ] ID:0  きら"


def test_make_stage_dnd_label_disabled_strike():
    df = _stage_df([(2, "削除済", False, 0, True)])
    label = stage_reorder.make_stage_dnd_label(
        2, df.loc[2], {2: "ライブ"},
    )
    # 打ち消し線でラップされる
    assert label == "~~[ライブ] ID:2  削除済~~"


def test_make_stage_dnd_label_unknown_kind():
    """kind_map に無い stage_id は [不明] を付与"""
    df = _stage_df([(7, "謎", False, 0, False)])
    label = stage_reorder.make_stage_dnd_label(7, df.loc[7], {})
    assert label == "[不明] ID:7  謎"


# ---------------------------------------------------------------------------
# apply_stage_reorder
# ---------------------------------------------------------------------------

def test_apply_stage_reorder_basic():
    """D&D の戻り値順に従って 表示順 が 0..N-1 で振り直される"""
    df = _stage_df([
        (0, "A", False, 0, False),
        (1, "B", False, 1, False),
        (2, "C", False, 2, False),
    ])
    # [C, A, B] の順に並べ直す
    stage_reorder.apply_stage_reorder(df, [2, 0, 1])
    assert df.at[2, "表示順"] == 0
    assert df.at[0, "表示順"] == 1
    assert df.at[1, "表示順"] == 2


def test_apply_stage_reorder_idempotent_same_order():
    """同じ順序を渡しても 表示順 は 0..N-1 で再正規化される"""
    df = _stage_df([
        (0, "A", False, 5, False),
        (1, "B", False, 9, False),
    ])
    stage_reorder.apply_stage_reorder(df, [0, 1])
    assert df.at[0, "表示順"] == 0
    assert df.at[1, "表示順"] == 1


def test_apply_stage_reorder_preserves_other_columns():
    """並び替え後も 他の列 (ステージ名/非活性化フラグ 等) は不変"""
    df = _stage_df([
        (0, "A", False, 0, False),
        (1, "B", True, 1, True),
    ])
    stage_reorder.apply_stage_reorder(df, [1, 0])
    assert df.at[0, "ステージ名"] == "A"
    assert df.at[1, "ステージ名"] == "B"
    assert df.at[1, "非活性化フラグ"] is True or df.at[1, "非活性化フラグ"] == True  # noqa: E712
    assert df.at[1, "特典会フラグ"] is True or df.at[1, "特典会フラグ"] == True  # noqa: E712
