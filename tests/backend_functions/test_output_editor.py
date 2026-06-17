"""output_editor.py: ⑥出力確認・編集 編集モードの保存ロジックのテスト"""

import json
import os

import pandas as pd
import pytest

from backend_functions import output_editor as _editor
from backend_functions import output_builder as _output


# ---------------------------------------------------------------------------
# validate_stage_master_edits
# ---------------------------------------------------------------------------

def _stage_df(rows):
    """rows = [(id, name, tok, order, disabled), ...]"""
    df = pd.DataFrame(
        {
            "ステージ名": [r[1] for r in rows],
            "特典会フラグ": [r[2] for r in rows],
            "表示順": [r[3] for r in rows],
            "非活性化フラグ": [r[4] for r in rows],
        },
        index=pd.Index([r[0] for r in rows], name="ステージID"),
    )
    return df


def test_validate_ok():
    df = _stage_df([
        (0, "A", False, 0, False),
        (1, "B", True, 1, False),
    ])
    assert _editor.validate_stage_master_edits(df) == []


def test_validate_empty_name():
    df = _stage_df([
        (0, "", False, 0, False),
        (1, "B", True, 1, False),
    ])
    errs = _editor.validate_stage_master_edits(df)
    assert len(errs) == 1
    assert "空" in errs[0]


def test_validate_duplicate_name_among_active():
    df = _stage_df([
        (0, "Same", False, 0, False),
        (1, "Same", False, 1, False),
    ])
    errs = _editor.validate_stage_master_edits(df)
    assert len(errs) == 1
    assert "重複" in errs[0]


def test_validate_duplicate_name_skipped_for_disabled():
    """非活性化ステージとの重複はバリデーション対象外"""
    df = _stage_df([
        (0, "Same", False, 0, False),
        (1, "Same", False, 1, True),     # 非活性化
    ])
    assert _editor.validate_stage_master_edits(df) == []


# ---------------------------------------------------------------------------
# save_event_edits: master_stage.csv 書き戻し + 表示順の正規化
# ---------------------------------------------------------------------------

def _setup_minimal_project(tmp_path):
    """1イベント / 1画像 / 2ステージ + IDマスタ確定済み の最小プロジェクト"""
    pj_path = tmp_path / "pj"
    event_dir = pj_path / "event_1" / "ライブ"
    os.makedirs(event_dir)

    # stage_0.json (トップレベル ステージID = 0)
    with open(event_dir / "stage_0.json", "w", encoding="utf-8") as f:
        json.dump({
            "ステージ名": "OldA",
            "ステージID": 0,
            "タイムテーブル": [
                {
                    "グループ名": "X", "グループ名_採用": "X",
                    "出番ID": 0, "グループID": 0,
                    "ライブステージ": {"from": "10:00", "to": "10:30"},
                    "備考": "",
                },
            ],
        }, f, ensure_ascii=False)
    # stage_1.json (トップレベル ステージID = 1)
    with open(event_dir / "stage_1.json", "w", encoding="utf-8") as f:
        json.dump({
            "ステージ名": "OldB",
            "ステージID": 1,
            "タイムテーブル": [
                {
                    "グループ名": "Y", "グループ名_採用": "Y",
                    "出番ID": 1, "グループID": 1,
                    "ライブステージ": {"from": "11:00", "to": "11:30"},
                    "備考": "",
                },
            ],
        }, f, ensure_ascii=False)

    project_info = {
        "project_name": "pj",
        "event_num": 1,
        "event_detail": [
            {
                "event_no": 0,
                "event_name": "event_1",
                "timetables": [
                    {
                        "image_no": 0,
                        "dir_name": "ライブ",
                        "display_name": "ライブ",
                        "kind": "live",
                        "stage_num": 2,
                        "stage_list": [
                            {"stage_no": 0, "stage_name": "OldA",
                             "kind": "live", "stage_id": 0},
                            {"stage_no": 1, "stage_name": "OldB",
                             "kind": "live", "stage_id": 1},
                        ],
                    }
                ],
            }
        ],
    }
    # project_info.json も書いておく(save_project_json が読み書きするため)
    with open(pj_path / "project_info.json", "w", encoding="utf-8") as f:
        json.dump(project_info, f, ensure_ascii=False)
    return str(pj_path), project_info


def test_save_event_edits_writes_master_stage_csv(tmp_path):
    pj_path, project_info = _setup_minimal_project(tmp_path)
    df_stage = _stage_df([
        (0, "OldA", False, 0, False),
        (1, "OldB", False, 1, False),
    ])
    _editor.save_event_edits(
        pj_path, "event_1", 0, project_info,
        edits={"stage": df_stage},
    )

    csv_path = os.path.join(pj_path, "event_1", "master_stage.csv")
    assert os.path.exists(csv_path)
    loaded = pd.read_csv(csv_path, index_col=0)
    assert "表示順" in loaded.columns
    assert "非活性化フラグ" in loaded.columns


def test_save_event_edits_normalizes_display_order(tmp_path):
    """表示順は 0 始まりの連番に正規化される (欠番や非連番を埋める)"""
    pj_path, project_info = _setup_minimal_project(tmp_path)
    df_stage = _stage_df([
        (0, "OldA", False, 5, False),     # 5
        (1, "OldB", False, 9, False),     # 9 → 0 始まり連番化されて 0, 1 になる
    ])
    _editor.save_event_edits(
        pj_path, "event_1", 0, project_info, edits={"stage": df_stage},
    )

    loaded = pd.read_csv(
        os.path.join(pj_path, "event_1", "master_stage.csv"), index_col=0,
    )
    # 表示順は 0, 1 に正規化
    assert sorted(loaded["表示順"].tolist()) == [0, 1]


def test_save_event_edits_propagates_stage_name_to_json_and_project_info(tmp_path):
    """ステージ名変更が stage_*.json トップレベルと project_info.stage_list に伝播する"""
    pj_path, project_info = _setup_minimal_project(tmp_path)
    df_stage = _stage_df([
        (0, "NewA", False, 0, False),     # OldA → NewA
        (1, "OldB", False, 1, False),
    ])
    _editor.save_event_edits(
        pj_path, "event_1", 0, project_info, edits={"stage": df_stage},
    )

    # stage_0.json のトップレベル ステージ名 が NewA に
    with open(os.path.join(pj_path, "event_1", "ライブ", "stage_0.json"),
              encoding="utf-8") as f:
        data = json.load(f)
    assert data["ステージ名"] == "NewA"

    # project_info.stage_list[0].stage_name も NewA に
    stage = project_info["event_detail"][0]["timetables"][0]["stage_list"][0]
    assert stage["stage_name"] == "NewA"
    # project_info.json も書き換わっている
    with open(os.path.join(pj_path, "project_info.json"), encoding="utf-8") as f:
        loaded_info = json.load(f)
    assert loaded_info["event_detail"][0]["timetables"][0]["stage_list"][0]["stage_name"] == "NewA"


# ---------------------------------------------------------------------------
# build_event_output: 非活性化ステージの除外
# ---------------------------------------------------------------------------

def test_build_event_output_keeps_disabled_stages_in_master(tmp_path):
    """非活性化ステージとそれに紐づく出番は `stage` / `live` (= master 実体) に保持される。
    集計 (duration_distribution / group_count / overlap_alerts / group_appearances) と
    表示・出力 (UI / Excel / Stella JSON) で必要に応じてフィルタする運用に変更。

    これにより `determine_id_master` / `save_event_edits` の再実行で
    非活性化ステージ行が CSV から消失しなくなる。"""
    pj_path, project_info = _setup_minimal_project(tmp_path)
    # master_stage.csv に stage_1 を 非活性化 で保存
    master_stage_path = os.path.join(pj_path, "event_1", "master_stage.csv")
    pd.DataFrame(
        {
            "ステージ名": ["OldA", "OldB"],
            "特典会フラグ": [False, False],
            "表示順": [0, 1],
            "非活性化フラグ": [False, True],
        },
        index=pd.Index([0, 1], name="ステージID"),
    ).to_csv(master_stage_path)

    result = _output.build_event_output(pj_path, "event_1", 0, project_info)
    assert result is not None
    df_stage = result["stage"]
    df_live = result["live"]
    # 非活性化された stage_id=1 も master として保持される (新仕様)
    assert 0 in df_stage.index.tolist()
    assert 1 in df_stage.index.tolist()
    assert bool(df_stage.loc[1, "非活性化フラグ"]) is True
    # 出番側 master にも stage_id=1 (OldB) に紐づく行が保持される
    assert (df_live["ステージID"] == 1).sum() >= 1
    # 集計はステージ非活性化を反映して除外される (グループY は OldB のみ出演)
    group_count = result["group_count"]
    # ステージID=1 の出演しか無いグループは集計に登場しない
    if "グループ名" in group_count.columns:
        assert "Y" not in group_count["グループ名"].tolist()


def test_determine_id_master_is_idempotent_for_disabled_stages(tmp_path):
    """determine_id_master を2回呼んでも非活性化ステージが master_stage.csv から消えない。
    auto-trigger で何度実行されても安全であることを保証する。"""
    pj_path, project_info = _setup_minimal_project(tmp_path)
    # 既存 master_stage.csv に stage_1 を 非活性化 で保存
    master_stage_path = os.path.join(pj_path, "event_1", "master_stage.csv")
    pd.DataFrame(
        {
            "ステージ名": ["OldA", "OldB"],
            "特典会フラグ": [False, False],
            "表示順": [0, 1],
            "非活性化フラグ": [False, True],
        },
        index=pd.Index([0, 1], name="ステージID"),
    ).to_csv(master_stage_path)

    # 初回確定 (build → determine_id_master)
    result1 = _output.build_event_output(pj_path, "event_1", 0, project_info)
    _output.determine_id_master({"event_1": result1}, pj_path, project_info)

    # 2回目: 再ビルド + 再確定
    result2 = _output.build_event_output(pj_path, "event_1", 0, project_info)
    _output.determine_id_master({"event_1": result2}, pj_path, project_info)

    # 確定後の master_stage.csv に非活性化ステージが残っている
    loaded = pd.read_csv(master_stage_path, index_col=0)
    assert 0 in loaded.index.tolist()
    assert 1 in loaded.index.tolist()
    assert bool(loaded.loc[1, "非活性化フラグ"]) is True


# ---------------------------------------------------------------------------
# Phase 3: validate_idolname_master_edits
# ---------------------------------------------------------------------------

def _idolname_df(rows):
    """rows = [(gid, name), ...]"""
    return pd.DataFrame(
        {"グループ名_採用": [r[1] for r in rows]},
        index=pd.Index([r[0] for r in rows], name="グループID"),
    )


def test_validate_idolname_ok():
    assert _editor.validate_idolname_master_edits(
        _idolname_df([(0, "A"), (1, "B")])
    ) == []


def test_validate_idolname_empty_name():
    errs = _editor.validate_idolname_master_edits(
        _idolname_df([(0, ""), (1, "B")])
    )
    assert len(errs) == 1 and "空" in errs[0]


def test_validate_idolname_duplicate():
    errs = _editor.validate_idolname_master_edits(
        _idolname_df([(0, "Same"), (1, "Same")])
    )
    assert len(errs) == 1 and "重複" in errs[0]


# ---------------------------------------------------------------------------
# Phase 4: validate_live_master_edits
# ---------------------------------------------------------------------------

def _live_df(rows):
    """rows = [(turn_id, stage_id, gid, from_str, dur, group_name, stage_name, tk_flag), ...]"""
    df = pd.DataFrame(
        {
            "ステージID":     [r[1] for r in rows],
            "グループID":     [r[2] for r in rows],
            "ライブ_from":    [r[3] for r in rows],
            "ライブ_長さ(分)":[r[4] for r in rows],
            "グループ名_raw": [r[5] for r in rows],
            "グループ名":     [r[5] for r in rows],
            "ステージ名":     [r[6] for r in rows],
            "特典会フラグ":   [r[7] for r in rows],
            "備考":           ["" for _ in rows],
        },
        index=pd.Index([r[0] for r in rows], name="出番ID"),
    )
    return df


def test_validate_live_ok():
    df_live = _live_df([(0, 0, 0, "10:00", 20, "A", "S1", False)])
    df_idol = _idolname_df([(0, "A")])
    assert _editor.validate_live_master_edits(df_live, df_idol) == []


def test_validate_live_bad_from_format():
    df_live = _live_df([(0, 0, 0, "10時00分", 20, "A", "S1", False)])
    df_idol = _idolname_df([(0, "A")])
    errs = _editor.validate_live_master_edits(df_live, df_idol)
    assert any("HH:MM" in e for e in errs)


def test_validate_live_non_positive_duration():
    df_live = _live_df([(0, 0, 0, "10:00", 0, "A", "S1", False)])
    df_idol = _idolname_df([(0, "A")])
    errs = _editor.validate_live_master_edits(df_live, df_idol)
    assert any("正の整数" in e for e in errs)


def test_validate_live_unknown_group_id():
    df_live = _live_df([(0, 0, 99, "10:00", 20, "?", "S1", False)])
    df_idol = _idolname_df([(0, "A")])
    errs = _editor.validate_live_master_edits(df_live, df_idol)
    assert any("グループID" in e for e in errs)


def _stage_master_df(rows):
    """rows = [(stage_id, stage_name, tokutenkai_flag), ...]"""
    return pd.DataFrame(
        {
            "ステージ名":   [r[1] for r in rows],
            "特典会フラグ": [r[2] for r in rows],
            "表示順":       list(range(len(rows))),
            "非活性化フラグ": [False] * len(rows),
        },
        index=pd.Index([r[0] for r in rows], name="ステージID"),
    )


def test_validate_live_accepts_stage_id_change_phase5():
    """Phase 5: ステージID 変更を許容する。"""
    df_live = _live_df([(0, 99, 0, "10:00", 20, "A", "S1", False)])
    df_original = _live_df([(0, 0, 0, "10:00", 20, "A", "S1", False)])
    df_idol = _idolname_df([(0, "A")])
    df_stage = _stage_master_df([(0, "S1", False), (99, "S2", False)])
    errs = _editor.validate_live_master_edits(
        df_live, df_idol, df_original, df_stage=df_stage,
    )
    assert errs == []


def test_validate_live_rejects_unknown_stage_id():
    df_live = _live_df([(0, 99, 0, "10:00", 20, "A", "S1", False)])
    df_idol = _idolname_df([(0, "A")])
    df_stage = _stage_master_df([(0, "S1", False)])
    errs = _editor.validate_live_master_edits(
        df_live, df_idol, df_live_original=None, df_stage=df_stage,
    )
    assert any("ステージID" in e and "未登録" in e for e in errs)


def test_validate_live_rejects_stage_id_tk_flag_mismatch():
    """ライブ行 (特典会フラグ=False) に 特典会ステージID を割り当てるとエラー。"""
    df_live = _live_df([(0, 1, 0, "10:00", 20, "A", "S1", False)])
    df_idol = _idolname_df([(0, "A")])
    df_stage = _stage_master_df([(0, "Live", False), (1, "Booth", True)])
    errs = _editor.validate_live_master_edits(
        df_live, df_idol, df_live_original=None, df_stage=df_stage,
    )
    assert any("特典会フラグ" in e and "区分" in e for e in errs)


# ---------------------------------------------------------------------------
# Phase 3: idolname の グループ名_採用 を stage_*.json タイムテーブル[] に伝播
# ---------------------------------------------------------------------------

def test_save_event_edits_propagates_idolname_to_json(tmp_path):
    pj_path, project_info = _setup_minimal_project(tmp_path)
    # 既存: グループID=0/X, グループID=1/Y → グループID=0 を XNEW にリネーム
    df_idol = _idolname_df([(0, "XNEW"), (1, "Y")])

    _editor.save_event_edits(
        pj_path, "event_1", 0, project_info,
        edits={"idolname": df_idol},
    )

    # stage_0.json の タイムテーブル[0].グループ名_採用 が更新される
    with open(os.path.join(pj_path, "event_1", "ライブ", "stage_0.json"),
              encoding="utf-8") as f:
        data = json.load(f)
    assert data["タイムテーブル"][0]["グループ名_採用"] == "XNEW"
    # 別ステージ (グループID=1/Y) は変更なし
    with open(os.path.join(pj_path, "event_1", "ライブ", "stage_1.json"),
              encoding="utf-8") as f:
        data2 = json.load(f)
    assert data2["タイムテーブル"][0]["グループ名_採用"] == "Y"

    # master_idolname.csv も書き出されている
    csv_path = os.path.join(pj_path, "event_1", "master_idolname.csv")
    assert os.path.exists(csv_path)
    loaded = pd.read_csv(csv_path, index_col=0)
    assert loaded.loc[0, "グループ名_採用"] == "XNEW"


# ---------------------------------------------------------------------------
# Phase 4: ライブ行の書き戻し
# ---------------------------------------------------------------------------

def test_save_event_edits_propagates_live_from_to_json(tmp_path):
    pj_path, project_info = _setup_minimal_project(tmp_path)
    df_live = _live_df([
        (0, 0, 0, "10:15", 25, "X", "OldA", False),    # 10:00 → 10:15, 30→25 min
        (1, 1, 1, "11:00", 30, "Y", "OldB", False),    # 11:00 → 11:00, 30→30 min
    ])
    df_idol = _idolname_df([(0, "X"), (1, "Y")])

    _editor.save_event_edits(
        pj_path, "event_1", 0, project_info,
        edits={"idolname": df_idol, "live": df_live},
    )

    with open(os.path.join(pj_path, "event_1", "ライブ", "stage_0.json"),
              encoding="utf-8") as f:
        data = json.load(f)
    assert data["タイムテーブル"][0]["ライブステージ"]["from"] == "10:15"
    assert data["タイムテーブル"][0]["ライブステージ"]["to"] == "10:40"

    # turn_id_data.csv も書き出されている
    assert os.path.exists(os.path.join(pj_path, "event_1", "turn_id_data.csv"))


def test_save_event_edits_group_id_change_updates_adopted_name(tmp_path):
    """グループID を別グループに変えると 採用名 (グループ名_採用) も追従して更新される。"""
    pj_path, project_info = _setup_minimal_project(tmp_path)
    # 出番ID=0 のグループID を 0 → 1 に変更
    df_live = _live_df([
        (0, 0, 1, "10:00", 30, "X", "OldA", False),    # gid 0→1
        (1, 1, 1, "11:00", 30, "Y", "OldB", False),
    ])
    df_idol = _idolname_df([(0, "X"), (1, "Y")])

    _editor.save_event_edits(
        pj_path, "event_1", 0, project_info,
        edits={"idolname": df_idol, "live": df_live},
    )

    with open(os.path.join(pj_path, "event_1", "ライブ", "stage_0.json"),
              encoding="utf-8") as f:
        data = json.load(f)
    # 出番ID=0 の グループID が 1 に、採用名 が Y に追従
    assert data["タイムテーブル"][0]["グループID"] == 1
    assert data["タイムテーブル"][0]["グループ名_採用"] == "Y"


# ---------------------------------------------------------------------------
# Phase 4: 特典会行の書き戻し (heiki 形式)
# ---------------------------------------------------------------------------

def _setup_heiki_project(tmp_path):
    """live_tokutenkai_heiki 形式の最小プロジェクト"""
    pj_path = tmp_path / "pj"
    event_dir = pj_path / "event_1" / "ライブ特典会"
    os.makedirs(event_dir)
    with open(event_dir / "stage_0.json", "w", encoding="utf-8") as f:
        json.dump({
            "ステージ名": "MAIN",
            "ステージID": 0,
            "タイムテーブル": [
                {
                    "グループ名": "X", "グループ名_採用": "X",
                    "出番ID": 0, "グループID": 0,
                    "ライブステージ": {"from": "10:00", "to": "10:20"},
                    "特典会": [
                        {"from": "10:30", "to": "11:00",
                         "ブース": "ブースA", "ステージID": 10,
                         "出番ID": 10, "対応出番ID": 0},
                    ],
                    "備考": "",
                },
            ],
        }, f, ensure_ascii=False)
    project_info = {
        "project_name": "pj",
        "event_num": 1,
        "event_detail": [{
            "event_no": 0,
            "event_name": "event_1",
            "timetables": [{
                "image_no": 0,
                "dir_name": "ライブ特典会",
                "display_name": "ライブ特典会",
                "kind": "live_tokutenkai_heiki",
                "stage_num": 1,
                "stage_list": [
                    {"stage_no": 0, "stage_name": "MAIN",
                     "kind": "live_tokutenkai_heiki", "stage_id": 0},
                ],
            }],
        }],
    }
    with open(pj_path / "project_info.json", "w", encoding="utf-8") as f:
        json.dump(project_info, f, ensure_ascii=False)
    return str(pj_path), project_info


def test_save_event_edits_propagates_tokutenkai_edit_via_corresponding_id(tmp_path):
    """特典会行 (特典会フラグ=True) の編集を 対応出番ID + ブース 一致で書き戻す。"""
    pj_path, project_info = _setup_heiki_project(tmp_path)
    # 出番ID=0=ライブ行, 出番ID=10=特典会行 (対応出番ID=0, ブース=ブースA)
    df_live = _live_df([
        (0, 0, 0, "10:00", 20, "X", "MAIN", False),
        (10, 10, 0, "10:45", 20, "X", "ブースA", True),     # from 10:30→10:45
    ])
    df_idol = _idolname_df([(0, "X")])

    _editor.save_event_edits(
        pj_path, "event_1", 0, project_info,
        edits={"idolname": df_idol, "live": df_live},
    )

    with open(os.path.join(pj_path, "event_1", "ライブ特典会", "stage_0.json"),
              encoding="utf-8") as f:
        data = json.load(f)
    tk = data["タイムテーブル"][0]["特典会"][0]
    assert tk["from"] == "10:45"
    assert tk["to"] == "11:05"   # 10:45 + 20分
    assert tk["対応出番ID"] == 0


# ---------------------------------------------------------------------------
# Phase 4: 対応出番ID 変更 (特典会要素の付け替え)
# ---------------------------------------------------------------------------

def _setup_heiki_two_lives_project(tmp_path):
    """1 stage_*.json に親ライブ 2 つ + booth 1 つ (出番ID=0 親, 出番ID=10 booth が親0 に紐づく)。
    親出番ID=1 もあり、テストでは booth を 0 → 1 に付け替える。
    """
    pj_path = tmp_path / "pj"
    event_dir = pj_path / "event_1" / "ライブ特典会"
    os.makedirs(event_dir)
    with open(event_dir / "stage_0.json", "w", encoding="utf-8") as f:
        json.dump({
            "ステージ名": "MAIN",
            "ステージID": 0,
            "タイムテーブル": [
                {
                    "グループ名": "X", "グループ名_採用": "X",
                    "出番ID": 0, "グループID": 0,
                    "ライブステージ": {"from": "10:00", "to": "10:20"},
                    "特典会": [
                        {"from": "10:30", "to": "11:00",
                         "ブース": "ブースA", "ステージID": 10,
                         "出番ID": 10, "対応出番ID": 0},
                    ],
                    "備考": "",
                },
                {
                    "グループ名": "Y", "グループ名_採用": "Y",
                    "出番ID": 1, "グループID": 1,
                    "ライブステージ": {"from": "11:00", "to": "11:20"},
                    "特典会": [],
                    "備考": "",
                },
            ],
        }, f, ensure_ascii=False)
    project_info = {
        "project_name": "pj",
        "event_num": 1,
        "event_detail": [{
            "event_no": 0,
            "event_name": "event_1",
            "timetables": [{
                "image_no": 0,
                "dir_name": "ライブ特典会",
                "display_name": "ライブ特典会",
                "kind": "live_tokutenkai_heiki",
                "stage_num": 1,
                "stage_list": [
                    {"stage_no": 0, "stage_name": "MAIN",
                     "kind": "live_tokutenkai_heiki", "stage_id": 0},
                ],
            }],
        }],
    }
    with open(pj_path / "project_info.json", "w", encoding="utf-8") as f:
        json.dump(project_info, f, ensure_ascii=False)
    return str(pj_path), project_info


def _setup_heiki_two_stages_project(tmp_path):
    """2 stage_*.json (別ファイル)。stage_0 に親=0, stage_1 に親=1。
    booth (出番ID=10, 対応出番ID=0) は最初 stage_0 にある。
    テストでは booth の 対応出番ID を 0 → 1 に変更 → stage_1.json に移動。
    """
    pj_path = tmp_path / "pj"
    event_dir = pj_path / "event_1" / "ライブ特典会"
    os.makedirs(event_dir)
    with open(event_dir / "stage_0.json", "w", encoding="utf-8") as f:
        json.dump({
            "ステージ名": "STAGE_A",
            "ステージID": 0,
            "タイムテーブル": [
                {
                    "グループ名": "X", "グループ名_採用": "X",
                    "出番ID": 0, "グループID": 0,
                    "ライブステージ": {"from": "10:00", "to": "10:20"},
                    "特典会": [
                        {"from": "10:30", "to": "11:00",
                         "ブース": "ブースA", "ステージID": 10,
                         "出番ID": 10, "対応出番ID": 0},
                    ],
                    "備考": "",
                },
            ],
        }, f, ensure_ascii=False)
    with open(event_dir / "stage_1.json", "w", encoding="utf-8") as f:
        json.dump({
            "ステージ名": "STAGE_B",
            "ステージID": 1,
            "タイムテーブル": [
                {
                    "グループ名": "Y", "グループ名_採用": "Y",
                    "出番ID": 1, "グループID": 1,
                    "ライブステージ": {"from": "11:00", "to": "11:20"},
                    "特典会": [],
                    "備考": "",
                },
            ],
        }, f, ensure_ascii=False)
    project_info = {
        "project_name": "pj",
        "event_num": 1,
        "event_detail": [{
            "event_no": 0,
            "event_name": "event_1",
            "timetables": [{
                "image_no": 0,
                "dir_name": "ライブ特典会",
                "display_name": "ライブ特典会",
                "kind": "live_tokutenkai_heiki",
                "stage_num": 2,
                "stage_list": [
                    {"stage_no": 0, "stage_name": "STAGE_A",
                     "kind": "live_tokutenkai_heiki", "stage_id": 0},
                    {"stage_no": 1, "stage_name": "STAGE_B",
                     "kind": "live_tokutenkai_heiki", "stage_id": 1},
                ],
            }],
        }],
    }
    with open(pj_path / "project_info.json", "w", encoding="utf-8") as f:
        json.dump(project_info, f, ensure_ascii=False)
    return str(pj_path), project_info


def _attach_corresp(df_live, corresp_dict):
    """df_live に 対応出番ID 列を追加するヘルパ。corresp_dict は {出番ID: 対応出番ID or None}"""
    df_live["対応出番ID"] = pd.Series(
        {k: v for k, v in corresp_dict.items()}, dtype="Int64",
    ).reindex(df_live.index)
    return df_live


def test_build_corresponding_turn_id_map_returns_booth_to_parent_map(tmp_path):
    pj_path, project_info = _setup_heiki_two_lives_project(tmp_path)
    m = _editor.build_corresponding_turn_id_map(
        pj_path, "event_1", 0, project_info,
    )
    assert m == {10: 0}


def test_save_event_edits_moves_tk_within_same_file_when_corresp_changed(tmp_path):
    """同 stage_*.json 内で 対応出番ID を 0 → 1 に変更 → 親0 の 特典会[] から削除、親1 に追加。"""
    pj_path, project_info = _setup_heiki_two_lives_project(tmp_path)
    df_live = _live_df([
        (0, 0, 0, "10:00", 20, "X", "MAIN", False),
        (1, 0, 1, "11:00", 20, "Y", "MAIN", False),
        (10, 10, 0, "10:30", 30, "X", "ブースA", True),  # 親0 → 親1 へ
    ])
    _attach_corresp(df_live, {0: pd.NA, 1: pd.NA, 10: 1})  # booth の 対応出番ID を 1 に変更
    df_idol = _idolname_df([(0, "X"), (1, "Y")])

    _editor.save_event_edits(
        pj_path, "event_1", 0, project_info,
        edits={"idolname": df_idol, "live": df_live},
    )

    with open(os.path.join(pj_path, "event_1", "ライブ特典会", "stage_0.json"),
              encoding="utf-8") as f:
        data = json.load(f)
    # 親0 (出番ID=0) の 特典会[] が空に
    assert data["タイムテーブル"][0]["特典会"] == []
    # 親1 (出番ID=1) の 特典会[] に元の booth 要素 (出番ID=10) が移動 & 対応出番ID 更新
    tk_list = data["タイムテーブル"][1]["特典会"]
    assert len(tk_list) == 1
    assert tk_list[0]["出番ID"] == 10           # booth 自身の出番ID は保持
    assert tk_list[0]["ステージID"] == 10       # ステージID も保持
    assert tk_list[0]["対応出番ID"] == 1        # 新親に更新
    # from/to は元の値が維持 (今回の編集で変えていない)
    # (ただし from=10:30 → to=11:00 は再計算で 30分→11:00 と一致)
    assert tk_list[0]["from"] == "10:30"
    assert tk_list[0]["to"] == "11:00"


def test_save_event_edits_moves_tk_across_files_when_corresp_changed(tmp_path):
    """別 stage_*.json 間で 対応出番ID 変更 → 旧ファイルから削除、新ファイルに追加。"""
    pj_path, project_info = _setup_heiki_two_stages_project(tmp_path)
    df_live = _live_df([
        (0, 0, 0, "10:00", 20, "X", "STAGE_A", False),
        (1, 1, 1, "11:00", 20, "Y", "STAGE_B", False),
        (10, 10, 0, "10:30", 30, "X", "ブースA", True),
    ])
    _attach_corresp(df_live, {0: pd.NA, 1: pd.NA, 10: 1})
    df_idol = _idolname_df([(0, "X"), (1, "Y")])

    _editor.save_event_edits(
        pj_path, "event_1", 0, project_info,
        edits={"idolname": df_idol, "live": df_live},
    )

    # 旧ファイル (stage_0): 親0 の 特典会[] が空
    with open(os.path.join(pj_path, "event_1", "ライブ特典会", "stage_0.json"),
              encoding="utf-8") as f:
        data_old = json.load(f)
    assert data_old["タイムテーブル"][0]["特典会"] == []

    # 新ファイル (stage_1): 親1 の 特典会[] に booth が追加
    with open(os.path.join(pj_path, "event_1", "ライブ特典会", "stage_1.json"),
              encoding="utf-8") as f:
        data_new = json.load(f)
    tk_list = data_new["タイムテーブル"][0]["特典会"]
    assert len(tk_list) == 1
    assert tk_list[0]["出番ID"] == 10
    assert tk_list[0]["対応出番ID"] == 1


def test_save_event_edits_move_with_simultaneous_value_edit(tmp_path):
    """対応出番ID 変更 + from/長さ 変更 が同時 → 移動先で from/to も更新される。"""
    pj_path, project_info = _setup_heiki_two_lives_project(tmp_path)
    df_live = _live_df([
        (0, 0, 0, "10:00", 20, "X", "MAIN", False),
        (1, 0, 1, "11:00", 20, "Y", "MAIN", False),
        (10, 10, 0, "11:45", 25, "X", "ブースA", True),  # from と長さも変更
    ])
    _attach_corresp(df_live, {0: pd.NA, 1: pd.NA, 10: 1})
    df_idol = _idolname_df([(0, "X"), (1, "Y")])

    _editor.save_event_edits(
        pj_path, "event_1", 0, project_info,
        edits={"idolname": df_idol, "live": df_live},
    )

    with open(os.path.join(pj_path, "event_1", "ライブ特典会", "stage_0.json"),
              encoding="utf-8") as f:
        data = json.load(f)
    assert data["タイムテーブル"][0]["特典会"] == []
    tk_list = data["タイムテーブル"][1]["特典会"]
    assert len(tk_list) == 1
    assert tk_list[0]["from"] == "11:45"
    assert tk_list[0]["to"] == "12:10"  # 11:45 + 25分
    assert tk_list[0]["対応出番ID"] == 1


def test_validate_live_rejects_invalid_corresponding_turn_id():
    df_live = _live_df([
        (0, 0, 0, "10:00", 20, "X", "MAIN", False),
        (10, 10, 0, "10:30", 30, "X", "ブースA", True),
    ])
    _attach_corresp(df_live, {0: pd.NA, 10: 99})  # 99 は存在しない
    df_idol = _idolname_df([(0, "X")])

    errs = _editor.validate_live_master_edits(df_live, df_idol)
    assert any("対応出番ID" in e for e in errs)


def test_validate_live_rejects_corresponding_id_pointing_to_booth_row():
    """対応出番ID は ライブ行の 出番ID を指す必要がある (別 booth 行は不可)。"""
    df_live = _live_df([
        (0, 0, 0, "10:00", 20, "X", "MAIN", False),
        (10, 10, 0, "10:30", 30, "X", "ブースA", True),
        (20, 10, 0, "11:30", 30, "X", "ブースA", True),
    ])
    _attach_corresp(df_live, {0: pd.NA, 10: 20, 20: 0})  # 10 が別 booth 20 を指す
    df_idol = _idolname_df([(0, "X")])
    errs = _editor.validate_live_master_edits(df_live, df_idol)
    assert any("対応出番ID" in e for e in errs)


def test_save_event_edits_propagate_safe_when_target_stage_missing(tmp_path):
    """ステージID が存在しない場合でも propagate は例外で落ちない (安全網)。

    バリデーション層で弾く前提だが、防御として無動作で返ることを確認する。
    """
    pj_path, project_info = _setup_minimal_project(tmp_path)
    df_live = _live_df([
        (0, 99, 0, "10:00", 30, "X", "PHANTOM", False),  # ステージID=99 (存在しない)
    ])
    df_idol = _idolname_df([(0, "X"), (1, "Y")])
    _editor.save_event_edits(
        pj_path, "event_1", 0, project_info,
        edits={"idolname": df_idol, "live": df_live},
    )


# ---------------------------------------------------------------------------
# Phase 5: 出番マスタの ステージID 変更 (stage_*.json 間エントリ移動)
# ---------------------------------------------------------------------------

def test_save_event_edits_phase5_moves_live_entry_between_files(tmp_path):
    """非heiki ライブ行の ステージID を変更すると、タイムテーブル[i] が
    別の stage_*.json に移動する。"""
    pj_path, project_info = _setup_minimal_project(tmp_path)
    # 出番ID=0 (元 stage_0/ID=0) の ステージID を 1 に変更 → stage_1.json に移動
    df_live = _live_df([
        (0, 1, 0, "10:00", 30, "X", "OldB", False),     # ID 0→1
        (1, 1, 1, "11:00", 30, "Y", "OldB", False),     # 変更なし
    ])
    df_idol = _idolname_df([(0, "X"), (1, "Y")])

    _editor.save_event_edits(
        pj_path, "event_1", 0, project_info,
        edits={"idolname": df_idol, "live": df_live},
    )

    # stage_0.json は空に、stage_1.json は 2 エントリに
    with open(os.path.join(pj_path, "event_1", "ライブ", "stage_0.json"),
              encoding="utf-8") as f:
        data0 = json.load(f)
    with open(os.path.join(pj_path, "event_1", "ライブ", "stage_1.json"),
              encoding="utf-8") as f:
        data1 = json.load(f)

    turn_ids_0 = [t["出番ID"] for t in data0["タイムテーブル"]]
    turn_ids_1 = [t["出番ID"] for t in data1["タイムテーブル"]]
    assert turn_ids_0 == []
    assert sorted(turn_ids_1) == [0, 1]


def test_save_event_edits_phase5_move_preserves_other_field_edits(tmp_path):
    """ステージID 変更と同時に from/長さ も編集した行は、移動後のエントリで反映される。"""
    pj_path, project_info = _setup_minimal_project(tmp_path)
    df_live = _live_df([
        (0, 1, 0, "12:00", 45, "X", "OldB", False),     # ステージID 0→1, 時刻も変更
        (1, 1, 1, "11:00", 30, "Y", "OldB", False),
    ])
    df_idol = _idolname_df([(0, "X"), (1, "Y")])

    _editor.save_event_edits(
        pj_path, "event_1", 0, project_info,
        edits={"idolname": df_idol, "live": df_live},
    )

    with open(os.path.join(pj_path, "event_1", "ライブ", "stage_1.json"),
              encoding="utf-8") as f:
        data1 = json.load(f)
    moved = [t for t in data1["タイムテーブル"] if t["出番ID"] == 0][0]
    assert moved["ライブステージ"]["from"] == "12:00"
    assert moved["ライブステージ"]["to"] == "12:45"


def test_save_event_edits_phase5_heiki_booth_stage_id_update_in_place(tmp_path):
    """heiki 形式の booth 行で ステージID を変更すると、特典会[].ステージID が in-place で更新される。

    booth 自身の親エントリ (= 親ライブ) は同じファイル内に留まる。
    """
    pj_path, project_info = _setup_heiki_project(tmp_path)
    # 出番ID=10 booth: ステージID 10 → 11
    df_live = _live_df([
        (0, 0, 0, "10:00", 20, "X", "MAIN", False),
        (10, 11, 0, "10:30", 30, "X", "ブースB", True),     # booth ステージID 10→11
    ])
    df_idol = _idolname_df([(0, "X")])

    _editor.save_event_edits(
        pj_path, "event_1", 0, project_info,
        edits={"idolname": df_idol, "live": df_live},
    )

    with open(os.path.join(pj_path, "event_1", "ライブ特典会", "stage_0.json"),
              encoding="utf-8") as f:
        data = json.load(f)
    tk = data["タイムテーブル"][0]["特典会"][0]
    # 特典会要素はファイルを跨がず in-place で ステージID のみ更新
    assert tk["ステージID"] == 11
    assert tk["出番ID"] == 10            # booth 出番ID は不変
    assert tk["対応出番ID"] == 0          # 親も不変


def test_build_event_output_sorts_df_stage_by_display_order(tmp_path):
    """df_stage は 表示順 昇順で返る"""
    pj_path, project_info = _setup_minimal_project(tmp_path)
    # 表示順を逆順 (1, 0) に設定
    pd.DataFrame(
        {
            "ステージ名": ["OldA", "OldB"],
            "特典会フラグ": [False, False],
            "表示順": [1, 0],
            "非活性化フラグ": [False, False],
        },
        index=pd.Index([0, 1], name="ステージID"),
    ).to_csv(os.path.join(pj_path, "event_1", "master_stage.csv"))

    result = _output.build_event_output(pj_path, "event_1", 0, project_info)
    assert result is not None
    df_stage = result["stage"]
    # 表示順 0 のものが先頭
    assert df_stage.iloc[0]["表示順"] == 0
    assert df_stage.iloc[0]["ステージ名"] == "OldB"
    assert df_stage.iloc[1]["表示順"] == 1
    assert df_stage.iloc[1]["ステージ名"] == "OldA"
