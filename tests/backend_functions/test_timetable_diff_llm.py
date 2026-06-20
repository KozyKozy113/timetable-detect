"""timetable_diff_llm のテスト（LLM呼び出しは行わない純粋ロジックのみ）。

- build_llm_timetable_input: ID系除外・連番付与・グループ名_採用優先
- detect_with_tokutenkai: kind / 特典会キーによる形式判定
- apply_proposal_to_stage_json: 変更／削除／追加 と ID方針の反映
"""

from backend_functions import timetable_diff_llm as _diff
from backend_functions.timetable_diff_llm import KEEP


# ---------------------------------------------------------------------------
# build_llm_timetable_input
# ---------------------------------------------------------------------------

def test_build_input_live_only_strips_ids_and_adds_index():
    stage_json = {
        "ステージ名": "A",
        "ステージID": 0,
        "タイムテーブル": [
            {
                "グループ名": "生OCR名",
                "グループ名_採用": "正式名",
                "ライブステージ": {"from": "10:00", "to": "10:20"},
                "出番ID": 5, "グループID": 12, "ステージID": 0,
            },
        ],
    }
    out = _diff.build_llm_timetable_input(stage_json)
    rec = out["タイムテーブル"][0]
    assert rec["連番"] == 0
    # グループ名_採用 があるので採用のみ、生グループ名は入れない
    assert rec["グループ名"] == "正式名"
    assert rec["ライブステージ"] == {"from": "10:00", "to": "10:20"}
    # ID系は全て除外
    for k in ("出番ID", "グループID", "ステージID", "対応出番ID", "コラボステージID"):
        assert k not in rec


def test_build_input_uses_raw_name_when_no_adopted():
    stage_json = {"タイムテーブル": [{"グループ名": "生OCR名", "ライブステージ": {"from": "1", "to": "2"}}]}
    out = _diff.build_llm_timetable_input(stage_json)
    assert out["タイムテーブル"][0]["グループ名"] == "生OCR名"


def test_build_input_tokutenkai_keeps_booth_and_index_only():
    stage_json = {
        "タイムテーブル": [
            {
                "グループ名_採用": "G",
                "ライブステージ": {"from": "10:00", "to": "10:20"},
                "特典会": [
                    {"from": "10:30", "to": "11:30", "ブース": "A",
                     "出番ID": 93, "ステージID": 4, "対応出番ID": 0},
                ],
                "出番ID": 0, "グループID": 1,
            },
        ],
    }
    out = _diff.build_llm_timetable_input(stage_json)
    tk = out["タイムテーブル"][0]["特典会"][0]
    assert tk == {"連番": 0, "from": "10:30", "to": "11:30", "ブース": "A"}


# ---------------------------------------------------------------------------
# detect_with_tokutenkai
# ---------------------------------------------------------------------------

def test_detect_with_tokutenkai_by_kind():
    assert _diff.detect_with_tokutenkai({}, {"kind": "live_tokutenkai_heiki"}) is True
    assert _diff.detect_with_tokutenkai({"タイムテーブル": [{"特典会": []}]}, {"kind": "live"}) is False


def test_detect_with_tokutenkai_fallback_by_key():
    assert _diff.detect_with_tokutenkai({"タイムテーブル": [{"特典会": []}]}) is True
    assert _diff.detect_with_tokutenkai({"タイムテーブル": [{"ライブステージ": {}}]}) is False


# ---------------------------------------------------------------------------
# apply_proposal_to_stage_json
# ---------------------------------------------------------------------------

def _live_stage():
    return {
        "ステージ名": "A", "ステージID": 0,
        "タイムテーブル": [
            {"グループ名": "Araw", "グループ名_採用": "A", "ライブステージ": {"from": "10:00", "to": "10:20"},
             "出番ID": 0, "グループID": 1},
            {"グループ名": "Braw", "グループ名_採用": "B", "ライブステージ": {"from": "10:20", "to": "10:40"},
             "出番ID": 1, "グループID": 2},
        ],
    }


def test_apply_change_time_keep_ids():
    ops = [{
        "種別": "変更", "対象連番": 0,
        "set_グループ名_採用": None,
        "set_ライブ時間": {"from": "10:05", "to": "10:25"},
        "set_出番ID": KEEP, "set_グループID": KEEP,
    }]
    out = _diff.apply_proposal_to_stage_json(_live_stage(), ops, with_tokutenkai=False)
    rec = out["タイムテーブル"][0]
    assert rec["ライブステージ"] == {"from": "10:05", "to": "10:25"}
    assert rec["出番ID"] == 0 and rec["グループID"] == 1  # 保持


def test_apply_change_name_renumber_nulls_ids():
    ops = [{
        "種別": "変更", "対象連番": 1,
        "set_グループ名_採用": "新グループ",
        "set_ライブ時間": {"from": "", "to": ""},
        "set_出番ID": None, "set_グループID": None,
    }]
    out = _diff.apply_proposal_to_stage_json(_live_stage(), ops, with_tokutenkai=False)
    rec = out["タイムテーブル"][1]
    assert rec["グループ名_採用"] == "新グループ"
    assert rec["グループ名"] == "Braw"  # 生OCR名は不変
    assert rec["出番ID"] is None and rec["グループID"] is None


def test_apply_change_manual_turn_id():
    ops = [{
        "種別": "変更", "対象連番": 0,
        "set_グループ名_採用": None, "set_ライブ時間": {"from": "", "to": ""},
        "set_出番ID": 99, "set_グループID": KEEP,
    }]
    out = _diff.apply_proposal_to_stage_json(_live_stage(), ops, with_tokutenkai=False)
    assert out["タイムテーブル"][0]["出番ID"] == 99


def test_apply_delete_whole_record():
    ops = [{"種別": "削除", "対象連番": 0, "削除種別": "全体"}]
    out = _diff.apply_proposal_to_stage_json(_live_stage(), ops, with_tokutenkai=False)
    assert len(out["タイムテーブル"]) == 1
    assert out["タイムテーブル"][0]["グループ名_採用"] == "B"


def test_apply_change_updates_raw_and_marks_correction():
    """別グループへの変更: rawを新規読みで更新し、採用名は補正用に空にする。"""
    ops = [{
        "種別": "変更", "対象連番": 0,
        "set_グループ名_raw": "新グループraw",
        "set_グループ名_採用": None,
        "needs_correction": True,
        "set_ライブ時間": {"from": "", "to": ""},
        "set_出番ID": None, "set_グループID": None,
    }]
    out = _diff.apply_proposal_to_stage_json(_live_stage(), ops, with_tokutenkai=False)
    rec = out["タイムテーブル"][0]
    assert rec["グループ名"] == "新グループraw"   # rawは更新
    assert rec["グループ名_採用"] == ""           # 補正ステップ用に空
    assert rec["出番ID"] is None and rec["グループID"] is None


def test_apply_change_rename_keep_id_sets_adopted():
    """同一グループの改名(グループID保持): rawと採用名を更新、IDは保持。"""
    ops = [{
        "種別": "変更", "対象連番": 0,
        "set_グループ名_raw": "改名後",
        "set_グループ名_採用": "改名後",
        "needs_correction": False,
        "set_ライブ時間": {"from": "", "to": ""},
        "set_出番ID": KEEP, "set_グループID": KEEP,
    }]
    out = _diff.apply_proposal_to_stage_json(_live_stage(), ops, with_tokutenkai=False)
    rec = out["タイムテーブル"][0]
    assert rec["グループ名"] == "改名後"
    assert rec["グループ名_採用"] == "改名後"
    assert rec["出番ID"] == 0 and rec["グループID"] == 1


def test_apply_add_with_correction_empties_adopted():
    """新規追加(マスタ未一致): rawに名を入れ採用名は空。"""
    ops = [{
        "種別": "追加", "対象連番": -1,
        "set_グループ名_raw": "新規raw",
        "set_グループ名_採用": None,
        "needs_correction": True,
        "set_ライブ時間": {"from": "11:00", "to": "11:20"},
        "set_出番ID": None, "set_グループID": None,
    }]
    out = _diff.apply_proposal_to_stage_json(_live_stage(), ops, with_tokutenkai=False)
    added = out["タイムテーブル"][-1]
    assert added["グループ名"] == "新規raw"
    assert added["グループ名_採用"] == ""
    assert added["出番ID"] is None and added["グループID"] is None


def test_apply_add_record_nulls_ids():
    ops = [{
        "種別": "追加", "対象連番": -1,
        "set_グループ名_採用": "新規G", "set_ライブ時間": {"from": "11:00", "to": "11:20"},
        "set_出番ID": None, "set_グループID": None,
    }]
    out = _diff.apply_proposal_to_stage_json(_live_stage(), ops, with_tokutenkai=False)
    added = out["タイムテーブル"][-1]
    assert added["グループ名_採用"] == "新規G"
    assert added["グループ名"] == "新規G"
    assert added["ライブステージ"] == {"from": "11:00", "to": "11:20"}
    assert added["出番ID"] is None and added["グループID"] is None


def _heiki_stage():
    return {
        "ステージ名": "A", "ステージID": 0,
        "タイムテーブル": [
            {
                "グループ名": "Araw", "グループ名_採用": "A",
                "ライブステージ": {"from": "10:00", "to": "10:20"},
                "特典会": [
                    {"from": "10:30", "to": "11:30", "ブース": "A", "出番ID": 90, "ステージID": 4, "対応出番ID": 0},
                    {"from": "11:40", "to": "12:40", "ブース": "B", "出番ID": 91, "ステージID": 5, "対応出番ID": 0},
                ],
                "出番ID": 0, "グループID": 1,
            },
        ],
    }


def test_apply_tokutenkai_only_delete():
    ops = [{"種別": "削除", "対象連番": 0, "削除種別": "特典会のみ", "特典会連番": 0}]
    out = _diff.apply_proposal_to_stage_json(_heiki_stage(), ops, with_tokutenkai=True)
    tk = out["タイムテーブル"][0]["特典会"]
    assert len(tk) == 1 and tk[0]["ブース"] == "B"
    # レコード自体は残る
    assert out["タイムテーブル"][0]["出番ID"] == 0


def test_apply_tokutenkai_change_nulls_child_ids():
    ops = [{
        "種別": "変更", "対象連番": 0,
        "set_グループ名_採用": None, "set_ライブ時間": {"from": "", "to": ""},
        "set_出番ID": KEEP, "set_グループID": KEEP,
        "特典会操作": [{"操作種別": "変更", "対象特典会連番": 0, "ブース": "Z", "from": "10:35", "to": "11:35"}],
    }]
    out = _diff.apply_proposal_to_stage_json(_heiki_stage(), ops, with_tokutenkai=True)
    tk0 = out["タイムテーブル"][0]["特典会"][0]
    assert tk0["ブース"] == "Z" and tk0["from"] == "10:35"
    assert tk0["出番ID"] is None and tk0["ステージID"] is None


def test_apply_parent_renumber_nulls_child_tokutenkai_ids():
    ops = [{
        "種別": "変更", "対象連番": 0,
        "set_グループ名_採用": "新", "set_ライブ時間": {"from": "", "to": ""},
        "set_出番ID": None, "set_グループID": None,
    }]
    out = _diff.apply_proposal_to_stage_json(_heiki_stage(), ops, with_tokutenkai=True)
    for tk in out["タイムテーブル"][0]["特典会"]:
        assert tk["出番ID"] is None and tk["ステージID"] is None


def test_apply_tokutenkai_add():
    ops = [{
        "種別": "変更", "対象連番": 0,
        "set_グループ名_採用": None, "set_ライブ時間": {"from": "", "to": ""},
        "set_出番ID": KEEP, "set_グループID": KEEP,
        "特典会操作": [{"操作種別": "追加", "対象特典会連番": -1, "ブース": "C", "from": "13:00", "to": "14:00"}],
    }]
    out = _diff.apply_proposal_to_stage_json(_heiki_stage(), ops, with_tokutenkai=True)
    tk = out["タイムテーブル"][0]["特典会"]
    assert len(tk) == 3
    assert tk[-1]["ブース"] == "C" and tk[-1]["出番ID"] is None and tk[-1]["対応出番ID"] is None


def test_apply_multiple_deletes_keep_index_stability():
    """複数レコード削除でも連番（元index）でズレなく削除される。"""
    stage = _live_stage()
    stage["タイムテーブル"].append(
        {"グループ名": "Craw", "グループ名_採用": "C", "ライブステージ": {"from": "10:40", "to": "11:00"},
         "出番ID": 2, "グループID": 3},
    )
    ops = [
        {"種別": "削除", "対象連番": 0, "削除種別": "全体"},
        {"種別": "削除", "対象連番": 2, "削除種別": "全体"},
    ]
    out = _diff.apply_proposal_to_stage_json(stage, ops, with_tokutenkai=False)
    names = [r["グループ名_採用"] for r in out["タイムテーブル"]]
    assert names == ["B"]


def test_apply_does_not_mutate_input():
    stage = _live_stage()
    ops = [{"種別": "削除", "対象連番": 0, "削除種別": "全体"}]
    _diff.apply_proposal_to_stage_json(stage, ops, with_tokutenkai=False)
    assert len(stage["タイムテーブル"]) == 2  # 元は不変
