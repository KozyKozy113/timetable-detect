"""空タイムテーブル用 DataFrame (empty_timetable_df) と
ID採番済判定 (project_repository.event_ids_assigned) のテスト。

④画面で OCR 結果が空 (タイムテーブル: []) のステージでも、
data_editor がカラム無しで行追加不能にならないよう、
正規カラムを備えた 0 行 DataFrame を供給する。
"""

import os

from backend_functions import project_repository as repo
from backend_functions import timetabledata


# ---------------------------------------------------------------------------
# empty_timetable_df: カラム構成
# ---------------------------------------------------------------------------

EXPECTED = {
    (True, True): [
        'グループ名', 'グループ名_採用', 'ライブ_from', 'ライブ_to', 'ライブ_長さ(分)',
        '特典会_from', '特典会_to', '特典会_長さ(分)', 'ブース',
        '出番ID', 'グループID', '特典会_出番ID', '特典会_ステージID',
        '備考', 'コラボグループID', 'コラボタイトル',
    ],
    (True, False): [
        'グループ名', 'グループ名_採用', 'ライブ_from', 'ライブ_to', 'ライブ_長さ(分)',
        '特典会_from', '特典会_to', '特典会_長さ(分)', 'ブース',
        '備考', 'コラボグループID', 'コラボタイトル',
    ],
    (False, True): [
        'グループ名', 'グループ名_採用', 'ライブ_from', 'ライブ_to', 'ライブ_長さ(分)',
        '出番ID', 'グループID', '備考', 'コラボグループID', 'コラボタイトル',
    ],
    (False, False): [
        'グループ名', 'グループ名_採用', 'ライブ_from', 'ライブ_to', 'ライブ_長さ(分)',
        '備考', 'コラボグループID', 'コラボタイトル',
    ],
}


def test_empty_df_columns_all_combinations():
    for (tokutenkai, id_assigned), cols in EXPECTED.items():
        df = timetabledata.empty_timetable_df(tokutenkai, id_assigned)
        assert list(df.columns) == cols, (tokutenkai, id_assigned)
        assert len(df) == 0


def test_empty_df_id_columns_dropped_when_not_assigned():
    df = timetabledata.empty_timetable_df(tokutenkai=True, id_assigned=False)
    for col in ('出番ID', 'グループID', '特典会_出番ID', '特典会_ステージID'):
        assert col not in df.columns
    # 特典会の時刻/ブース列は ID 未採番でも残る
    for col in ('特典会_from', '特典会_to', '特典会_長さ(分)', 'ブース'):
        assert col in df.columns


def test_empty_df_int_dtypes():
    df = timetabledata.empty_timetable_df(tokutenkai=True, id_assigned=True)
    for col in ('ライブ_長さ(分)', '出番ID', 'グループID', '特典会_ステージID'):
        assert str(df[col].dtype) == 'Int64'


# ---------------------------------------------------------------------------
# json_to_df との整合 (ドリフトガード): 通常パスのドロップ後カラムと一致する
# ---------------------------------------------------------------------------

def test_matches_json_to_df_with_ids():
    json_with_ids = {
        "タイムテーブル": [{
            "グループ名": "A", "グループ名_採用": "A", "グループID": 1, "出番ID": 10,
            "ライブステージ": {"from": "11:00", "to": "11:20"},
            "特典会": [{"from": "11:40", "to": "12:00", "ブース": "B",
                       "出番ID": 20, "ステージID": 3}],
        }],
    }
    df = timetabledata.json_to_df(json_with_ids, tokutenkai=True)
    expected = timetabledata.empty_timetable_df(tokutenkai=True, id_assigned=True)
    assert list(df.columns) == list(expected.columns)


def test_matches_json_to_df_without_ids():
    json_no_ids = {
        "タイムテーブル": [{
            "グループ名": "A", "グループ名_採用": "A",
            "ライブステージ": {"from": "11:00", "to": "11:20"},
            "特典会": [{"from": "11:40", "to": "12:00", "ブース": "B"}],
        }],
    }
    df = timetabledata.json_to_df(json_no_ids, tokutenkai=True)
    expected = timetabledata.empty_timetable_df(tokutenkai=True, id_assigned=False)
    assert list(df.columns) == list(expected.columns)


def test_matches_json_to_df_live_without_ids():
    json_no_ids = {
        "タイムテーブル": [{
            "グループ名": "A", "グループ名_採用": "A",
            "ライブステージ": {"from": "11:00", "to": "11:20"},
        }],
    }
    df = timetabledata.json_to_df(json_no_ids, tokutenkai=False)
    expected = timetabledata.empty_timetable_df(tokutenkai=False, id_assigned=False)
    assert list(df.columns) == list(expected.columns)


# ---------------------------------------------------------------------------
# json_to_df の id_assigned 明示制御 (種別単位)
# ---------------------------------------------------------------------------

def test_json_to_df_id_assigned_true_keeps_id_cols_without_data():
    """採番済種別: IDがデータに無くても (全NaN) ID列を残す。"""
    json_no_ids = {
        "タイムテーブル": [{
            "グループ名": "A", "グループ名_採用": "A",
            "ライブステージ": {"from": "11:00", "to": "11:20"},
            "特典会": [{"from": "11:40", "to": "12:00", "ブース": "B"}],
        }],
    }
    df = timetabledata.json_to_df(json_no_ids, tokutenkai=True, id_assigned=True)
    for col in ('出番ID', 'グループID', '特典会_出番ID', '特典会_ステージID'):
        assert col in df.columns
        assert str(df[col].dtype) == 'Int64'
    # 空df (採番済) とカラム構成が一致する
    expected = timetabledata.empty_timetable_df(tokutenkai=True, id_assigned=True)
    assert list(df.columns) == list(expected.columns)


def test_json_to_df_id_assigned_false_drops_id_cols_with_data():
    """未採番種別: IDがデータに有っても ID列を落とす。"""
    json_with_ids = {
        "タイムテーブル": [{
            "グループ名": "A", "グループ名_採用": "A", "グループID": 1, "出番ID": 10,
            "ライブステージ": {"from": "11:00", "to": "11:20"},
            "特典会": [{"from": "11:40", "to": "12:00", "ブース": "B",
                       "出番ID": 20, "ステージID": 3}],
        }],
    }
    df = timetabledata.json_to_df(json_with_ids, tokutenkai=True, id_assigned=False)
    for col in ('出番ID', 'グループID', '特典会_出番ID', '特典会_ステージID'):
        assert col not in df.columns
    expected = timetabledata.empty_timetable_df(tokutenkai=True, id_assigned=False)
    assert list(df.columns) == list(expected.columns)


def test_json_to_df_id_assigned_none_is_data_driven():
    """id_assigned=None (ビルド系既定): 従来どおりデータ駆動。"""
    json_with_ids = {
        "タイムテーブル": [{
            "グループ名": "A", "グループ名_採用": "A", "グループID": 1, "出番ID": 10,
            "ライブステージ": {"from": "11:00", "to": "11:20"},
        }],
    }
    df = timetabledata.json_to_df(json_with_ids, tokutenkai=False)
    assert '出番ID' in df.columns and 'グループID' in df.columns


# ---------------------------------------------------------------------------
# img_type_ids_assigned: 種別単位の採番判定
# ---------------------------------------------------------------------------

def _project_info_live_assigned_tokutenkai_not():
    """ライブは stage_id 採番済、特典会は未採番の project_info。"""
    return {
        "event_detail": [{
            "event_name": "ev",
            "timetables": [
                {"dir_name": "ライブ", "stage_list": [
                    {"stage_no": 0, "stage_name": "A", "stage_id": 5},
                    {"stage_no": 1, "stage_name": "B"},
                ]},
                {"dir_name": "特典会", "stage_list": [
                    {"stage_no": 0, "stage_name": "C"},
                ]},
            ],
        }],
    }


def test_img_type_ids_assigned_live_true_tokutenkai_false():
    pi = _project_info_live_assigned_tokutenkai_not()
    assert repo.img_type_ids_assigned(pi, 0, "ライブ") is True
    assert repo.img_type_ids_assigned(pi, 0, "特典会") is False


def test_img_type_ids_assigned_unknown_type_false():
    pi = _project_info_live_assigned_tokutenkai_not()
    assert repo.img_type_ids_assigned(pi, 0, "存在しない種別") is False


# ---------------------------------------------------------------------------
# event_ids_assigned: 3点CSVの有無
# ---------------------------------------------------------------------------

def _touch(path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("")


def test_event_ids_assigned_all_present(tmp_path):
    base = tmp_path / "ev"
    os.makedirs(base)
    for fname in ("master_stage.csv", "master_idolname.csv", "turn_id_data.csv"):
        _touch(base / fname)
    assert repo.event_ids_assigned(str(tmp_path), "ev") is True


def test_event_ids_assigned_missing_one(tmp_path):
    base = tmp_path / "ev"
    os.makedirs(base)
    _touch(base / "master_stage.csv")
    _touch(base / "master_idolname.csv")
    # turn_id_data.csv 欠落
    assert repo.event_ids_assigned(str(tmp_path), "ev") is False


def test_event_ids_assigned_none(tmp_path):
    os.makedirs(tmp_path / "ev")
    assert repo.event_ids_assigned(str(tmp_path), "ev") is False
