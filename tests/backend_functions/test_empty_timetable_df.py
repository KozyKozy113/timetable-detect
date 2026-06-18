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
