"""json_to_df が ID=None（採番前）のレコードを変換できることの回帰テスト。

⑤変更比較でグループ名変更等により 出番ID/グループID/特典会IDが None になった
stage_n.json を ④画面で DataFrame 表示する際、int(None) で TypeError にならないこと。
"""

import pandas as pd

from backend_functions import timetabledata


def test_json_to_df_live_only_with_null_ids():
    json_data = {
        "ステージ名": "A", "ステージID": 0,
        "タイムテーブル": [
            {
                "グループ名": "G", "グループ名_採用": "G",
                "ライブステージ": {"from": "10:00", "to": "10:20"},
                "出番ID": None, "グループID": None,
            },
        ],
    }
    # ④画面と同様 id_assigned を明示（採番済種別としてID列を保持）。
    # 旧実装では int(None) で TypeError になっていた。
    df = timetabledata.json_to_df(json_data, tokutenkai=False, id_assigned=True)
    assert len(df) == 1
    assert pd.isna(df.iloc[0]["出番ID"])
    assert pd.isna(df.iloc[0]["グループID"])


def test_json_to_df_tokutenkai_with_null_ids():
    json_data = {
        "ステージ名": "A", "ステージID": 0,
        "タイムテーブル": [
            {
                "グループ名": "G", "グループ名_採用": "G",
                "ライブステージ": {"from": "10:00", "to": "10:20"},
                "特典会": [
                    {"from": "10:30", "to": "11:30", "ブース": "A",
                     "出番ID": None, "ステージID": None, "対応出番ID": None},
                ],
                "出番ID": None, "グループID": None,
            },
        ],
    }
    # 例外が出ずに変換できること（旧実装では TypeError）
    df = timetabledata.json_to_df(json_data, tokutenkai=True, id_assigned=True)
    assert len(df) == 1
    assert df.iloc[0]["ブース"] == "A"
    assert pd.isna(df.iloc[0]["出番ID"])
    assert pd.isna(df.iloc[0]["特典会_出番ID"])
