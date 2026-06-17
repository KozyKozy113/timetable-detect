"""output_builder.py のリファクタリング対象関数のテスト"""

import os

import pandas as pd

from backend_functions import output_builder as _output


# ---------------------------------------------------------------------------
# find_or_create_stage_id
# ---------------------------------------------------------------------------

def test_find_or_create_stage_id_existing_returns_existing_id():
    stage_master = {
        0: {"ステージ名": "メインステージ", "特典会フラグ": False, "表示順": 0, "非活性化フラグ": False},
        1: {"ステージ名": "サブステージ", "特典会フラグ": False, "表示順": 1, "非活性化フラグ": False},
    }

    stage_id, next_id = _output.find_or_create_stage_id(
        stage_master, "サブステージ", False, next_id=2,
    )

    assert stage_id == 1
    assert next_id == 2  # next_idは変化しない
    # stage_masterは変化しない
    assert stage_master[1]["ステージ名"] == "サブステージ"


def test_find_or_create_stage_id_new_appends_and_increments():
    stage_master = {
        0: {"ステージ名": "メインステージ", "特典会フラグ": False, "表示順": 0, "非活性化フラグ": False},
    }

    stage_id, next_id = _output.find_or_create_stage_id(
        stage_master, "新ブース", True, next_id=1,
    )

    assert stage_id == 1
    assert next_id == 2
    assert stage_master[1]["ステージ名"] == "新ブース"
    assert stage_master[1]["特典会フラグ"] is True
    # 新規ステージの表示順は既存最大値 + 1
    assert stage_master[1]["表示順"] == 1
    assert stage_master[1]["非活性化フラグ"] is False


def test_find_or_create_stage_id_from_empty_master():
    stage_master = {}

    stage_id, next_id = _output.find_or_create_stage_id(
        stage_master, "初登場ステージ", False, next_id=0,
    )

    assert stage_id == 0
    assert next_id == 1
    assert stage_master[0]["ステージ名"] == "初登場ステージ"
    assert stage_master[0]["特典会フラグ"] is False
    assert stage_master[0]["表示順"] == 0
    assert stage_master[0]["非活性化フラグ"] is False


def test_find_or_create_stage_id_existing_stage_id_hits_master():
    """existing_stage_id 指定時はステージ名一致を経由せずIDで引き当て、
    ステージ名が異なれば**マスタを更新**する。"""
    stage_master = {
        0: {"ステージ名": "旧名", "特典会フラグ": False, "表示順": 0, "非活性化フラグ": False},
    }

    stage_id, next_id = _output.find_or_create_stage_id(
        stage_master, "新名", False, next_id=1, existing_stage_id=0,
    )

    assert stage_id == 0
    assert next_id == 1
    # マスタ側のステージ名が新しい名前に更新される(編集モードの反映)
    assert stage_master[0]["ステージ名"] == "新名"


def test_find_or_create_stage_id_existing_stage_id_handles_string_keys():
    """CSV 経由で読み込まれた stage_master はキーが文字列になっているケース。"""
    stage_master = {
        "0": {"ステージ名": "メインステージ", "特典会フラグ": False, "表示順": 0, "非活性化フラグ": False},
    }

    stage_id, next_id = _output.find_or_create_stage_id(
        stage_master, "メインステージ", False, next_id=1, existing_stage_id=0,
    )

    assert stage_id == 0
    assert next_id == 1
    # 文字列キーでも引き当て成功
    assert stage_master["0"]["ステージ名"] == "メインステージ"


# ---------------------------------------------------------------------------
# load_existing_masters
# ---------------------------------------------------------------------------

def test_load_existing_masters_pre_confirm(project_pre_confirm):
    pj_path, _ = project_pre_confirm
    output_path = os.path.join(pj_path, "event_1")

    stage_master, next_stage_id, idolname_df, next_artist_id = \
        _output.load_existing_masters(output_path)

    assert stage_master == {}
    assert next_stage_id == 0
    assert list(idolname_df.columns) == ["グループ名_採用"]
    assert idolname_df.index.name == "グループID"
    assert len(idolname_df) == 0
    assert next_artist_id == 0


def test_load_existing_masters_post_confirm(project_post_confirm):
    pj_path, _ = project_post_confirm
    output_path = os.path.join(pj_path, "event_1")

    stage_master, next_stage_id, idolname_df, next_artist_id = \
        _output.load_existing_masters(output_path)

    # master_stage.csv に "0,メインステージ,False" と "1,特典会ブース,True"
    # json.loads(df.T.to_json()) によりキーは文字列になる(既存挙動)
    # 後方互換: 表示順 / 非活性化フラグ が無い旧CSVには自動補完される
    assert stage_master["0"]["ステージ名"] == "メインステージ"
    assert stage_master["0"]["特典会フラグ"] is False
    assert stage_master["0"]["表示順"] == 0
    assert stage_master["0"]["非活性化フラグ"] is False
    assert stage_master["1"]["ステージ名"] == "特典会ブース"
    assert stage_master["1"]["特典会フラグ"] is True
    assert stage_master["1"]["表示順"] == 1
    assert stage_master["1"]["非活性化フラグ"] is False
    assert next_stage_id == 2

    # master_idolname.csv は "グループ名" カラムを "グループ名_採用" にリネーム
    assert list(idolname_df.columns) == ["グループ名_採用"]
    assert idolname_df.index.name == "グループID"
    assert idolname_df.loc[0, "グループ名_採用"] == "アルファ"
    assert idolname_df.loc[1, "グループ名_採用"] == "ベータ"
    assert next_artist_id == 2


# ---------------------------------------------------------------------------
# build_event_output
# ---------------------------------------------------------------------------

EXPECTED_LIVE_COLUMNS = [
    "ライブ_from", "ライブ_to", "ライブ_長さ(分)", "グループID", "ステージID",
    "グループ名_raw", "グループ名", "ステージ名", "備考",
    "コラボタイトル",
]


def test_build_event_output_pre_confirm(project_pre_confirm):
    pj_path, project_info_json = project_pre_confirm

    result = _output.build_event_output(pj_path, "event_1", 0, project_info_json)

    assert result is not None
    df_stage = result["stage"]
    df_idolname = result["idolname"]
    df_live = result["live"]

    # ステージマスタ: 0=メインステージ, 1=特典会ブース
    assert df_stage.index.name == "ステージID"
    assert sorted(df_stage.index.tolist()) == [0, 1]
    assert df_stage.loc[0, "ステージ名"] == "メインステージ"
    assert not df_stage.loc[0, "特典会フラグ"]
    assert df_stage.loc[1, "ステージ名"] == "特典会ブース"
    assert bool(df_stage.loc[1, "特典会フラグ"]) is True

    # アーティストマスタ: ソート順で 0=アルファ, 1=ガンマ, 2=ベータ
    assert df_idolname.index.name == "グループID"
    assert len(df_idolname) == 3
    assert sorted(df_idolname["グループ名_採用"].tolist()) == ["アルファ", "ガンマ", "ベータ"]

    # 出番データ: 5行(ライブ3 + 特典会2)
    assert list(df_live.columns) == EXPECTED_LIVE_COLUMNS
    assert df_live.index.name == "出番ID"
    assert len(df_live) == 5
    assert sorted(df_live.index.tolist()) == [0, 1, 2, 3, 4]


def test_build_event_output_post_confirm(project_post_confirm):
    pj_path, project_info_json = project_post_confirm

    result = _output.build_event_output(pj_path, "event_1", 0, project_info_json)

    assert result is not None
    df_idolname = result["idolname"]
    df_live = result["live"]

    # アーティストマスタ: 既存 0=アルファ, 1=ベータ + 新規 2=ガンマ
    assert df_idolname.loc[0, "グループ名_採用"] == "アルファ"
    assert df_idolname.loc[1, "グループ名_採用"] == "ベータ"
    assert df_idolname.loc[2, "グループ名_採用"] == "ガンマ"

    # 出番ID: 既存(0,1,2,3)維持 + ガンマに新規ID 4
    assert sorted(df_live.index.tolist()) == [0, 1, 2, 3, 4]
    gamma_row = df_live[df_live["グループ名"] == "ガンマ"]
    assert len(gamma_row) == 1
    assert gamma_row.index[0] == 4

    # アルファの既存IDが維持されているか確認
    alpha_live = df_live[(df_live["グループ名"] == "アルファ") & (df_live["ステージ名"] == "メインステージ")]
    assert len(alpha_live) == 1
    assert alpha_live.index[0] == 0


def test_build_event_output_tokutenkai_heiki(project_tokutenkai_heiki):
    pj_path, project_info_json = project_tokutenkai_heiki

    result = _output.build_event_output(pj_path, "event_1", 0, project_info_json)

    assert result is not None
    df_stage = result["stage"]
    df_live = result["live"]

    # ステージマスタ: メインステージ(ライブ) + ブースX + ブースY = 3行
    stage_names = df_stage["ステージ名"].tolist()
    assert "メインステージ" in stage_names
    assert "ブースX" in stage_names
    assert "ブースY" in stage_names

    # メインステージは特典会フラグFalse、ブースはTrue
    main_row = df_stage[df_stage["ステージ名"] == "メインステージ"].iloc[0]
    assert not main_row["特典会フラグ"]
    booth_rows = df_stage[df_stage["ステージ名"].isin(["ブースX", "ブースY"])]
    assert booth_rows["特典会フラグ"].all()

    # 出番データ: ライブ3 + 特典会3 = 6行
    assert len(df_live) == 6
    # ライブ行: ステージ名がメインステージ
    live_rows = df_live[df_live["ステージ名"] == "メインステージ"]
    assert len(live_rows) == 3
    # 特典会行: ステージ名がブースX/ブースY
    tokutenkai_rows = df_live[df_live["ステージ名"].isin(["ブースX", "ブースY"])]
    assert len(tokutenkai_rows) == 3


def test_build_event_output_no_data(tmp_path):
    """JSON ファイルが1つも存在しないイベントは None を返す"""
    project_info_json = {
        "project_name": "empty",
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
                        "format": "通常",
                        "stage_num": 1,
                        "stage_list": [
                            {"stage_no": 0, "stage_name": "main", "kind": "live"}
                        ],
                    }
                ],
            }
        ],
    }
    os.makedirs(tmp_path / "event_1" / "ライブ", exist_ok=True)

    result = _output.build_event_output(
        str(tmp_path), "event_1", 0, project_info_json,
    )

    assert result is None


# ---------------------------------------------------------------------------
# build_all_event_outputs
# ---------------------------------------------------------------------------

def test_build_all_event_outputs_returns_data_per_event(project_pre_confirm):
    pj_path, project_info_json = project_pre_confirm

    result = _output.build_all_event_outputs(pj_path, project_info_json)

    assert "event_1" in result
    assert "stage" in result["event_1"]
    assert "idolname" in result["event_1"]
    assert "live" in result["event_1"]


def test_build_all_event_outputs_preserves_empty_for_no_data(tmp_path):
    """データなしイベントは空dictとして保持(キーは残す)"""
    def _live_entry():
        return {
            "image_no": 0,
            "dir_name": "ライブ",
            "display_name": "ライブ",
            "kind": "live",
            "format": "通常",
            "stage_num": 1,
            "stage_list": [{"stage_no": 0, "stage_name": "main", "kind": "live"}],
        }

    project_info_json = {
        "project_name": "two_events",
        "event_num": 2,
        "event_detail": [
            {
                "event_no": 0,
                "event_name": "event_1",
                "timetables": [_live_entry()],
            },
            {
                "event_no": 1,
                "event_name": "event_2",
                "timetables": [_live_entry()],
            },
        ],
    }
    os.makedirs(tmp_path / "event_1" / "ライブ", exist_ok=True)
    os.makedirs(tmp_path / "event_2" / "ライブ", exist_ok=True)

    result = _output.build_all_event_outputs(str(tmp_path), project_info_json)

    assert set(result.keys()) == {"event_1", "event_2"}
    assert result["event_1"] == {}
    assert result["event_2"] == {}


# ---------------------------------------------------------------------------
# 空dictイベント混在時の後段関数の挙動
# ---------------------------------------------------------------------------

def _make_two_event_project(tmp_path):
    """event_1=データあり, event_2=空dict のテストデータを作る"""
    def _live_entry():
        return {
            "image_no": 0,
            "dir_name": "ライブ",
            "display_name": "ライブ",
            "kind": "live",
            "format": "通常",
            "stage_num": 1,
            "stage_list": [{"stage_no": 0, "stage_name": "メイン", "kind": "live"}],
        }

    project_info_json = {
        "project_name": "p",
        "event_num": 2,
        "event_detail": [
            {
                "event_no": 0, "event_name": "event_1",
                "timetables": [_live_entry()],
            },
            {
                "event_no": 1, "event_name": "event_2",
                "timetables": [_live_entry()],
            },
        ],
    }
    os.makedirs(tmp_path / "event_1", exist_ok=True)
    os.makedirs(tmp_path / "event_2", exist_ok=True)
    output_df = {
        "event_1": {
            "stage": pd.DataFrame(
                {"ステージ名": ["メイン"], "特典会フラグ": [False]},
                index=pd.Index([0], name="ステージID"),
            ),
            "idolname": pd.DataFrame(
                {"グループ名_採用": ["ZZグループ"]},
                index=pd.Index([0], name="グループID"),
            ),
            "live": pd.DataFrame(
                {
                    "ライブ_from": ["11:00"], "ライブ_長さ(分)": [30],
                    "グループID": [0], "ステージID": [0],
                    "グループ名_raw": ["ZZグループ"], "グループ名": ["ZZグループ"],
                    "ステージ名": ["メイン"], "備考": [""],
                },
                index=pd.Index([0], name="出番ID"),
            ),
        },
        "event_2": {},
    }
    return project_info_json, output_df


def test_determine_id_master_skips_empty_event(tmp_path):
    project_info_json, output_df = _make_two_event_project(tmp_path)

    _output.determine_id_master(output_df, str(tmp_path), project_info_json)

    # event_1 はCSV出力される
    assert (tmp_path / "event_1" / "master_stage.csv").exists()
    assert (tmp_path / "event_1" / "master_idolname.csv").exists()
    # event_2 はスキップされてCSVは作られない(KeyErrorも起きない)
    assert not (tmp_path / "event_2" / "master_stage.csv").exists()


def test_export_excel_skips_empty_event(tmp_path):
    project_info_json, output_df = _make_two_event_project(tmp_path)
    event_list = ["event_1", "event_2"]

    output_path = _output.export_excel(output_df, str(tmp_path), event_list)

    # 出力ファイルが作られ、event_1 のシートだけが含まれる
    from openpyxl import load_workbook
    wb = load_workbook(output_path)
    assert "event_1" in wb.sheetnames
    assert "event_2" not in wb.sheetnames


# ---------------------------------------------------------------------------
# kind ベース 特典会フラグ導出の回帰テスト (Phase 6)
# ---------------------------------------------------------------------------

import json as _json


def _write_stage_json(stage_dir, group_name: str, time_from: str = "11:00") -> None:
    os.makedirs(stage_dir, exist_ok=True)
    payload = {
        "タイムテーブル": [
            {
                "グループ名": group_name,
                "グループ名_採用": group_name,
                "ライブステージ": {"from": time_from, "to": "11:30"},
                "備考": "",
            }
        ]
    }
    with open(os.path.join(stage_dir, "stage_0.json"), "w", encoding="utf-8") as f:
        _json.dump(payload, f, ensure_ascii=False)


def _make_single_image_project(tmp_path, dir_name: str, kind: str,
                               include_format: bool = True) -> dict:
    """1 イベント / 1 画像 / 1 ステージのプロジェクトを作る"""
    entry = {
        "image_no": 0,
        "dir_name": dir_name,
        "display_name": dir_name,
        "kind": kind,
        "stage_num": 1,
        "stage_list": [{"stage_no": 0, "stage_name": "ステージA", "kind": kind}],
    }
    if include_format:
        entry["format"] = "通常"
    project_info_json = {
        "project_name": "p",
        "event_num": 1,
        "event_detail": [
            {
                "event_no": 0, "event_name": "event_1",
                "timetables": [entry],
            }
        ],
    }
    _write_stage_json(os.path.join(tmp_path, "event_1", dir_name), "アルファ")
    return project_info_json


def test_custom_dir_name_with_kind_live_is_not_tokutenkai(tmp_path):
    """カスタム dir_name + kind=live → 特典会フラグ=False。

    旧仕様 (event_type == "特典会" 文字列一致) でも False だったので結論は変わらないが、
    新仕様が kind ベースで判定していることを担保する。
    """
    pij = _make_single_image_project(tmp_path, dir_name="縁日", kind="live")

    result = _output.build_event_output(str(tmp_path), "event_1", 0, pij)

    assert result is not None
    df_stage = result["stage"]
    row = df_stage[df_stage["ステージ名"] == "ステージA"].iloc[0]
    assert not row["特典会フラグ"]


def test_custom_dir_name_with_kind_tokutenkai_is_tokutenkai(tmp_path):
    """カスタム dir_name + kind=tokutenkai → 特典会フラグ=True。

    旧仕様では event_type=="特典会" の完全一致でしか特典会判定できず、
    "縁日" のようなカスタム名は常にライブ扱いだった。Plan §現状の問題点 1 のバグ修正。
    """
    pij = _make_single_image_project(tmp_path, dir_name="縁日", kind="tokutenkai")

    result = _output.build_event_output(str(tmp_path), "event_1", 0, pij)

    assert result is not None
    df_stage = result["stage"]
    row = df_stage[df_stage["ステージ名"] == "ステージA"].iloc[0]
    assert bool(row["特典会フラグ"]) is True


def test_custom_dir_name_with_kind_heiki_no_format_field(tmp_path):
    """カスタム dir_name + kind=live_tokutenkai_heiki は format フィールドを持たない設計だが、
    build_event_output が動作することを担保する (kind ベースで分岐するため)。
    """
    pij = _make_single_image_project(
        tmp_path, dir_name="特殊併記", kind="live_tokutenkai_heiki",
        include_format=False,
    )
    # 併記タイテ JSON (実フォーマットに合わせる: 各行に「ライブステージ」と「特典会」配列)
    stage_dir = os.path.join(tmp_path, "event_1", "特殊併記")
    payload = {
        "タイムテーブル": [
            {
                "グループ名": "アルファ",
                "グループ名_採用": "アルファ",
                "ライブステージ": {"from": "11:00", "to": "11:30"},
                "特典会": [
                    {"from": "11:40", "to": "12:10", "ブース": "ブースZ"},
                ],
                "備考": "",
            }
        ]
    }
    with open(os.path.join(stage_dir, "stage_0.json"), "w", encoding="utf-8") as f:
        _json.dump(payload, f, ensure_ascii=False)

    result = _output.build_event_output(str(tmp_path), "event_1", 0, pij)
    assert result is not None
    df_stage = result["stage"]
    # 親ステージはライブ枠 → 特典会フラグ False
    main_stage = df_stage[df_stage["ステージ名"] == "ステージA"].iloc[0]
    assert not main_stage["特典会フラグ"]
    # ブースZ は特典会扱い → 特典会フラグ True
    booth = df_stage[df_stage["ステージ名"] == "ブースZ"].iloc[0]
    assert bool(booth["特典会フラグ"]) is True


# ---------------------------------------------------------------------------
# build_duration_distribution / build_group_appearance_count
# ---------------------------------------------------------------------------

def _make_stage_df(rows):
    """rows = [(stage_id, stage_name, tokutenkai_flg), ...]"""
    df = pd.DataFrame(
        {
            "ステージ名": [r[1] for r in rows],
            "特典会フラグ": [r[2] for r in rows],
        },
        index=pd.Index([r[0] for r in rows], name="ステージID"),
    )
    return df


def _make_live_df(rows):
    """rows = [(出番ID, ライブ_長さ(分), グループID, グループ名, ステージID), ...]"""
    df = pd.DataFrame(
        {
            "ライブ_from": ["11:00"] * len(rows),
            "ライブ_長さ(分)": [r[1] for r in rows],
            "グループID": [r[2] for r in rows],
            "ステージID": [r[4] for r in rows],
            "グループ名_raw": [r[3] for r in rows],
            "グループ名": [r[3] for r in rows],
            "ステージ名": [""] * len(rows),
            "備考": [""] * len(rows),
        },
        index=pd.Index([r[0] for r in rows], name="出番ID"),
    )
    return df


def test_build_duration_distribution_splits_by_tokutenkai_flag():
    df_stage = _make_stage_df([
        (0, "メイン", False),
        (1, "ブースA", True),
    ])
    df_live = _make_live_df([
        (0, 20, 0, "A", 0),
        (1, 20, 1, "B", 0),
        (2, 30, 2, "C", 0),
        (3, 20, 0, "A", 1),
        (4, 25, 1, "B", 1),
        (5, 25, 1, "B", 1),
    ])

    result = _output.build_duration_distribution(df_live, df_stage)

    assert result.index.name == "長さ(分)"
    assert list(result.columns) == ["ライブステージ", "特典会ステージ"]
    assert result.loc[20, "ライブステージ"] == 2
    assert result.loc[20, "特典会ステージ"] == 1
    assert result.loc[25, "ライブステージ"] == 0
    assert result.loc[25, "特典会ステージ"] == 2
    assert result.loc[30, "ライブステージ"] == 1
    assert result.loc[30, "特典会ステージ"] == 0
    assert list(result.index) == [20, 25, 30]


def test_build_duration_distribution_empty_returns_empty_with_two_cols():
    df_stage = _make_stage_df([(0, "メイン", False)])
    df_live = _make_live_df([])

    result = _output.build_duration_distribution(df_live, df_stage)

    assert list(result.columns) == ["ライブステージ", "特典会ステージ"]
    assert len(result) == 0
    assert result.index.name == "長さ(分)"


def test_build_duration_distribution_only_live_still_has_tokutenkai_col():
    df_stage = _make_stage_df([(0, "メイン", False)])
    df_live = _make_live_df([
        (0, 30, 0, "A", 0),
        (1, 30, 1, "B", 0),
    ])

    result = _output.build_duration_distribution(df_live, df_stage)

    assert list(result.columns) == ["ライブステージ", "特典会ステージ"]
    assert result.loc[30, "ライブステージ"] == 2
    assert result.loc[30, "特典会ステージ"] == 0


def test_build_group_appearance_count_splits_by_tokutenkai_flag():
    df_stage = _make_stage_df([
        (0, "メイン", False),
        (1, "ブース", True),
    ])
    df_live = _make_live_df([
        (0, 30, 10, "アルファ", 0),
        (1, 30, 10, "アルファ", 0),
        (2, 25, 11, "ベータ", 0),
        (3, 20, 10, "アルファ", 1),
        (4, 20, 12, "ガンマ", 1),
    ])

    result = _output.build_group_appearance_count(df_live, df_stage)

    assert result.index.name == "グループID"
    assert list(result.columns) == [
        "グループ名", "ライブ出演回数", "特典会出演回数", "合計",
    ]
    assert result.loc[10, "グループ名"] == "アルファ"
    assert result.loc[10, "ライブ出演回数"] == 2
    assert result.loc[10, "特典会出演回数"] == 1
    assert result.loc[10, "合計"] == 3
    assert result.loc[11, "ライブ出演回数"] == 1
    assert result.loc[11, "特典会出演回数"] == 0
    assert result.loc[11, "合計"] == 1
    assert result.loc[12, "ライブ出演回数"] == 0
    assert result.loc[12, "特典会出演回数"] == 1
    assert result.loc[12, "合計"] == 1
    assert list(result.index) == [10, 11, 12]


def test_build_group_appearance_count_empty_returns_empty():
    df_stage = _make_stage_df([(0, "メイン", False)])
    df_live = _make_live_df([])

    result = _output.build_group_appearance_count(df_live, df_stage)

    assert list(result.columns) == [
        "グループ名", "ライブ出演回数", "特典会出演回数", "合計",
    ]
    assert len(result) == 0
    assert result.index.name == "グループID"


def test_build_event_output_includes_duration_and_group_count(project_pre_confirm):
    pj_path, project_info_json = project_pre_confirm

    result = _output.build_event_output(pj_path, "event_1", 0, project_info_json)

    assert "duration_distribution" in result
    assert "group_count" in result
    dist = result["duration_distribution"]
    grp = result["group_count"]
    assert list(dist.columns) == ["ライブステージ", "特典会ステージ"]
    assert list(grp.columns) == [
        "グループ名", "ライブ出演回数", "特典会出演回数", "合計",
    ]


# ---------------------------------------------------------------------------
# build_overlap_alerts / build_group_appearances
# ---------------------------------------------------------------------------

def _make_live_df_full(rows):
    """rows = [(出番ID, グループID, グループ名, ステージID, ステージ名,
              ライブ_from, ライブ_to, 長さ(分), 備考), ...]"""
    df = pd.DataFrame(
        {
            "ライブ_from": [r[5] for r in rows],
            "ライブ_to": [r[6] for r in rows],
            "ライブ_長さ(分)": [r[7] for r in rows],
            "グループID": [r[1] for r in rows],
            "ステージID": [r[3] for r in rows],
            "グループ名_raw": [r[2] for r in rows],
            "グループ名": [r[2] for r in rows],
            "ステージ名": [r[4] for r in rows],
            "備考": [r[8] for r in rows],
        },
        index=pd.Index([r[0] for r in rows], name="出番ID"),
    )
    return df


def test_build_overlap_alerts_detects_overlapping_pair():
    df_stage = _make_stage_df([
        (0, "メイン", False),
        (1, "サブ", False),
    ])
    df_live = _make_live_df_full([
        (0, 10, "アルファ", 0, "メイン", "11:00", "11:30", 30, ""),
        (1, 10, "アルファ", 1, "サブ", "11:15", "11:45", 30, ""),
        (2, 11, "ベータ", 0, "メイン", "12:00", "12:30", 30, ""),
    ])

    result = _output.build_overlap_alerts(df_live, df_stage)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["グループID"] == 10
    assert row["グループ名"] == "アルファ"
    assert {row["出番ID_1"], row["出番ID_2"]} == {0, 1}
    assert row["重複(分)"] == 15


def test_build_overlap_alerts_no_overlap_returns_empty():
    df_stage = _make_stage_df([(0, "メイン", False)])
    df_live = _make_live_df_full([
        (0, 10, "アルファ", 0, "メイン", "11:00", "11:30", 30, ""),
        (1, 10, "アルファ", 0, "メイン", "12:00", "12:30", 30, ""),
    ])

    result = _output.build_overlap_alerts(df_live, df_stage)

    assert len(result) == 0
    assert list(result.columns) == [
        "グループID", "グループ名",
        "出番ID_1", "ステージID_1", "ステージ名_1", "開始_1", "終了_1",
        "出番ID_2", "ステージID_2", "ステージ名_2", "開始_2", "終了_2",
        "重複(分)",
    ]


def test_build_overlap_alerts_touching_intervals_not_overlap():
    """11:00-11:30 と 11:30-12:00 は重複扱いしない (境界一致は OK)。"""
    df_stage = _make_stage_df([(0, "メイン", False)])
    df_live = _make_live_df_full([
        (0, 10, "アルファ", 0, "メイン", "11:00", "11:30", 30, ""),
        (1, 10, "アルファ", 0, "メイン", "11:30", "12:00", 30, ""),
    ])

    result = _output.build_overlap_alerts(df_live, df_stage)

    assert len(result) == 0


def test_build_overlap_alerts_cross_stage_kind():
    """ライブステージと特典会ステージを跨いだ重複も検出する。"""
    df_stage = _make_stage_df([
        (0, "メイン", False),       # ライブ
        (1, "ブース", True),        # 特典会
    ])
    df_live = _make_live_df_full([
        (0, 10, "アルファ", 0, "メイン", "11:00", "11:30", 30, ""),
        (1, 10, "アルファ", 1, "ブース", "11:15", "11:45", 30, ""),
    ])

    result = _output.build_overlap_alerts(df_live, df_stage)

    assert len(result) == 1
    assert result.iloc[0]["重複(分)"] == 15


def test_build_overlap_alerts_three_way_overlap_yields_three_pairs():
    df_stage = _make_stage_df([(0, "メイン", False)])
    df_live = _make_live_df_full([
        (0, 10, "アルファ", 0, "メイン", "11:00", "12:00", 60, ""),
        (1, 10, "アルファ", 0, "メイン", "11:15", "12:15", 60, ""),
        (2, 10, "アルファ", 0, "メイン", "11:30", "12:30", 60, ""),
    ])

    result = _output.build_overlap_alerts(df_live, df_stage)

    # 3つすべて互いに重なる → C(3,2) = 3 ペア
    assert len(result) == 3


def test_build_overlap_alerts_empty_returns_empty():
    df_stage = _make_stage_df([(0, "メイン", False)])
    df_live = _make_live_df_full([])

    result = _output.build_overlap_alerts(df_live, df_stage)
    assert len(result) == 0


def test_build_group_appearances_returns_all_groups():
    df_stage = _make_stage_df([
        (0, "メイン", False),
        (1, "ブース", True),
    ])
    df_live = _make_live_df_full([
        (0, 10, "アルファ", 0, "メイン", "11:00", "11:30", 30, ""),
        (1, 10, "アルファ", 1, "ブース", "12:00", "12:30", 30, "特典会"),
        (2, 11, "ベータ", 0, "メイン", "13:00", "13:25", 25, ""),
    ])

    result = _output.build_group_appearances(df_live, df_stage)

    assert result.index.name == "出番ID"
    assert list(result.columns) == [
        "グループID", "グループ名", "ステージID", "ステージ名",
        "ライブ_from", "ライブ_to", "ライブ_長さ(分)", "備考",
    ]
    # 全3件保持される
    assert len(result) == 3
    # グループID=10 で絞り込めば 2 件
    alpha = result[result["グループID"] == 10]
    assert len(alpha) == 2
    assert list(alpha["ステージ名"]) == ["メイン", "ブース"]


def test_build_group_appearances_empty_returns_empty():
    df_stage = _make_stage_df([(0, "メイン", False)])
    df_live = _make_live_df_full([])

    result = _output.build_group_appearances(df_live, df_stage)
    assert len(result) == 0
    assert list(result.columns) == [
        "グループID", "グループ名", "ステージID", "ステージ名",
        "ライブ_from", "ライブ_to", "ライブ_長さ(分)", "備考",
    ]


def test_build_event_output_includes_overlap_and_appearances(project_pre_confirm):
    pj_path, project_info_json = project_pre_confirm

    result = _output.build_event_output(pj_path, "event_1", 0, project_info_json)

    assert "overlap_alerts" in result
    assert "group_appearances" in result
    assert list(result["group_appearances"].columns) == [
        "グループID", "グループ名", "ステージID", "ステージ名",
        "ライブ_from", "ライブ_to", "ライブ_長さ(分)", "備考",
    ]


def test_listup_new_idolname_skips_empty_event(tmp_path, monkeypatch):
    project_info_json, output_df = _make_two_event_project(tmp_path)
    event_list = ["event_1", "event_2"]

    # idolname.detect_new_data の依存をスタブ化
    from backend_functions import idolname as _idolname_mod
    monkeypatch.setattr(_idolname_mod, "detect_new_data", lambda names: list(names))

    df = _output.listup_new_idolname(output_df, event_list)

    # event_1 の "ZZグループ" だけ抽出される(event_2は空dictで無視)
    assert "ZZグループ" in df["グループ名"].tolist()
