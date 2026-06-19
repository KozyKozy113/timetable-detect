"""get_idolname_confirmed_list のテスト。

「確定」の判定を IDマスタ確定済フラグ (3点CSVの存在) に置き換え、
グループ一覧のソースを master_idolname.csv とした挙動を検証する。
未確定 (3点CSV未保存) のときは空リストを返し、呼び出し側で全マスタ補正へ
フォールバックさせる。
"""

import os

from backend_functions import ocr_service as ocr


_ID_CSVS = ("master_stage.csv", "master_idolname.csv", "turn_id_data.csv")


def _write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _make_event(base, idolname_csv_text):
    """3点CSVを揃えたイベントディレクトリを作る。master_idolname の中身は指定。"""
    os.makedirs(base)
    _write(os.path.join(base, "master_stage.csv"), "ステージID,ステージ名\n0,メイン\n")
    _write(os.path.join(base, "turn_id_data.csv"), "出番ID,グループ名\n0,A\n")
    _write(os.path.join(base, "master_idolname.csv"), idolname_csv_text)


def test_confirmed_reads_master_idolname_saiyou_column(tmp_path):
    """確定済 + 列名が グループ名_採用 (現行保存形式)。"""
    base = tmp_path / "ev"
    _make_event(base, "グループID,グループ名_採用\n0,アルファ\n1,ベータ\n")
    result = ocr.get_idolname_confirmed_list(str(tmp_path), "ev", {})
    assert sorted(result) == ["アルファ", "ベータ"]


def test_unconfirmed_returns_empty_when_csv_missing(tmp_path):
    """3点CSVが揃っていない (未確定) → 空リスト。"""
    base = tmp_path / "ev"
    os.makedirs(base)
    # master_idolname.csv はあるが他が無い → event_ids_assigned False
    _write(os.path.join(base, "master_idolname.csv"), "グループID,グループ名_採用\n0,アルファ\n")
    result = ocr.get_idolname_confirmed_list(str(tmp_path), "ev", {})
    assert result == []


def test_confirmed_but_no_saiyou_column_returns_empty(tmp_path):
    """確定済だが グループ名_採用 列が無い (旧 グループ名 形式含む) → 空リスト。

    旧列名はサポートしないため、呼び出し側で全マスタ補正へフォールバックさせる。
    """
    base = tmp_path / "ev"
    _make_event(base, "グループID,グループ名\n0,アルファ\n")
    result = ocr.get_idolname_confirmed_list(str(tmp_path), "ev", {})
    assert result == []


def test_confirmed_skips_blank_and_dedups(tmp_path):
    """空欄/重複は除外・重複排除される。"""
    base = tmp_path / "ev"
    _make_event(
        base,
        "グループID,グループ名_採用\n0,アルファ\n1,\n2,アルファ\n3,ベータ\n",
    )
    result = ocr.get_idolname_confirmed_list(str(tmp_path), "ev", {})
    assert sorted(result) == ["アルファ", "ベータ"]
