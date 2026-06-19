"""ocr_service.autodetect_collab_single / _all のテスト。

stage_N.json を直接読み書きするバッチ用コラボ検出が、
timetabledata.autodetect_collab_groups と同じ採番ルール
(同一 ライブ_from を 2件以上で同一 コラボグループID 採番・冪等) を
json 入出力で正しく満たすことを検証する。
"""

import json
import os

from backend_functions import ocr_service as ocr


def _write_stage(base, stage_no, items):
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, f"stage_{stage_no}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"ステージ名": "メイン", "タイムテーブル": items}, f, ensure_ascii=False)
    return path


def _read_items(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["タイムテーブル"]


def _item(name, live_from, cgid=None, turn_id=None):
    return {
        "グループ名": name,
        "グループ名_採用": name,
        "ライブステージ": {"from": live_from, "to": ""},
        "コラボグループID": cgid,
        "出番ID": turn_id,
    }


def test_same_from_two_rows_get_same_cgid(tmp_path):
    """同じ ライブ_from の行が2件 → 同一 コラボグループID を採番。"""
    base = tmp_path / "ev" / "live"
    path = _write_stage(str(base), 0, [
        _item("A", "12:00"),
        _item("B", "12:00"),
        _item("C", "13:00"),
    ])
    ocr.autodetect_collab_single(0, str(tmp_path), "ev", "live")
    items = _read_items(path)
    assert items[0]["コラボグループID"] == items[1]["コラボグループID"]
    assert items[0]["コラボグループID"] is not None
    # 単独行 (13:00) は採番されない
    assert items[2]["コラボグループID"] is None


def test_single_row_not_grouped(tmp_path):
    """同一 from が1件のみ → 採番しない (冪等)。"""
    base = tmp_path / "ev" / "live"
    path = _write_stage(str(base), 0, [
        _item("A", "12:00"),
        _item("B", "13:00"),
    ])
    ocr.autodetect_collab_single(0, str(tmp_path), "ev", "live")
    items = _read_items(path)
    assert all(it["コラボグループID"] is None for it in items)


def test_existing_cgid_preserved(tmp_path):
    """既に コラボグループID が入っている行は変更しない。"""
    base = tmp_path / "ev" / "live"
    path = _write_stage(str(base), 0, [
        _item("A", "12:00", cgid=5),
        _item("B", "12:00", cgid=5),
        _item("C", "14:00"),
        _item("D", "14:00"),
    ])
    ocr.autodetect_collab_single(0, str(tmp_path), "ev", "live")
    items = _read_items(path)
    assert items[0]["コラボグループID"] == 5
    assert items[1]["コラボグループID"] == 5
    # 新規グループは既存最大値+1=6 から採番
    assert items[2]["コラボグループID"] == items[3]["コラボグループID"] == 6


def test_clear_turn_id_clears_only_new_groups(tmp_path):
    """clear_turn_id=True で新規採番行の 出番ID のみクリアする。"""
    base = tmp_path / "ev" / "live"
    path = _write_stage(str(base), 0, [
        _item("A", "12:00", turn_id=10),
        _item("B", "12:00", turn_id=11),
        _item("C", "15:00", turn_id=12),
    ])
    ocr.autodetect_collab_single(0, str(tmp_path), "ev", "live", clear_turn_id=True)
    items = _read_items(path)
    # コラボ採番された A/B は出番IDクリア
    assert items[0]["出番ID"] is None
    assert items[1]["出番ID"] is None
    # 単独行 C は維持
    assert items[2]["出番ID"] == 12


def test_all_iterates_stages(tmp_path):
    """autodetect_collab_all が全ステージを処理する。"""
    base = tmp_path / "ev" / "live"
    p0 = _write_stage(str(base), 0, [_item("A", "12:00"), _item("B", "12:00")])
    p1 = _write_stage(str(base), 1, [_item("C", "13:00"), _item("D", "13:00")])
    ocr.autodetect_collab_all(str(tmp_path), "ev", "live", stage_num=2)
    i0, i1 = _read_items(p0), _read_items(p1)
    assert i0[0]["コラボグループID"] == i0[1]["コラボグループID"] is not None
    assert i1[0]["コラボグループID"] == i1[1]["コラボグループID"] is not None


def test_missing_file_noop(tmp_path):
    """ファイルが無くても例外を出さない。"""
    ocr.autodetect_collab_single(0, str(tmp_path), "ev", "live")
