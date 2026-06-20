"""ocr_service.fill_empty_adopted_names のテスト。

⑤変更比較で新規検出したグループ（採用名を空にしたレコード）にだけ補正を適用し、
既存の採用名は触らないことを確認する。確定リスト経路で検証（埋め込みモデル不要）。
"""

from backend_functions import ocr_service


def test_fills_only_empty_adopted_names_with_confirmed_list():
    stage_json = {
        "タイムテーブル": [
            # 既存の採用名は維持される
            {"グループ名": "Araw", "グループ名_採用": "既存採用A"},
            # 採用名が空 → 確定リストから補正される
            {"グループ名": "アイドルカレッジ", "グループ名_採用": ""},
            # 採用名が None → 補正される
            {"グループ名": "わーすた", "グループ名_採用": None},
        ],
    }
    confirmed = ["アイドルカレッジ", "わーすた", "別グループ"]
    out = ocr_service.fill_empty_adopted_names(stage_json, confirmed_list=confirmed)
    rows = out["タイムテーブル"]
    assert rows[0]["グループ名_採用"] == "既存採用A"        # 既存は不変
    assert rows[1]["グループ名_採用"] == "アイドルカレッジ"  # 補正で充当
    assert rows[2]["グループ名_採用"] == "わーすた"


def test_skips_records_without_raw_name():
    stage_json = {"タイムテーブル": [{"グループ名": "", "グループ名_採用": ""}]}
    out = ocr_service.fill_empty_adopted_names(stage_json, confirmed_list=["X"])
    assert out["タイムテーブル"][0]["グループ名_採用"] == ""
