"""stella_reserve.py (Phase 4 Reserve-First 採番ロジック) のテスト。

純粋ロジック (bundleId 算出 / 採番計画 / バリデーション) と、GitHub I/O を
monkeypatch したオーケストレーション (採番成功 / push 失敗ロールバック) を検証する。
"""

import json
import os

import pytest

from backend_functions import stella_reserve as sr
from backend_functions import github_ops


# ---------------------------------------------------------------------------
# project_info_json ビルダ
# ---------------------------------------------------------------------------

def _meta(date="20260504", dow=1, **extra):
    m = {
        "date": date,
        "dow": dow,
        "openTime": "",
        "closeTime": "",
        "notificationVersion": "1",
        "notification": "",
        "_last_pushed_notification": None,
    }
    m.update(extra)
    return m


def _pij(event_metas, *, live_name="テストライブ", genre=2, release=0, pref=13):
    """event_metas は各 event の stella_metadata dict のリスト。"""
    return {
        "project_name": "pj",
        "event_num": len(event_metas),
        "stella_project_meta": {
            "liveName": live_name,
            "genre": genre,
            "release": release,
            "pref": pref,
        },
        "event_detail": [
            {"event_no": i, "event_name": f"event_{i + 1}", "stella_metadata": m}
            for i, m in enumerate(event_metas)
        ],
    }


# ---------------------------------------------------------------------------
# compute_dow
# ---------------------------------------------------------------------------

def test_compute_dow_monday():
    assert sr.compute_dow("20260504") == 1  # 2026-05-04 は月曜


def test_compute_dow_sunday():
    assert sr.compute_dow("20260510") == 7  # 2026-05-10 は日曜


def test_compute_dow_invalid_returns_none():
    assert sr.compute_dow("") is None
    assert sr.compute_dow("not-a-date") is None


# ---------------------------------------------------------------------------
# is_reserved
# ---------------------------------------------------------------------------

def test_is_reserved():
    assert sr.is_reserved({"liveId": 547}) is True
    assert sr.is_reserved({"liveId": "547"}) is True
    assert sr.is_reserved({}) is False
    assert sr.is_reserved({"liveId": None}) is False
    assert sr.is_reserved({"liveId": ""}) is False


# ---------------------------------------------------------------------------
# compute_bundle_id (Phase 4-5)
# ---------------------------------------------------------------------------

def test_compute_bundle_id_single_event_is_empty():
    assert sr.compute_bundle_id(1, [], [603]) == ""


def test_compute_bundle_id_all_new_uses_min_new_id():
    assert sr.compute_bundle_id(2, [], [605, 604]) == "604"


def test_compute_bundle_id_existing_non_empty_wins():
    assert sr.compute_bundle_id(3, ["604", ""], [606]) == "604"


def test_compute_bundle_id_existing_all_empty_falls_back_to_new():
    # event_num=1 → 2 になった直後など、既存 bundleId が "" のみ
    assert sr.compute_bundle_id(2, [""], [610]) == "610"


# ---------------------------------------------------------------------------
# build_live_list_entry
# ---------------------------------------------------------------------------

def test_build_live_list_entry_full_shape():
    pm = {"liveName": "歌舞伎町UP GATE", "genre": 2, "release": 2, "pref": 13}
    entry = sr.build_live_list_entry(547, "547", pm, _meta(date="20260504", dow=1))
    assert entry == {
        "liveId": 547,
        "bundleId": "547",
        "liveName": "歌舞伎町UP GATE",
        "release": 2,
        "pref": 13,
        "date": "20260504",
        "dow": 1,
        "genre": 2,
        "serchKey": "0",
    }


def test_build_live_list_entry_recomputes_dow_when_missing():
    pm = {"liveName": "x", "genre": 2, "release": 0, "pref": 13}
    entry = sr.build_live_list_entry(1, "", pm, _meta(date="20260510", dow=None))
    assert entry["dow"] == 7  # date から再計算


# ---------------------------------------------------------------------------
# validate_for_reservation
# ---------------------------------------------------------------------------

def test_validate_ok():
    pij = _pij([_meta()])
    assert sr.validate_for_reservation(pij) == []


def test_validate_missing_live_name():
    pij = _pij([_meta()], live_name="")
    errors = sr.validate_for_reservation(pij)
    assert any("liveName" in e for e in errors)


def test_validate_missing_date():
    pij = _pij([_meta(date="", dow=None)])
    errors = sr.validate_for_reservation(pij)
    assert any("公演日" in e for e in errors)


def test_validate_skips_date_check_for_reserved_event():
    # 既採番 event は date 未入力でもチェック対象外
    pij = _pij([_meta(date="", dow=None, liveId=500, bundleId="")])
    assert sr.validate_for_reservation(pij) == []


# ---------------------------------------------------------------------------
# plan_reservations
# ---------------------------------------------------------------------------

def test_plan_single_event_new():
    pij = _pij([_meta()])
    plan = sr.plan_reservations(pij, max_existing_live_id=602)
    assert plan.assignments == {0: 603}
    assert plan.bundle_id == ""  # event_num==1
    assert len(plan.entries) == 1
    assert plan.entries[0]["liveId"] == 603
    assert plan.entries[0]["bundleId"] == ""
    assert plan.rebundled_event_nos == []


def test_plan_two_events_all_new():
    pij = _pij([_meta(date="20260504", dow=1), _meta(date="20260505", dow=2)])
    plan = sr.plan_reservations(pij, max_existing_live_id=602)
    assert plan.assignments == {0: 603, 1: 604}
    assert plan.bundle_id == "603"  # 新規採番の最小値
    assert {e["liveId"] for e in plan.entries} == {603, 604}
    assert all(e["bundleId"] == "603" for e in plan.entries)


def test_plan_partial_keeps_reserved_assigns_unreserved():
    # event0 は既採番 (liveId=600, bundleId=600), event1 のみ未採番
    pij = _pij([
        _meta(date="20260504", dow=1, liveId=600, bundleId="600"),
        _meta(date="20260505", dow=2),
    ])
    plan = sr.plan_reservations(pij, max_existing_live_id=602)
    assert plan.assignments == {1: 603}  # event0 は保持
    assert plan.bundle_id == "600"  # 既存非空 bundleId に揃える
    # event1 の新規エントリのみ (event0 は bundleId 不変なので再書込不要)
    assert [e["liveId"] for e in plan.entries] == [603]
    assert plan.entries[0]["bundleId"] == "600"
    assert plan.rebundled_event_nos == []


def test_plan_event_num_grew_rebundles_existing_empty():
    # 元 event_num=1 で採番済 (bundleId="") → event 追加して再採番
    pij = _pij([
        _meta(date="20260504", dow=1, liveId=600, bundleId=""),
        _meta(date="20260505", dow=2),
    ])
    plan = sr.plan_reservations(pij, max_existing_live_id=602)
    assert plan.assignments == {1: 603}
    assert plan.bundle_id == "603"  # 既存は "" のみ → 新規最小値
    # event0 (bundleId "" → "603") も再書込対象
    assert plan.rebundled_event_nos == [0]
    by_id = {e["liveId"]: e for e in plan.entries}
    assert by_id[600]["bundleId"] == "603"
    assert by_id[603]["bundleId"] == "603"


def test_plan_noop_when_all_reserved_no_rebundle():
    pij = _pij([
        _meta(liveId=600, bundleId="600"),
        _meta(liveId=601, bundleId="600"),
    ])
    plan = sr.plan_reservations(pij, max_existing_live_id=602)
    assert plan.assignments == {}
    assert plan.is_noop is True


# ---------------------------------------------------------------------------
# max_live_id
# ---------------------------------------------------------------------------

def _write_live_list(path, ids):
    with open(path, "w", encoding="utf-8-sig") as f:
        json.dump({"liveList": [{"liveId": i} for i in ids]}, f)


def test_max_live_id(tmp_path):
    p = tmp_path / "liveList.json"
    _write_live_list(p, [1, 602, 300])
    assert sr.max_live_id(str(p)) == 602


def test_max_live_id_empty_or_missing(tmp_path):
    assert sr.max_live_id(str(tmp_path / "nope.json")) == 0
    p = tmp_path / "liveList.json"
    _write_live_list(p, [])
    assert sr.max_live_id(str(p)) == 0


# ---------------------------------------------------------------------------
# clear_all_reservations
# ---------------------------------------------------------------------------

def test_clear_all_reservations():
    pij = _pij([
        _meta(liveId=600, bundleId="600", jsonVersion=3,
              _last_pushed_notification="old"),
        _meta(liveId=601, bundleId="600", jsonVersion=1),
    ])
    sr.clear_all_reservations(pij)
    for ev in pij["event_detail"]:
        m = ev["stella_metadata"]
        assert "liveId" not in m
        assert "bundleId" not in m
        assert "jsonVersion" not in m
        assert m["_last_pushed_notification"] is None
        # 入力情報は保持
        assert m["date"] == "20260504"


# ---------------------------------------------------------------------------
# reserve_live_ids オーケストレーション (github_ops を monkeypatch)
# ---------------------------------------------------------------------------

class _FakeGit:
    """github_ops の関数群を差し替える簡易フェイク。

    内側リポは tmp ディレクトリ + liveList.json で表現。push 成功/失敗を切替可能。
    """

    def __init__(self, repo_dir, existing_ids, fail_push=False):
        self.repo_dir = str(repo_dir)
        self.live_list_path = os.path.join(self.repo_dir, "liveList.json")
        self.fail_push = fail_push
        self.reset_called_with = None
        self.commit_messages = []
        _write_live_list(self.live_list_path, existing_ids)
        # before スナップショット (ロールバック検証用)
        with open(self.live_list_path, encoding="utf-8-sig") as f:
            self._snapshot = f.read()

    def clone_or_pull(self, local_path=None, branch=None):
        return "BEFORE_HASH"

    def commit_and_push(self, files, message, local_path=None, branch=None):
        self.commit_messages.append(message)
        self.committed_files = list(files)
        if self.fail_push:
            raise github_ops.GithubPushError("push failed (simulated)")
        return "AFTER_HASH"

    def reset_hard(self, commit_hash, local_path=None):
        self.reset_called_with = commit_hash
        # 実際の git reset --hard 同様、ファイルをスナップショットに戻す
        with open(self.live_list_path, "w", encoding="utf-8-sig") as f:
            f.write(self._snapshot)


@pytest.fixture
def patch_git(monkeypatch):
    def _apply(fake):
        monkeypatch.setattr(github_ops, "clone_or_pull", fake.clone_or_pull)
        monkeypatch.setattr(github_ops, "commit_and_push", fake.commit_and_push)
        monkeypatch.setattr(github_ops, "reset_hard", fake.reset_hard)
        return fake
    return _apply


def _make_pj_dir(tmp_path, pij):
    pj_dir = tmp_path / "pj"
    pj_dir.mkdir()
    with open(pj_dir / "project_info.json", "w", encoding="utf-8") as f:
        json.dump(pij, f, ensure_ascii=False)
    return str(pj_dir)


def test_reserve_success_writes_metadata_and_live_list(tmp_path, patch_git):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    fake = patch_git(_FakeGit(repo_dir, existing_ids=[600, 601, 602]))

    pij = _pij([_meta(date="20260504", dow=1), _meta(date="20260505", dow=2)])
    pj_path = _make_pj_dir(tmp_path, pij)

    result = sr.reserve_live_ids(pij, pj_path, repo_path=str(repo_dir))

    assert result.pushed is True
    # メタデータ確定
    m0 = pij["event_detail"][0]["stella_metadata"]
    m1 = pij["event_detail"][1]["stella_metadata"]
    assert m0["liveId"] == 603 and m0["bundleId"] == "603"
    assert m1["liveId"] == 604 and m1["bundleId"] == "603"
    # liveList.json に追記されている
    ids = {e["liveId"] for e in sr._read_live_list(fake.live_list_path)}
    assert {600, 601, 602, 603, 604} <= ids
    # project_info.json が保存されている
    with open(os.path.join(pj_path, "project_info.json"), encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["event_detail"][0]["stella_metadata"]["liveId"] == 603


def test_reserve_push_failure_rolls_back_and_keeps_local_clean(tmp_path, patch_git):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    fake = patch_git(_FakeGit(repo_dir, existing_ids=[600], fail_push=True))

    pij = _pij([_meta(date="20260504", dow=1)])
    pj_path = _make_pj_dir(tmp_path, pij)

    with pytest.raises(github_ops.GithubPushError):
        sr.reserve_live_ids(pij, pj_path, repo_path=str(repo_dir))

    # ロールバックが呼ばれた
    assert fake.reset_called_with == "BEFORE_HASH"
    # ローカル project_info_json は無変更 (liveId 未確定)
    assert "liveId" not in pij["event_detail"][0]["stella_metadata"]
    # liveList.json はスナップショットに戻っている
    ids = {e["liveId"] for e in sr._read_live_list(fake.live_list_path)}
    assert ids == {600}
    # project_info.json は保存されていない (採番前の状態)
    with open(os.path.join(pj_path, "project_info.json"), encoding="utf-8") as f:
        saved = json.load(f)
    assert "liveId" not in saved["event_detail"][0]["stella_metadata"]


def test_reserve_validation_error_does_not_touch_git(tmp_path, patch_git):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    fake = patch_git(_FakeGit(repo_dir, existing_ids=[600]))

    pij = _pij([_meta()], live_name="")  # liveName 未入力
    pj_path = _make_pj_dir(tmp_path, pij)

    with pytest.raises(sr.ReservationValidationError):
        sr.reserve_live_ids(pij, pj_path, repo_path=str(repo_dir))
    assert fake.commit_messages == []


def test_reserve_noop_when_all_reserved(tmp_path, patch_git):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    fake = patch_git(_FakeGit(repo_dir, existing_ids=[600, 601]))

    pij = _pij([
        _meta(liveId=600, bundleId="600"),
        _meta(liveId=601, bundleId="600"),
    ])
    pj_path = _make_pj_dir(tmp_path, pij)

    result = sr.reserve_live_ids(pij, pj_path, repo_path=str(repo_dir))
    assert result.pushed is False
    assert fake.commit_messages == []  # push していない


def test_re_reserve_assigns_only_unreserved(tmp_path, patch_git):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    fake = patch_git(_FakeGit(repo_dir, existing_ids=[600, 601, 602]))

    pij = _pij([
        _meta(date="20260504", dow=1, liveId=600, bundleId="600"),
        _meta(date="20260505", dow=2),  # 後追加 event
    ])
    pj_path = _make_pj_dir(tmp_path, pij)

    result = sr.re_reserve(pij, pj_path, repo_path=str(repo_dir))
    assert result.pushed is True
    assert pij["event_detail"][0]["stella_metadata"]["liveId"] == 600  # 保持
    assert pij["event_detail"][1]["stella_metadata"]["liveId"] == 603  # 追番


# ---------------------------------------------------------------------------
# liveListPanel2.json 連携 (採番と同時に追記)
# ---------------------------------------------------------------------------

def test_reserve_writes_and_commits_panel2(tmp_path, patch_git):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    fake = patch_git(_FakeGit(repo_dir, existing_ids=[600, 601, 602]))

    pij = _pij([_meta(date="20260504", dow=1), _meta(date="20260505", dow=2)])
    pj_path = _make_pj_dir(tmp_path, pij)

    result = sr.reserve_live_ids(pij, pj_path, repo_path=str(repo_dir))
    assert result.pushed is True
    # 採番 push に liveList.json と liveListPanel2.json の両方が含まれる
    assert set(fake.committed_files) == {"liveList.json", "liveListPanel2.json"}
    # Panel2 に bundle [603, 604] が 2026/5 へ追記されている
    panel = _read_panel(os.path.join(str(repo_dir), "liveListPanel2.json"))
    y = next(y for y in panel if y["year"] == 2026)
    may = next(m for m in y["monthList"] if m["month"] == 5)
    assert [603, 604] in may["liveIdlist"]


def test_reserve_panel2_skipped_when_noop(tmp_path, patch_git):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    fake = patch_git(_FakeGit(repo_dir, existing_ids=[600]))
    pij = _pij([_meta(liveId=600, bundleId="")])  # 採番済 → no-op
    pj_path = _make_pj_dir(tmp_path, pij)

    result = sr.reserve_live_ids(pij, pj_path, repo_path=str(repo_dir))
    assert result.pushed is False
    assert fake.commit_messages == []


def _read_panel(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# is_linked / build_current_live_list_entries
# ---------------------------------------------------------------------------

def test_is_linked():
    assert sr.is_linked(_pij([_meta()])) is False
    assert sr.is_linked(_pij([_meta(liveId=600, bundleId="")])) is True


def test_build_current_live_list_entries_skips_unreserved():
    pij = _pij([
        _meta(liveId=600, bundleId="600"),
        _meta(),  # 未採番 → 除外
    ], live_name="X", release=2)
    entries = sr.build_current_live_list_entries(pij)
    assert [e["liveId"] for e in entries] == [600]
    assert entries[0]["release"] == 2
    assert entries[0]["bundleId"] == "600"


def test_build_current_live_list_entries_release_override():
    pij = _pij([_meta(liveId=600, bundleId="")], release=0)
    entries = sr.build_current_live_list_entries(pij, release_override=2)
    assert entries[0]["release"] == 2
    # 実 project_meta は変更しない
    assert pij["stella_project_meta"]["release"] == 0


# ---------------------------------------------------------------------------
# resync_live_list
# ---------------------------------------------------------------------------

def _seed_repo_with_entries(repo_dir, entries):
    p = os.path.join(str(repo_dir), "liveList.json")
    with open(p, "w", encoding="utf-8-sig") as f:
        json.dump({"liveList": entries}, f, ensure_ascii=False)


def _seed_panel_for(repo_dir, pij):
    """pij の採番済 bundle を反映した liveListPanel2.json を repo に配置する。"""
    from backend_functions import stella_panel
    live_ids, first_date = sr.reserved_bundle(pij)
    live_dates = {}
    for ev in pij["event_detail"]:
        m = ev["stella_metadata"]
        if m.get("liveId") not in (None, ""):
            live_dates[int(m["liveId"])] = m.get("date", "")
    panel = []
    if live_ids and first_date:
        stella_panel.upsert_bundle(panel, live_ids, first_date, live_dates)
    stella_panel.write_panel(
        os.path.join(str(repo_dir), "liveListPanel2.json"), panel,
    )


def test_resync_noop_when_unlinked(tmp_path, patch_git):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    fake = patch_git(_FakeGit(repo_dir, existing_ids=[]))
    pij = _pij([_meta()])  # 未採番
    pj_path = _make_pj_dir(tmp_path, pij)

    result = sr.resync_live_list(pij, pj_path, repo_path=str(repo_dir))
    assert result.pushed is False
    assert fake.commit_messages == []


def test_resync_noop_when_no_diff(tmp_path, patch_git):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    fake = patch_git(_FakeGit(repo_dir, existing_ids=[]))
    pij = _pij([_meta(liveId=600, bundleId="600")], live_name="X", release=2)
    pj_path = _make_pj_dir(tmp_path, pij)
    # 既存 liveList / Panel2 が現在値と完全一致 → 差分なし
    _seed_repo_with_entries(repo_dir, sr.build_current_live_list_entries(pij))
    _seed_panel_for(repo_dir, pij)

    result = sr.resync_live_list(pij, pj_path, repo_path=str(repo_dir))
    assert result.pushed is False
    assert fake.commit_messages == []


def test_resync_pushes_changed_release(tmp_path, patch_git):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    fake = patch_git(_FakeGit(repo_dir, existing_ids=[]))
    pij = _pij([_meta(liveId=600, bundleId="600")], live_name="X", release=0)
    pj_path = _make_pj_dir(tmp_path, pij)
    # 既存は release=0、ローカルで release=2 に変えた想定 (date は不変 → Panel2 据え置き)
    _seed_repo_with_entries(repo_dir, sr.build_current_live_list_entries(pij))
    _seed_panel_for(repo_dir, pij)
    pij["stella_project_meta"]["release"] = 2

    result = sr.resync_live_list(pij, pj_path, repo_path=str(repo_dir))
    assert result.pushed is True
    assert result.updated_live_ids == [600]
    # liveList.json に release=2 が反映
    saved = sr._read_live_list(fake.live_list_path)
    assert saved[0]["release"] == 2
    # date 不変なので Panel2 は push 対象外
    assert fake.committed_files == ["liveList.json"]


def test_resync_release_override_persists_on_success(tmp_path, patch_git):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    fake = patch_git(_FakeGit(repo_dir, existing_ids=[]))
    pij = _pij([_meta(liveId=600, bundleId="600")], live_name="X", release=0)
    pj_path = _make_pj_dir(tmp_path, pij)
    _seed_repo_with_entries(repo_dir, sr.build_current_live_list_entries(pij))
    _seed_panel_for(repo_dir, pij)

    result = sr.resync_live_list(pij, pj_path, repo_path=str(repo_dir), release_override=2)
    assert result.pushed is True
    # release が project_meta に確定 + 保存される
    assert pij["stella_project_meta"]["release"] == 2
    with open(os.path.join(pj_path, "project_info.json"), encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk["stella_project_meta"]["release"] == 2


def test_resync_push_failure_rolls_back_no_persist(tmp_path, patch_git):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    fake = patch_git(_FakeGit(repo_dir, existing_ids=[], fail_push=True))
    pij = _pij([_meta(liveId=600, bundleId="600")], live_name="X", release=0)
    pj_path = _make_pj_dir(tmp_path, pij)
    _seed_repo_with_entries(repo_dir, sr.build_current_live_list_entries(pij))

    with pytest.raises(github_ops.GithubPushError):
        sr.resync_live_list(pij, pj_path, repo_path=str(repo_dir), release_override=2)
    assert fake.reset_called_with == "BEFORE_HASH"
    # release は確定していない
    assert pij["stella_project_meta"]["release"] == 0


def test_resync_backfills_missing_panel_entry(tmp_path, patch_git):
    # 採番だけ先に済ませ Panel2 への記入が漏れたプロジェクトのバックフィル
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    fake = patch_git(_FakeGit(repo_dir, existing_ids=[]))
    pij = _pij([_meta(date="20260504", dow=1, liveId=600, bundleId="600")], live_name="X")
    pj_path = _make_pj_dir(tmp_path, pij)
    # liveList は既に正しい / Panel2 は未記入 (seed しない)
    _seed_repo_with_entries(repo_dir, sr.build_current_live_list_entries(pij))

    result = sr.resync_live_list(pij, pj_path, repo_path=str(repo_dir))
    assert result.pushed is True
    # liveList は差分なし → Panel2 のみ push
    assert fake.committed_files == ["liveListPanel2.json"]
    panel = _read_panel(os.path.join(str(repo_dir), "liveListPanel2.json"))
    y = next(y for y in panel if y["year"] == 2026)
    may = next(m for m in y["monthList"] if m["month"] == 5)
    assert may["liveIdlist"] == [[600]]


def test_resync_repositions_panel_on_date_change(tmp_path, patch_git):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    fake = patch_git(_FakeGit(repo_dir, existing_ids=[]))
    # 既存: 2026/05/04 で採番済 → liveList / Panel2 を 5月で seed
    pij = _pij([_meta(date="20260504", dow=1, liveId=600, bundleId="600")], live_name="X")
    pj_path = _make_pj_dir(tmp_path, pij)
    _seed_repo_with_entries(repo_dir, sr.build_current_live_list_entries(pij))
    _seed_panel_for(repo_dir, pij)

    # 公演日を 7月へ変更
    pij["event_detail"][0]["stella_metadata"]["date"] = "20260712"
    pij["event_detail"][0]["stella_metadata"]["dow"] = 7

    result = sr.resync_live_list(pij, pj_path, repo_path=str(repo_dir))
    assert result.pushed is True
    # liveList と Panel2 の両方が push される
    assert set(fake.committed_files) == {"liveList.json", "liveListPanel2.json"}
    # Panel2 で bundle [600] が 5月から消え 7月へ移動
    panel = _read_panel(os.path.join(str(repo_dir), "liveListPanel2.json"))
    y = next(y for y in panel if y["year"] == 2026)
    months = {m["month"]: m["liveIdlist"] for m in y["monthList"]}
    assert 5 not in months          # 5月は空になり消える
    assert months[7] == [[600]]     # 7月へ移動
    assert y["monthEnableList"][4] is False  # 5月 off
    assert y["monthEnableList"][6] is True   # 7月 on
