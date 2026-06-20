"""stella_push.py (Phase 6 / 7-2 ⑥-D GitHub Push) のテスト。

バリデーション・バージョンインクリメント・PR/直接 push・push 失敗ロールバックを、
github_ops を monkeypatch して検証する。
"""

import json
import os

import pandas as pd
import pytest

from backend_functions import stella_push as sp
from backend_functions import github_ops


# ---------------------------------------------------------------------------
# project_info_json / event_output ビルダ
# ---------------------------------------------------------------------------

def _meta(**extra):
    m = {
        "date": "20260504",
        "dow": 1,
        "openTime": "12",
        "closeTime": "23",
        "notificationVersion": "1",
        "notification": "",
        "_last_pushed_notification": None,
    }
    m.update(extra)
    return m


def _pij(event_metas, *, live_name="テストライブ"):
    return {
        "project_name": "pj",
        "event_num": len(event_metas),
        "stella_project_meta": {
            "liveName": live_name, "genre": 2, "release": 0, "pref": 13,
        },
        "event_detail": [
            {"event_no": i, "event_name": f"event_{i + 1}", "stella_metadata": m}
            for i, m in enumerate(event_metas)
        ],
    }


def _event_output():
    """build_event_output 相当の最小データ (stage/idolname/live)。"""
    df_stage = pd.DataFrame(
        {"ステージ名": ["メイン"], "表示順": [0]}, index=[0],
    )
    df_idol = pd.DataFrame({"グループ名_採用": ["グループA"]}, index=[0])
    df_live = pd.DataFrame({
        "出番ID": [0],
        "グループID": [0],
        "ステージID": [0],
        "ライブ_from": ["13:00"],
        "ライブ_長さ(分)": [30],
    })
    return {"stage": df_stage, "idolname": df_idol, "live": df_live}


# ---------------------------------------------------------------------------
# validate_for_push
# ---------------------------------------------------------------------------

def test_validate_ok_single_reserved():
    pij = _pij([_meta(liveId=547, bundleId="")])
    assert sp.validate_for_push(pij, 0) == []


def test_validate_unreserved_blocks():
    pij = _pij([_meta()])
    errors = sp.validate_for_push(pij, 0)
    assert any("採番" in e for e in errors)


def test_validate_multi_event_missing_bundle_id():
    pij = _pij([
        _meta(liveId=547, bundleId=""),
        _meta(liveId=548, bundleId=""),
    ])
    errors = sp.validate_for_push(pij, 0)
    assert any("bundleId" in e for e in errors)


def test_validate_multi_event_with_bundle_id_ok():
    pij = _pij([
        _meta(liveId=547, bundleId="547"),
        _meta(liveId=548, bundleId="547"),
    ])
    assert sp.validate_for_push(pij, 0) == []


# ---------------------------------------------------------------------------
# 補助関数
# ---------------------------------------------------------------------------

def test_feature_branch_name():
    assert sp.feature_branch_name(547, 3) == "stella/live547-v3"


def test_build_commit_message():
    assert "live547" in sp.build_commit_message(547, 2, "X")


# ---------------------------------------------------------------------------
# push オーケストレーション (github_ops を monkeypatch)
# ---------------------------------------------------------------------------

class _FakeGit:
    def __init__(self, repo_dir, fail_push=False):
        self.repo_dir = str(repo_dir)
        self.fail_push = fail_push
        self.reset_called = []
        self.pushes = []          # (files, branch, remote_branch)
        self.prs = []             # (head, base)

    def clone_or_pull(self, local_path=None, branch=None):
        return "BEFORE_HASH"

    def commit_and_push(self, files, message, local_path=None, branch=None,
                        remote_branch=None):
        self.pushes.append((list(files), branch, remote_branch))
        if self.fail_push:
            raise github_ops.GithubPushError("push failed (simulated)")
        return "AFTER_HASH"

    def create_pull_request(self, head, base, title, body):
        self.prs.append((head, base))
        return f"https://github.com/x/y/pull/1?{head}"

    def reset_hard(self, commit_hash, local_path=None):
        self.reset_called.append(commit_hash)


@pytest.fixture
def patch_git(monkeypatch):
    def _apply(fake):
        monkeypatch.setattr(github_ops, "clone_or_pull", fake.clone_or_pull)
        monkeypatch.setattr(github_ops, "commit_and_push", fake.commit_and_push)
        monkeypatch.setattr(github_ops, "create_pull_request", fake.create_pull_request)
        monkeypatch.setattr(github_ops, "reset_hard", fake.reset_hard)
        return fake
    return _apply


def _make_pj_dir(tmp_path, pij):
    pj_dir = tmp_path / "pj"
    (pj_dir / "event_1").mkdir(parents=True)
    with open(pj_dir / "project_info.json", "w", encoding="utf-8") as f:
        json.dump(pij, f, ensure_ascii=False)
    return str(pj_dir)


def test_push_pr_mode_creates_pr_and_increments_version(tmp_path, patch_git):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    fake = patch_git(_FakeGit(repo_dir))

    pij = _pij([_meta(liveId=547, bundleId="")])
    pj_path = _make_pj_dir(tmp_path, pij)

    result = sp.push_stella_json(
        pij, pj_path, "event_1", _event_output(), mode="pr", repo_path=str(repo_dir),
    )

    # 初回 push → jsonVersion=1
    assert result.json_version == 1
    assert result.pr_url is not None
    assert result.branch == "stella/live547-v1"
    # フィーチャブランチへ push
    assert fake.pushes[0][2] == "stella/live547-v1"
    # PR 作成
    assert fake.prs == [("stella/live547-v1", "main")]
    # PR は未マージ → ローカルは origin に戻す
    assert fake.reset_called == ["BEFORE_HASH"]
    # メタデータ確定
    m = pij["event_detail"][0]["stella_metadata"]
    assert m["jsonVersion"] == 1
    # live{id}.json が本アプリ配下と内側リポの両方に出力されている
    assert os.path.exists(os.path.join(pj_path, "event_1", "live547.json"))
    assert os.path.exists(os.path.join(str(repo_dir), "live547.json"))
    # project_info.json が保存されている
    with open(os.path.join(pj_path, "project_info.json"), encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["event_detail"][0]["stella_metadata"]["jsonVersion"] == 1


def test_push_direct_mode_pushes_to_main(tmp_path, patch_git):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    fake = patch_git(_FakeGit(repo_dir))

    pij = _pij([_meta(liveId=547, bundleId="", jsonVersion=4)])
    pj_path = _make_pj_dir(tmp_path, pij)

    result = sp.push_stella_json(
        pij, pj_path, "event_1", _event_output(), mode="direct",
        repo_path=str(repo_dir),
    )

    assert result.mode == "direct"
    assert result.json_version == 5  # 4 → 5
    assert result.pr_url is None
    # remote_branch 指定なし (main:main)
    assert fake.pushes[0][2] is None
    assert fake.prs == []
    # 直接 push は reset しない
    assert fake.reset_called == []


def test_push_increments_notification_version_on_change(tmp_path, patch_git):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    patch_git(_FakeGit(repo_dir))

    pij = _pij([_meta(
        liveId=547, bundleId="", jsonVersion=2,
        notification="新しいお知らせ", notificationVersion="1",
        _last_pushed_notification=None,
    )])
    pj_path = _make_pj_dir(tmp_path, pij)

    result = sp.push_stella_json(
        pij, pj_path, "event_1", _event_output(), mode="direct",
        repo_path=str(repo_dir),
    )
    assert result.notification_version == "2"  # notification 変更 → +1
    m = pij["event_detail"][0]["stella_metadata"]
    assert m["_last_pushed_notification"] == "新しいお知らせ"


def test_push_failure_rolls_back_and_keeps_metadata(tmp_path, patch_git):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    fake = patch_git(_FakeGit(repo_dir, fail_push=True))

    pij = _pij([_meta(liveId=547, bundleId="", jsonVersion=3)])
    pj_path = _make_pj_dir(tmp_path, pij)

    with pytest.raises(github_ops.GithubPushError):
        sp.push_stella_json(
            pij, pj_path, "event_1", _event_output(), mode="pr",
            repo_path=str(repo_dir),
        )

    # ロールバックが呼ばれた
    assert fake.reset_called == ["BEFORE_HASH"]
    # メタデータの version は進んでいない
    m = pij["event_detail"][0]["stella_metadata"]
    assert m["jsonVersion"] == 3
    # 未 push のファイルを本アプリ配下に残さない
    assert not os.path.exists(os.path.join(pj_path, "event_1", "live547.json"))


def test_push_validation_error_does_not_touch_git(tmp_path, patch_git):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    fake = patch_git(_FakeGit(repo_dir))

    pij = _pij([_meta()])  # 未採番
    pj_path = _make_pj_dir(tmp_path, pij)

    with pytest.raises(sp.PushValidationError):
        sp.push_stella_json(
            pij, pj_path, "event_1", _event_output(), mode="pr",
            repo_path=str(repo_dir),
        )
    assert fake.pushes == []


# ---------------------------------------------------------------------------
# 一括 Push (push_all_stella_json)
# ---------------------------------------------------------------------------

def _make_pj_dir_multi(tmp_path, pij):
    pj_dir = tmp_path / "pj"
    for ev in pij["event_detail"]:
        (pj_dir / ev["event_name"]).mkdir(parents=True)
    with open(pj_dir / "project_info.json", "w", encoding="utf-8") as f:
        json.dump(pij, f, ensure_ascii=False)
    return str(pj_dir)


def test_bulk_helper_branch_name():
    assert sp.bundle_branch_name([548, 547], 3) == "stella/bundle-547_548-v3"


def test_bulk_push_pr_single_commit_single_pr(tmp_path, patch_git):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    fake = patch_git(_FakeGit(repo_dir))

    pij = _pij([
        _meta(liveId=547, bundleId="547"),
        _meta(liveId=548, bundleId="547", jsonVersion=2),
    ])
    pj_path = _make_pj_dir_multi(tmp_path, pij)
    outputs = {"event_1": _event_output(), "event_2": _event_output()}

    result = sp.push_all_stella_json(
        pij, pj_path, outputs, mode="pr", repo_path=str(repo_dir),
    )

    # 1 コミットに全 live{id}.json が含まれる
    assert len(fake.pushes) == 1
    files, _, remote_branch = fake.pushes[0]
    assert set(files) == {"live547.json", "live548.json"}
    # 単一フィーチャブランチ + 単一 PR (最大 jsonVersion=3: event_2 が 2→3)
    assert remote_branch == "stella/bundle-547_548-v3"
    assert fake.prs == [("stella/bundle-547_548-v3", "main")]
    assert result.pr_url is not None
    assert len(result.events) == 2
    # PR は未マージ → ローカルを戻す
    assert fake.reset_called == ["BEFORE_HASH"]
    # 両イベントの version 確定
    assert pij["event_detail"][0]["stella_metadata"]["jsonVersion"] == 1
    assert pij["event_detail"][1]["stella_metadata"]["jsonVersion"] == 3
    # 両ファイルが内側リポに出力されている
    assert os.path.exists(os.path.join(str(repo_dir), "live547.json"))
    assert os.path.exists(os.path.join(str(repo_dir), "live548.json"))


def test_bulk_push_direct_pushes_all_to_main(tmp_path, patch_git):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    fake = patch_git(_FakeGit(repo_dir))

    pij = _pij([
        _meta(liveId=547, bundleId="547"),
        _meta(liveId=548, bundleId="547"),
    ])
    pj_path = _make_pj_dir_multi(tmp_path, pij)
    outputs = {"event_1": _event_output(), "event_2": _event_output()}

    result = sp.push_all_stella_json(
        pij, pj_path, outputs, mode="direct", repo_path=str(repo_dir),
    )
    assert result.mode == "direct"
    assert fake.pushes[0][2] is None  # main:main
    assert fake.prs == []
    assert fake.reset_called == []


def test_bulk_push_aborts_if_any_event_unreserved(tmp_path, patch_git):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    fake = patch_git(_FakeGit(repo_dir))

    pij = _pij([
        _meta(liveId=547, bundleId="547"),
        _meta(),  # 未採番
    ])
    pj_path = _make_pj_dir_multi(tmp_path, pij)
    outputs = {"event_1": _event_output(), "event_2": _event_output()}

    with pytest.raises(sp.PushValidationError) as exc:
        sp.push_all_stella_json(
            pij, pj_path, outputs, mode="pr", repo_path=str(repo_dir),
        )
    # event_2 のエラーが含まれ、git は一切触られない
    assert any("event_2" in e for e in exc.value.errors)
    assert fake.pushes == []
    # event_1 の version も進んでいない (原子性)
    assert "jsonVersion" not in pij["event_detail"][0]["stella_metadata"]


def test_bulk_push_failure_rolls_back(tmp_path, patch_git):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    fake = patch_git(_FakeGit(repo_dir, fail_push=True))

    pij = _pij([
        _meta(liveId=547, bundleId="547", jsonVersion=1),
        _meta(liveId=548, bundleId="547", jsonVersion=1),
    ])
    pj_path = _make_pj_dir_multi(tmp_path, pij)
    outputs = {"event_1": _event_output(), "event_2": _event_output()}

    with pytest.raises(github_ops.GithubPushError):
        sp.push_all_stella_json(
            pij, pj_path, outputs, mode="pr", repo_path=str(repo_dir),
        )
    assert fake.reset_called == ["BEFORE_HASH"]
    # version は据え置き
    assert pij["event_detail"][0]["stella_metadata"]["jsonVersion"] == 1
    assert pij["event_detail"][1]["stella_metadata"]["jsonVersion"] == 1


# ---------------------------------------------------------------------------
# release_override (Push に release 変更を同梱)
# ---------------------------------------------------------------------------

def _write_live_list(path, ids):
    with open(path, "w", encoding="utf-8-sig") as f:
        json.dump({"liveList": [{"liveId": i} for i in ids]}, f, ensure_ascii=False)

def test_push_release_override_bundles_livelist_and_persists(tmp_path, patch_git):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    fake = patch_git(_FakeGit(repo_dir))
    # 既存 liveList を用意 (resync の比較用は不要だが update_live_list の入出力先)
    _write_live_list(os.path.join(str(repo_dir), "liveList.json"), [547])

    pij = _pij([_meta(liveId=547, bundleId="")], live_name="X")
    pij["stella_project_meta"]["release"] = 0
    pj_path = _make_pj_dir(tmp_path, pij)

    result = sp.push_stella_json(
        pij, pj_path, "event_1", _event_output(), mode="direct",
        repo_path=str(repo_dir), release_override=2,
    )
    assert result.mode == "direct"
    # commit 対象に live{id}.json と liveList.json の両方が含まれる
    files, _, _ = fake.pushes[0]
    assert set(files) == {"live547.json", "liveList.json"}
    # release が確定保存される
    assert pij["stella_project_meta"]["release"] == 2
    # liveList.json に release=2 が反映
    saved = {e["liveId"]: e for e in sp.stella_reserve._read_live_list(
        os.path.join(str(repo_dir), "liveList.json"))}
    assert saved[547]["release"] == 2


def test_push_release_override_not_persisted_on_failure(tmp_path, patch_git):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    fake = patch_git(_FakeGit(repo_dir, fail_push=True))
    _write_live_list(os.path.join(str(repo_dir), "liveList.json"), [547])

    pij = _pij([_meta(liveId=547, bundleId="")], live_name="X")
    pij["stella_project_meta"]["release"] = 0
    pj_path = _make_pj_dir(tmp_path, pij)

    with pytest.raises(github_ops.GithubPushError):
        sp.push_stella_json(
            pij, pj_path, "event_1", _event_output(), mode="pr",
            repo_path=str(repo_dir), release_override=2,
        )
    # release は確定していない
    assert pij["stella_project_meta"]["release"] == 0
