"""Stella `live{id}.json` の GitHub Push オーケストレーション (Phase 6 / 7-2 ⑥-D)。

⑥で確定したタイテを Stella JSON に変換し、`data/timetableproj/` 配下へ書き出して
GitHub へ push する。Push 方式は PR 作成 (推奨) と default ブランチ直接 push の 2 つ。

採番 (`stella_reserve`) と同じく Reserve-First / スナップショット → 試行 →
確定/巻き戻し (Phase 6-6) のパターンに従う。バージョン番号 (jsonVersion /
notificationVersion) は **push 成功後にのみ** インクリメントし、失敗時はローカルを
一切変更しない。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd

from backend_functions import github_ops
from backend_functions import project_repository as repo
from backend_functions import stella_export
from backend_functions import stella_reserve


# ---------------------------------------------------------------------------
# 例外
# ---------------------------------------------------------------------------

class PushValidationError(ValueError):
    """Push 前バリデーションに失敗した (未採番 / bundleId 未設定 等)。"""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


# ---------------------------------------------------------------------------
# 結果データ
# ---------------------------------------------------------------------------

@dataclass
class PushResult:
    """`push_stella_json()` の結果。"""

    live_id: int
    json_version: int
    notification_version: str
    mode: str               # "pr" | "direct"
    branch: str             # push 先リモートブランチ名
    local_path: str         # 本アプリ配下に残した live{id}.json のパス
    pr_url: str | None = None


@dataclass
class BulkPushResult:
    """`push_all_stella_json()` の結果 (プロジェクト全体一括 Push)。"""

    events: list[dict]      # [{event_name, live_id, json_version, notification_version, local_path}, ...]
    mode: str               # "pr" | "direct"
    branch: str             # push 先 (PR は単一フィーチャブランチ / direct は default)
    pr_url: str | None = None
    notified: bool = False  # notificationData4.json を同梱したか


@dataclass
class NotificationPush:
    """一括 Push に同梱する notificationData4.json への追記指示 (アプリトップのお知らせ)。

    追記の ON/OFF・文面編集・重複時の選択は UI 側で確定し、本オブジェクトとして渡す。
    「取りやめ」を選んだ場合は UI が `notification=None` を渡す (= タイテ Push のみ実行)。
    """

    message: str
    message_en: str
    date: str                     # "M/D" (UI で確定済み・編集後の値)
    live_ids: list[int]
    dedup_strategy: str = "none"  # "none" | "keep" (過去を残す) | "replace" (過去を削除)


# ---------------------------------------------------------------------------
# バリデーション
# ---------------------------------------------------------------------------

def _is_reserved(meta: dict) -> bool:
    v = meta.get("liveId")
    if v in (None, ""):
        return False
    try:
        int(v)
        return True
    except (ValueError, TypeError):
        return False


def validate_for_push(project_info_json: dict, event_no: int) -> list[str]:
    """Push 前バリデーション。エラーメッセージの list を返す (空なら OK)。

    - liveId 未採番 → ①画面で採番が必要 (Phase 4-4)
    - event_num > 1 かつ bundleId が空 → ①画面で再採番が必要 (Phase 4-5)
    """
    errors: list[str] = []
    meta = repo.get_stella_metadata(project_info_json, event_no)
    if not _is_reserved(meta):
        errors.append("liveId が未採番です。①画面で採番してください。")
        return errors  # 採番前は bundleId 判定もできないので打ち切り

    event_num = len(project_info_json.get("event_detail", []))
    bundle_id = meta.get("bundleId")
    if event_num > 1 and bundle_id in (None, ""):
        errors.append("bundleId が未設定です。①画面で再採番してください。")
    return errors


# ---------------------------------------------------------------------------
# commit メッセージ / PR 本文
# ---------------------------------------------------------------------------

def _live_name(project_info_json: dict) -> str:
    pm = repo.get_stella_project_meta(project_info_json)
    return str(pm.get("liveName", "") or "").strip() or "(no name)"


def build_commit_message(live_id: int, json_version: int, live_name: str) -> str:
    return f"[stella] update live{live_id} (v{json_version}) {live_name}"


def build_pr_body(
    live_id: int, json_version: int, notification_version: str, live_name: str,
) -> str:
    return (
        f"Stella タイムテーブル更新\n\n"
        f"- liveName: {live_name}\n"
        f"- liveId: {live_id}\n"
        f"- jsonVersion: {json_version}\n"
        f"- notificationVersion: {notification_version}\n"
    )


def feature_branch_name(live_id: int, json_version: int) -> str:
    return f"stella/live{live_id}-v{json_version}"


def bundle_branch_name(live_ids: list[int], max_json_version: int) -> str:
    """一括 Push 用の単一フィーチャブランチ名 (全 liveId をまとめる)。"""
    ids = "_".join(str(i) for i in sorted(live_ids))
    return f"stella/bundle-{ids}-v{max_json_version}"


# ---------------------------------------------------------------------------
# オーケストレーション (共通ヘルパ)
# ---------------------------------------------------------------------------

def _prepare_event_push(
    project_info_json: dict,
    pj_path: str,
    event_name: str,
    event_no: int,
    event_output: dict[str, pd.DataFrame],
    repo_path: str,
) -> dict:
    """1 イベント分の Stella JSON を新バージョンで構築し、**内側リポにのみ**書き出す
    (git add 対象)。push はまだ行わない。本アプリ配下へのコピーは push 成功が確定して
    から `_persist_local_copies()` で保存する (失敗時に未 push のファイルを残さないため)。
    """
    meta = repo.get_stella_metadata(project_info_json, event_no)
    live_id = int(meta["liveId"])
    new_meta = stella_export.increment_versions_on_push(meta)
    stella_json = stella_export.build_stella_json(event_output, new_meta)

    stella_export.write_stella_json(stella_json, repo_path, live_id=live_id)

    return {
        "event_no": event_no,
        "event_name": event_name,
        "live_id": live_id,
        "new_meta": new_meta,
        "stella_json": stella_json,
        "local_dir": os.path.join(pj_path, event_name),
        "fname": f"live{live_id}.json",
        "local_path": None,
        "json_version": int(new_meta["jsonVersion"]),
        "notification_version": str(new_meta["notificationVersion"]),
    }


def _persist_local_copies(records: list[dict]) -> None:
    """push 成功後、本アプリ配下 (`pj_path/event_name/live{id}.json`) へ
    push したものと同一の JSON を保存する。各 record の `local_path` を更新する。

    PR モードでは内側リポの作業ツリーが reset --hard で push 前に戻るため、
    手元に残す確定コピーはこの本アプリ配下のファイルが正となる。
    """
    for rec in records:
        rec["local_path"] = stella_export.write_stella_json(
            rec["stella_json"], rec["local_dir"], live_id=rec["live_id"],
        )


def _commit_versions(project_info_json: dict, records: list[dict]) -> None:
    """push 成功後、各イベントの version 系メタデータを確定書き込みする。"""
    for rec in records:
        new_meta = rec["new_meta"]
        repo.set_stella_metadata(project_info_json, rec["event_no"], {
            "jsonVersion": new_meta["jsonVersion"],
            "notificationVersion": new_meta["notificationVersion"],
            "_last_pushed_notification": new_meta["_last_pushed_notification"],
        })


def _stage_release_livelist(
    project_info_json: dict, repo_path: str, release_override: int | None,
) -> list[str]:
    """release_override 指定時、liveList.json を新 release で更新して内側リポへ書き出し、
    commit 対象に追加するファイル名 (`["liveList.json"]`) を返す。指定なしなら空リスト。

    `stella_project_meta.release` 自体の確定は push 成功後に呼び出し側で行う
    (失敗時にローカルへ反映しないため)。
    """
    if release_override is None:
        return []
    entries = stella_reserve.build_current_live_list_entries(
        project_info_json, release_override=release_override,
    )
    if not entries:
        return []
    live_list_path = os.path.join(repo_path, "liveList.json")
    stella_export.update_live_list(live_list_path, entries)
    return ["liveList.json"]


def _stage_notification(
    notification: "NotificationPush | None", repo_path: str,
) -> list[str]:
    """notification 指定時、notificationData4.json の `notificationList` 先頭へ追記して
    内側リポへ書き出し、commit 対象ファイル名 (`["notificationData4.json"]`) を返す。
    指定なしなら空リスト。

    dedup_strategy:
      - "replace": 既存の重複エントリ (liveId 交差) を削除してから先頭追加
      - "keep" / "none": 既存を残して先頭追加
    重複判定・選択は UI 側で確定済みである前提 (本関数は strategy に従うのみ)。
    """
    if notification is None:
        return []
    notif_path = os.path.join(repo_path, stella_export.NOTIFICATION_FILENAME)
    data = stella_export.read_notification_data(notif_path)
    remove_indices: list[int] = []
    if notification.dedup_strategy == "replace":
        remove_indices = stella_export.find_duplicate_notifications(
            data.get("notificationList", []), notification.live_ids,
        )
    entry = stella_export.build_notification_entry(
        notification.live_ids, notification.date,
        notification.message, notification.message_en,
    )
    stella_export.prepend_notification(data, entry, remove_indices=remove_indices)
    stella_export.write_notification_data(notif_path, data)
    return [stella_export.NOTIFICATION_FILENAME]


# ---------------------------------------------------------------------------
# オーケストレーション (単一イベント)
# ---------------------------------------------------------------------------

def push_stella_json(
    project_info_json: dict,
    pj_path: str,
    event_name: str,
    event_output: dict[str, pd.DataFrame],
    *,
    mode: str = "pr",
    branch: str | None = None,
    repo_path: str | None = None,
    release_override: int | None = None,
) -> PushResult:
    """⑥の出力データを Stella JSON 化して GitHub へ push する (Phase 6 / 7-2 ⑥-D)。

    フロー:
      1. バリデーション (採番済 / bundleId)
      2. clone_or_pull で内側リポを最新化 (before_hash を退避)
      3. jsonVersion / notificationVersion をメモリ上で算出 (increment_versions_on_push)
      4. 新バージョンで Stella JSON を構築し、本アプリ配下 + 内側リポへ書き出し
      5. commit → push
           - mode="direct": default ブランチへ直接 push
           - mode="pr":     フィーチャブランチへ push → PR 作成 → ローカルは origin に戻す
      6. 成功: stella_metadata の version 系を確定書き戻し project_info.json 保存
         失敗: 内側リポを before_hash へ巻き戻し、ローカルは無変更で例外送出

    Args:
        event_output: `build_event_output()` の戻り値 (stage/idolname/live を含む)。
        mode: "pr" (PR 作成、推奨) または "direct" (直接 push)。
        release_override: 指定時、この Push で `release` を変更し liveList.json も
            同一コミット/PR で更新する (公開状態の切替を Push に同梱)。

    Raises:
        PushValidationError / github_ops.GithubAuthError / GithubPushError。
    """
    if mode not in ("pr", "direct"):
        raise ValueError(f"未知の push mode: {mode}")

    event_no = repo.get_event_no_by_event_name(project_info_json, event_name)
    if event_no is None:
        raise PushValidationError([f"event_name={event_name} が見つかりません"])

    errors = validate_for_push(project_info_json, event_no)
    if errors:
        raise PushValidationError(errors)

    if branch is None:
        branch = github_ops.DEFAULT_BRANCH
    if repo_path is None:
        repo_path = github_ops.LOCAL_REPO_PATH

    live_name = _live_name(project_info_json)

    # 1. pull + スナップショット
    before_hash = github_ops.clone_or_pull(local_path=repo_path, branch=branch)

    # 2-3. 新バージョンで JSON 構築 + 書き出し
    rec = _prepare_event_push(
        project_info_json, pj_path, event_name, event_no, event_output, repo_path,
    )
    live_id = rec["live_id"]
    json_version = rec["json_version"]
    notif_version = rec["notification_version"]
    message = build_commit_message(live_id, json_version, live_name)

    # release 変更を同梱する場合は liveList.json も commit 対象に加える
    extra_files = _stage_release_livelist(project_info_json, repo_path, release_override)
    push_files = [rec["fname"]] + extra_files

    pr_url: str | None = None
    pushed_branch = branch
    try:
        if mode == "direct":
            github_ops.commit_and_push(
                push_files, message, local_path=repo_path, branch=branch,
            )
        else:
            pushed_branch = feature_branch_name(live_id, json_version)
            github_ops.commit_and_push(
                push_files, message, local_path=repo_path, branch=branch,
                remote_branch=pushed_branch,
            )
            pr_url = github_ops.create_pull_request(
                pushed_branch, branch, message,
                build_pr_body(live_id, json_version, notif_version, live_name),
            )
            # PR は未マージのため、ローカル内側リポは origin/branch に戻して一致させる
            github_ops.reset_hard(before_hash, local_path=repo_path)
    except Exception:
        try:
            github_ops.reset_hard(before_hash, local_path=repo_path)
        except Exception:
            pass
        raise

    # 4. push 成功 → 本アプリ配下へ確定コピー + version 系を確定保存
    _persist_local_copies([rec])
    _commit_versions(project_info_json, [rec])
    if release_override is not None:
        repo.set_stella_project_meta(project_info_json, {"release": int(release_override)})
    repo.save_project_json(pj_path, project_info_json)

    return PushResult(
        live_id=live_id,
        json_version=json_version,
        notification_version=notif_version,
        mode=mode,
        branch=pushed_branch,
        local_path=rec["local_path"],
        pr_url=pr_url,
    )


# ---------------------------------------------------------------------------
# オーケストレーション (プロジェクト全体一括)
# ---------------------------------------------------------------------------

def push_all_stella_json(
    project_info_json: dict,
    pj_path: str,
    event_outputs: dict[str, dict],
    *,
    mode: str = "pr",
    branch: str | None = None,
    repo_path: str | None = None,
    release_override: int | None = None,
    notification: "NotificationPush | None" = None,
) -> BulkPushResult:
    """プロジェクト全イベントの live{id}.json を 1 コミットでまとめて push する。

    PR モードでは全イベントを **単一フィーチャブランチ・単一 PR** にまとめる。
    pull は最初の 1 回だけ行い、全イベント分の JSON を構築してから 1 コミットで push する。
    バリデーションは全対象イベントを先に検査し、1 件でも NG なら push せず中断する
    (1 PR の原子性を保つため)。version インクリメント / ロールバックは単一 push と同方針。

    Args:
        event_outputs: `{event_name: build_event_output() 戻り値}`。出力済イベントのみ。
        mode: "pr" (推奨) または "direct"。
        release_override: 指定時、この Push で `release` を変更し liveList.json も
            同一コミット/PR で更新する。
        notification: 指定時、notificationData4.json の `notificationList` 先頭へお知らせを
            追記し、同一コミット/PR に同梱する (一括 Push のみ対応)。None なら追記しない。

    Raises:
        PushValidationError: 対象なし / いずれかのイベントが採番・bundleId 検証に失敗。
        github_ops.GithubAuthError / GithubPushError。
    """
    if mode not in ("pr", "direct"):
        raise ValueError(f"未知の push mode: {mode}")
    if not event_outputs:
        raise PushValidationError(["出力データのあるイベントがありません"])

    if branch is None:
        branch = github_ops.DEFAULT_BRANCH
    if repo_path is None:
        repo_path = github_ops.LOCAL_REPO_PATH

    # 全対象イベントを先にバリデーション (1 件でも NG なら中断)
    targets: list[tuple[str, int, dict]] = []
    errors: list[str] = []
    for event_name, event_output in event_outputs.items():
        event_no = repo.get_event_no_by_event_name(project_info_json, event_name)
        if event_no is None:
            errors.append(f"{event_name}: event が見つかりません")
            continue
        ev_errors = validate_for_push(project_info_json, event_no)
        if ev_errors:
            errors.extend(f"{event_name}: {e}" for e in ev_errors)
            continue
        targets.append((event_name, event_no, event_output))
    if errors:
        raise PushValidationError(errors)
    if not targets:
        raise PushValidationError(["Push 対象イベントがありません"])

    live_name = _live_name(project_info_json)

    # 1. pull + スナップショット (1 回だけ)
    before_hash = github_ops.clone_or_pull(local_path=repo_path, branch=branch)

    # 2-3. 全イベント分の JSON を構築 + 書き出し
    records = [
        _prepare_event_push(
            project_info_json, pj_path, ev_name, ev_no, ev_out, repo_path,
        )
        for ev_name, ev_no, ev_out in targets
    ]

    extra_files = _stage_release_livelist(project_info_json, repo_path, release_override)
    notif_files = _stage_notification(notification, repo_path)
    fnames = [rec["fname"] for rec in records] + extra_files + notif_files
    live_ids = [rec["live_id"] for rec in records]
    max_version = max(rec["json_version"] for rec in records)
    message = (
        f"[stella] update {len(records)} lives "
        f"({','.join(str(i) for i in sorted(live_ids))}) {live_name}"
    )
    if notif_files:
        message += " + notify"

    pr_url: str | None = None
    pushed_branch = branch
    try:
        if mode == "direct":
            github_ops.commit_and_push(
                fnames, message, local_path=repo_path, branch=branch,
            )
        else:
            pushed_branch = bundle_branch_name(live_ids, max_version)
            github_ops.commit_and_push(
                fnames, message, local_path=repo_path, branch=branch,
                remote_branch=pushed_branch,
            )
            body_lines = "\n".join(
                f"- live{rec['live_id']}.json (v{rec['json_version']})"
                for rec in records
            )
            pr_url = github_ops.create_pull_request(
                pushed_branch, branch,
                f"[stella] {live_name} 全{len(records)}イベント更新",
                f"Stella タイムテーブル一括更新\n\n- liveName: {live_name}\n{body_lines}\n",
            )
            github_ops.reset_hard(before_hash, local_path=repo_path)
    except Exception:
        try:
            github_ops.reset_hard(before_hash, local_path=repo_path)
        except Exception:
            pass
        raise

    # 4. push 成功 → 本アプリ配下へ確定コピー + 全イベントの version 系を確定保存
    _persist_local_copies(records)
    _commit_versions(project_info_json, records)
    if release_override is not None:
        repo.set_stella_project_meta(project_info_json, {"release": int(release_override)})
    repo.save_project_json(pj_path, project_info_json)

    return BulkPushResult(
        events=[
            {
                "event_name": rec["event_name"],
                "live_id": rec["live_id"],
                "json_version": rec["json_version"],
                "notification_version": rec["notification_version"],
                "local_path": rec["local_path"],
            }
            for rec in records
        ],
        mode=mode,
        branch=pushed_branch,
        pr_url=pr_url,
        notified=bool(notif_files),
    )
