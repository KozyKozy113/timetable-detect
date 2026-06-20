"""Stella liveId Reserve-First 採番ロジック (Phase 4)。

GitHub 側 `liveList.json` への push が成功してから手元の
`stella_metadata.liveId` を確定する「Reserve-First」方式。手元の liveId は
常に「GitHub に書き込めた liveId だけ」という不変条件を保つ。

本モジュールは以下を提供する:
  - `reserve_live_ids()`        : 未採番 event を採番して push、成功時に確定書き戻し
  - `re_reserve()`             : 採番済保持 + 未採番追番 + bundleId 再計算 (案B)
                                  ※ reserve_live_ids が両者を統一的に扱うため別名
  - `clear_all_reservations()` : 採番情報をローカルで全消去 (push しない)
  - bundleId 算出 / liveList エントリ構築 / Push 失敗時のスナップショット & ロールバック

純粋ロジック (`plan_reservations` / `compute_bundle_id` / `build_live_list_entry`
等) と GitHub I/O を伴うオーケストレーション (`reserve_live_ids`) を分離している。
GitHub 操作は `github_ops` モジュール経由 (テストでは monkeypatch 可能)。
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from datetime import datetime

from backend_functions import github_ops
from backend_functions import project_repository as repo
from backend_functions import stella_export
from backend_functions import stella_panel


SERCH_KEY_DEFAULT = "0"


# ---------------------------------------------------------------------------
# 例外
# ---------------------------------------------------------------------------

class ReservationValidationError(ValueError):
    """採番前バリデーションに失敗した (liveName 未入力 / date 未入力 等)。

    `errors` 属性に人間可読のエラーメッセージ list を持つ。
    """

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


# ---------------------------------------------------------------------------
# 結果データ
# ---------------------------------------------------------------------------

@dataclass
class ReservationPlan:
    """採番計画 (純粋ロジックの出力)。

    - assignments: 新規採番する event_no → liveId
    - bundle_id: 全 event に適用する bundleId (event_num==1 のときは "")
    - entries: liveList.json に書き込む/更新するエントリ群 (新規 + bundleId 変更分)
    - rebundled_event_nos: 既採番だが bundleId が変わるため再書込される event_no
    """

    assignments: dict[int, int] = field(default_factory=dict)
    bundle_id: str = ""
    entries: list[dict] = field(default_factory=list)
    rebundled_event_nos: list[int] = field(default_factory=list)

    @property
    def is_noop(self) -> bool:
        """書き込むべき変更が無い (= push 不要) か。"""
        return not self.entries


@dataclass
class ReservationResult:
    """`reserve_live_ids()` の結果。"""

    plan: ReservationPlan
    pushed: bool
    commit_hash: str | None = None


@dataclass
class ResyncResult:
    """`resync_live_list()` の結果。"""

    pushed: bool
    updated_live_ids: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 純粋ヘルパ (I/O なし)
# ---------------------------------------------------------------------------

def _to_int_or_none(v) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def is_reserved(meta: dict) -> bool:
    """stella_metadata が採番済 (liveId が整数として入っている) か。"""
    return _to_int_or_none(meta.get("liveId")) is not None


def compute_dow(date_str: str) -> int | None:
    """`YYYYMMDD` 文字列から dow (月=1 .. 日=7) を返す。不正/空なら None。"""
    if not date_str:
        return None
    try:
        return datetime.strptime(str(date_str), "%Y%m%d").isoweekday()
    except (ValueError, TypeError):
        return None


def compute_bundle_id(
    event_num: int,
    reserved_bundle_ids: list,
    newly_assigned_ids: list[int],
) -> str:
    """bundleId を算出する (Phase 4-5)。

    Args:
        event_num: イベント総数。
        reserved_bundle_ids: 既採番 event が持つ bundleId のリスト ("" 含む)。
        newly_assigned_ids: 今回新規採番する liveId のリスト。

    Returns:
        - event_num == 1               → "" (空文字)
        - 既採番 event に非空 bundleId  → その先頭値に揃える
        - それ以外 (全未採番 / 全 "")   → 新規採番 liveId の最小値 (str)
    """
    if event_num <= 1:
        return ""
    non_empty = [str(b) for b in reserved_bundle_ids if b not in (None, "")]
    if non_empty:
        return non_empty[0]
    if newly_assigned_ids:
        return str(min(newly_assigned_ids))
    return ""


def build_live_list_entry(
    live_id: int,
    bundle_id: str,
    project_meta: dict,
    event_meta: dict,
) -> dict:
    """liveList.json の 1 エントリをフル構築する (Phase 4-1)。

    フィールド順は既存 liveList.json に合わせる:
    `{liveId, bundleId, liveName, release, pref, date, dow, genre, serchKey}`。
    """
    date = str(event_meta.get("date", "") or "")
    dow = event_meta.get("dow")
    if _to_int_or_none(dow) is None:
        dow = compute_dow(date)
    else:
        dow = int(dow)
    return {
        "liveId": int(live_id),
        "bundleId": str(bundle_id),
        "liveName": str(project_meta.get("liveName", "") or ""),
        "release": int(project_meta.get("release", 0)),
        "pref": int(project_meta.get("pref", 13)),
        "date": date,
        "dow": dow,
        "genre": int(project_meta.get("genre", 2)),
        "serchKey": SERCH_KEY_DEFAULT,
    }


def validate_for_reservation(project_info_json: dict) -> list[str]:
    """採番ボタン押下前のバリデーション (Phase 4-2)。

    必須項目: `liveName` + 未採番を含む全 event の `date`。
    既採番 event は date 未入力でも既に push 済のためチェック対象外とする。
    エラーメッセージの list を返す (空なら OK)。
    """
    errors: list[str] = []
    pm = repo.get_stella_project_meta(project_info_json)
    if not str(pm.get("liveName", "") or "").strip():
        errors.append("ライブ名 (liveName) を入力してください")

    for ev in project_info_json.get("event_detail", []):
        ev_no = ev["event_no"]
        meta = repo.get_stella_metadata(project_info_json, ev_no)
        if is_reserved(meta):
            continue
        if not str(meta.get("date", "") or "").strip():
            name = ev.get("event_name", f"event_{ev_no}")
            errors.append(f"{name} の公演日 (date) を入力してください")
    return errors


def plan_reservations(
    project_info_json: dict,
    max_existing_live_id: int,
) -> ReservationPlan:
    """採番計画を組み立てる純粋関数 (I/O なし)。

    未採番 event に `max_existing_live_id + 1` から連番を仮割当し、bundleId を
    再計算する。既採番 event の liveId は保持し (案B)、bundleId が変わる場合のみ
    エントリ再構築対象に含める。
    """
    pm = repo.get_stella_project_meta(project_info_json)
    events = project_info_json.get("event_detail", [])
    event_num = len(events)

    metas = {ev["event_no"]: repo.get_stella_metadata(project_info_json, ev["event_no"])
             for ev in events}

    # 未採番 event を event_no 昇順に採番
    assignments: dict[int, int] = {}
    next_id = int(max_existing_live_id) + 1
    for ev in sorted(events, key=lambda e: e["event_no"]):
        ev_no = ev["event_no"]
        if not is_reserved(metas[ev_no]):
            assignments[ev_no] = next_id
            next_id += 1

    reserved_bundle_ids = [
        metas[ev["event_no"]].get("bundleId")
        for ev in events
        if is_reserved(metas[ev["event_no"]])
    ]
    bundle_id = compute_bundle_id(
        event_num, reserved_bundle_ids, list(assignments.values()),
    )

    entries: list[dict] = []
    rebundled: list[int] = []
    for ev in events:
        ev_no = ev["event_no"]
        meta = metas[ev_no]
        if ev_no in assignments:
            entries.append(
                build_live_list_entry(assignments[ev_no], bundle_id, pm, meta)
            )
        elif is_reserved(meta):
            current_bundle = str(meta.get("bundleId") or "")
            if current_bundle != bundle_id:
                entries.append(
                    build_live_list_entry(int(meta["liveId"]), bundle_id, pm, meta)
                )
                rebundled.append(ev_no)

    return ReservationPlan(
        assignments=assignments,
        bundle_id=bundle_id,
        entries=entries,
        rebundled_event_nos=rebundled,
    )


def build_commit_message(plan: ReservationPlan, project_meta: dict) -> str:
    """採番 push の commit メッセージを生成する。"""
    live_name = str(project_meta.get("liveName", "") or "").strip() or "(no name)"
    new_ids = sorted(plan.assignments.values())
    parts = []
    if new_ids:
        parts.append("reserve liveId " + ",".join(str(i) for i in new_ids))
    if plan.rebundled_event_nos:
        parts.append(f"rebundle (bundleId={plan.bundle_id or 'empty'})")
    summary = " + ".join(parts) if parts else "update liveList"
    return f"[stella] {summary} for {live_name}"


# ---------------------------------------------------------------------------
# liveList.json 読み取り
# ---------------------------------------------------------------------------

def _read_live_list(live_list_path: str) -> list[dict]:
    if not os.path.exists(live_list_path):
        return []
    with open(live_list_path, encoding="utf-8-sig") as f:
        data = json.load(f)
    return data.get("liveList", [])


def max_live_id(live_list_path: str) -> int:
    """liveList.json 内の最大 liveId を返す。空/未取得なら 0。"""
    live_list = _read_live_list(live_list_path)
    ids = [_to_int_or_none(e.get("liveId")) for e in live_list]
    ids = [i for i in ids if i is not None]
    return max(ids) if ids else 0


# ---------------------------------------------------------------------------
# 採番計画のメタデータ反映 (push 成功後のみ呼ぶ)
# ---------------------------------------------------------------------------

def project_bundle(project_info_json: dict, plan: ReservationPlan) -> tuple[list[int], str]:
    """採番後のプロジェクトの bundle (全 liveId) と最初の日付を返す (liveListPanel2 用)。

    各 event の liveId は plan の新規割当を優先し、無ければ既採番値を使う。
    `first_date` は全 event の `date` の最小値 (複数日イベントの最初の日付)。
    """
    live_ids: list[int] = []
    dates: list[str] = []
    for ev in project_info_json.get("event_detail", []):
        ev_no = ev["event_no"]
        meta = repo.get_stella_metadata(project_info_json, ev_no)
        if ev_no in plan.assignments:
            live_ids.append(int(plan.assignments[ev_no]))
        elif is_reserved(meta):
            live_ids.append(int(meta["liveId"]))
        else:
            continue
        d = str(meta.get("date", "") or "")
        if d:
            dates.append(d)
    live_ids.sort()
    first_date = min(dates) if dates else ""
    return live_ids, first_date


def reserved_bundle(project_info_json: dict) -> tuple[list[int], str]:
    """採番済みイベントのみの bundle (全 liveId) と最初の日付を返す。

    `project_bundle()` の plan 無し版。Panel2 の再配置 (resync) で使う。
    """
    live_ids: list[int] = []
    dates: list[str] = []
    for ev in project_info_json.get("event_detail", []):
        meta = repo.get_stella_metadata(project_info_json, ev["event_no"])
        if not is_reserved(meta):
            continue
        live_ids.append(int(meta["liveId"]))
        d = str(meta.get("date", "") or "")
        if d:
            dates.append(d)
    live_ids.sort()
    return live_ids, (min(dates) if dates else "")


def _resync_panel(
    project_info_json: dict, repo_path: str, live_list_path: str,
) -> bool:
    """採番済みプロジェクトの bundle を現在の date で Panel2 に再配置する。

    date 変更で配置 (年月 / 月内順) が変わった場合に追従。変更があれば書き込んで True。
    """
    live_ids, first_date = reserved_bundle(project_info_json)
    if not live_ids or not first_date:
        return False
    live_dates = {
        int(e["liveId"]): str(e.get("date", "") or "")
        for e in _read_live_list(live_list_path)
        if "liveId" in e
    }
    panel_path = os.path.join(repo_path, stella_panel.PANEL2_FILENAME)
    panel = stella_panel.read_panel(panel_path)
    new_panel = copy.deepcopy(panel)
    stella_panel.upsert_bundle(new_panel, live_ids, first_date, live_dates)
    if new_panel == panel:
        return False
    stella_panel.write_panel(panel_path, new_panel)
    return True


def _update_panel(
    project_info_json: dict,
    plan: ReservationPlan,
    repo_path: str,
    live_list_path: str,
) -> None:
    """liveListPanel2.json にプロジェクトの bundle を追記 / 再配置する。

    日付参照 (月内の並べ替え) は更新後の liveList.json から構築する
    (新規採番分の date も含めるため、update_live_list 実行後に呼ぶ)。
    """
    live_ids, first_date = project_bundle(project_info_json, plan)
    if not live_ids or not first_date:
        return
    live_dates = {
        int(e["liveId"]): str(e.get("date", "") or "")
        for e in _read_live_list(live_list_path)
        if "liveId" in e
    }
    panel_path = os.path.join(repo_path, stella_panel.PANEL2_FILENAME)
    panel = stella_panel.read_panel(panel_path)
    stella_panel.upsert_bundle(panel, live_ids, first_date, live_dates)
    stella_panel.write_panel(panel_path, panel)


def _apply_plan_to_metadata(project_info_json: dict, plan: ReservationPlan) -> None:
    """push 成功後、採番計画を `event_detail[i].stella_metadata` に確定書き込みする。

    - 新規採番 event: liveId / bundleId / dow を書き込み
    - bundleId 変更 (rebundle) event: bundleId 更新 + jsonVersion +1 (Phase 4-4-3)
    """
    for ev_no, live_id in plan.assignments.items():
        meta = repo.get_stella_metadata(project_info_json, ev_no)
        meta["liveId"] = int(live_id)
        meta["bundleId"] = plan.bundle_id
        if _to_int_or_none(meta.get("dow")) is None:
            meta["dow"] = compute_dow(str(meta.get("date", "") or ""))

    for ev_no in plan.rebundled_event_nos:
        meta = repo.get_stella_metadata(project_info_json, ev_no)
        meta["bundleId"] = plan.bundle_id
        meta["jsonVersion"] = (_to_int_or_none(meta.get("jsonVersion")) or 0) + 1


# ---------------------------------------------------------------------------
# オーケストレーション (GitHub I/O を伴う)
# ---------------------------------------------------------------------------

def reserve_live_ids(
    project_info_json: dict,
    pj_path: str,
    *,
    branch: str | None = None,
    repo_path: str | None = None,
) -> ReservationResult:
    """Reserve-First 方式で未採番 event の liveId を採番する (Phase 4-4)。

    フロー:
      1. バリデーション (liveName / 全 event date)
      2. clone_or_pull で内側リポを最新化 (before_hash を退避)
      3. liveList.json の max(liveId) を取得 → 未採番 event に連番仮割当 + bundleId 再計算
      4. liveList.json を更新 → commit → push
      5. 成功: stella_metadata に liveId / bundleId / dow を書き戻し project_info.json 保存
         失敗: 内側リポを before_hash へ reset --hard、ローカルは一切無変更で例外送出

    既採番 event は保持し未採番のみ採番するため、新規採番 / 再採番の両方を本関数で扱う。

    Returns:
        ReservationResult。採番対象が無い場合は pushed=False の no-op を返す。

    Raises:
        ReservationValidationError: 必須項目未入力。
        github_ops.GithubAuthError / GithubPushError: 認証 / push 失敗 (ロールバック済)。
    """
    errors = validate_for_reservation(project_info_json)
    if errors:
        raise ReservationValidationError(errors)

    if branch is None:
        branch = github_ops.DEFAULT_BRANCH
    if repo_path is None:
        repo_path = github_ops.LOCAL_REPO_PATH

    # 1. pull (内側リポを origin と一致させ、スナップショットを取る)
    before_hash = github_ops.clone_or_pull(local_path=repo_path, branch=branch)
    live_list_path = os.path.join(repo_path, "liveList.json")

    # 2. 採番計画
    plan = plan_reservations(project_info_json, max_live_id(live_list_path))
    if plan.is_noop:
        # 採番済かつ bundleId 変更も無い → push 不要
        return ReservationResult(plan=plan, pushed=False, commit_hash=before_hash)

    # 3. liveList.json 更新 (+ 新規採番があれば liveListPanel2.json も追記) → commit → push
    stella_export.update_live_list(live_list_path, plan.entries)
    push_files = ["liveList.json"]
    if plan.assignments:
        _update_panel(project_info_json, plan, repo_path, live_list_path)
        push_files.append(stella_panel.PANEL2_FILENAME)
    pm = repo.get_stella_project_meta(project_info_json)
    message = build_commit_message(plan, pm)
    try:
        commit_hash = github_ops.commit_and_push(
            push_files, message, local_path=repo_path, branch=branch,
        )
    except Exception:
        # commit_and_push は push 失敗時に内側リポを巻き戻すが、commit 自体の
        # 失敗等で working tree が変わったままになるケースも防御的に巻き戻す。
        try:
            github_ops.reset_hard(before_hash, local_path=repo_path)
        except Exception:
            pass
        raise

    # 4. push 成功 → ローカル確定 (ここで初めて project_info.json を変更)
    _apply_plan_to_metadata(project_info_json, plan)
    repo.save_project_json(pj_path, project_info_json)

    return ReservationResult(plan=plan, pushed=True, commit_hash=commit_hash)


def re_reserve(
    project_info_json: dict,
    pj_path: str,
    *,
    branch: str | None = None,
    repo_path: str | None = None,
) -> ReservationResult:
    """再採番 (案B: 既採番保持 + 未採番追番 + bundleId 再計算) — Phase 4-4-3。

    `reserve_live_ids()` が既採番 event を自然に保持しつつ未採番のみ採番するため、
    実体は同一処理。意図を明示するための別名 (UI のラベル切替に対応)。
    """
    return reserve_live_ids(
        project_info_json, pj_path, branch=branch, repo_path=repo_path,
    )


def clear_all_reservations(project_info_json: dict) -> None:
    """全 event の採番情報をローカルで消去する (Phase 4-4-4)。

    クリア対象: `liveId` / `bundleId` / `jsonVersion` / `_last_pushed_notification`。
    `liveList.json` 上の旧エントリは削除しない (過去 push 済のため。release=0 なら
    Stella 側に影響しない方針)。push は行わず、保存は呼び出し側の責務。
    """
    for ev in project_info_json.get("event_detail", []):
        meta = repo.get_stella_metadata(project_info_json, ev["event_no"])
        for key in ("liveId", "bundleId", "jsonVersion"):
            meta.pop(key, None)
        meta["_last_pushed_notification"] = None


# ---------------------------------------------------------------------------
# liveList 再同期 (採番後の release / pref / liveName / date 等の反映)
# ---------------------------------------------------------------------------

def is_linked(project_info_json: dict) -> bool:
    """「GitHub とリンク済み」= 採番済みイベントが 1 つでもあるか。

    リンク済みなら liveList.json に各イベント行が存在するため、release 等の変更を
    `resync_live_list()` で反映できる。リンク前はローカル保存のみで GitHub に出さない。
    """
    return any(
        is_reserved(repo.get_stella_metadata(project_info_json, ev["event_no"]))
        for ev in project_info_json.get("event_detail", [])
    )


def build_current_live_list_entries(
    project_info_json: dict,
    *,
    release_override: int | None = None,
) -> list[dict]:
    """採番済み全イベントの liveList エントリを **現在のメタデータ値** で構築する。

    liveId / bundleId は各イベントの既存値を保持し、liveName / genre / release /
    pref / date / dow / serchKey を現在の `stella_project_meta` / `stella_metadata`
    から作り直す。`release_override` 指定時はその release 値で構築する
    (実 `stella_project_meta` は変更しない — 確定は push 成功後に呼び出し側で行う)。
    """
    pm = dict(repo.get_stella_project_meta(project_info_json))
    if release_override is not None:
        pm["release"] = int(release_override)
    entries: list[dict] = []
    for ev in project_info_json.get("event_detail", []):
        meta = repo.get_stella_metadata(project_info_json, ev["event_no"])
        if not is_reserved(meta):
            continue
        live_id = int(meta["liveId"])
        bundle_id = str(meta.get("bundleId") or "")
        entries.append(build_live_list_entry(live_id, bundle_id, pm, meta))
    return entries


def resync_live_list(
    project_info_json: dict,
    pj_path: str,
    *,
    release_override: int | None = None,
    branch: str | None = None,
    repo_path: str | None = None,
) -> ResyncResult:
    """採番済みイベントの liveList エントリを現在値で GitHub に再反映する (C-1)。

    `liveName` / `release` / `pref` / `genre` / `date` / `dow` は `liveList.json` のみが
    持つフィールドで、採番後の変更は採番フローでは反映されない。本関数が現在のメタデータ
    値でエントリを作り直し、差分があれば default ブランチへ直接 push する。

    - 既存エントリと差分が無ければ push せず no-op を返す。
    - 未リンク (採番済みイベントなし) なら no-op。
    - `release_override` 指定時は push 成功後に `stella_project_meta.release` を確定保存する。
    - push 失敗時は内側リポを巻き戻し、ローカル project_info.json は無変更で例外送出。

    Raises:
        github_ops.GithubAuthError / GithubPushError。
    """
    if branch is None:
        branch = github_ops.DEFAULT_BRANCH
    if repo_path is None:
        repo_path = github_ops.LOCAL_REPO_PATH

    entries = build_current_live_list_entries(
        project_info_json, release_override=release_override,
    )
    if not entries:
        return ResyncResult(pushed=False)  # 未リンク

    before_hash = github_ops.clone_or_pull(local_path=repo_path, branch=branch)
    live_list_path = os.path.join(repo_path, "liveList.json")
    existing_by_id = {e.get("liveId"): e for e in _read_live_list(live_list_path)}
    changed = [e for e in entries if existing_by_id.get(e["liveId"]) != e]

    push_files: list[str] = []
    if changed:
        stella_export.update_live_list(live_list_path, changed)
        push_files.append("liveList.json")

    # Panel2 の再配置 (date 変更で表示エリア / 月内順が変わった場合に追従)
    if _resync_panel(project_info_json, repo_path, live_list_path):
        push_files.append(stella_panel.PANEL2_FILENAME)

    if not push_files:
        return ResyncResult(pushed=False)  # 差分なし

    live_name = str(
        repo.get_stella_project_meta(project_info_json).get("liveName", "") or ""
    ).strip() or "(no name)"
    parts = []
    if changed:
        parts.append("liveList (" + ",".join(str(e["liveId"]) for e in changed) + ")")
    if stella_panel.PANEL2_FILENAME in push_files:
        parts.append("panel2")
    message = f"[stella] resync {' + '.join(parts)} {live_name}"
    try:
        github_ops.commit_and_push(
            push_files, message, local_path=repo_path, branch=branch,
        )
    except Exception:
        try:
            github_ops.reset_hard(before_hash, local_path=repo_path)
        except Exception:
            pass
        raise

    if release_override is not None:
        repo.set_stella_project_meta(project_info_json, {"release": int(release_override)})
        repo.save_project_json(pj_path, project_info_json)

    return ResyncResult(pushed=True, updated_live_ids=[e["liveId"] for e in changed])
