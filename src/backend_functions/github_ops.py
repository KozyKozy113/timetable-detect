"""GitHub 連携操作 (Stella timetableproj 用)

ローカル: data/timetableproj/ にネスト git として clone する
認証: .env の GITHUB_TOKEN / GITHUB_USER_NAME / GITHUB_USER_EMAIL を python-dotenv 経由で参照
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DIR_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.normpath(os.path.join(DIR_PATH, "..", "..", "data"))

REPO_OWNER = "ys0512"
REPO_NAME = "timetableproj"
REPO_FULL_NAME = f"{REPO_OWNER}/{REPO_NAME}"
REPO_URL = f"https://github.com/{REPO_FULL_NAME}.git"
LOCAL_REPO_PATH = os.path.join(DATA_PATH, "timetableproj")
DEFAULT_BRANCH = "main"


@dataclass(frozen=True)
class GithubCredentials:
    token: str
    user_name: str
    user_email: str


class GithubAuthError(RuntimeError):
    pass


class GithubPushError(RuntimeError):
    pass


def get_credentials_from_env(require_user_info: bool = True) -> GithubCredentials:
    """環境変数 (.env) から GitHub 認証情報を取得する。

    .env の読み込みは呼び出し側の責務 (既存 AWS 認証と同じ)。
    値が無い場合は GithubAuthError を投げる。

    Args:
        require_user_info: True なら GITHUB_USER_NAME / GITHUB_USER_EMAIL も必須。
            False なら token のみ必須（clone/pull 単体用途）。
    """
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    user_name = os.environ.get("GITHUB_USER_NAME", "").strip()
    user_email = os.environ.get("GITHUB_USER_EMAIL", "").strip()
    if not token:
        raise GithubAuthError(
            "GITHUB_TOKEN が未設定です。.env に PAT を追加してください。"
        )
    if require_user_info and (not user_name or not user_email):
        raise GithubAuthError(
            "GITHUB_USER_NAME / GITHUB_USER_EMAIL が未設定です。.env を確認してください。"
        )
    return GithubCredentials(token=token, user_name=user_name, user_email=user_email)


def credentials_available(require_user_info: bool = True) -> bool:
    """UI 側の有効/無効判定用。例外を投げずに bool を返す。"""
    try:
        get_credentials_from_env(require_user_info=require_user_info)
        return True
    except GithubAuthError:
        return False


def _build_authenticated_url(token: str) -> str:
    return f"https://x-access-token:{token}@github.com/{REPO_FULL_NAME}.git"


def is_repo_cloned(local_path: str = LOCAL_REPO_PATH) -> bool:
    """指定パスが git リポとして clone 済か (.git ディレクトリ有無で判定)。"""
    return os.path.isdir(os.path.join(local_path, ".git"))


def clone_or_pull(
    local_path: str = LOCAL_REPO_PATH,
    branch: str = DEFAULT_BRANCH,
) -> str:
    """初回 clone もしくは pull を行い、完了後の HEAD コミット hash を返す。

    - clone 済: remote URL を認証付きで上書き → fetch → reset --hard origin/{branch}
      (ローカル変更は破棄、内側リポは常にリモートと一致)
    - 未 clone: 認証付き URL で full clone
    """
    from git import Repo

    creds = get_credentials_from_env(require_user_info=False)
    auth_url = _build_authenticated_url(creds.token)

    Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    if is_repo_cloned(local_path):
        repo = Repo(local_path)
        with repo.remotes.origin.config_writer as cw:
            cw.set("url", auth_url)
        repo.remotes.origin.fetch()
        repo.git.checkout(branch)
        repo.git.reset("--hard", f"origin/{branch}")
    else:
        repo = Repo.clone_from(auth_url, local_path, branch=branch)

    if creds.user_name and creds.user_email:
        with repo.config_writer() as cw:
            cw.set_value("user", "name", creds.user_name)
            cw.set_value("user", "email", creds.user_email)

    return repo.head.commit.hexsha


def commit_and_push(
    files: Iterable[str],
    message: str,
    local_path: str = LOCAL_REPO_PATH,
    branch: str = DEFAULT_BRANCH,
    remote_branch: str | None = None,
) -> str:
    """指定ファイル群を commit & push する。成功時は push 後の HEAD hash を返す。

    files は local_path からの相対パス。
    `remote_branch` を指定すると `branch:remote_branch` の refspec で push する
    (PR 用フィーチャブランチへの push に使う)。省略時は `branch:branch`。
    Push に失敗した場合は内側リポを push 前の HEAD まで `reset --hard` し、
    GithubPushError を投げる (Phase 6-6 ロールバック設計)。
    """
    from git import Repo
    from git.exc import GitCommandError

    if not is_repo_cloned(local_path):
        raise GithubPushError(
            f"{local_path} に git リポがありません。先に clone_or_pull() を実行してください。"
        )

    repo = Repo(local_path)
    before_hash = repo.head.commit.hexsha
    target = remote_branch or branch

    try:
        repo.index.add(list(files))
        repo.index.commit(message)
    except GitCommandError as e:
        raise GithubPushError(f"commit 失敗: {e}") from e

    try:
        repo.remotes.origin.push(refspec=f"{branch}:{target}").raise_if_error()
        return repo.head.commit.hexsha
    except Exception as e:
        try:
            repo.git.reset("--hard", before_hash)
        except GitCommandError:
            pass
        raise GithubPushError(f"push 失敗 (ロールバック済): {e}") from e


def reset_hard(commit_hash: str, local_path: str = LOCAL_REPO_PATH) -> None:
    """内側リポを指定コミットまで `reset --hard` で巻き戻す (ロールバック用)。

    commit_and_push() の push 失敗時ロールバックと同じ操作を、採番フロー
    (stella_reserve) など外側から呼べるよう公開ヘルパとして切り出したもの。
    外側 timetable-detect リポには一切影響しない (内側リポのみに作用)。
    """
    from git import Repo

    repo = Repo(local_path)
    repo.git.reset("--hard", commit_hash)


def create_pull_request(
    head_branch: str,
    base_branch: str,
    title: str,
    body: str,
) -> str:
    """PR を作成して URL を返す。PyGithub の REST API 経由。

    PyGithub の例外は `GithubPushError` に包んで投げ直す (UI で握れるように)。
    特に 403 (権限不足) は、PAT に Pull requests の書き込み権限が無いケースを案内する。
    """
    from github import Github, GithubException

    creds = get_credentials_from_env()
    gh = Github(creds.token)
    try:
        repo = gh.get_repo(REPO_FULL_NAME)
        pr = repo.create_pull(
            title=title, body=body, head=head_branch, base=base_branch,
        )
        return pr.html_url
    except GithubException as e:
        status = getattr(e, "status", None)
        if status == 403:
            raise GithubPushError(
                "PR の作成権限がありません (403)。フィーチャブランチへの push は成功して "
                f"いるため、GitHub 上にブランチ `{head_branch}` は残っています。\n"
                "PAT に PR 作成権限を付与してください:\n"
                "  - Fine-grained PAT: ys0512/timetableproj に "
                "「Pull requests: Read and write」を追加\n"
                "  - Classic PAT: 「repo」スコープ\n"
                "権限付与後に再実行するか、当面は「Push (直接)」をご利用ください。"
            ) from e
        raise GithubPushError(f"PR 作成に失敗しました ({status}): {e}") from e
