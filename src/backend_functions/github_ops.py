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
) -> str:
    """指定ファイル群を commit & push する。成功時は push 後の HEAD hash を返す。

    files は local_path からの相対パス。
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

    try:
        repo.index.add(list(files))
        repo.index.commit(message)
    except GitCommandError as e:
        raise GithubPushError(f"commit 失敗: {e}") from e

    try:
        repo.remotes.origin.push(refspec=f"{branch}:{branch}").raise_if_error()
        return repo.head.commit.hexsha
    except Exception as e:
        try:
            repo.git.reset("--hard", before_hash)
        except GitCommandError:
            pass
        raise GithubPushError(f"push 失敗 (ロールバック済): {e}") from e


def create_pull_request(
    head_branch: str,
    base_branch: str,
    title: str,
    body: str,
) -> str:
    """PR を作成して URL を返す。PyGithub の REST API 経由。"""
    from github import Github

    creds = get_credentials_from_env()
    gh = Github(creds.token)
    repo = gh.get_repo(REPO_FULL_NAME)
    pr = repo.create_pull(title=title, body=body, head=head_branch, base=base_branch)
    return pr.html_url
