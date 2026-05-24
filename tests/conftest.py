import json
import shutil
from pathlib import Path

import pytest

from backend_functions import project_migration as _migration


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "output_builder"


def _load_project(name: str, dst: Path) -> tuple[str, dict]:
    src = FIXTURE_ROOT / name
    shutil.copytree(src, dst, dirs_exist_ok=True)
    with open(dst / "project_info.json", encoding="utf-8") as f:
        project_info_json = json.load(f)
    # フィクスチャは旧スキーマで保存しているため、読み込み時に新スキーマへ変換する
    project_info_json = _migration.migrate_project_info(project_info_json)
    return str(dst), project_info_json


@pytest.fixture
def project_pre_confirm(tmp_path):
    """IDマスタ確定前の通常形式プロジェクト"""
    return _load_project("project_pre_confirm", tmp_path / "project")


@pytest.fixture
def project_post_confirm(tmp_path):
    """IDマスタ確定後の通常形式プロジェクト"""
    return _load_project("project_post_confirm", tmp_path / "project")


@pytest.fixture
def project_tokutenkai_heiki(tmp_path):
    """特典会併記形式プロジェクト"""
    return _load_project("project_tokutenkai_heiki", tmp_path / "project")
