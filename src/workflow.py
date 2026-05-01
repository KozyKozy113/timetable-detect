"""
ワークフロー層。

UIアクション（ボタンクリック等）に対応するユースケースを、
フレームワーク非依存で実装する。
UIコールバックとバックエンドサービスの橋渡しを担う。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import pandas as pd
from PIL import Image

from backend_functions import project_repository as repo
from backend_functions import s3access

if TYPE_CHECKING:
    from app_state import AppState


@dataclass
class WorkflowResult:
    """ワークフローの実行結果"""
    success: bool
    data: Any = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# ProjectWorkflow
# ---------------------------------------------------------------------------

class ProjectWorkflow:
    """プロジェクト管理のワークフロー"""

    def __init__(self, data_path: str) -> None:
        self._data_path = data_path

    def create_project(self, pj_name: str, state: AppState,
                       existing_names: list[str]) -> WorkflowResult:
        """新規プロジェクトを作成し、stateを更新する"""
        if pj_name in existing_names:
            return WorkflowResult(success=False, error="既に存在する名前のプロジェクトです")

        project_info_json, pm, pm_s3 = repo.create_project_data(
            self._data_path, pj_name,
            state.project.project_master, state.project.project_master_s3,
        )
        state.project.project_master = pm
        state.project.project_master_s3 = pm_s3

        return self.load_project(pj_name, state)

    def load_project(self, pj_name: str, state: AppState) -> WorkflowResult:
        """既存プロジェクトを読み込み、stateを更新する"""
        pj_path = os.path.join(self._data_path, "projects", pj_name)

        s3access.get_master()
        s3access.get_project_data(pj_name)

        project_master = pd.read_csv(
            os.path.join(self._data_path, "master", "projects_master.csv"),
            index_col=0,
        )
        project_info = project_master.loc[pj_name]

        state.project.pj_name = pj_name
        state.project.pj_path = pj_path
        state.project.project_master = project_master

        if not pd.isna(project_info["event_type"]):
            state.project.event_type = project_info["event_type"]
        else:
            state.project.event_type = "対バン"

        if not pd.isna(project_info["event_num"]):
            state.project.event_num = int(project_info["event_num"])
        else:
            state.project.event_num = 1

        for i in range(state.project.event_num):
            os.makedirs(os.path.join(pj_path, "event_{}".format(i + 1)), exist_ok=True)

        state.project.project_info_json = repo.get_project_json(pj_path)

        # crop / OCR 状態のリセット
        state.crop.images_eachstage = []
        state.ocr.time_axis_detect = None
        state.ocr.timeline_eachstage = []

        return WorkflowResult(success=True)

    def update_project_setting(self, state: AppState,
                               event_type: str, event_num: int) -> WorkflowResult:
        """プロジェクト設定を変更"""
        updated_json, updated_master = repo.apply_project_setting(
            self._data_path,
            state.project.pj_name,
            state.project.project_info_json,
            state.project.project_master,
            event_type,
            event_num,
        )
        state.project.project_info_json = updated_json
        state.project.project_master = updated_master
        return WorkflowResult(success=True)

    def register_image(self, state: AppState, event_name: str,
                       img_type: str, img_format: str,
                       file_data: bytes,
                       img_type_alternative: str = "") -> WorkflowResult:
        """タイムテーブル画像を登録

        Returns:
            WorkflowResult with data={"resolved_img_type": str}
        """
        pij = state.project.project_info_json
        pj_path = state.project.pj_path
        event_no = repo.get_event_no_by_event_name(pij, event_name)

        if img_type in ["ライブ", "特典会"]:
            repo.register_timetable_image(
                pj_path, event_name, event_no, img_type, img_format, file_data, pij,
            )
            resolved = img_type

        elif img_type == "両方(特典会別添え)":
            for t in ["ライブ", "特典会"]:
                repo.register_timetable_image(
                    pj_path, event_name, event_no, t, img_format, file_data, pij,
                )
            resolved = "ライブ"

        elif img_type == "両方(特典会併記)":
            resolved = "ライブ特典会"
            repo.register_timetable_image(
                pj_path, event_name, event_no, resolved, "特典会併記", file_data, pij,
            )

        elif img_type in ["その他", "その他(特典会併記)"]:
            resolved = img_type_alternative
            fmt = "特典会併記" if img_type == "その他(特典会併記)" else img_format
            repo.register_timetable_image(
                pj_path, event_name, event_no, resolved, fmt, file_data, pij,
            )

        else:
            return WorkflowResult(success=False, error=f"不明な画像種別: {img_type}")

        repo.save_project_json(pj_path, pij)
        state.project.project_master = repo.update_timestamp(
            state.project.project_master, state.project.pj_name, self._data_path,
        )
        state.project.project_info_json = pij

        return WorkflowResult(success=True, data={"resolved_img_type": resolved})

    def save_ticket_urls(self, state: AppState, scope: str,
                         urls_data: dict) -> WorkflowResult:
        """チケットURL設定を保存"""
        pij = repo.build_ticket_urls_data(
            state.project.project_info_json, scope, urls_data,
        )
        repo.save_project_json(state.project.pj_path, pij)
        state.project.project_master = repo.update_timestamp(
            state.project.project_master, state.project.pj_name, self._data_path,
        )
        state.project.project_info_json = pij
        return WorkflowResult(success=True)


# ---------------------------------------------------------------------------
# ImageWorkflow
# ---------------------------------------------------------------------------

class ImageWorkflow:
    """画像処理のワークフロー"""

    def detect_stage_lines(self, image: Image.Image,
                           params: Any) -> WorkflowResult:
        """ステージ線を検出し、結果を返す"""
        raise NotImplementedError

    def split_evenly(self, image: Image.Image,
                     stage_num: int) -> WorkflowResult:
        """画像を均等分割"""
        raise NotImplementedError

    def replace_stage_images_from_new_raw(self, state: AppState,
                                          new_image: Image.Image,
                                          event_name: str,
                                          img_type: str) -> WorkflowResult:
        """新画像からbboxで各ステージ画像を切り出して置き換える"""
        raise NotImplementedError

    def save_time_axis(self, state: AppState,
                       time_start: str, top: int, height: int,
                       total_duration: float) -> WorkflowResult:
        """時間軸設定を保存"""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# OcrWorkflow
# ---------------------------------------------------------------------------

class OcrWorkflow:
    """OCR実行のワークフロー"""

    def run_ocr_single(self, state: AppState, mode: str,
                       stage_no: int, user_prompt: str,
                       ticket_urls: list[str] | None = None) -> WorkflowResult:
        """1ステージのOCR実行"""
        raise NotImplementedError

    def run_ocr_all(self, state: AppState, mode: str,
                    user_prompt: str,
                    ticket_urls: list[str] | None = None) -> WorkflowResult:
        """全ステージの並列OCR実行"""
        raise NotImplementedError

    def run_batch(self, state: AppState,
                  targets: list[dict],
                  options: dict) -> WorkflowResult:
        """一括OCR実行"""
        raise NotImplementedError

    def correct_idol_names(self, state: AppState,
                           stage_no: int | None = None) -> WorkflowResult:
        """グループ名補正（stage_no=Noneで全ステージ）"""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# OutputWorkflow
# ---------------------------------------------------------------------------

class OutputWorkflow:
    """出力のワークフロー"""

    def build_all_events(self, state: AppState) -> WorkflowResult:
        """全イベントの出力データを構築"""
        raise NotImplementedError

    def export_excel(self, state: AppState) -> WorkflowResult:
        """Excelファイル出力"""
        raise NotImplementedError

    def generate_timetable_image(self, state: AppState,
                                 stage_no: int,
                                 time_match: bool) -> WorkflowResult:
        """読み取り結果から構造化タイムテーブル画像を生成"""
        raise NotImplementedError

    def update_idol_name_master(self, state: AppState,
                                new_names: list[str]) -> WorkflowResult:
        """グループ名マスタに新規名を追加"""
        raise NotImplementedError
