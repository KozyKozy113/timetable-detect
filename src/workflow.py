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
from backend_functions import image_processing as _imgproc
from backend_functions import time_axis as _time_axis
from backend_functions import ocr_service as _ocr
from backend_functions import output_builder as _output
from backend_functions import s3access
from frontend_functions import timetable_difference

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

    def delete_image(self, state: AppState, event_no: int,
                     img_type: str) -> WorkflowResult:
        """登録済み画像を削除し、project_info_jsonを更新する

        Returns:
            WorkflowResult with data={"remaining_types": list[str]}
        """
        pij = state.project.project_info_json
        repo.delete_timetable_image(pij, event_no, img_type)
        repo.save_project_json(state.project.pj_path, pij)
        state.project.project_master = repo.update_timestamp(
            state.project.project_master, state.project.pj_name, self._data_path,
        )
        remaining = repo.get_event_type_list(pij, event_no)
        return WorkflowResult(success=True, data={"remaining_types": remaining})


# ---------------------------------------------------------------------------
# ImageWorkflow
# ---------------------------------------------------------------------------

class ImageWorkflow:
    """画像処理のワークフロー"""

    def __init__(self, data_path: str) -> None:
        self._data_path = data_path

    def detect_stage_lines(self, state: AppState, image: Image.Image,
                           params: _imgproc.StageLineDetectParams,
                           event_name: str, img_type: str) -> WorkflowResult:
        """ステージ線を検出し、state.cropを更新する"""
        # raw_cropped.png を保存
        img_path = os.path.join(
            state.project.pj_path, event_name, img_type, "raw_cropped.png",
        )
        state.crop.cropped_image.save(img_path)

        result = _imgproc.detect_stageline(image, params, state.crop.crop_box)
        state.crop.stage_line_list = result.stage_line_list
        state.crop.images_eachstage = result.images_eachstage
        state.crop.images_eachstage_bbox = result.images_eachstage_bbox
        state.crop.annotated_image = result.annotated_image
        return WorkflowResult(success=True, data=result)

    def split_evenly(self, state: AppState, image: Image.Image,
                     stage_num: int,
                     event_name: str, img_type: str) -> WorkflowResult:
        """画像を均等分割し、state.cropを更新する"""
        # raw_cropped.png を保存
        img_path = os.path.join(
            state.project.pj_path, event_name, img_type, "raw_cropped.png",
        )
        state.crop.cropped_image.save(img_path)

        images, bboxes = _imgproc.split_image_evenly(
            image, stage_num, state.crop.crop_box,
        )
        state.crop.images_eachstage = images
        state.crop.images_eachstage_bbox = bboxes
        return WorkflowResult(success=True)

    def save_stage_images(self, state: AppState,
                          event_name: str, img_type: str,
                          accept_flags: list[bool] | None = None) -> WorkflowResult:
        """ステージ画像を確定保存し、project_info_jsonを更新する"""
        event_no = repo.get_event_no_by_event_name(
            state.project.project_info_json, event_name,
        )
        _imgproc.save_stage_images(
            state.project.pj_path, event_name, img_type,
            state.crop.images_eachstage, state.crop.images_eachstage_bbox,
            state.crop.cropped_image, state.crop.crop_box,
            state.project.project_info_json, event_no,
            accept_flags=accept_flags,
        )
        repo.save_project_json(state.project.pj_path, state.project.project_info_json)
        state.project.project_master = repo.update_timestamp(
            state.project.project_master, state.project.pj_name, self._data_path,
        )
        return WorkflowResult(success=True)

    def replace_stage_images_from_new_raw(self, state: AppState,
                                          new_image_path: str,
                                          event_name: str,
                                          img_type: str) -> WorkflowResult:
        """新画像からbboxで各ステージ画像を切り出して置き換える"""
        event_no = repo.get_event_no_by_event_name(
            state.project.project_info_json, event_name,
        )
        error_msg = _imgproc.replace_stage_images_from_new_raw(
            new_image_path, state.project.pj_path, event_name,
            img_type, state.project.project_info_json, event_no,
        )
        if error_msg:
            return WorkflowResult(success=False, error=error_msg)
        state.project.project_master = repo.update_timestamp(
            state.project.project_master, state.project.pj_name, self._data_path,
        )
        return WorkflowResult(success=True)

    def save_time_axis(self, state: AppState,
                       time_start: Any, top: int, height: int,
                       total_duration: float,
                       event_name: str, img_type: str) -> WorkflowResult:
        """時間軸設定を保存"""
        event_no = repo.get_event_no_by_event_name(
            state.project.project_info_json, event_name,
        )
        time_pixel_dict = _time_axis.build_time_pixel_config(
            time_start, top, height, total_duration,
        )
        state.project.project_info_json["event_detail"][event_no]\
            ["timetables"][img_type]["time_pixel"] = time_pixel_dict
        repo.save_project_json(state.project.pj_path, state.project.project_info_json)
        state.project.project_master = repo.update_timestamp(
            state.project.project_master, state.project.pj_name, self._data_path,
        )
        return WorkflowResult(success=True)

    @staticmethod
    def output_difference_image(new_image_file: Any,
                                pj_path: str, event_name: str,
                                img_type: str) -> WorkflowResult:
        """差分画像を生成して返す"""
        old_image_path = os.path.join(pj_path, event_name, img_type, "raw.png")
        old_image = Image.open(old_image_path)
        difference_image = timetable_difference.output_difference(
            Image.open(new_image_file), old_image,
        )
        return WorkflowResult(success=True, data=difference_image)


# ---------------------------------------------------------------------------
# OcrWorkflow
# ---------------------------------------------------------------------------

class OcrWorkflow:
    """OCR実行のワークフロー"""

    def __init__(self, data_path: str) -> None:
        self._data_path = data_path

    def run_ocr_single(self, state: AppState, mode: str,
                       stage_no: int, user_prompt: str,
                       event_name: str, img_type: str,
                       ticket_urls: list[str] | None = None,
                       ensure_addtime_fn: Any = None) -> WorkflowResult:
        """1ステージのOCR実行"""
        if mode == "notime" and ensure_addtime_fn is not None:
            img_path = os.path.join(
                state.project.pj_path, event_name, img_type,
                f"stage_{stage_no}_addtime.png",
            )
            if not os.path.exists(img_path):
                ensure_addtime_fn(stage_no)

        _ocr.run_ocr_single_stage(
            mode, stage_no, user_prompt,
            state.project.pj_path, event_name, img_type,
            state.project.project_info_json, ticket_urls,
        )
        return WorkflowResult(success=True)

    def run_ocr_all(self, state: AppState, mode: str,
                    user_prompt: str, stage_num: int,
                    event_name: str, img_type: str,
                    ticket_urls: list[str] | None = None,
                    ensure_addtime_fn: Any = None) -> WorkflowResult:
        """全ステージの並列OCR実行"""
        _ocr.run_ocr_all_stages(
            mode, user_prompt,
            state.project.pj_path, event_name, img_type,
            state.project.project_info_json, stage_num, ticket_urls,
            ensure_addtime_fn=ensure_addtime_fn,
        )
        return WorkflowResult(success=True)

    def run_batch(self, state: AppState,
                  together_targets: dict[str, bool],
                  options: dict,
                  ensure_addtime_fn: Any = None,
                  get_ticket_urls_fn: Any = None) -> WorkflowResult:
        """一括OCR実行（get_timetabledata_together相当）"""
        event_list = repo.get_event_name_list(state.project.project_info_json)
        state.project.project_info_json = _ocr.run_batch_ocr(
            event_list, state.project.project_info_json, state.project.pj_path,
            together_targets,
            ocr_stage=options.get("ocr_stage", False),
            ocr_timetable=options.get("ocr_timetable", False),
            correct=options.get("correct", False),
            correct_in_confirmed=options.get("correct_in_confirmed", False),
            ocr_stage_prompt=options.get("ocr_stage_prompt", ""),
            ocr_user_prompt=options.get("ocr_user_prompt", ""),
            use_ticket_urls=options.get("use_ticket_urls", True),
            ensure_addtime_fn=ensure_addtime_fn,
            get_ticket_urls_fn=get_ticket_urls_fn,
        )
        return WorkflowResult(success=True)

    def correct_idol_names_single(self, state: AppState,
                                  stage_no: int,
                                  event_name: str, img_type: str,
                                  use_confirmed: bool,
                                  confirmed_list: list[str] | None = None,
                                  ticket_performers: list[str] | None = None) -> WorkflowResult:
        """1ステージのグループ名を補正"""
        _ocr.correct_idol_names_single(
            stage_no, state.project.pj_path, event_name, img_type,
            use_confirmed, confirmed_list, ticket_performers,
        )
        return WorkflowResult(success=True)

    def correct_idol_names_all(self, state: AppState,
                               event_name: str, img_type: str,
                               stage_num: int,
                               use_confirmed: bool,
                               confirmed_list: list[str] | None = None,
                               ticket_performers: list[str] | None = None) -> WorkflowResult:
        """全ステージのグループ名を補正"""
        _ocr.correct_idol_names_all(
            state.project.pj_path, event_name, img_type, stage_num,
            use_confirmed, confirmed_list, ticket_performers,
        )
        return WorkflowResult(success=True)

    def detect_stage_names(self, state: AppState,
                           user_prompt: str,
                           event_name: str, img_type: str,
                           stage_num: int) -> WorkflowResult:
        """ステージ名をOCRで読み取り、project_info_jsonを更新"""
        state.project.project_info_json = _ocr.detect_stage_names(
            state.project.pj_path, event_name, img_type, stage_num,
            user_prompt, state.project.project_info_json,
        )
        repo.save_project_json(state.project.pj_path, state.project.project_info_json)
        state.project.project_master = repo.update_timestamp(
            state.project.project_master, state.project.pj_name, self._data_path,
        )
        return WorkflowResult(success=True)

    def set_stage_name(self, state: AppState,
                       stage_no: int, stage_name: str,
                       event_name: str, img_type: str) -> WorkflowResult:
        """ステージ名を手動設定"""
        state.project.project_info_json = _ocr.set_stage_name(
            state.project.pj_path, event_name, img_type,
            stage_no, stage_name, state.project.project_info_json,
        )
        repo.save_project_json(state.project.pj_path, state.project.project_info_json)
        state.project.project_master = repo.update_timestamp(
            state.project.project_master, state.project.pj_name, self._data_path,
        )
        return WorkflowResult(success=True)

    def booth_name_add_prefix(self, state: AppState,
                              stage_no: int,
                              df_timetable: pd.DataFrame,
                              event_name: str, img_type: str) -> WorkflowResult:
        """ブース名にステージ名を接頭辞として付与。更新後のDFを返す"""
        event_no = repo.get_event_no_by_event_name(
            state.project.project_info_json, event_name,
        )
        stage_name = repo.get_stage_name(
            state.project.project_info_json, event_no, img_type, stage_no,
        )
        updated_df = _ocr.booth_name_add_prefix(df_timetable, stage_name)
        return WorkflowResult(success=True, data=updated_df)

    def save_timetable_data(self, state: AppState,
                            stage_no: int,
                            df_timetable: pd.DataFrame,
                            stage_name: str,
                            event_name: str, img_type: str,
                            is_tokutenkai_heiki: bool) -> WorkflowResult:
        """編集結果を保存。更新後のDFを返す"""
        updated_df = _ocr.save_timetable_data(
            stage_no, df_timetable, stage_name,
            state.project.pj_path, event_name, img_type,
            is_tokutenkai_heiki,
        )
        return WorkflowResult(success=True, data=updated_df)

    def generate_timetable_picture(self, state: AppState,
                                   stage_no: int,
                                   event_name: str, img_type: str,
                                   time_match: bool,
                                   converter: Any = None) -> WorkflowResult:
        """読み取り結果からタイムテーブル画像を生成"""
        output_path = _ocr.generate_timetable_picture(
            stage_no, state.project.pj_path, event_name, img_type,
            time_match, converter,
        )
        state.project.project_master = repo.update_timestamp(
            state.project.project_master, state.project.pj_name, self._data_path,
        )
        return WorkflowResult(success=True, data=output_path)

    def detect_timeline(self, state: AppState,
                        stage_no: int,
                        event_name: str, img_type: str,
                        params: _imgproc.TimelineDetectParams,
                        converter: Any) -> WorkflowResult:
        """横線時刻推定（タイムライン検出）"""
        img_path = os.path.join(
            state.project.pj_path, event_name, img_type,
            f"stage_{stage_no}.png",
        )
        result = _imgproc.detect_timeline_onlyonestage(
            img_path, params, converter.pix_to_time, converter.time_length_to_pix,
        )
        return WorkflowResult(success=True, data=result)


# ---------------------------------------------------------------------------
# OutputWorkflow
# ---------------------------------------------------------------------------

class OutputWorkflow:
    """出力のワークフロー"""

    def __init__(self, data_path: str) -> None:
        self._data_path = data_path

    def determine_id_master(self, state: AppState) -> WorkflowResult:
        """IDマスタを確定して保存する"""
        _output.determine_id_master(
            state.output.output_df,
            state.project.pj_path,
            state.project.project_info_json,
        )
        state.project.project_master = repo.update_timestamp(
            state.project.project_master, state.project.pj_name, self._data_path,
        )
        return WorkflowResult(success=True)

    def save_to_s3(self, state: AppState) -> WorkflowResult:
        """プロジェクトデータをS3にアップロードする"""
        _output.save_to_s3(state.project.pj_name)
        return WorkflowResult(success=True)

    def export_excel(self, state: AppState) -> WorkflowResult:
        """Excelファイルを出力"""
        event_list = repo.get_event_name_list(state.project.project_info_json)
        output_path = _output.export_excel(
            state.output.output_df, state.project.pj_path, event_list,
        )
        return WorkflowResult(success=True, data=output_path)

    def listup_new_idolname(self, state: AppState) -> WorkflowResult:
        """新規登場グループ名をリストアップしてstateに格納"""
        event_list = repo.get_event_name_list(state.project.project_info_json)
        state.output.new_idolname = _output.listup_new_idolname(
            state.output.output_df, event_list,
        )
        return WorkflowResult(success=True)

    def update_idol_name_master(self, state: AppState,
                                df_new_idolname: pd.DataFrame) -> WorkflowResult:
        """グループ名マスタに新規名を追加しS3同期"""
        _output.update_master_idolname(df_new_idolname, self._data_path)
        return WorkflowResult(success=True)
