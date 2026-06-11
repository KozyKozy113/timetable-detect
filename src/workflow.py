"""
ワークフロー層。

UIアクション（ボタンクリック等）に対応するユースケースを、
フレームワーク非依存で実装する。
UIコールバックとバックエンドサービスの橋渡しを担う。
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import pandas as pd
from PIL import Image

from backend_functions import project_repository as repo
from backend_functions import image_processing as _imgproc
from backend_functions import time_axis as _time_axis
from backend_functions import ocr_service as _ocr
from backend_functions import output_builder as _output
from backend_functions import output_editor as _output_editor
from backend_functions import event_timetable_picture as _etp
from backend_functions import s3access
from backend_functions import stella_export as _stella
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
                       kind: str, img_format: str | None,
                       dir_name: str,
                       file_data: bytes,
                       overwrite: bool = False) -> WorkflowResult:
        """タイムテーブル画像を 1 件登録する。

        Args:
            kind: "live" | "tokutenkai" | "live_tokutenkai_heiki"
            img_format: kind=live_tokutenkai_heiki のとき None。それ以外は "通常"/"ライムライト式"。
            dir_name: 保存先サブフォルダ名 (UI 表示名兼用)
            overwrite: True なら既存同 dir_name エントリの派生物をクリーンアップしてから登録

        Returns:
            WorkflowResult with data={"dir_name": str}
        """
        pij = state.project.project_info_json
        pj_path = state.project.pj_path
        event_no = repo.get_event_no_by_event_name(pij, event_name)

        if overwrite:
            repo.cleanup_image_artifacts(pj_path, event_name, dir_name)

        repo.register_timetable_image(
            pj_path, event_name, event_no,
            dir_name=dir_name, kind=kind, img_format=img_format,
            file_data=file_data, project_info_json=pij,
        )

        repo.save_project_json(pj_path, pij)
        state.project.project_master = repo.update_timestamp(
            state.project.project_master, state.project.pj_name, self._data_path,
        )
        state.project.project_info_json = pij

        return WorkflowResult(success=True, data={"dir_name": dir_name})

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

    def move_timetable_up(self, state: AppState, event_no: int,
                          image_no: int) -> WorkflowResult:
        """登録済み画像を 1 つ前に移動する。"""
        pij = state.project.project_info_json
        repo.move_timetable_up(pij, event_no, image_no)
        repo.save_project_json(state.project.pj_path, pij)
        state.project.project_master = repo.update_timestamp(
            state.project.project_master, state.project.pj_name, self._data_path,
        )
        return WorkflowResult(success=True)

    def move_timetable_down(self, state: AppState, event_no: int,
                            image_no: int) -> WorkflowResult:
        """登録済み画像を 1 つ後ろに移動する。"""
        pij = state.project.project_info_json
        repo.move_timetable_down(pij, event_no, image_no)
        repo.save_project_json(state.project.pj_path, pij)
        state.project.project_master = repo.update_timestamp(
            state.project.project_master, state.project.pj_name, self._data_path,
        )
        return WorkflowResult(success=True)

    def reset_timetable_order(self, state: AppState,
                              event_no: int) -> WorkflowResult:
        """timetables[] をデフォルトバケット順に並べ直す。"""
        pij = state.project.project_info_json
        repo.reset_timetable_order(pij, event_no)
        repo.save_project_json(state.project.pj_path, pij)
        state.project.project_master = repo.update_timestamp(
            state.project.project_master, state.project.pj_name, self._data_path,
        )
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
        entry = repo.get_image_entry_by_dir_name(
            state.project.project_info_json, event_no, img_type,
        )
        if entry is None:
            return WorkflowResult(success=False, error=f"画像が見つかりません: {img_type}")
        entry["time_pixel"] = time_pixel_dict
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

    def adopt_raw_idol_names_all(self, state: AppState,
                                 event_name: str, img_type: str,
                                 stage_num: int) -> WorkflowResult:
        """全ステージで グループ名_採用 を グループ名 (raw) で上書きする。"""
        _ocr.adopt_raw_idol_names_all(
            state.project.pj_path, event_name, img_type, stage_num,
        )
        return WorkflowResult(success=True)

    def adopt_raw_idol_names_event(self, state: AppState,
                                   event_name: str) -> WorkflowResult:
        """イベント配下の全 (event_type, stage) で raw → 採用 をコピーし、
        タイテ画像 (stage_N_timetable.png + 集約画像) を再生成する。
        """
        pij = state.project.project_info_json
        event_no = repo.get_event_no_by_event_name(pij, event_name)
        if event_no is None:
            return WorkflowResult(success=False, error="event_no が解決できません")
        _ocr.adopt_raw_idol_names_event(
            state.project.pj_path, event_name, event_no, pij,
        )
        # 各 (img_type, stage_no) の stage_N_timetable.png を再生成
        for img_type in repo.get_event_type_list(pij, event_no):
            entry = repo.get_image_entry_by_dir_name(pij, event_no, img_type)
            if entry is None:
                continue
            converter = _time_axis.TimeAxisConverter.from_project_info(
                pij, event_no, img_type,
            )
            for stage_no in range(int(entry.get("stage_num", 0) or 0)):
                _ocr.generate_timetable_picture(
                    stage_no, state.project.pj_path, event_name, img_type,
                    time_match=converter is not None,
                    time_axis_converter=converter,
                    project_info_json=pij,
                    event_no=event_no,
                )
        # 集約画像を再生成
        _etp.regenerate_all_event_images(
            state.project.pj_path, event_name, pij,
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
        """編集結果を保存。更新後のDFを返す

        保存前に **同 event の他ステージとの 出番ID 衝突** を検証する。
        衝突があれば保存を中止し、エラーメッセージを返す
        (コラボはステージを跨がない設計のため、跨いだ 出番ID は整合性違反)。
        """
        from backend_functions import timetabledata as _td
        collisions = _td.detect_cross_stage_turn_id_collision(
            state.project.pj_path, event_name, img_type, stage_no,
            df_timetable, state.project.project_info_json,
        )
        if collisions:
            lines = [
                "出番IDが他ステージと衝突しています。コラボはステージを跨げないため、"
                "重複している 出番ID を別IDに振り直してから保存してください:",
            ]
            seen: set[tuple[int, str, int, str]] = set()
            for c in collisions:
                key = (c["出番ID"], c["他種別"], c["他ステージNo"], c["場所"])
                if key in seen:
                    continue
                seen.add(key)
                lines.append(
                    f"  出番ID={c['出番ID']} → 他ステージ {c['他種別']}/stage_{c['他ステージNo']} ({c['場所']})"
                )
            return WorkflowResult(success=False, error="\n".join(lines))

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
            project_info_json=state.project.project_info_json,
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
        # ステージID確定でステージマスタ参照が初めて成立するため、
        # 全イベントの集約画像を再生成する。
        for event_name in repo.get_event_name_list(state.project.project_info_json):
            _etp.regenerate_all_event_images(
                state.project.pj_path, event_name,
                state.project.project_info_json,
            )
        state.project.project_master = repo.update_timestamp(
            state.project.project_master, state.project.pj_name, self._data_path,
        )
        return WorkflowResult(success=True)

    def regenerate_event_type_images(
        self, state: AppState, event_name: str, img_type: str,
        *, include_cross_type: bool = True,
    ) -> WorkflowResult:
        """1種別の集約画像 (種別単位) を再生成。
        include_cross_type=True で種別横断も同時に再生成。
        """
        _etp.regenerate_event_type_images(
            state.project.pj_path, event_name, img_type,
            state.project.project_info_json,
            include_cross_type=include_cross_type,
        )
        return WorkflowResult(success=True)

    def regenerate_event_cross_image(
        self, state: AppState, event_name: str,
    ) -> WorkflowResult:
        """1イベントの種別横断画像のみを再生成。"""
        _etp.save_event_image(
            state.project.pj_path, event_name,
            state.project.project_info_json,
        )
        return WorkflowResult(success=True)

    def regenerate_all_event_images(
        self, state: AppState, event_name: str,
    ) -> WorkflowResult:
        """1イベントの全集約画像を再生成。"""
        _etp.regenerate_all_event_images(
            state.project.pj_path, event_name,
            state.project.project_info_json,
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

    # ---------------------------------------------------------------------------
    # Phase 3 / 5 / 7: Stella JSON
    # ---------------------------------------------------------------------------

    def save_stella_metadata(
        self, state: AppState, event_name: str, metadata: dict,
    ) -> WorkflowResult:
        """⑥-A の入力を `event_detail[i].stella_metadata` に書き戻して保存する。"""
        pij = state.project.project_info_json
        event_no = repo.get_event_no_by_event_name(pij, event_name)
        if event_no is None:
            return WorkflowResult(success=False, error=f"event_name={event_name} が見つかりません")
        repo.set_stella_metadata(pij, event_no, metadata)
        repo.save_project_json(state.project.pj_path, pij)
        state.project.project_master = repo.update_timestamp(
            state.project.project_master, state.project.pj_name, self._data_path,
        )
        return WorkflowResult(success=True)

    def build_stella_json(
        self, state: AppState, event_name: str,
    ) -> WorkflowResult:
        """⑥-C: Stella JSON を組み立てて dict を返す (ファイル書き出しはしない)。"""
        data = state.output.output_df.get(event_name)
        if not data:
            return WorkflowResult(success=False, error="出力データがありません")
        pij = state.project.project_info_json
        event_no = repo.get_event_no_by_event_name(pij, event_name)
        if event_no is None:
            return WorkflowResult(success=False, error=f"event_name={event_name} が見つかりません")
        metadata = repo.get_stella_metadata(pij, event_no)
        stella_json = _stella.build_stella_json(data, metadata)
        return WorkflowResult(success=True, data=stella_json)

    def export_stella_json(
        self, state: AppState, event_name: str,
    ) -> WorkflowResult:
        """⑥-C: Stella JSON をプロジェクト配下 (`event_N/live{liveId}.json`) に書き出す。"""
        result = self.build_stella_json(state, event_name)
        if not result.success:
            return result
        out_dir = os.path.join(state.project.pj_path, event_name)
        path = _stella.write_stella_json(result.data, out_dir)
        return WorkflowResult(success=True, data={"json": result.data, "path": path})

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

    # ---------------------------------------------------------------------------
    # 編集モード (ステージ / グループ / 出番マスタ)
    # ---------------------------------------------------------------------------

    def is_id_master_confirmed(self, state: AppState, event_name: str) -> bool:
        """指定イベントの IDマスタ確定済みか判定。
        master_stage.csv / master_idolname.csv / turn_id_data.csv が全て存在すれば確定済み。
        """
        output_path = os.path.join(state.project.pj_path, event_name)
        required = ["master_stage.csv", "master_idolname.csv", "turn_id_data.csv"]
        return all(os.path.exists(os.path.join(output_path, f)) for f in required)

    def enter_output_edit_mode(self, state: AppState, event_name: str) -> WorkflowResult:
        """編集モード開始。output_df[event_name] のディープコピーを edits[event_name] に格納する。

        live については、編集UIでライブ/特典会行の区別が必要なため、
        stage マスタの `特典会フラグ` を `ステージID` で join して付与する。
        """
        data = state.output.output_df.get(event_name)
        if not data:
            return WorkflowResult(
                success=False, error="編集対象データがありません",
            )
        edits: dict = {
            "stage": data["stage"].copy(deep=True),
            "idolname": data["idolname"].copy(deep=True),
        }
        live = data["live"].copy(deep=True)
        stage_flags = data["stage"][["特典会フラグ"]]
        live = live.join(stage_flags, on="ステージID", how="left")
        live["特典会フラグ"] = live["特典会フラグ"].fillna(False).astype(bool)

        # 特典会行に 対応出番ID 列を付与 (Phase 4: 親ライブとの紐付け)
        # heiki (live_tokutenkai_heiki) を含むイベントのみ追加する。
        # 非heiki イベントでは booth-別出番が存在しないため列ごと省く。
        event_no = repo.get_event_no_by_event_name(
            state.project.project_info_json, event_name,
        )
        corresp_map = _output_editor.build_corresponding_turn_id_map(
            state.project.pj_path, event_name, event_no,
            state.project.project_info_json,
        )
        if corresp_map:
            live["対応出番ID"] = pd.Series(
                [corresp_map.get(int(idx)) for idx in live.index],
                index=live.index,
                dtype="Int64",
            )
            # 対応出番ID の差分検知用 baseline (UI には出さない)
            edits["_live_baseline"] = live[["対応出番ID"]].copy()

        edits["live"] = live

        # ステージマスタ D&D ラベルに種別名を併記するためのマップ
        edits["stage_kind_map"] = _output_editor.build_stage_kind_map(
            state.project.pj_path, event_name, event_no,
            state.project.project_info_json,
        )

        state.output.edits[event_name] = edits
        return WorkflowResult(success=True)

    def cancel_output_edit_mode(self, state: AppState, event_name: str) -> WorkflowResult:
        """編集モードキャンセル。作業コピーを破棄する。"""
        state.output.edits.pop(event_name, None)
        return WorkflowResult(success=True)

    def save_output_edits(self, state: AppState, event_name: str) -> WorkflowResult:
        """編集結果を保存。
        バリデーション → master CSV + stage_*.json + project_info への書き戻し →
        output_df を再構築 → 編集モード解除。
        """
        edits = state.output.edits.get(event_name)
        if not edits:
            return WorkflowResult(success=False, error="編集中ではありません")

        # バリデーション
        validation_errors: list[str] = []
        if "stage" in edits and edits["stage"] is not None:
            validation_errors += _output_editor.validate_stage_master_edits(edits["stage"])
        if "idolname" in edits and edits["idolname"] is not None:
            validation_errors += _output_editor.validate_idolname_master_edits(edits["idolname"])
        if "live" in edits and edits["live"] is not None:
            original_data = state.output.output_df.get(event_name) or {}
            original_live = original_data.get("live")
            idol_for_validation = (
                edits["idolname"] if "idolname" in edits and edits["idolname"] is not None
                else original_data.get("idolname")
            )
            stage_for_validation = (
                edits["stage"] if "stage" in edits and edits["stage"] is not None
                else original_data.get("stage")
            )
            if idol_for_validation is not None:
                validation_errors += _output_editor.validate_live_master_edits(
                    edits["live"], idol_for_validation, original_live,
                    df_stage=stage_for_validation,
                )
        if validation_errors:
            return WorkflowResult(
                success=False,
                error="\n".join(validation_errors),
            )

        event_no = repo.get_event_no_by_event_name(
            state.project.project_info_json, event_name,
        )
        _output_editor.save_event_edits(
            state.project.pj_path, event_name, event_no,
            state.project.project_info_json, edits,
        )
        # 編集モード解除 + output_df 再構築
        state.output.edits.pop(event_name, None)
        state.output.output_df = _output.build_all_event_outputs(
            state.project.pj_path, state.project.project_info_json,
        )
        # 編集内容を全タイムテーブル画像に反映
        # 1. 各 (img_type, stage_no) の stage_N_timetable.png を再生成
        # 2. 種別単位 + 種別横断の集約画像を再生成
        self._regenerate_stage_timetable_pictures(state, event_name, event_no)
        _etp.regenerate_all_event_images(
            state.project.pj_path, event_name,
            state.project.project_info_json,
        )
        state.project.project_master = repo.update_timestamp(
            state.project.project_master, state.project.pj_name, self._data_path,
        )
        return WorkflowResult(success=True)

    def _regenerate_stage_timetable_pictures(
        self, state: AppState, event_name: str, event_no: int,
    ) -> None:
        """イベント配下の全 (img_type, stage_no) の stage_N_timetable.png を再生成する。

        TimeAxisConverter が利用可能な image entry では time_match=True で生成し、
        無い場合は ocr_service 側が time_match=False 経路にフォールバックする。

        (img_type, stage_no) ごとに独立した処理のため、並列実行する。
        """
        pij = state.project.project_info_json
        tasks: list[tuple[str, Any, int]] = []
        for img_type in repo.get_event_type_list(pij, event_no):
            entry = repo.get_image_entry_by_dir_name(pij, event_no, img_type)
            if entry is None:
                continue
            converter = _time_axis.TimeAxisConverter.from_project_info(
                pij, event_no, img_type,
            )
            for stage_no in range(entry.get("stage_num", 0)):
                tasks.append((img_type, converter, stage_no))

        if not tasks:
            return

        pj_path = state.project.pj_path

        def _run(img_type: str, converter: Any, stage_no: int) -> None:
            _ocr.generate_timetable_picture(
                stage_no, pj_path, event_name, img_type,
                time_match=converter is not None,
                time_axis_converter=converter,
                project_info_json=pij,
                event_no=event_no,
            )

        max_workers = min(len(tasks), 8)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run, *t) for t in tasks]
            for future in futures:
                future.result()
