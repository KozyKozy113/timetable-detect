"""
ワークフロー層。

UIアクション（ボタンクリック等）に対応するユースケースを、
フレームワーク非依存で実装する。
UIコールバックとバックエンドサービスの橋渡しを担う。
"""

from __future__ import annotations

import json
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
from backend_functions import timetable_diff_llm as _diff_llm
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

    def delete_project(self, pj_name: str, state: AppState) -> WorkflowResult:
        """プロジェクトを削除する。

        順序:
            1) ローカル本体ディレクトリ削除 + ローカル両マスタCSVから行削除
            2) S3 projects/<pj_name>/ プレフィックス削除
            3) S3 master/projects_master_s3.csv に「行を削除した状態の写し」を上書き

        途中失敗時は WorkflowResult.error にどこまで進んだかを格納する。
        ユーザーが意図的に中断した場合に S3 がバックアップとして残るよう
        「ローカル先」順を採用している。全ステップが冪等なため、エラー時は
        再度同じ pj_name で削除を実行すれば収束する。
        """
        pm = state.project.project_master
        pm_s3 = state.project.project_master_s3
        exists = (
            (pm is not None and pj_name in pm.index)
            or (pm_s3 is not None and pj_name in pm_s3.index)
        )
        if not exists:
            return WorkflowResult(success=False, error="存在しないプロジェクトです")

        # 1) ローカル削除
        try:
            new_pm, new_pm_s3 = repo.delete_project_data(
                self._data_path, pj_name, pm, pm_s3,
            )
            state.project.project_master = new_pm
            state.project.project_master_s3 = new_pm_s3
        except Exception as e:
            return WorkflowResult(
                success=False,
                error=f"ローカル削除に失敗しました: {e}",
            )

        # 2) S3 オブジェクト削除
        try:
            s3access.delete_project_from_s3(pj_name)
        except Exception as e:
            return WorkflowResult(
                success=False,
                error=(
                    "S3 プロジェクトデータの削除に失敗しました"
                    "（ローカルは削除済み）。再度削除を実行してください: "
                    f"{e}"
                ),
            )

        # 3) S3 master/projects_master_s3.csv 上書き
        try:
            s3access.put_projects_master_s3()
        except Exception as e:
            return WorkflowResult(
                success=False,
                error=(
                    "S3 projects_master_s3.csv の上書きに失敗しました"
                    "（ローカル・S3 本体は削除済み）。再度削除を実行してください: "
                    f"{e}"
                ),
            )

        # 削除したのが選択中プロジェクトなら state をクリア
        if state.project.pj_name == pj_name:
            from app_state import CropState, OcrState, OutputState
            state.project.pj_name = None
            state.project.pj_path = None
            state.project.project_info_json = None
            state.project.event_type = None
            state.project.event_num = 1
            state.crop = CropState()
            state.ocr = OcrState()
            state.output = OutputState()

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

    def output_difference_image(self, state: AppState,
                                new_image_file: Any, event_name: str,
                                img_type: str) -> WorkflowResult:
        """既存画像・全体差分・ステージ別差分をまとめて生成して返す。

        戻り値 data は dict:
            old_image / new_image / diff_image / stages
        （詳細は timetable_difference.analyze_difference_by_stage を参照）
        """
        pj_path = state.project.pj_path
        old_image_path = os.path.join(pj_path, event_name, img_type, "raw.png")
        if not os.path.exists(old_image_path):
            return WorkflowResult(success=False, error=f"既存画像が見つかりません: {old_image_path}")
        old_image = Image.open(old_image_path)

        event_no = repo.get_event_no_by_event_name(
            state.project.project_info_json, event_name,
        )
        entry = repo.get_image_entry_by_dir_name(
            state.project.project_info_json, event_no, img_type,
        )
        stage_list = entry.get("stage_list", []) if entry is not None else []

        result = timetable_difference.analyze_difference_by_stage(
            Image.open(new_image_file), old_image, stage_list,
        )
        return WorkflowResult(success=True, data=result)

    # ------------------------------------------------------------------
    # ⑤変更比較：LLMによる変更案の生成・反映
    # ------------------------------------------------------------------

    def _resolve_known_groups(self, state: AppState, event_name: str) -> list[str]:
        """既知グループ名一覧を解決する。

        master_idolname.csv を優先し、無ければ当該イベント配下の全 stage_*.json の
        グループ名_採用 の和集合をフォールバックとして返す。
        """
        output_path = os.path.join(state.project.pj_path, event_name)
        try:
            _, _, idolname_df, _ = _output.load_existing_masters(output_path)
        except Exception:
            idolname_df = None
        if idolname_df is not None and len(idolname_df) > 0 and "グループ名_採用" in idolname_df.columns:
            return [str(n) for n in idolname_df["グループ名_採用"].dropna().tolist()]

        names: set[str] = set()
        event_dir = os.path.join(state.project.pj_path, event_name)
        if os.path.isdir(event_dir):
            for img_type in os.listdir(event_dir):
                type_dir = os.path.join(event_dir, img_type)
                if not os.path.isdir(type_dir):
                    continue
                for fn in os.listdir(type_dir):
                    if fn.startswith("stage_") and fn.endswith(".json"):
                        try:
                            with open(os.path.join(type_dir, fn), encoding="utf-8") as f:
                                sj = json.load(f)
                        except Exception:
                            continue
                        for rec in sj.get("タイムテーブル", []) or []:
                            v = rec.get("グループ名_採用")
                            if v:
                                names.add(v)
        return sorted(names)

    def propose_changes_for_stage(self, state: AppState,
                                  event_name: str, img_type: str, stage_no: int,
                                  diff_result: dict) -> WorkflowResult:
        """1ステージの変更案をLLMで生成して返す（生成のみ・副作用なし）。

        stage_n.json が無いステージは対象外。
        """
        pj_path = state.project.pj_path
        json_path = os.path.join(pj_path, event_name, img_type, f"stage_{stage_no}.json")
        if not os.path.exists(json_path):
            return WorkflowResult(success=False, error=f"タイムテーブル未作成のため対象外（stage_{stage_no}）")
        with open(json_path, encoding="utf-8") as f:
            stage_json = json.load(f)

        # 形式判定（stage_list[].kind を第一に）
        event_no = repo.get_event_no_by_event_name(state.project.project_info_json, event_name)
        entry = repo.get_image_entry_by_dir_name(state.project.project_info_json, event_no, img_type)
        stage_entry = None
        if entry is not None:
            for s in entry.get("stage_list", []):
                if s.get("stage_no") == stage_no:
                    stage_entry = s
                    break
        with_tokutenkai = _diff_llm.detect_with_tokutenkai(stage_json, stage_entry)

        # old_crop / new_crop を diff_result から取得
        stage_diff = None
        for s in (diff_result.get("stages") or []):
            if s.get("stage_no") == stage_no:
                stage_diff = s
                break
        if stage_diff is None:
            return WorkflowResult(success=False, error=f"差分結果にステージ {stage_no} が見つかりません")

        known_groups = self._resolve_known_groups(state, event_name)
        name_to_group_id = self._resolve_group_name_id_map(state, event_name)
        try:
            proposal = _diff_llm.propose_changes(
                stage_diff.get("old_crop"), stage_diff.get("new_crop"),
                stage_json, known_groups, with_tokutenkai,
            )
        except Exception as e:
            return WorkflowResult(success=False, error=f"変更案の生成に失敗しました（stage_{stage_no}）: {e}")

        return WorkflowResult(success=True, data={
            "stage_no": stage_no,
            "with_tokutenkai": with_tokutenkai,
            "stage_json": stage_json,
            "proposal": proposal,
            "known_groups": known_groups,
            "name_to_group_id": name_to_group_id,
        })

    def _resolve_group_name_id_map(self, state: AppState, event_name: str) -> dict:
        """グループ名_採用 → グループID のマップを master_idolname.csv から構築する（無ければ空）。"""
        output_path = os.path.join(state.project.pj_path, event_name)
        try:
            _, _, idolname_df, _ = _output.load_existing_masters(output_path)
        except Exception:
            return {}
        name_to_id: dict = {}
        if idolname_df is not None and "グループ名_採用" in idolname_df.columns:
            for gid, row in idolname_df.iterrows():
                name = row.get("グループ名_採用")
                if name not in (None, ""):
                    try:
                        name_to_id[str(name)] = int(gid)
                    except (ValueError, TypeError):
                        continue
        return name_to_id

    def propose_changes_all_diff_stages(self, state: AppState,
                                        event_name: str, img_type: str,
                                        diff_stages: list, diff_result: dict) -> WorkflowResult:
        """差分ありステージ（stage_n.json有り）の変更案を並列生成して集約する。"""
        stage_nos = [s["stage_no"] for s in diff_stages]
        results: dict = {}
        skipped: list = []

        def _work(stage_no):
            return stage_no, self.propose_changes_for_stage(
                state, event_name, img_type, stage_no, diff_result,
            )

        with ThreadPoolExecutor(max_workers=5) as ex:
            for stage_no, res in ex.map(_work, stage_nos):
                if res.success:
                    results[stage_no] = res.data
                else:
                    skipped.append((stage_no, res.error))
        return WorkflowResult(success=True, data={"results": results, "skipped": skipped})

    def apply_all_change_proposals(self, state: AppState,
                                   event_name: str, img_type: str,
                                   new_image_path: str | None,
                                   edited_by_stage: dict) -> WorkflowResult:
        """全差分ステージの解決済み操作を一括反映する（全体一括・原子的）。

        edited_by_stage: {stage_no: {
            "with_tokutenkai": bool,
            "resolved_ops": [ ... ],                # timetable_diff_llm の解決済み操作
            "master_name_updates": {group_id: new_name},  # 任意
        }}
        new_image_path: アップロードされた新規画像のファイルパス（raw差し替え用）。
        """
        pj_path = state.project.pj_path
        pij = state.project.project_info_json
        event_no = repo.get_event_no_by_event_name(pij, event_name)

        # 1. 各 stage_n.json を編集して保存
        for stage_no, info in edited_by_stage.items():
            json_path = os.path.join(pj_path, event_name, img_type, f"stage_{stage_no}.json")
            if not os.path.exists(json_path):
                continue
            with open(json_path, encoding="utf-8") as f:
                stage_json = json.load(f)
            new_json = _diff_llm.apply_proposal_to_stage_json(
                stage_json, info.get("resolved_ops", []), info.get("with_tokutenkai", False),
            )
            # 新規検出グループの採用名を補正ステップで生成（採用名が空のレコードのみ対象）
            new_json = _ocr.fill_empty_adopted_names(new_json)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(new_json, f, indent=4, ensure_ascii=False)

        # 2. マスタ更新（グループ名変更かつグループID保持時）
        name_updates: dict = {}
        for info in edited_by_stage.values():
            name_updates.update(info.get("master_name_updates") or {})
        if name_updates:
            self._update_idolname_master(state, event_name, event_no, name_updates)

        # 3. raw.png / 各ステージ切り出し画像を新規画像で一度だけ差し替え
        if new_image_path:
            tmp_path = self._prepare_resized_new_image(pj_path, event_name, img_type, new_image_path)
            try:
                err = _imgproc.replace_stage_images_from_new_raw(
                    tmp_path, pj_path, event_name, img_type, pij, event_no,
                )
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            if err:
                return WorkflowResult(success=False, error=err)

        # 4. 可視化画像の再生成（per-stage ＋ 集約）
        entry = repo.get_image_entry_by_dir_name(pij, event_no, img_type)
        if entry is not None:
            converter = _time_axis.TimeAxisConverter.from_project_info(pij, event_no, img_type)
            for stage_no in range(int(entry.get("stage_num", 0) or 0)):
                _ocr.generate_timetable_picture(
                    stage_no, pj_path, event_name, img_type,
                    time_match=converter is not None,
                    time_axis_converter=converter,
                    project_info_json=pij, event_no=event_no,
                )
        _etp.regenerate_all_event_images(pj_path, event_name, pij)

        state.project.project_master = repo.update_timestamp(
            state.project.project_master, state.project.pj_name, self._data_path,
        )
        return WorkflowResult(success=True)

    def _prepare_resized_new_image(self, pj_path: str, event_name: str,
                                   img_type: str, new_image_path: str) -> str:
        """新規画像を既存raw.pngサイズへ揃えて一時ファイルに保存し、そのパスを返す。

        replace_stage_images_from_new_raw は新旧同サイズを要求するため、bbox整合を保つ。
        """
        base_dir = os.path.join(pj_path, event_name, img_type)
        new_img = Image.open(new_image_path).convert("RGB")
        old_raw = os.path.join(base_dir, "raw.png")
        if os.path.exists(old_raw):
            with Image.open(old_raw) as oi:
                target_size = oi.size
            if new_img.size != target_size:
                new_img = new_img.resize(target_size, Image.LANCZOS)
        tmp_path = os.path.join(base_dir, "_new_raw_tmp.png")
        new_img.save(tmp_path)
        return tmp_path

    def _update_idolname_master(self, state: AppState, event_name: str,
                                event_no: int, name_updates: dict) -> None:
        """master_idolname.csv の指定グループIDの名称を更新し、全 stage_*.json へ伝播する。"""
        output_path = os.path.join(state.project.pj_path, event_name)
        csv_path = os.path.join(output_path, "master_idolname.csv")
        if not os.path.exists(csv_path):
            return
        df = pd.read_csv(csv_path, index_col=0)
        col = "グループ名_採用" if "グループ名_採用" in df.columns else "グループ名"
        for gid, new_name in name_updates.items():
            try:
                gid_int = int(gid)
            except (ValueError, TypeError):
                continue
            if gid_int in df.index:
                df.at[gid_int, col] = new_name
        df.to_csv(csv_path)

        df_norm = df.rename(columns={col: "グループ名_採用"})
        _output_editor._propagate_idolname_master_to_json(
            state.project.pj_path, event_name, event_no,
            state.project.project_info_json, df_norm,
        )


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
            autodetect_collab=options.get("autodetect_collab", False),
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
                           stage_num: int,
                           detect_color: bool = True) -> WorkflowResult:
        """ステージ名をOCRで読み取り、project_info_jsonを更新。

        detect_color=True で 特典会種別以外のステージカラーも併せて推定・永続化する。
        """
        state.project.project_info_json = _ocr.detect_stage_names(
            state.project.pj_path, event_name, img_type, stage_num,
            user_prompt, state.project.project_info_json,
            detect_color=detect_color,
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

    def determine_id_master(
        self, state: AppState,
        target_event_names: list[str] | None = None,
        only_event_types: list[str] | None = None,
    ) -> WorkflowResult:
        """IDマスタを確定して保存する。

        `target_event_names` を指定すると当該イベントのみ確定・画像再生成・
        timestamp bump の対象とする (auto-trigger からの呼び出し用)。
        None の場合は全イベントを対象 (従来の手動ボタン互換)。

        `only_event_types` を指定すると stage_id 書き戻し対象の種別を限定する
        (採番済み種別のみ確定。④保存からの種別スコープ確定用)。None で全種別。
        """
        targets = (
            target_event_names
            if target_event_names is not None
            else repo.get_event_name_list(state.project.project_info_json)
        )
        if not targets:
            return WorkflowResult(success=True)

        _output.determine_id_master(
            state.output.output_df,
            state.project.pj_path,
            state.project.project_info_json,
            target_event_names=targets,
            only_event_types=only_event_types,
        )
        # ステージID確定でステージマスタ参照が成立したイベントの集約画像を再生成する。
        for event_name in targets:
            _etp.regenerate_all_event_images(
                state.project.pj_path, event_name,
                state.project.project_info_json,
            )
        # 実際に永続化したイベントがあるときのみ timestamp を bump する
        # (auto-trigger の空振りで S3 同期が頻発しないようにするため)
        state.project.project_master = repo.update_timestamp(
            state.project.project_master, state.project.pj_name, self._data_path,
        )
        return WorkflowResult(success=True)

    def recommit_event_ids_after_edit(
        self, state: AppState, event_name: str, img_type: str,
    ) -> WorkflowResult:
        """④保存後に、採番済み種別のみを再ビルドして ID マスタを再確定する。

        - 編集中の種別 (`img_type`) が未採番なら no-op (json保存のみ・⑥まで遅延)。
        - 採番済みなら、イベント内の **採番済み種別のみ** に限定して再ビルド・永続化する
          (未採番種別=後から追加した特典会等は巻き込まない)。
        - 内部整合異常 (`detect_id_anomalies`) があれば永続化せず success=False で返す
          (= 永続化をブロック)。
        - master差分 (`detect_master_diff`) は非ブロックの通知として `data["notices"]` に格納。

        戻り値の `data` は常に `{"notices": [...]}` 形式 (異常ブロック時は空)。
        """
        pj_path = state.project.pj_path
        pij = state.project.project_info_json
        event_no = repo.get_event_no_by_event_name(pij, event_name)
        if event_no is None:
            return WorkflowResult(success=True, data={"notices": [], "regenerated": False})

        # 編集中種別が未採番 → 採番処理を行わない (⑥まで遅延)
        if not repo.img_type_ids_assigned(pij, event_no, img_type):
            return WorkflowResult(success=True, data={"notices": [], "regenerated": False})

        # 採番済みの種別のみを対象にする
        assigned_types = [
            et for et in repo.get_event_type_list(pij, event_no)
            if repo.img_type_ids_assigned(pij, event_no, et)
        ]
        if not assigned_types:
            return WorkflowResult(success=True, data={"notices": [], "regenerated": False})

        # ビルド可否 (採番済み種別の範囲で グループ名_採用 欠損がないこと)
        if _ocr.check_event_has_empty_adopted_idol_names(
            pj_path, event_name, event_no, pij, only_event_types=assigned_types,
        ):
            return WorkflowResult(success=True, data={"notices": [], "regenerated": False})

        data = _output.build_event_output(
            pj_path, event_name, event_no, pij, only_event_types=assigned_types,
        )
        if not data:
            return WorkflowResult(success=True, data={"notices": [], "regenerated": False})

        # 内部整合異常 → ブロック
        anomalies = _output.detect_id_anomalies(data)
        if anomalies:
            return WorkflowResult(
                success=False,
                error=(
                    "ID異常を検出したため永続化を中止しました"
                    " (json保存は完了。修正して再保存してください):\n"
                    + "\n".join(f"  - {m}" for m in anomalies)
                ),
                data={"notices": [], "regenerated": False},
            )

        # master差分 → 非ブロック通知 (永続化前に旧 master を読む)
        output_path = os.path.join(pj_path, event_name)
        notices = _output.detect_master_diff(data, output_path)

        state.output.output_df[event_name] = data
        result = self.determine_id_master(
            state, target_event_names=[event_name], only_event_types=assigned_types,
        )
        # determine_id_master が集約画像をフル再生成済み (regenerate_all_event_images)。
        return WorkflowResult(
            success=result.success, error=result.error,
            data={"notices": notices, "regenerated": True},
        )

    def list_events_with_unconfirmed_ids(self, state: AppState) -> list[str]:
        """`output_df` 上で未確定 (= 永続化されていない) IDを持つイベント名一覧を返す。

        判定条件 (どちらかを満たせば未確定):
          1. master_stage.csv / master_idolname.csv / turn_id_data.csv のいずれかが欠落
          2. 既存 CSV と in-memory `output_df` の index 集合に差分がある
             (= 新規ステージ/グループ/出番ID が追加されている)

        `build_event_output` は呼び出し時に全行へ ID を採番するため、
        `output_df` 側に NaN が残ることはない前提だが、保険として
        NaN 含む場合も未確定として扱う。
        """
        result: list[str] = []
        pj_path = state.project.pj_path
        if not pj_path:
            return result
        for event_name, data in (state.output.output_df or {}).items():
            if not data:
                continue
            if self._event_has_unconfirmed_ids(pj_path, event_name, data):
                result.append(event_name)
        return result

    @staticmethod
    def _event_has_unconfirmed_ids(
        pj_path: str, event_name: str, data: dict,
    ) -> bool:
        output_path = os.path.join(pj_path, event_name)
        stage_csv = os.path.join(output_path, "master_stage.csv")
        idolname_csv = os.path.join(output_path, "master_idolname.csv")
        live_csv = os.path.join(output_path, "turn_id_data.csv")

        # 条件1: ファイル欠落 (採番アウトプットの有無は repo に集約)
        if not repo.event_ids_assigned(pj_path, event_name):
            return True

        # 条件2: index集合の差分検出 (in-memory に新規IDがある)
        def _index_diff(df_in_memory, csv_path) -> bool:
            if df_in_memory is None:
                return False
            try:
                df_disk = pd.read_csv(csv_path, index_col=0)
            except (OSError, ValueError, pd.errors.ParserError):
                return True
            disk_ids = set(df_disk.index.tolist())
            mem_ids = set(df_in_memory.index.tolist())
            return bool(mem_ids - disk_ids)

        if _index_diff(data.get("stage"), stage_csv):
            return True
        if _index_diff(data.get("idolname"), idolname_csv):
            return True
        if _index_diff(data.get("live"), live_csv):
            return True

        # 保険: NaN index (採番失敗) も未確定として扱う
        for key in ("stage", "idolname", "live"):
            df = data.get(key)
            if df is None:
                continue
            try:
                if df.index.isna().any():
                    return True
            except (AttributeError, TypeError):
                continue
        return False

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
