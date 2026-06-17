"""
アプリケーション状態の型付き定義。

フレームワーク非依存の純粋なPythonオブジェクトとして、
アプリケーション全体の状態を管理する。
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from PIL import Image


@dataclass
class ProjectState:
    """プロジェクト関連の状態"""
    pj_name: str | None = None
    pj_path: str | None = None
    project_info_json: dict | None = None
    project_master: pd.DataFrame | None = None
    project_master_s3: pd.DataFrame | None = None
    event_type: str | None = None
    event_num: int = 1


@dataclass
class CropState:
    """画像切り取り関連の状態"""
    crop_tgt_event: str | None = None
    crop_tgt_img_type: str | None = None
    cropped_image: Image.Image | None = None
    crop_box: dict | None = None
    images_eachstage: list[Image.Image] = field(default_factory=list)
    images_eachstage_bbox: list[dict] = field(default_factory=list)
    stage_crop_rects: list[dict] = field(default_factory=list)
    stage_line_list: pd.DataFrame | None = None
    annotated_image: Image.Image | None = None


@dataclass
class OcrState:
    """OCR関連の状態"""
    ocr_tgt_event: str | None = None
    ocr_tgt_img_type: str | None = None
    ocr_tgt_image_info: dict | None = None
    ocr_tgt_stage_num: int = 0
    timeline_eachstage: list = field(default_factory=list)
    time_axis_detect: Any = None
    df_timetables: list[pd.DataFrame] = field(default_factory=list)
    correct_idolname_in_confirmed_list: bool = False
    ocr_output_picture_time_match: bool = True


@dataclass
class OutputState:
    """出力関連の状態"""
    output_df: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)
    new_idolname: pd.DataFrame | None = None
    # ⑥出力確認・編集 編集モードの作業コピー (event_name -> {"stage": df, ...})
    edits: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)
    # Phase 3: Stella JSON トップレベルメタデータ (event_name -> dict)
    stella_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class UIState:
    """UI関連の状態 (ページ遷移検知など)"""
    last_page: str | None = None


@dataclass
class AppState:
    """アプリケーション全体の状態（フレームワーク非依存）"""
    project: ProjectState = field(default_factory=ProjectState)
    crop: CropState = field(default_factory=CropState)
    ocr: OcrState = field(default_factory=OcrState)
    output: OutputState = field(default_factory=OutputState)
    ui: UIState = field(default_factory=UIState)
