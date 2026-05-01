"""
時間軸とピクセル座標の相互変換。

time_pixel設定（time_start, start_pix, total_pix, total_duration）を
もとに、ピクセル値と時刻の変換を行う純粋なロジック。
Streamlitに依存しない。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Optional

import numpy as np


@dataclass
class TimePixelConfig:
    """時間軸のピクセル設定"""
    time_start: time       # 基準時刻
    start_pix: int         # 基準時刻のピクセル位置
    total_pix: int         # 時間軸全体のピクセル高さ
    total_duration: float  # 時間軸全体の分数


class TimeAxisConverter:
    """時間軸とピクセルの相互変換器"""

    def __init__(self, config: TimePixelConfig) -> None:
        self._config = config

    @property
    def config(self) -> TimePixelConfig:
        return self._config

    @classmethod
    def from_project_info(
        cls,
        project_info_json: dict,
        event_no: int,
        img_type: str,
    ) -> Optional[TimeAxisConverter]:
        """project_info_jsonからコンバータを構築する。

        time_pixel未設定の場合はNoneを返す。
        """
        time_format = "%H:%M"
        try:
            time_pixel = project_info_json["event_detail"][event_no]["timetables"][img_type]["time_pixel"]
            config = TimePixelConfig(
                time_start=datetime.strptime(time_pixel["time_start"], time_format).time(),
                start_pix=time_pixel["start_pix"],
                total_pix=time_pixel["total_pix"],
                total_duration=time_pixel["total_duration"],
            )
            return cls(config)
        except KeyError:
            return None

    def pix_to_time(self, pix: float) -> time:
        """ピクセル値を時刻に変換する（5分単位で丸め）"""
        c = self._config
        minutes = np.round(
            (pix - c.start_pix) / (c.total_pix / c.total_duration * 5)
        ) * 5
        return (
            datetime(2024, 1, 1, c.time_start.hour, c.time_start.minute)
            + timedelta(minutes=float(minutes))
        ).time()

    def time_to_pix(self, tgt_time: time) -> int:
        """時刻をピクセル値に変換する"""
        c = self._config
        minutes = (
            datetime.combine(datetime.today(), tgt_time)
            - datetime.combine(datetime.today(), c.time_start)
        ).total_seconds() / 60
        return c.start_pix + int(minutes * c.total_pix / c.total_duration)

    def time_length_to_pix(self, minutes: float, int_flag: bool = True) -> int | float:
        """時間の長さ（分）をピクセル幅に変換する"""
        c = self._config
        value = minutes * c.total_pix / c.total_duration
        if int_flag:
            return int(value)
        return value


def build_time_pixel_config(
    time_start: time,
    top: int,
    height: int,
    total_duration: float,
) -> dict:
    """project_info_json保存用のtime_pixel辞書を構築して返す。

    I/Oは行わない。呼び出し側がproject_info_jsonに書き込み・保存する。
    """
    time_format = "%H:%M"
    return {
        "time_start": time_start.strftime(time_format),
        "start_pix": top,
        "total_pix": height,
        "total_duration": total_duration,
    }
