from pydantic import BaseModel
from typing import List

class LiveStage(BaseModel):
    """開始時刻と終了時刻を持つライブのステージ情報を表す。「10:30」のように「hh:mm」形式でそれぞれ出力する。"""
    time_from: str
    time_to: str

class Tokutenkai(BaseModel):
    """特典会情報。開始時刻と終了時刻、およびその実施場所を持つ。開始時刻と終了時刻は「10:30」のように「hh:mm」形式でそれぞれ出力する。"""
    time_from: str
    time_to: str
    booth: str

class ArtistLive(BaseModel):
    """ある出演者のライブステージ情報（グループ名とライブステージの時間）を表す。"""
    artist_name: str
    live_stage: LiveStage

class ArtistLiveTokutenkai(BaseModel):
    """ある出演者のライブステージ情報と特典会情報（グループ名とライブステージの時間、特典会の時間と場所）を表す。"""
    artist_name: str
    live_stage: LiveStage
    tokutenkai: List[Tokutenkai]

class TimetableLive(BaseModel):
    """あるステージのステージ名と、出演者一覧およびそれぞれの出演者のライブの時間情報を表す。"""
    stage_name: str
    timetable: List[ArtistLive]

class TimetableLiveTokutenkai(BaseModel):
    """あるステージのステージ名と、出演者一覧およびそれぞれの出演者のライブ・特典会の情報を表す。"""
    stage_name: str
    timetable: List[ArtistLiveTokutenkai]
