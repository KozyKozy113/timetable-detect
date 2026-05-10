# app.py リファクタリング計画

## 概要

`src/app.py`（2,254行）はStreamlit UIの描画、ビジネスロジック、データアクセス、画像処理がすべて混在しており、責務が過大になっている。
本計画では、**フロントエンド（Streamlit）から切り離せる実装とデータ保持**をバックエンドモジュールとして抽出し、app.pyをUI層に限定する。

さらに、**将来的にStreamlitから別のフレームワーク（FastAPI + React等）への移行**を見据え、UI層自体も疎結合・ステートレスな設計にする。具体的には、アプリケーション状態の型付き管理、ワークフロー層の導入、Streamlit固有のパターン（`st.session_state`, `st.stop()`, `@st.cache_data` 等）の抽象化を行う。

### 目標
- app.pyをUI描画とコールバックの薄いレイヤーに限定する（目安: ~800行）
- ビジネスロジックを独立したモジュールに分離し、テスト可能にする
- `st.session_state`への依存を最小化し、バックエンドの関数は引数で必要な値を受け取る設計にする
- **アプリケーション状態を型付きオブジェクトとして管理し、UIフレームワークのState APIに非依存にする**
- **UIコールバックとビジネスロジックの間にワークフロー層を導入し、フレームワーク移行時の変更範囲をUI層のみに限定する**

### 方針
- 既存の`backend_functions/`配下に新モジュールを追加する形で進める
- 既存モジュール（`gpt_ocr.py`, `timetabledata.py`, `idolname.py`, `s3access.py`, `ticket_scraper.py`）は変更しない
- 段階的に移行可能な設計とし、一度にすべてを変更しない
- **バックエンドモジュールからは `import streamlit` を一切含めない**
- **app.py（UI層）のみがStreamlitに依存し、将来的にapp.pyのみの差し替えでフレームワーク移行が完了する構成を目指す**

---

## 1. 現状分析

### 1.1 app.py の責務マップ

| 行範囲 | 責務 | 種別 | 行数 |
|---------|------|------|------|
| 1-66 | インポート・初期化・セッション初期化 | UI + データ | 66 |
| 67-221 | プロジェクト管理（作成/読込/設定） | **ビジネスロジック** | 155 |
| 222-348 | 画像登録・イベント情報アクセサ | **ビジネスロジック** + UI | 127 |
| 349-566 | 画像切り取り・ステージ分割 | **画像処理** | 218 |
| 574-636 | 時間軸変換（ピクセル⇔時刻） | **ユーティリティ** | 63 |
| 638-940 | OCR実行・タイムライン検出 | **画像処理 + OCR連携** | 303 |
| 942-993 | グループ名補正 | **ビジネスロジック** | 52 |
| 995-1101 | ステージ名管理・バッチ処理・保存 | **ビジネスロジック** | 107 |
| 1102-1130 | 出力生成（画像化） | **ビジネスロジック** | 29 |
| 1131-1230 | 画像差分・置き換え | **画像処理 + ファイル操作** | 100 |
| 1231-1320 | 出力生成（Excel・S3・マスタ更新） | **ビジネスロジック** | 90 |
| 1322-2276 | Streamlit UI（7セクション） | UI | 955 |

**ビジネスロジック合計: 約1,210行（52%）** — これがフロントエンドから分離可能な部分。

### 1.2 `st.session_state` 依存の分類

app.pyの関数が`st.session_state`から読み取る主な値:

| session_state キー | 用途 | 依存関数数 |
|---|---|---|
| `pj_path` | プロジェクトディレクトリパス | ~30 |
| `project_info_json` | プロジェクト設定JSON | ~25 |
| `project_master` | プロジェクト一覧CSV | ~5 |
| `ocr_tgt_event` / `ocr_tgt_img_type` | 操作対象のイベント/種別 | ~15 |
| `crop_tgt_event` / `crop_tgt_img_type` | 切り取り対象のイベント/種別 | ~10 |
| `ocr_tgt_stage_num` / `ocr_tgt_image_info` | ステージ数・画像情報 | ~10 |
| `images_eachstage` / `stage_crop_rects` | 切り取り結果の画像リスト | ~8 |
| `timeline_eachstage` | タイムライン検出結果 | ~5 |
| `df_timetables` | 読み取り結果DataFrame | ~5 |
| `x_edge_threshold_*` / `y_edge_threshold_*` 等 | 画像処理パラメータ | ~5 |

### 1.3 Streamlit固有パターンの依存マップ

フレームワーク移行時に対処が必要なStreamlit固有パターン:

| パターン | 使用箇所 | 移行時の課題 |
|---------|---------|-------------|
| `st.session_state` (グローバル可変辞書) | ~120箇所 | 型なし・スコープなしの状態管理をフレームワーク非依存の型付きオブジェクトに置換 |
| `on_click` / `on_change` コールバック | ~28箇所 | Streamlit独自のコールバックモデル。他フレームワークではイベントハンドラやAPIエンドポイントに対応 |
| `st.stop()` (描画中断) | 4箇所 (L1248,1373,1392,2115) | Streamlit固有のフロー制御例外。条件分岐に変換が必要 |
| `@st.cache_data` | 1箇所 (L222) | `functools.lru_cache`等の汎用キャッシュに置換 |
| `st.warning/error/success` (ビジネスロジック内) | L72,237,256,267,281,295,604 | ビジネスロジックから通知を除去し、戻り値で結果を返す設計に変更 |
| `st_cropper` (サードパーティ) | L1427 | Streamlit専用の画像クロッパー。他フレームワーク用のクロッパーに差し替え |
| `AgGrid` (サードパーティ) | L1893 | Streamlit専用のデータグリッド。他フレームワーク用のグリッドに差し替え |
| UIコンテナへの直接書き込み | L237,256,267,281,295,316,444 | 関数がモジュールスコープのUI変数（`col_file_uploader`, `edge_result`）を参照 |
| 動的session_stateキー | ~15箇所 | `f"ocr_user_prompt_stage{i}"` 等のフォーマット文字列でキーを動的生成 |

### 1.4 既存モジュール構成

```
src/
├── app.py                          (2,254行 - 本リファクタリング対象)
├── backend_functions/
│   ├── gpt_ocr.py                 (31KB - GPT Vision API連携)
│   ├── timetabledata.py           (18KB - JSON⇔DataFrame変換)
│   ├── idolname.py                (12KB - グループ名マッチング)
│   ├── s3access.py                (9KB - AWS S3同期)
│   └── ticket_scraper.py          (5KB - チケットサイトスクレイピング)
├── frontend_functions/
│   ├── timetablepicture.py        (7KB - タイムテーブル画像生成)
│   └── timetable_difference.py    (1KB - 差分画像生成)
└── gpt_output_format/
    └── timetable_format.py        (Pydanticモデル)
```

---

## 2. リファクタリング後のモジュール構成

```
src/
├── app.py                          (UI層のみ ~800行、唯一のStreamlit依存)
├── app_state.py                    (NEW: 型付きアプリケーション状態)
├── workflow.py                     (NEW: ワークフロー層)
├── backend_functions/
│   ├── gpt_ocr.py                 (既存・変更なし)
│   ├── timetabledata.py           (既存・変更なし)
│   ├── idolname.py                (既存・変更なし)
│   ├── s3access.py                (既存・変更なし)
│   ├── ticket_scraper.py          (既存・変更なし)
│   ├── project_repository.py      (NEW: プロジェクトデータ管理)
│   ├── image_processing.py        (NEW: 画像処理)
│   ├── time_axis.py               (NEW: 時間軸変換)
│   ├── ocr_service.py             (NEW: OCR実行オーケストレーション)
│   └── output_builder.py          (NEW: 出力データ構築)
├── frontend_functions/             (既存・変更なし)
└── gpt_output_format/              (既存・変更なし)
```

**レイヤー構造:**

```
┌───────────────────────────────────────────────────────┐
│  app.py (UI層)                                        │
│  - Streamlit固有のUI描画・入力・レイアウト             │
│  - session_stateとAppStateの同期                       │
│  - ワークフロー呼び出しと結果のUI反映                  │
│  ※ フレームワーク移行時に差し替える唯一のファイル      │
├───────────────────────────────────────────────────────┤
│  app_state.py (状態層)                                │
│  - 型付きアプリケーション状態の定義                    │
│  - フレームワーク非依存                               │
├───────────────────────────────────────────────────────┤
│  workflow.py (ワークフロー層)                          │
│  - UIアクションに対応するユースケース                  │
│  - AppStateとバックエンドサービスの橋渡し              │
│  - フレームワーク非依存                               │
├───────────────────────────────────────────────────────┤
│  backend_functions/ (サービス層)                       │
│  - project_repository, image_processing, time_axis,   │
│    ocr_service, output_builder                        │
│  - gpt_ocr, timetabledata, idolname, s3access         │
│  - 純粋なビジネスロジック・データアクセス              │
│  - フレームワーク非依存                               │
└───────────────────────────────────────────────────────┘
```

---

## 3. 新規モジュール詳細設計

### 3.1 `project_repository.py` — プロジェクトデータ管理

**責務**: プロジェクトのCRUD操作、project_info.jsonの読み書き、マスタCSVの管理

**現在app.pyにある対象関数**:
- `make_project()` (L67-99)
- `set_project()` の非UIロジック部分 (L101-128)
- `get_project_json()` (L130-134)
- `set_project_json()` (L143-147)
- `update_project_timestamp()` (L136-141)
- `determine_project_setting()` (L200-221)
- `save_ticket_urls()` (L149-180)
- `get_ticket_urls_for_event()` (L182-198)
- `get_event_name()` (L319-320)
- `get_event_name_list()` (L322-323)
- `get_event_type_list()` (L325-334)
- `get_event_no_by_event_name()` (L336-341)
- `get_stage_name_list()` (L343-344)
- `get_stage_name()` (L346-347)

**設計**:

```python
class ProjectRepository:
    """プロジェクトデータへのアクセスを一元管理する"""

    def __init__(self, data_path: str):
        self.data_path = data_path

    # --- プロジェクトライフサイクル ---
    def create_project(self, pj_name: str, project_master: pd.DataFrame) -> dict:
        """新規プロジェクトを作成し、project_info_jsonを返す"""

    def load_project(self, pj_name: str) -> dict:
        """プロジェクト設定を読み込む（S3同期はapp.py側で呼ぶ）"""

    def load_project_master(self) -> pd.DataFrame:
        """projects_master.csvを読み込む"""

    # --- project_info.json操作 ---
    def get_project_json(self, pj_path: str) -> dict:
        """project_info.jsonを読み込む"""

    def save_project_json(self, pj_path: str, json_data: dict) -> None:
        """project_info.jsonを書き込み、タイムスタンプを更新"""

    def update_timestamp(self, project_master: pd.DataFrame, pj_name: str) -> None:
        """更新日時をCSVに反映"""

    # --- プロジェクト設定 ---
    def apply_project_setting(self, pj_path: str, project_info_json: dict,
                               event_type: str, event_num: int) -> dict:
        """イベント形式・数の変更を適用し、更新後のJSONを返す"""

    def save_ticket_urls(self, project_info_json: dict, scope: str,
                          urls_data: dict) -> dict:
        """チケットURL設定を適用し、更新後のJSONを返す"""

    def get_ticket_urls_for_event(self, project_info_json: dict,
                                   event_name: str) -> list:
        """指定イベントのチケットURLを取得"""

    # --- イベント/ステージ情報アクセサ ---
    def get_event_name(self, project_info_json: dict, event_no: int) -> str:

    def get_event_name_list(self, project_info_json: dict) -> list[str]:

    def get_event_type_list(self, project_info_json: dict, event_no: int) -> list[str]:

    def get_event_no_by_event_name(self, project_info_json: dict,
                                     event_name: str) -> int | None:

    def get_stage_name_list(self, project_info_json: dict,
                             event_no: int, img_type: str) -> list[str]:

    def get_stage_name(self, project_info_json: dict,
                        event_no: int, img_type: str, stage_no: int) -> str:

    # --- 画像登録 ---
    def register_timetable_image(self, pj_path: str, event_name: str,
                                    img_type: str, img_format: str,
                                    file_data: bytes,
                                    project_info_json: dict) -> dict:
        """画像ファイルを保存しproject_info_jsonを更新して返す"""
```

**ポイント**:
- `st.session_state` に依存しない。必要な値（`pj_path`, `project_info_json`等）は引数で受け取る
- JSON/CSVの読み書きを一箇所に集約し、エンコーディングやパス構築を統一する
- app.py側では `repo = ProjectRepository(DATA_PATH)` でインスタンスを生成し、session_stateの値を引数として渡す
- `determine_timetable_image()`の6分岐で共通する画像保存処理を`register_timetable_image()`に集約

---

### 3.2 `image_processing.py` — 画像処理

**責務**: OpenCVによる画像処理（エッジ検出、ハフ変換、画像分割）。Streamlitに一切依存しない。

**現在app.pyにある対象関数**:
- `detect_stageline()` (L385-446) — 縦線検出
- `detect_timeline_onlyonestage()` (L820-934) — 横線検出
- `get_x_freq()` (L355-374) — 矩形のX座標頻度分析
- `get_image_eachstage_byocr()` (L449-478) — OCRベースの画像分割
- `get_image_eachstage_for_croppedimage_byevenly()` (L501-515) — 均等分割
- `replace_stage_images_from_new_raw()` (L1170-1203) — bboxによる新画像からのステージ画像置き換え（画像クロップ・addtime削除）

**設計**:

```python
@dataclass
class LineDetectionParams:
    """エッジ検出・ハフ変換のパラメータ"""
    edge_threshold_1: int
    edge_threshold_2: int
    hough_threshold: int
    hough_gap: int
    minlength_rate: float
    identify_interval: int

@dataclass
class StageLineResult:
    """縦線検出の結果"""
    line_list: pd.DataFrame          # 検出された線の一覧
    stage_images: list[Image.Image]  # ステージごとの画像
    crop_rects: list[dict]           # 各ステージの矩形座標
    debug_image: Image.Image         # 線描画済みのデバッグ用画像

@dataclass
class TimelineResult:
    """横線検出の結果"""
    timeline_df: pd.DataFrame        # y座標・長さ・時刻のDataFrame
    annotated_image_path: str        # 時刻注釈付き画像のパス

def detect_stage_lines(image: Image.Image, params: LineDetectionParams) -> StageLineResult:
    """画像から縦線を検出し、ステージごとに分割する（純粋な画像処理）"""

def detect_timeline(image_path: str, params: LineDetectionParams,
                     pix_to_time_fn, time_length_to_pix_fn,
                     ignoretime_threshold: int) -> TimelineResult:
    """画像から横線を検出し、時刻を推定する（純粋な画像処理）"""

def split_image_evenly(image: Image.Image, stage_num: int,
                        overlap_ratio: float = 0.05) -> tuple[list[Image.Image], list[dict]]:
    """画像を均等幅で分割する"""

def split_image_by_xpoints(image: Image.Image, xpoints: list[int],
                            stage_num: int) -> list[Image.Image]:
    """指定X座標で画像を分割する"""

def get_rectangle_x_frequency(image: np.ndarray,
                                stage_num: int) -> pd.Series:
    """矩形のX座標出現頻度を取得"""

def crop_stages_by_bbox(image: Image.Image,
                         stage_list: list[dict]) -> list[tuple[int, Image.Image]]:
    """bboxを使って画像から各ステージ領域をクロップする。
    戻り値: (stage_no, cropped_image) のリスト。bbox未設定のステージはスキップ。"""

def crop_by_raw_crop_box(image: Image.Image,
                          raw_crop_box: dict) -> Image.Image:
    """raw_crop_box（left, top, width, height）で画像をクロップする。"""
```

**ポイント**:
- 現在の`detect_stageline()`はUIコンテナ(`edge_result`)への書き込みと`st.session_state`への保存を含むが、新モジュールでは結果をデータクラスで返すだけにする
- `detect_timeline_onlyonestage()`の時刻変換部分は、コールバック関数（`pix_to_time_fn`）として外から注入する
- パラメータは個別のsession_state変数ではなく`LineDetectionParams`にまとめて渡す

---

### 3.3 `time_axis.py` — 時間軸変換

**責務**: ピクセル座標と時刻の相互変換

**現在app.pyにある対象関数**:
- `pix_to_time()` (L591-604)
- `time_to_pix()` (L606-619)
- `time_length_to_pix()` (L622-636)
- `save_time_pixel()` (L574-582)

**設計**:

```python
@dataclass
class TimePixelConfig:
    """時間軸の設定（project_info.jsonのtime_pixel相当）"""
    time_start: str     # "HH:MM"
    start_pix: int
    total_pix: int
    total_duration: int  # 分

class TimeAxisConverter:
    """ピクセル座標と時刻の相互変換"""

    def __init__(self, config: TimePixelConfig):
        self.config = config

    def pix_to_time(self, pix: int) -> time:
        """ピクセル値を時刻に変換"""

    def time_to_pix(self, tgt_time: time) -> int:
        """時刻をピクセル値に変換"""

    def time_length_to_pix(self, minutes: float, int_flag: bool = True) -> int | float:
        """時間の長さをピクセル値に変換"""

    @staticmethod
    def create_time_pixel_config(time_start: time, top: int,
                                   height: int, total_duration: float) -> dict:
        """保存用のtime_pixel辞書を作成"""
```

**ポイント**:
- 現在は`st.session_state`からイベント/画像種別を辿って`time_pixel`を取得しているが、新モジュールではconfigを直接受け取る
- 純粋関数として実装し、テスト容易にする

---

### 3.4 `ocr_service.py` — OCR実行オーケストレーション

**責務**: GPT OCRの呼び出し管理、並列実行、結果保存

**現在app.pyにある対象関数**:
- `get_timetabledata_onestage()` (L638-693)
- `get_timetabledata_onestage_worker()` (L695-762) — 既にsession_state非依存
- `get_timetabledata_allstages()` (L764-800)
- `get_timetabledata_together()` (L1044-1078)
- `get_stagelist()` (L995-1018)
- `save_timetable_data_onlyonestage()` (L1080-1096)
- `save_timetable_data_eachstage()` (L1098-1100)
- `idolname_correct_onlyonestage()` (L942-966)
- `idolname_correct_eachstage()` (L968-974)
- `get_idolname_confirmed_list()` (L976-993)
- `set_stage_name()` (L1020-1032)
- `booth_name_add_prefix_onlyonestage()` (L1034-1038)

**設計**:

```python
class OcrService:
    """OCR実行とタイムテーブルデータの管理"""

    def __init__(self, project_repo: ProjectRepository):
        self.repo = project_repo

    # --- OCR実行 ---
    def read_single_stage(self, mode: str, stage_no: int, user_prompt: str,
                           pj_path: str, event_name: str, img_type: str,
                           project_info_json: dict,
                           ticket_urls: list[str] | None = None) -> dict:
        """1ステージのOCRを実行し、結果JSONを保存して返す"""
        # 現在のget_timetabledata_onestage_workerを移植

    def read_all_stages(self, mode: str, user_prompt: str,
                         pj_path: str, event_name: str, img_type: str,
                         project_info_json: dict, stage_num: int,
                         ticket_urls: list[str] | None = None,
                         max_workers: int = 10) -> None:
        """全ステージを並列OCR実行"""

    def read_stage_names(self, pj_path: str, event_name: str,
                          img_type: str, stage_num: int,
                          user_prompt: str) -> tuple[list[str], str]:
        """ステージ名一覧をOCRで読み取る"""

    # --- グループ名補正 ---
    def correct_idol_names_single(self, pj_path: str, event_name: str,
                                    img_type: str, stage_no: int,
                                    use_confirmed_list: bool,
                                    confirmed_list: list[str] | None = None) -> None:
        """1ステージのグループ名を補正"""

    def correct_idol_names_all(self, pj_path: str, event_name: str,
                                 img_type: str, stage_num: int,
                                 use_confirmed_list: bool) -> None:
        """全ステージのグループ名を補正"""

    def get_confirmed_idol_names(self, pj_path: str, event_name: str,
                                   project_info_json: dict) -> list[str]:
        """確定済みグループ名の一覧を取得"""

    # --- データ保存 ---
    def save_timetable_data(self, pj_path: str, event_name: str,
                              img_type: str, stage_no: int,
                              df_timetable: pd.DataFrame,
                              stage_name: str, format_type: str) -> None:
        """編集後のタイムテーブルデータを保存"""

    def set_stage_name(self, pj_path: str, event_name: str,
                        img_type: str, stage_no: int, stage_name: str,
                        project_info_json: dict) -> dict:
        """ステージ名を更新"""

    def add_booth_prefix(self, df_timetable: pd.DataFrame,
                          stage_name: str) -> pd.DataFrame:
        """特典会ブース名にステージ名プレフィックスを付与"""
```

**ポイント**:
- `get_timetabledata_onestage_worker()`は既にsession_state非依存なので、ほぼそのまま移植可能
- `get_timetabledata_together()`のイベント横断バッチ処理もここに含める
- グループ名補正は`idolname`モジュールを内部で呼ぶ

---

### 3.5 `output_builder.py` — 出力データ構築

**責務**: ステージマスタ・アーティストマスタ・出番データの組み立て、Excel出力

**現在app.pyにある対象関数**:
- `determine_id_master()` (L1143-1165)
- `output_data_for_stella()` (L1170-1179)
- `save_dataframe_to_excel()` (L1181-1193)
- `listup_new_idolname()` (L1195-1201)
- `update_master_idolname()` (L1203-1222)
- `output_timetable_picture_onlyonestage()` (L1102-1125)
- UI⑥セクション内のデータ集計ロジック (L2029-2168, 約140行)

**設計**:

```python
@dataclass
class EventOutputData:
    """1イベント分の出力データ"""
    stage_df: pd.DataFrame
    idolname_df: pd.DataFrame
    live_df: pd.DataFrame

class OutputBuilder:
    """出力データの構築"""

    def __init__(self, project_repo: ProjectRepository):
        self.repo = project_repo

    def build_event_output(self, pj_path: str, event_name: str,
                             event_no: int, project_info_json: dict) -> EventOutputData:
        """1イベント分のステージマスタ・アーティストマスタ・出番データを構築する
        
        現在app.pyのUI⑥セクション(L2038-2158)にインラインで書かれている
        データ集計ロジックをこのメソッドに集約する。
        """

    def save_id_masters(self, pj_path: str, project_info_json: dict,
                          output_data: dict[str, EventOutputData]) -> None:
        """IDマスタを確定して保存する（現determine_id_master相当）"""

    def export_to_excel(self, pj_path: str,
                          output_data: dict[str, EventOutputData]) -> str:
        """Excelファイルを出力しパスを返す（現output_data_for_stella相当）"""

    @staticmethod
    def save_dataframe_to_excel(wb: Workbook, sheet_name: str,
                                  df: pd.DataFrame, position: tuple) -> None:
        """DataFrameをExcelワークブックの指定位置に書き込む"""

    def generate_timetable_image(self, pj_path: str, event_name: str,
                                   img_type: str, stage_no: int,
                                   project_info_json: dict,
                                   time_match: bool,
                                   time_axis: 'TimeAxisConverter | None') -> None:
        """読み取り結果から構造化タイムテーブル画像を生成"""

    def list_new_idol_names(self, output_data: dict[str, EventOutputData]) -> list[str]:
        """新規登場グループ名をリストアップ"""

    def update_idol_name_master(self, data_path: str,
                                  new_names: list[str]) -> None:
        """グループ名マスタに新規名を追加しS3同期"""
```

**ポイント**:
- **最大のリファクタリング効果**はUI⑥セクションの集計ロジック抽出にある。現在140行以上のデータ組み立てコードがStreamlit UIコードの中に埋まっている。これを`build_event_output()`に集約する
- Excel出力はStreamlitに依存しないため、完全にバックエンドで処理可能

---

### 3.6 `app_state.py` — 型付きアプリケーション状態

**責務**: アプリケーション全体の状態を型付きオブジェクトとして定義する。`st.session_state`の型なし辞書から脱却し、状態の構造と型をコードで明示する。

**設計上の考え方**:

現在の `st.session_state` は以下の問題を抱えている:
1. **型なし**: `st.session_state.pj_name` が `str | None` なのか `str` なのか、コードを読まないと分からない
2. **スコープなし**: プロジェクト設定、切り取りパラメータ、OCR設定が全て同じフラットな名前空間にある
3. **フレームワーク依存**: `st.session_state` はStreamlit固有のAPI

これを型付きdataclassに置き換えることで、IDEの補完・型チェック・リファクタリングサポートが効くようになり、かつフレームワーク非依存になる。

```python
from dataclasses import dataclass, field
from PIL import Image
import pandas as pd
from typing import Any


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
    crop_rect: dict | None = None
    images_eachstage: list[Image.Image] = field(default_factory=list)
    stage_crop_rects: list[dict] = field(default_factory=list)
    stage_line_list: pd.DataFrame | None = None


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


@dataclass
class OutputState:
    """出力関連の状態"""
    output_df: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)
    new_idolname: pd.DataFrame | None = None


@dataclass
class AppState:
    """アプリケーション全体の状態（フレームワーク非依存）"""
    project: ProjectState = field(default_factory=ProjectState)
    crop: CropState = field(default_factory=CropState)
    ocr: OcrState = field(default_factory=OcrState)
    output: OutputState = field(default_factory=OutputState)
```

**UIパラメータの扱い**:

スライダー値やチェックボックスの状態など、UIウィジェットに直接紐づくパラメータは `AppState` には含めない。これらはUI層が管理し、ワークフロー呼び出し時に引数として渡す。

```python
# UIパラメータの例（app_state.pyではなく、ワークフロー呼び出し時に直接渡す）
# st.session_state.x_edge_threshold_1 → LineDetectionParams.edge_threshold_1
# st.session_state.ocr_user_prompt_stage{i} → workflow.run_ocr(user_prompt=...)
```

**`st.session_state` との同期（app.py側で実装）**:

```python
# app.py 内
class StreamlitStateSync:
    """AppStateとst.session_stateの双方向同期"""

    @staticmethod
    def load(state: AppState) -> None:
        """st.session_stateからAppStateにロードする（アプリ起動時）"""
        if "pj_name" in st.session_state:
            state.project.pj_name = st.session_state.pj_name
            state.project.pj_path = st.session_state.pj_path
            state.project.project_info_json = st.session_state.project_info_json
            # ... 以下同様

    @staticmethod
    def save(state: AppState) -> None:
        """AppStateからst.session_stateに書き戻す"""
        st.session_state.pj_name = state.project.pj_name
        st.session_state.pj_path = state.project.pj_path
        st.session_state.project_info_json = state.project.project_info_json
        # ... 以下同様
```

**ポイント**:
- `AppState` 自体はフレームワーク非依存の純粋なPythonオブジェクト
- `StreamlitStateSync` はapp.pyの中に置き、Streamlitとの橋渡しを担う
- フレームワーク移行時は `StreamlitStateSync` を削除し、新フレームワークに合った状態管理に差し替えるだけでよい
- AppStateは**論理的な状態のみ**を持ち、UIウィジェットの状態（スライダー値等）は持たない

---

### 3.7 `workflow.py` — ワークフロー層

**責務**: UIアクション（ボタンクリック等）に対応するユースケースを、フレームワーク非依存で実装する。

**設計上の考え方**:

現在のapp.pyでは、コールバック関数が以下を同時に行っている:
1. `st.session_state` からの入力値取得
2. ビジネスロジックの呼び出し
3. `st.session_state` への結果格納
4. `st.warning()` 等によるUI通知

リファクタリング後は、このうち(2)と(3)をワークフロー層に移動する。(1)と(4)のみがUI層（app.py）に残る。

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkflowResult:
    """ワークフローの実行結果"""
    success: bool
    data: Any = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


class ProjectWorkflow:
    """プロジェクト管理のワークフロー"""

    def __init__(self, repo: ProjectRepository):
        self.repo = repo

    def create_project(self, pj_name: str, state: AppState,
                        existing_names: list[str]) -> WorkflowResult:
        """新規プロジェクトを作成し、stateを更新する"""
        if pj_name in existing_names:
            return WorkflowResult(success=False, error="既に存在する名前のプロジェクトです")
        project_info = self.repo.create_project(pj_name, state.project.project_master)
        state.project.project_info_json = project_info
        state.project.pj_name = pj_name
        state.project.pj_path = os.path.join(self.repo.data_path, "projects", pj_name)
        return WorkflowResult(success=True)

    def load_project(self, pj_name: str, state: AppState) -> WorkflowResult:
        """既存プロジェクトを読み込み、stateを更新する"""
        # S3同期、project_info_json読込、各種state初期化を一括実行
        ...
        return WorkflowResult(success=True)

    def update_project_setting(self, state: AppState,
                                 event_type: str, event_num: int) -> WorkflowResult:
        """プロジェクト設定を変更"""
        updated_json = self.repo.apply_project_setting(
            state.project.pj_path, state.project.project_info_json,
            event_type, event_num
        )
        state.project.project_info_json = updated_json
        return WorkflowResult(success=True)

    def register_image(self, state: AppState, event_name: str,
                         img_type: str, img_format: str,
                         file_data: bytes) -> WorkflowResult:
        """タイムテーブル画像を登録"""
        updated_json = self.repo.register_timetable_image(
            state.project.pj_path, event_name, img_type,
            img_format, file_data, state.project.project_info_json
        )
        state.project.project_info_json = updated_json
        state.crop.crop_tgt_event = event_name
        state.crop.crop_tgt_img_type = img_type
        state.ocr.ocr_tgt_event = event_name
        state.ocr.ocr_tgt_img_type = img_type
        return WorkflowResult(success=True, data="画像を登録しました")


class ImageWorkflow:
    """画像処理のワークフロー"""

    def detect_stage_lines(self, image: Image.Image,
                             params: LineDetectionParams) -> WorkflowResult:
        """ステージ線を検出し、結果を返す"""
        result = detect_stage_lines(image, params)
        return WorkflowResult(success=True, data=result)

    def split_evenly(self, image: Image.Image,
                       stage_num: int) -> WorkflowResult:
        """画像を均等分割"""
        images, rects = split_image_evenly(image, stage_num)
        return WorkflowResult(success=True, data={"images": images, "rects": rects})

    def replace_stage_images_from_new_raw(self, state: AppState, repo: ProjectRepository,
                                            new_image: Image.Image, event_name: str,
                                            img_type: str) -> WorkflowResult:
        """新画像からbboxで各ステージ画像を切り出して置き換える。
        raw.pngを新画像で上書きし、raw_cropped.pngとstage_X.pngを再生成。
        _addtime.pngが存在する場合は削除。"""

    def save_time_axis(self, state: AppState, repo: ProjectRepository,
                         time_start: str, top: int, height: int,
                         total_duration: float) -> WorkflowResult:
        """時間軸設定を保存"""
        config = TimeAxisConverter.create_time_pixel_config(
            time_start, top, height, total_duration
        )
        event_no = repo.get_event_no_by_event_name(
            state.project.project_info_json, state.crop.crop_tgt_event
        )
        state.project.project_info_json["event_detail"][event_no]\
            ["timetables"][state.crop.crop_tgt_img_type]["time_pixel"] = config
        repo.save_project_json(state.project.pj_path, state.project.project_info_json)
        return WorkflowResult(success=True)


class OcrWorkflow:
    """OCR実行のワークフロー"""

    def __init__(self, ocr_service: OcrService, repo: ProjectRepository):
        self.ocr = ocr_service
        self.repo = repo

    def run_ocr_single(self, state: AppState, mode: str,
                         stage_no: int, user_prompt: str,
                         ticket_urls: list[str] | None = None) -> WorkflowResult:
        """1ステージのOCR実行"""
        result = self.ocr.read_single_stage(
            mode, stage_no, user_prompt,
            state.project.pj_path, state.ocr.ocr_tgt_event,
            state.ocr.ocr_tgt_img_type, state.project.project_info_json,
            ticket_urls
        )
        return WorkflowResult(success=True, data=result)

    def run_ocr_all(self, state: AppState, mode: str,
                      user_prompt: str,
                      ticket_urls: list[str] | None = None) -> WorkflowResult:
        """全ステージの並列OCR実行"""
        self.ocr.read_all_stages(
            mode, user_prompt,
            state.project.pj_path, state.ocr.ocr_tgt_event,
            state.ocr.ocr_tgt_img_type, state.project.project_info_json,
            state.ocr.ocr_tgt_stage_num, ticket_urls
        )
        return WorkflowResult(success=True)

    def run_batch(self, state: AppState,
                    targets: list[dict],
                    options: dict) -> WorkflowResult:
        """一括OCR実行（現get_timetabledata_together相当）"""
        ...
        return WorkflowResult(success=True)


class OutputWorkflow:
    """出力のワークフロー"""

    def __init__(self, output_builder: OutputBuilder, repo: ProjectRepository):
        self.builder = output_builder
        self.repo = repo

    def build_all_events(self, state: AppState) -> WorkflowResult:
        """全イベントの出力データを構築"""
        event_list = self.repo.get_event_name_list(state.project.project_info_json)
        output = {}
        for i, event_name in enumerate(event_list):
            event_data = self.builder.build_event_output(
                state.project.pj_path, event_name, i,
                state.project.project_info_json
            )
            if event_data:
                output[event_name] = event_data
        return WorkflowResult(success=True, data=output)

    def export_excel(self, state: AppState) -> WorkflowResult:
        """Excelファイル出力"""
        file_path = self.builder.export_to_excel(
            state.project.pj_path, state.output.output_df
        )
        return WorkflowResult(success=True, data=file_path)
```

**`WorkflowResult` による通知の分離**:

現在ビジネスロジック内にある `st.warning()` / `st.error()` は、`WorkflowResult` の `warnings` / `error` フィールドで返す。UI層がこれを受け取って適切な方法で表示する。

```python
# Before (現状): ビジネスロジックがUIに直接通知
def make_project(pj_name=None):
    if pj_name in pj_name_list:
        with project_setting:
            st.error("既に存在する名前のプロジェクトです")  # ← UI依存

# After: ワークフローが結果を返し、UI側で表示
# workflow.py
def create_project(self, pj_name, state, existing_names):
    if pj_name in existing_names:
        return WorkflowResult(success=False, error="既に存在する名前のプロジェクトです")
    ...

# app.py (Streamlit UI)
def on_create_project():
    result = project_wf.create_project(st.session_state.new_pj_name, app_state, pj_name_list)
    StreamlitStateSync.save(app_state)
    if not result.success:
        st.error(result.error)
    for warning in result.warnings:
        st.warning(warning)
```

**ポイント**:
- ワークフロー関数はStreamlitを一切importしない
- `AppState`を引数で受け取り、直接更新する（session_stateの仲介なし）
- 結果は`WorkflowResult`で返し、成功/失敗/警告をUI側で表示する
- フレームワーク移行時、ワークフロー層はそのまま再利用でき、UI層のみを差し替える

---

## 4. app.py 変更後の構造

リファクタリング後の`app.py`は以下の構造になる:

```python
# === インポート ===
import streamlit as st
from st_aggrid import AgGrid
from streamlit_cropper import st_cropper

from app_state import AppState, ProjectState, CropState, OcrState
from workflow import (
    ProjectWorkflow, ImageWorkflow, OcrWorkflow, OutputWorkflow,
    WorkflowResult
)
from backend_functions.project_repository import ProjectRepository
from backend_functions.image_processing import LineDetectionParams
from backend_functions.time_axis import TimeAxisConverter, TimePixelConfig
from backend_functions.ocr_service import OcrService
from backend_functions.output_builder import OutputBuilder

# === サービス初期化 ===
repo = ProjectRepository(DATA_PATH)
ocr_service = OcrService(repo)
output_builder = OutputBuilder(repo)

# === ワークフロー初期化 ===
project_wf = ProjectWorkflow(repo)
image_wf = ImageWorkflow()
ocr_wf = OcrWorkflow(ocr_service, repo)
output_wf = OutputWorkflow(output_builder, repo)

# === AppState初期化・同期 ===
if "app_state" not in st.session_state:
    st.session_state.app_state = AppState()
    # 初回のマスタデータ読込
    ...
app_state: AppState = st.session_state.app_state


# === UIコールバック（薄い橋渡し） ===
def on_create_project():
    """ワークフロー呼び出し → 結果をUIに反映"""
    result = project_wf.create_project(
        st.session_state.new_pj_name, app_state, pj_name_list
    )
    if not result.success:
        st.error(result.error)

def on_detect_stagelines():
    """UIウィジェットからパラメータを収集 → ワークフロー呼び出し"""
    params = LineDetectionParams(
        edge_threshold_1=st.session_state.x_edge_threshold_1,
        edge_threshold_2=st.session_state.x_edge_threshold_2,
        hough_threshold=st.session_state.x_hough_threshold,
        hough_gap=st.session_state.x_hough_gap,
        minlength_rate=st.session_state.x_minlength_rate,
        identify_interval=st.session_state.x_identify_interval,
    )
    result = image_wf.detect_stage_lines(app_state.crop.cropped_image, params)
    if result.success:
        app_state.crop.images_eachstage = result.data.stage_images
        app_state.crop.stage_crop_rects = result.data.crop_rects
        app_state.crop.stage_line_list = result.data.line_list

def on_run_ocr(mode, stage_no, user_prompt):
    """OCR実行のコールバック"""
    ticket_urls = repo.get_ticket_urls_for_event(
        app_state.project.project_info_json, app_state.ocr.ocr_tgt_event
    ) if st.session_state.use_ticket_urls else None
    result = ocr_wf.run_ocr_single(app_state, mode, stage_no, user_prompt, ticket_urls)
    for warning in result.warnings:
        st.warning(warning)


# === UI描画 ===
# ①プロジェクト設定
# ②画像登録
# ③画像切り取り
# ④読み取り
# ⑤変更比較
# ⑥出力
# ⑦マスタ更新
```

**コールバックの責務を3行で表現できるルール**:

```
1. UIウィジェットからパラメータを収集（st.session_stateからの読み取り）
2. ワークフロー関数を呼ぶ（app_stateを渡す）
3. 結果をUIに反映（st.error/warning/success、もしくは描画更新のトリガー）
```

---

## 5. Streamlit依存の解消戦略

### 5.1 `st.session_state` → `AppState`

**現状**: 約120箇所で `st.session_state.xxx` を直接読み書き

**方針**:
- ビジネスロジック・ワークフロー層では `AppState` オブジェクトを使う
- UIウィジェットの値（スライダー、テキスト入力等）は `st.session_state` のキーに直接紐付く（Streamlit固有）ため、コールバック内で `AppState` またはワークフロー引数に変換する
- `AppState` オブジェクト自体は `st.session_state.app_state` に格納して永続化する

```python
# Before: グローバル辞書アクセス
pj_path = st.session_state.pj_path
json_data = st.session_state.project_info_json

# After: 型付きオブジェクトアクセス
pj_path = app_state.project.pj_path        # IDE補完が効く
json_data = app_state.project.project_info_json  # 型チェックが効く
```

### 5.2 `st.stop()` → 条件付きレンダリング

**現状**: 4箇所で `st.stop()` を使って以降の描画を中断

```python
# Before: st.stop()で以降のUI全体を中断
if st.session_state.pj_name is None:
    st.stop()
```

**方針**: `st.stop()` を使わず、条件分岐でUI描画の範囲を制御する。

```python
# After: 条件分岐でセクション単位の表示を制御
if app_state.project.pj_name is not None:
    render_image_upload_section()
    render_crop_section()
    render_ocr_section()
    render_output_section()
```

これによりUIの描画フローが明示的になり、他フレームワークでのルーティングやコンポーネント分割に対応しやすくなる。

### 5.3 `@st.cache_data` → 汎用キャッシュ

**現状**: `get_image()` (L222) に `@st.cache_data` を使用

```python
# Before
@st.cache_data
def get_image(img_path):
    return Image.open(img_path)
```

**方針**: `functools.lru_cache` で置き換える。ただし `Image` オブジェクトはhashableでないため、パスをキーとしたキャッシュ辞書で管理する。

```python
# After: backend_functions/image_processing.py
_image_cache: dict[str, Image.Image] = {}

def get_image(img_path: str) -> Image.Image:
    if img_path not in _image_cache:
        _image_cache[img_path] = Image.open(img_path)
    return _image_cache[img_path]
```

### 5.4 ビジネスロジック内の `st.warning/error` → `WorkflowResult`

**現状**: 7箇所でビジネスロジック関数内から直接 `st.warning()` / `st.error()` を呼んでいる

**方針**: すべて `WorkflowResult` の `warnings` / `error` フィールドで返す。UIコールバック側でのみ表示処理を行う。

```python
# Before (pix_to_time L604)
except KeyError:
    st.warning("基準時間を設定してください")

# After (time_axis.py)
except KeyError:
    return None  # 呼び出し側で None を受け取りUI側で警告表示
```

### 5.5 UIコンテナへの直接参照 → コールバックでの描画

**現状**: モジュールスコープの変数 `col_file_uploader`, `edge_result`, `timetable_compare_col` を関数内から参照

**方針**: バックエンド関数はUIコンテナを参照せず、結果を返すだけにする。描画はコールバック関数またはUI描画セクションで行う。

### 5.6 サードパーティStreamlitコンポーネント

| コンポーネント | 用途 | 移行戦略 |
|--------------|------|---------|
| `st_cropper` | 画像の矩形選択 | 入力: `Image`, 出力: `(left, top, width, height)` の仕様のみ定義。実装はフレームワークごとに異なる |
| `AgGrid` | DataFrameの編集可能テーブル | 入力: `DataFrame`, 出力: 編集後`DataFrame` の仕様のみ定義。Streamlitでは`AgGrid`、Reactでは`ag-grid`等 |

これらはフレームワーク移行時にUI層ごと差し替わるコンポーネントなので、バックエンドとの境界（入出力の仕様）を明確にしておけばよい。

### 5.7 動的session_stateキー → ステージインデックスによるアクセス

**現状**: `f"ocr_user_prompt_stage{i}"` のようにフォーマット文字列でsession_stateキーを動的生成

**方針**: UIウィジェットのキーとしては引き続き使うが、ワークフローに渡す際はインデックスを使って明示的に値を取得し、引数として渡す。

```python
# Before: ワークフロー側で動的キーを知る必要がある
user_prompt = st.session_state["ocr_user_prompt_stage{}".format(i)]

# After: コールバック側で値を取得してワークフローに渡す
def on_run_ocr_stage(stage_no):
    user_prompt = st.session_state[f"ocr_user_prompt_stage{stage_no}"]
    ocr_wf.run_ocr_single(app_state, mode, stage_no, user_prompt)
```

---

## 6. 実施フェーズ

段階的に移行を行い、各フェーズ完了時にアプリが正常動作することを確認する。

### フェーズ0: `app_state.py` と `workflow.py` の基盤構築

**理由**: 以降のすべてのフェーズで使用する基盤モジュール。先に構造だけ作り、中身はフェーズ1以降で段階的に充実させる。

**内容**:
1. `app_state.py` を作成し、`AppState` とサブ状態のdataclassを定義
2. `workflow.py` を作成し、`WorkflowResult` と各Workflowクラスの空の骨格を定義
3. app.pyで `AppState` を `st.session_state.app_state` に格納する初期化コードを追加
4. **この時点では既存のコードは変更しない**（新モジュールの追加のみ）

**削減行数**: 0行（基盤追加のみ、既存コード変更なし）

---

### フェーズ1: `project_repository.py` の抽出

**理由**: 最も多くの関数が依存しており、かつロジックが単純（JSONの読み書き、辞書の操作）なため、リスクが低い。

**対象関数（app.pyから移動）**:
| 関数名 | 行 | 移動先メソッド |
|--------|-----|---------------|
| `get_project_json()` | L130-134 | `repo.get_project_json()` |
| `set_project_json()` | L143-147 | `repo.save_project_json()` |
| `update_project_timestamp()` | L136-141 | `repo.update_timestamp()` |
| `get_event_name()` | L319-320 | `repo.get_event_name()` |
| `get_event_name_list()` | L322-323 | `repo.get_event_name_list()` |
| `get_event_type_list()` | L325-334 | `repo.get_event_type_list()` |
| `get_event_no_by_event_name()` | L336-341 | `repo.get_event_no_by_event_name()` |
| `get_stage_name_list()` | L343-344 | `repo.get_stage_name_list()` |
| `get_stage_name()` | L346-347 | `repo.get_stage_name()` |
| `save_ticket_urls()` | L149-180 | `repo.save_ticket_urls()` |
| `get_ticket_urls_for_event()` | L182-198 | `repo.get_ticket_urls_for_event()` |
| `make_project()` | L67-99 | `repo.create_project()` |
| `determine_project_setting()` | L200-221 | `repo.apply_project_setting()` |

**手順**:
1. `project_repository.py`を作成し、上記関数をクラスメソッドとして移植（`st.session_state`依存を引数に変換）
2. `workflow.py`の`ProjectWorkflow`に対応するメソッドを実装
3. app.pyで`repo = ProjectRepository(DATA_PATH)`を初期化
4. app.py内の各呼び出し箇所を`repo.xxx()`に置き換え（session_stateの値を引数として渡す）
5. **`st.error()` / `st.warning()` がビジネスロジック内にある箇所は、WorkflowResultで返す形に変更**
6. 動作確認

**削減行数**: 約200行

---

### フェーズ2: `time_axis.py` の抽出

**理由**: 依存関係が少なく、最も純粋な計算ロジック。

**対象関数**:
| 関数名 | 行 | 移動先 |
|--------|-----|--------|
| `pix_to_time()` | L591-604 | `TimeAxisConverter.pix_to_time()` |
| `time_to_pix()` | L606-619 | `TimeAxisConverter.time_to_pix()` |
| `time_length_to_pix()` | L622-636 | `TimeAxisConverter.time_length_to_pix()` |
| `save_time_pixel()` | L574-582 | `TimeAxisConverter.create_time_pixel_config()` |

**手順**:
1. `time_axis.py`を作成
2. app.pyで時間軸が必要な箇所で`TimeAxisConverter`を生成して使用
3. `detect_timeline_onlyonestage()`内での`pix_to_time`呼び出しもコールバックに変更
4. **`st.warning("基準時間を設定してください")`をKeyError時のNone返却に変更し、UI側で警告表示**

**削減行数**: 約60行

---

### フェーズ3: `image_processing.py` の抽出

**理由**: 画像処理ロジックはStreamlitに本質的に依存しないが、現在はsession_stateの読み書きとUI描画が混在している。フェーズ1,2が完了していると依存が整理されている。

**対象関数**:
| 関数名 | 行 | 備考 |
|--------|-----|------|
| `detect_stageline()` | L385-446 | `edge_result`コンテナへの描画を除去 |
| `detect_timeline_onlyonestage()` | L820-934 | session_state依存を排除 |
| `get_x_freq()` | L355-374 | そのまま移植可能 |
| `get_image_eachstage_byocr()` | L449-478 | そのまま移植可能 |
| `get_image_eachstage_for_croppedimage_byevenly()` | L501-515 | session_state依存を排除 |
| `determine_image_eachstage()` | L517-540 | 画像保存+JSON更新の分離 |
| `determine_image_eachstage_without_nocheck()` | L542-566 | 同上 |
| `replace_stage_images_from_new_raw()` | L1170-1203 | 画像クロップ・ファイル操作・UIを分離 |

**手順**:
1. `image_processing.py`を作成
2. 各関数から`st.session_state`への参照を除去し、引数とreturnに変更
3. UI描画（`with edge_result:` 等）はapp.py側に残す
4. app.pyのコールバックを「パラメータ収集→ワークフロー呼び出し→結果をAppStateに格納→UI描画」に変更
5. **`@st.cache_data` の `get_image()` を `image_processing.py` に移動し、汎用キャッシュに変更**

**削減行数**: 約300行

---

### フェーズ4: `ocr_service.py` の抽出

**理由**: フェーズ1-3完了後、OCR関連関数のsession_state依存がほぼ解消されている。

**対象関数**:
| 関数名 | 行 | 備考 |
|--------|-----|------|
| `get_timetabledata_onestage_worker()` | L695-762 | 既にsession_state非依存 |
| `get_timetabledata_onestage()` | L638-693 | workerを呼ぶ形に統合 |
| `get_timetabledata_allstages()` | L764-800 | 並列実行管理 |
| `get_timetabledata_together()` | L1044-1078 | バッチ処理 |
| `get_stagelist()` | L995-1018 | |
| `idolname_correct_onlyonestage()` | L942-966 | |
| `idolname_correct_eachstage()` | L968-974 | |
| `get_idolname_confirmed_list()` | L976-993 | |
| `save_timetable_data_onlyonestage()` | L1080-1096 | |
| `set_stage_name()` | L1020-1032 | |
| `booth_name_add_prefix_onlyonestage()` | L1034-1038 | |

**削減行数**: 約350行

---

### フェーズ5: `output_builder.py` の抽出

**理由**: UI⑥セクションのインラインロジックの抽出は影響範囲が大きいため最後に行う。

**対象**:
| 対象 | 行 | 備考 |
|------|-----|------|
| `determine_id_master()` | L1143-1165 | |
| `output_data_for_stella()` | L1170-1179 | |
| `save_dataframe_to_excel()` | L1181-1193 | |
| `listup_new_idolname()` | L1195-1201 | |
| `update_master_idolname()` | L1203-1222 | |
| `output_timetable_picture_onlyonestage()` | L1102-1125 | |
| UI⑥セクションの集計ロジック | L2038-2158 | **最重要** |

**手順**:
1. `output_builder.py`を作成
2. UI⑥セクションの集計ロジック（ステージマスタ・アーティストマスタ・出番データの組み立て）を`build_event_output()`に抽出
3. app.pyのUI⑥セクションは`output_builder.build_event_output()`の結果を`st.dataframe()`で表示するだけにする
4. Excel出力、マスタ更新もOutputBuilderに移動

**削減行数**: 約240行

---

### フェーズ6: UI層の最終整理

**理由**: フェーズ1-5でバックエンド抽出が完了した後、UI層自体の疎結合化を仕上げる。

#### Step 6A: ProjectWorkflowのコールバック書き換え ✅ 完了

ProjectWorkflow実装済み。`make_project()`, `set_project()`, `determine_project_setting()`,
`determine_timetable_image()`, `save_ticket_urls()` 等をワークフロー経由に統一。

#### Step 6B: ImageWorkflowの実装 + 画像系コールバック書き換え ✅ 完了

ImageWorkflow 6メソッド実装（`detect_stage_lines`, `split_evenly`, `save_stage_images`,
`replace_stage_images_from_new_raw`, `save_time_axis`, `output_difference_image`）。
app.pyの画像系コールバック7関数を書き換え。

#### Step 6C: OcrWorkflowの実装 + OCR系コールバック書き換え ✅ 完了

OcrWorkflow 11メソッド実装。app.pyのOCR系コールバック17関数を書き換え。

#### Step 6D: OutputWorkflowの実装 + 出力系コールバック書き換え ✅ 完了

OutputWorkflow 5メソッド実装。app.pyの出力系コールバック5関数を書き換え。

---

#### Step 6E: 不要コードの削除 ← **次のタスク** (Step 6Fを先に実施)

後続ステップでのコード見通しを改善するため、不要なコード・コメントアウトブロックを先に削除する。

| 対象 | 行 | 理由 |
|------|-----|------|
| `get_image_eachstage_for_linecroppedimage_byocr()` | L266-282 | 旧UIフロー用。現在のUIから呼ばれていない |
| `get_image_eachstage_for_linecroppedimage_byevenly()` | L284-305 | `#使ってない` とコメントあり |
| 旧UI実装の大量コメントアウトブロック | L931-1053 | 約120行の旧UI実装コメント |

**推定削減行数**: 約160行

---

#### Step 6F: サイドバーナビゲーション導入 + UI描画のセクション関数化 + `st.stop()` の除去 ✅ 完了

現在縦に連なっている7つのセクション（①〜⑦）を、**サイドバーによる画面切り替え方式**に変更する。
一度に1画面分のみ描画されるため、パフォーマンスが向上し、処理フェーズの切り替えが明確になる。

**設計方針:**
- プロジェクト選択/作成はサイドバーに常時表示する
- ①設定〜⑦マスタ更新をラジオボタンで切り替え、選択中のフェーズのみメインエリアに描画する
- 前提条件を満たしていないフェーズにも遷移可能とし、画面内に警告メッセージを表示する
- `st.stop()` は全箇所除去し、各セクション関数内の early return に置き換える

**レイアウト構成:**

```
┌──────────────┬──────────────────────────────────┐
│ サイドバー    │  メインエリア                     │
│              │                                  │
│ [プロジェクト │  選択中のフェーズの               │
│  作成/呼出]  │  コンテンツのみ表示               │
│              │                                  │
│ 選択中: XXX  │                                  │
│              │                                  │
│ ── ナビ ──   │                                  │
│ ○ ①設定     │                                  │
│ ○ ②登録     │                                  │
│ ● ③切取     │                                  │
│ ○ ④読取     │                                  │
│ ○ ⑤比較     │                                  │
│ ○ ⑥出力     │                                  │
│ ○ ⑦マスタ   │                                  │
└──────────────┴──────────────────────────────────┘
```

**サイドバー（常時表示）の内容:**
- プロジェクト作成（テキスト入力 + 作成ボタン）
- プロジェクト呼出（セレクトボックス + 呼出ボタン）
- 選択中プロジェクト名の表示
- フェーズ切り替えラジオボタン（①設定〜⑦マスタ）

**セクション関数一覧:**

| 関数名 | 対応セクション | 前提条件（未達時は警告表示してreturn） |
|--------|--------------|--------------------------------------|
| `render_project_setting()` | ①プロジェクト設定 | なし（常にアクセス可能） |
| `render_image_upload()` | ②画像登録 | プロジェクト選択済み |
| `render_crop_section()` | ③画像切り取り | 画像登録済み + 対象イベントに画像あり |
| `render_ocr_section()` | ④読み取り | 画像登録済み + ステージ分割済み |
| `render_comparison_section()` | ⑤変更比較 | 画像登録済み + 対象イベントに画像あり |
| `render_output_section()` | ⑥出力 | 読み取りデータあり |
| `render_master_update_section()` | ⑦マスタ更新 | プロジェクト選択済み |

**メインエリアの描画フロー:**

```python
# === サイドバー ===
with st.sidebar:
    st.markdown("### プロジェクト")
    col_makepj = st.columns((5, 1))
    with col_makepj[0]:
        st.text_input("新しいプロジェクト名", key="new_pj_name")
    with col_makepj[1]:
        st.button("作成", on_click=make_project)
    col_setpj = st.columns((5, 1))
    with col_setpj[0]:
        st.selectbox("既存プロジェクト", pj_name_list, key="exist_pj_name")
    with col_setpj[1]:
        st.button("呼出", on_click=set_project, args=(st.session_state.exist_pj_name,))

    if app_state.project.pj_name is not None:
        st.success(f"選択中: {app_state.project.pj_name}")

    st.divider()
    page = st.radio("処理フェーズ", [
        "①設定", "②画像登録", "③画像切り取り",
        "④読み取り", "⑤変更比較", "⑥出力", "⑦マスタ更新",
    ])

# === メインエリア ===
if app_state.project.pj_name is None and page != "①設定":
    st.info("サイドバーからプロジェクトを選択または作成してください")
elif page == "①設定":
    render_project_setting()
elif page == "②画像登録":
    render_image_upload()
elif page == "③画像切り取り":
    render_crop_section()
elif page == "④読み取り":
    render_ocr_section()
elif page == "⑤変更比較":
    render_comparison_section()
elif page == "⑥出力":
    render_output_section()
elif page == "⑦マスタ更新":
    render_master_update_section()
```

**`st.stop()` の除去方針（7箇所すべてearly returnに置換）:**

| 現在の行 | 現在の場所 | 対応方法 |
|---------|----------|---------|
| L631 | ①プロジェクト設定 | メインエリアの条件分岐で対応（プロジェクト未選択時にst.info表示） |
| L756 | ②画像登録確認 | `render_crop_section()`等の冒頭で画像登録有無をチェック |
| L776 | ③切り取り | `render_crop_section()`内でearly return |
| L1090 | ④読み取り | `render_ocr_section()`内でearly return |
| L1108 | ④読み取り | `render_ocr_section()`内でearly return |
| L1406 | ⑤変更比較 | `render_comparison_section()`内でearly return |
| L1500 | ⑥出力 | `render_output_section()`内でearly return |

```python
# Before: st.stop()で以降全体を停止
if len(event_type_list) == 0:
    st.warning("画像を登録するか他のイベントを選択してください")
    st.stop()

# After: セクション関数内でearly return
def render_crop_section():
    event_type_list = get_event_type_list(...)
    if len(event_type_list) == 0:
        st.warning("画像を登録するか他のイベントを選択してください")
        return
    # ... 以降の描画処理
```

**①設定ページに含まれる内容:**
- イベント形式の選択（対バン/フェス）
- イベント数の入力
- プロジェクト設定反映ボタン
- チケットURL設定（expanderのまま）

**注意点:**
- コールバック関数内で参照している `edge_result`, `col_file_uploader`, `timetable_compare_col` 等のモジュールスコープUI変数は、各セクション関数のローカル変数に変更する。コールバックからの参照は `st.session_state` 経由のコンテナ参照か、コールバック内での描画に切り替える
- `st.set_page_config()` の `layout="wide"` は維持
- タイトルとヘッダー部分はメインエリア上部に残す

**推定削減行数**: 約50行（st.stop() + st.divider() + コンテナ宣言の除去。セクション関数化自体は行数を減らさないが見通しを大幅改善）

---

#### Step 6G: UIヘルパーの整理 ✅ 完了

app.pyに残っている `_repo.*` 薄ラッパー群を整理した。

- `update_project_timestamp()` — 全ワークフロー内部で `repo.update_timestamp()` を呼んでいるため削除
- `set_project_json()` — 同上。呼び出し元がなくなったため削除
- `get_project_json()` — 呼び出し元がなくなったため削除
- `delete_uploaded_image()` — `ProjectWorkflow.delete_image()` を追加し、ワークフロー経由に書き換え
- `get_event_name()` 等のrepoアクセサ6個 — UI用として残す（各所で多用）
- `get_idolname_confirmed_list()` — コールバック内ヘルパーとして残す
- `_get_time_axis_converter()` — 同上

---

#### Step 6H: `@st.cache_data` の汎用キャッシュ化 ✅ 完了

`@st.cache_data` の `get_image()` を `image_processing.py` 内の辞書キャッシュ (`_image_cache`) に移動。
app.pyの `get_image()` は `_imgproc.get_image()` への委譲のみとなり、Streamlit固有デコレータへの依存を解消。

---

**推奨実施順序**: 6E → 6F → 6G → 6H

**現在のapp.py**: 1,291行（目標 ~800行）

**Step 6F 実施内容まとめ:**
- サイドバーにプロジェクト作成/呼出 + フェーズ切り替えラジオを配置
- 7つのrender関数に分割: `render_project_setting()`, `render_image_upload()`, `render_crop_section()`, `render_ocr_section()`, `render_comparison_section()`, `render_output_section()`, `render_master_update_section()`
- `st.stop()` 7箇所を全てearly return/continueに置換
- `st.divider()` 5箇所（セクション間）を除去
- モジュールスコープUIコンテナ参照4箇所を解消（`project_setting`, `col_file_uploader`, `edge_result`, `timetable_compare_col`）
- `st.toast()`によるコールバックフィードバック、`app_state`によるwidgetデフォルト復元を実装
- `set_crop_image()`/`set_ocr_image()`にapp_state同期を追加

---

## 7. 注意事項

### 7.1 `st.session_state` の扱い

バックエンドモジュール・ワークフロー層は`st.session_state`を一切参照しない。app.pyの薄いコールバック関数が以下の役割を担う:

```python
# Before (現状): ロジックがsession_stateに直接依存
def detect_stageline(image):
    img_path = os.path.join(st.session_state.pj_path, st.session_state.crop_tgt_event, ...)
    st.session_state.cropped_image.save(img_path)
    # ... 100行のロジック ...
    st.session_state.images_eachstage = ...
    with edge_result:
        st.image(image_copy)

# After (リファクタリング後): コールバック → ワークフロー → AppState
def on_detect_stagelines():
    # 1. UIパラメータの収集
    params = LineDetectionParams(
        edge_threshold_1=st.session_state.x_edge_threshold_1,
        edge_threshold_2=st.session_state.x_edge_threshold_2,
        hough_threshold=st.session_state.x_hough_threshold,
        hough_gap=st.session_state.x_hough_gap,
        minlength_rate=st.session_state.x_minlength_rate,
        identify_interval=st.session_state.x_identify_interval,
    )
    # 2. ワークフロー呼び出し（AppStateを渡す）
    result = image_wf.detect_stage_lines(app_state.crop.cropped_image, params)
    # 3. 結果をAppStateに反映
    if result.success:
        app_state.crop.images_eachstage = result.data.stage_images
        app_state.crop.stage_crop_rects = result.data.crop_rects
        app_state.crop.stage_line_list = result.data.line_list
    # 4. UI通知（エラー・警告があれば）
    for warning in result.warnings:
        st.warning(warning)
```

### 7.2 `_worker` パターンの解消

現在、`get_timetabledata_onestage()`（session_state依存）と`get_timetabledata_onestage_worker()`（非依存）が二重に存在している。リファクタリング後は`ocr_service`側に非依存版のみを持ち、app.pyのコールバックが値を渡す形に統一する。

### 7.3 UIコンテナ参照の分離

現在、一部の関数がモジュールスコープのUIコンテナ変数（`edge_result`, `col_file_uploader`, `timetable_compare_col`等）を参照している:

- `detect_stageline()` → `edge_result` (L444)
- `determine_timetable_image()` → `col_file_uploader` (L237, 256, 267, 281, 295)
- `delete_uploaded_image()` → `col_file_uploader` (L316)
- `output_difference_image()` → `timetable_compare_col` (L1167)
- `replace_stage_images_from_new_raw()` → UIコンテナ依存なし（st.warning使用のみ）

これらはリファクタリング時にapp.pyのコールバック側に移動する。

### 7.4 画像登録関数の分岐整理

`determine_timetable_image()` (L226-300) は画像種別ごとに6つの分岐があるが、処理の大部分（ファイル保存・JSON更新・session_state更新）は共通。リファクタリング時に以下のように整理する:

```python
# project_repository.py
def register_timetable_image(self, pj_path: str, event_name: str,
                                img_type: str, img_format: str,
                                file_data: bytes,
                                project_info_json: dict) -> dict:
    """画像ファイルを保存しproject_info_jsonを更新して返す"""
```

app.py側で種別の判定と分岐だけを行い、実際の保存処理は`ProjectRepository`に委譲する。

### 7.5 フレームワーク移行時の変更範囲

リファクタリング完了後、Streamlitから別フレームワークに移行する際の変更範囲:

| モジュール | 変更 | 内容 |
|-----------|------|------|
| `app.py` | **全面差し替え** | 新フレームワークのUI層として書き直す |
| `app_state.py` | 変更なし | フレームワーク非依存 |
| `workflow.py` | 変更なし | フレームワーク非依存 |
| `backend_functions/*.py` | 変更なし | フレームワーク非依存 |
| `frontend_functions/*.py` | 変更なし | 画像生成ロジックのみ |

つまり、**app.pyのみを差し替えれば移行が完了する**状態になる。

### 7.6 移行先フレームワークへの対応例

**FastAPI + React の場合**:

```python
# api.py (app.pyの代わり)
from fastapi import FastAPI, UploadFile
from workflow import ProjectWorkflow, ImageWorkflow, OcrWorkflow

app = FastAPI()
# AppStateはサーバーサイドセッション or データベースで管理

@app.post("/api/projects")
async def create_project(pj_name: str):
    result = project_wf.create_project(pj_name, session_state, existing_names)
    if not result.success:
        raise HTTPException(400, detail=result.error)
    return {"status": "ok"}

@app.post("/api/ocr/{stage_no}")
async def run_ocr(stage_no: int, mode: str, user_prompt: str):
    result = ocr_wf.run_ocr_single(session_state, mode, stage_no, user_prompt)
    return {"status": "ok", "warnings": result.warnings}
```

ワークフロー層のインターフェースがそのままAPIエンドポイントにマッピングできる。

---

## 8. リファクタリング効果の見積もり

| 指標 | Before | After |
|------|--------|-------|
| app.py行数 | 2,178行 | ~800行 |
| app.pyのビジネスロジック行数 | ~1,146行 | ~0行 |
| st.session_state参照箇所 | ~120箇所 | ~40箇所（UIウィジェットキーとAppState格納のみ） |
| テスト可能な関数の割合 | ~20% | ~95% |
| 新規モジュール数 | 0 | 7 (backend×5 + app_state + workflow) |
| フレームワーク移行時の変更ファイル数 | 全ファイル | 1ファイル (app.pyのみ) |
| ビジネスロジック内のst.warning/error | 7箇所 | 0箇所 |

---

## 9. フェーズごとの完了基準

各フェーズの完了時に以下を確認する:

1. `streamlit run src/app.py` でアプリが正常に起動すること
2. プロジェクトの作成・読込・設定変更が動作すること
3. 画像のアップロード・切り取り・分割が動作すること
4. OCR読み取り・グループ名補正が動作すること
5. 出力（表示・Excel・S3）が動作すること
6. 新モジュール（`app_state.py`, `workflow.py` を除く）に`import streamlit`が含まれていないこと — ✅ 達成済み
7. **`app_state.py` と `workflow.py` に `import streamlit` が含まれていないこと** — ✅ 達成済み
8. **ビジネスロジック関数が `st.warning()` / `st.error()` / `st.success()` を直接呼んでいないこと** — ✅ 達成済み
9. **`st.stop()` がapp.pyから除去されていること（フェーズ6完了時）** — ✅ Step 6Fで達成済み
