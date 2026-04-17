# アーキテクチャ

本ドキュメントでは、プロジェクトのモジュール構成とデータフローについて説明します。

## 目次
- [システム概要](#システム概要)
- [モジュール構成](#モジュール構成)
- [データフロー](#データフロー)
- [依存関係](#依存関係)
- [S3同期の仕組み](#s3同期の仕組み)

---

## システム概要

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit Web UI                         │
│                         (src/app.py)                            │
└─────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  Backend      │     │  Frontend     │     │  GPT Output   │
│  Functions    │     │  Functions    │     │  Format       │
└───────────────┘     └───────────────┘     └───────────────┘
        │                       │                       │
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ - gpt_ocr     │     │ - timetable   │     │ - timetable   │
│ - idolname    │     │   picture     │     │   _format.py  │
│ - timetable   │     │ - timetable   │     │  (Pydantic)   │
│   data        │     │   _difference │     │               │
│ - s3access    │     │ - Fonts/      │     └───────────────┘
│ - ticket_     │     └───────────────┘
│   scraper     │
└───────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                      External Services                         │
├───────────────────────────────────────────────────────────────┤
│  OpenAI API          │  AWS S3              │ Ticket Sites    │
│  (GPT-5, gpt-4o-mini)│  (idol-timetable)    │ (TicketDive等)  │
└───────────────────────────────────────────────────────────────┘
```

---

## モジュール構成

### メインアプリケーション

#### `src/app.py`
- **役割**: Streamlit Webアプリケーションのエントリーポイント
- **行数**: 約2,150行
- **主要機能**:
  - UI構築とユーザーインタラクション
  - セッション状態の管理
  - 各機能モジュールの統合

### バックエンドモジュール

#### `src/backend_functions/gpt_ocr.py`
- **役割**: GPT Vision APIを使用したOCR処理
- **主要機能**:
  - 画像のBase64エンコード
  - プロンプト管理
  - API呼び出し（Structured Output形式）
  - レスポンス解析

#### `src/backend_functions/idolname.py`
- **役割**: グループ名のマッチングと補正
- **主要機能**:
  - OpenAI Embeddings APIによるベクトル生成
  - FAISSによるベクトル検索
  - Levenshtein距離による文字列類似度計算
  - マスタデータの管理

#### `src/backend_functions/timetabledata.py`
- **役割**: タイムテーブルデータの加工
- **主要機能**:
  - JSON ↔ DataFrame変換
  - 時間計算（duration、add_minutes_to_time）
  - ライブ/特典会データの分離
  - ID付与処理

#### `src/backend_functions/s3access.py`
- **役割**: AWS S3との同期
- **主要機能**:
  - プロジェクトデータのアップロード/ダウンロード
  - マスタデータの同期
  - バージョン管理

#### `src/backend_functions/ticket_scraper.py`
- **役割**: チケットサイトからの出演者情報取得
- **主要機能**:
  - Webスクレイピング（requests + BeautifulSoup）
  - LLMによる出演者抽出
  - 結果のキャッシュ

### フロントエンドモジュール

#### `src/frontend_functions/timetablepicture.py`
- **役割**: タイムテーブル画像の生成
- **主要機能**:
  - JSONデータからの可視化画像生成
  - 時間軸の描画
  - グループ枠の配置

#### `src/frontend_functions/timetable_difference.py`
- **役割**: タイムテーブル画像の差分検出
- **主要機能**:
  - 画像のリサイズ
  - ピクセル単位の差分計算
  - 差分ヒートマップ生成

### データスキーマ

#### `src/gpt_output_format/timetable_format.py`
- **役割**: GPT出力のPydanticモデル定義
- **定義クラス**:
  - `LiveStage`: ライブの時間範囲
  - `Tokutenkai`: 特典会情報
  - `ArtistLive`: アーティスト+ライブ情報
  - `ArtistLiveTokutenkai`: アーティスト+ライブ+特典会
  - `TimetableLive`: ステージ名+ライブ一覧
  - `TimetableLiveTokutenkai`: ステージ名+ライブ+特典会一覧

### プロンプト

#### `src/prompt_system/`
- **役割**: GPTへの指示プロンプトを格納
- **ファイル一覧**:
  - `fes_stagelist.txt`: ステージ名抽出用
  - `fes_timetable_singlestage.txt`: 通常形式タイテ用
  - `fes_timetable_singlestage_notime_live.txt`: ライムライト式（ライブ）用
  - `fes_timetable_singlestage_notime_tokutenkai.txt`: ライムライト式（特典会）用
  - `fes_timetable_singlestage_liveandtokutenkai.txt`: 特典会併記用
  - `taiban.txt`: 対バン形式用
  - `fes_info.txt`: フェス情報抽出用
  - `fes_timetable_live.txt`: 追加のライブ用プロンプト

---

## データフロー

### 読み取り処理の全体フロー

```
┌──────────────┐
│  画像入力    │
│  (raw.png)   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  領域切り出し │
│(raw_cropped) │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ ステージ分割 │
│(stage_X.png) │
└──────┬───────┘
       │
       ├─────────────────┐
       │                 │
       ▼                 ▼
┌──────────────┐   ┌──────────────┐
│ 通常形式     │   │ライムライト式 │
│              │   │              │
│  GPT OCR     │   │ 横線検出     │
│              │   │     ↓        │
│              │   │ 時刻追記     │
│              │   │ (_addtime)   │
│              │   │     ↓        │
│              │   │  GPT OCR     │
└──────┬───────┘   └──────┬───────┘
       │                 │
       └────────┬────────┘
                │
                ▼
       ┌──────────────┐
       │  JSONデータ   │
       │ (stage_X.json)│
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │ グループ名   │
       │ 補正処理     │
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │ 可視化画像   │
       │ (_timetable) │
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │  出力        │
       │ (Excel/S3)   │
       └──────────────┘
```

### データ変換フロー

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   画像      │ ──▶ │    JSON     │ ──▶ │  DataFrame  │
│ (stage_X.png)│     │(stage_X.json)│     │             │
└─────────────┘     └─────────────┘     └─────────────┘
                            │                   │
                            │                   │
                            ▼                   ▼
                    ┌─────────────┐     ┌─────────────┐
                    │  可視化     │     │   Excel     │
                    │  画像生成   │     │   出力      │
                    └─────────────┘     └─────────────┘
```

---

## 依存関係

### 外部ライブラリ

| カテゴリ | ライブラリ | 用途 |
|---------|-----------|------|
| UI | streamlit | Webアプリケーション |
| UI | streamlit-cropper | 画像切り出しUI |
| AI | openai | GPT API呼び出し |
| 画像処理 | Pillow (PIL) | 画像操作 |
| 画像処理 | opencv-python (cv2) | エッジ検出、線分検出 |
| データ処理 | pandas | DataFrameの操作 |
| データ処理 | numpy | 数値計算 |
| ベクトル検索 | faiss-cpu | 類似度検索 |
| 文字列処理 | Levenshtein | 編集距離計算 |
| スクレイピング | beautifulsoup4 | HTML解析 |
| スクレイピング | requests | HTTP通信 |
| クラウド | boto3 | AWS S3操作 |
| バリデーション | pydantic | データモデル |
| Excel | openpyxl | Excel出力 |

### モジュール間の依存

```
app.py
├── backend_functions/
│   ├── gpt_ocr.py
│   │   └── ticket_scraper.py
│   │   └── gpt_output_format/timetable_format.py
│   ├── timetabledata.py
│   ├── idolname.py
│   └── s3access.py
│       └── idolname.py
└── frontend_functions/
    ├── timetablepicture.py
    └── timetable_difference.py
```

---

## S3同期の仕組み

### バケット構成

```
s3://idol-timetable/
├── master/
│   ├── master_version_s3.json       # マスタバージョン管理
│   ├── projects_master_s3.csv       # プロジェクト一覧
│   ├── idolname_embedding_data.csv  # グループ名+埋め込み
│   └── idolname_latest.csv          # グループ名一覧
└── projects/
    └── {project_name}/
        └── (プロジェクトデータ一式)
```

### 同期フロー

#### 起動時のマスタ取得

```
1. S3から master_version_s3.json をダウンロード
       ↓
2. ローカルの master_version.json と比較
       ↓
3. 差分があるマスタファイルをダウンロード
       ↓
4. ローカルの master_version.json を更新
```

#### プロジェクト読み込み時

```
1. projects_master.csv と projects_master_s3.csv を比較
       ↓
2. S3側が新しい場合、プロジェクトデータをダウンロード
       ↓
3. ローカルの projects_master.csv を更新
```

#### プロジェクト保存時

```
1. ローカルのプロジェクトフォルダを S3 にアップロード
       ↓
2. projects_master_s3.csv を更新
       ↓
3. projects_master_s3.csv を S3 にアップロード
```

### バージョン管理

`master_version_s3.json` の形式:
```json
{
    "idolname_embedding_data.csv": "2024/01/15 12:30:45.123456",
    "idolname_latest.csv": "2024/01/15 12:30:45.123456"
}
```

- 日時文字列による単純比較でバージョン判定
- S3側のタイムスタンプが新しい場合にダウンロード

---

## セッション状態

Streamlitの `st.session_state` で管理される主要な状態:

| 状態名 | 型 | 説明 |
|--------|-----|------|
| `pj_name` | str | 現在のプロジェクト名 |
| `pj_path` | str | プロジェクトのパス |
| `project_info_json` | dict | プロジェクト設定 |
| `cropped_image` | Image | 切り出し後の画像 |
| `images_eachstage` | list[Image] | ステージごとの画像 |
| `timeline_eachstage` | list[DataFrame] | 検出した横線の情報 |
| `df_timetables` | list[DataFrame] | 読み取り結果のDataFrame |
| `output_df` | dict | 出力用のDataFrame群 |

---

## 並列処理

タイムテーブル読み取りは `concurrent.futures.ThreadPoolExecutor` で並列化:

```python
def get_timetabledata_allstages(mode, user_prompt, ticket_urls=None):
    max_workers = 10
    stage_nums = list(range(st.session_state.ocr_tgt_stage_num))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                get_timetabledata_onestage_worker,
                mode, i, user_prompt, ...
            )
            for i in stage_nums
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
```

- 最大10スレッドで並列実行
- `st.session_state` 非依存のワーカー関数を使用
- APIレート制限に注意
