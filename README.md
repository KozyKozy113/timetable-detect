# タイムテーブル読み取りアプリ

アイドルフェス・対バンライブのタイムテーブル画像を構造化データ（JSON/Excel）に変換するStreamlitアプリケーションです。

## 概要

- タイムテーブル画像をアップロードし、GPT（Vision API）を活用してOCR・構造化を行います
- 複数ステージの同時進行タイムテーブル（フェス形式）に対応
- グループ名のマスタデータによる自動補正機能
- AWS S3によるプロジェクトデータの同期
- Stella（ライブ管理アプリ）向けExcel出力

## 主な機能

| 機能 | 説明 |
|------|------|
| プロジェクト管理 | イベント単位でプロジェクトを作成・管理 |
| 画像登録 | ライブ/特典会/両方など種別を指定して画像登録 |
| 画像加工 | 領域切り出し、ステージ分割、時間軸設定 |
| タイテ読み取り | GPT OCRによるタイムテーブル情報抽出 |
| グループ名補正 | ベクトル検索+編集距離によるマスタ参照補正 |
| 差分検出 | 更新されたタイテ画像の差分可視化 |
| データ出力 | Excel/JSON形式での出力、S3同期 |

## 動作環境

- Python 3.10以上
- OpenAI API Key（GPT-4V / GPT-5 対応）
- AWS認証情報（S3同期を使用する場合）

## インストール

```bash
# リポジトリのクローン
git clone <repository-url>
cd timetable-detect

# 依存パッケージのインストール
pip install -r requirements.txt

# 環境変数の設定
cp .env.example .env
# .env ファイルを編集し、OPENAI_API_KEY を設定
```

## 起動方法

```bash
cd timetable-detect
streamlit run src/app.py
```

ブラウザで `http://localhost:8501` が自動的に開きます。

## 基本的な使い方

アプリは7つのステップで構成されています：

### ①プロジェクトの設定
- 新規プロジェクトの作成または既存プロジェクトの呼び出し
- イベント形式（フェス/対バン）とイベント数の設定
- チケットサイトURLの登録（出演者情報取得用）

### ②タイムテーブル画像の登録
- 画像のアップロード
- 種別（ライブ/特典会/両方）と形式（通常/ライムライト式）の選択

### ③タイムテーブル画像の切り取り
- (ⅰ) タイムテーブル領域の切り出し
- (ⅱ) ステージごとの画像分割（縦線検出 or 均等分割）
- (ⅲ) 時間軸の基準位置設定

### ④タイムテーブル画像の読み取り
- ステージ名の自動読み取り
- タイムテーブル情報（グループ名・出演時間）の抽出
- グループ名のマスタ参照補正
- 読み取り結果の手動編集

### ⑤タイムテーブル画像の追加・変更
- 更新されたタイテ画像との差分確認

### ⑥タイムテーブル情報の出力
- ステージマスタ/グループマスタ/出番データの生成
- Excel形式でのダウンロード
- S3へのプロジェクトデータアップロード

### ⑦マスタのアップデート
- 新規登場グループ名のリストアップ
- グループ名マスタへの追加

## ディレクトリ構成

```
timetable-detect/
├── src/
│   ├── app.py                      # メインアプリケーション
│   ├── backend_functions/          # バックエンド機能
│   │   ├── gpt_ocr.py             # GPT OCR処理
│   │   ├── timetabledata.py       # タイムテーブルデータ処理
│   │   ├── idolname.py            # グループ名マッチング
│   │   ├── s3access.py            # AWS S3連携
│   │   └── ticket_scraper.py      # チケットサイトスクレイピング
│   ├── frontend_functions/         # フロントエンド機能
│   │   ├── timetablepicture.py    # タイムテーブル画像生成
│   │   ├── timetable_difference.py # 差分画像生成
│   │   └── Fonts/                 # フォントファイル
│   ├── gpt_output_format/          # GPT出力スキーマ
│   │   └── timetable_format.py    # Pydanticモデル定義
│   └── prompt_system/              # GPTプロンプトテンプレート
│       ├── fes_stagelist.txt
│       ├── fes_timetable_singlestage.txt
│       └── ...
├── data/
│   ├── master/                     # マスタデータ
│   │   ├── projects_master.csv
│   │   ├── idolname_embedding_data.csv
│   │   └── ...
│   └── projects/                   # プロジェクトデータ
│       └── {project_name}/
├── docs/                           # ドキュメント
│   ├── workflow.md                # ワークフロー詳細
│   ├── data_structure.md          # データ構造仕様
│   ├── gpt_ocr.md                 # GPT OCR詳細
│   └── architecture.md            # アーキテクチャ図
├── requirements.txt
└── README.md
```

## ドキュメント

詳細なドキュメントは `docs/` ディレクトリを参照してください：

- [ワークフロー詳細](docs/workflow.md) - 各ステップの処理内容
- [データ構造仕様](docs/data_structure.md) - プロジェクトデータ/パラメータ
- [GPT OCR詳細](docs/gpt_ocr.md) - プロンプト/読み取り処理
- [アーキテクチャ](docs/architecture.md) - モジュール構成/データフロー

## 対応タイムテーブル形式

| 形式 | 説明 |
|------|------|
| 通常形式 | 各グループ枠に時間が記載されている形式 |
| ライムライト式 | 枠外の時間軸から時間を読み取る形式 |
| 特典会併記 | ライブと特典会が同一画像に記載されている形式 |

## 技術スタック

- **フロントエンド**: Streamlit, streamlit-cropper, st-aggrid
- **AI/LLM**: OpenAI API (GPT-5, gpt-4o-mini)
- **画像処理**: PIL/Pillow, OpenCV
- **データ処理**: Pandas, NumPy
- **ベクトル検索**: FAISS
- **文字列マッチング**: Levenshtein
- **Webスクレイピング**: BeautifulSoup, requests
- **クラウドストレージ**: AWS S3 (boto3)
- **データバリデーション**: Pydantic

## ライセンス

Private

## 作者

kkoaz
