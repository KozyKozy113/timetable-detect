# ワークフロー詳細

`src/app.py` における各ステップの処理内容を詳細に説明します。

## 目次
- [①プロジェクトの設定](#プロジェクトの設定)
- [②タイムテーブル画像の登録](#タイムテーブル画像の登録)
- [③タイムテーブル画像の切り取り](#タイムテーブル画像の切り取り)
- [④タイムテーブル画像の読み取り](#タイムテーブル画像の読み取り)
- [⑤タイムテーブル画像の追加・変更](#タイムテーブル画像の追加変更)
- [⑥タイムテーブル情報の出力](#タイムテーブル情報の出力)
- [⑦マスタのアップデート](#マスタのアップデート)

---

## ①プロジェクトの設定

### 概要
イベント単位でプロジェクトを作成・管理します。一つのプロジェクトは一つのイベント（フェス/対バン）に紐付きます。

### 主要関数

#### `make_project(pj_name=None)`
新規プロジェクトを作成します。

**処理内容:**
1. プロジェクト名の重複チェック
2. `data/projects/{pj_name}/` ディレクトリの作成
3. `projects_master.csv` への登録（作成日時、更新日時）
4. `project_info.json` の初期化
5. `set_project()` の呼び出し

#### `set_project(pj_name)`
既存プロジェクトを読み込みます。

**処理内容:**
1. S3からマスタデータの取得（`s3access.get_master()`）
2. S3からプロジェクトデータの取得（`s3access.get_project_data()`）
3. プロジェクト設定の読み込み（イベント形式、イベント数）
4. イベントフォルダの作成
5. `project_info.json` の読み込み
6. 画像関連の状態初期化

#### `determine_project_setting()`
プロジェクト設定を確定・保存します。

**処理内容:**
1. イベント形式（フェス/対バン）の保存
2. イベント数の保存
3. 各イベントフォルダの作成
4. `project_info.json` の更新

### チケットURL設定

#### `save_ticket_urls()`
チケットサイトURLを保存します。

**スコープオプション:**
- `プロジェクト共通`: 全イベントで同じURLを使用
- `イベントごと`: イベント別にURLを設定

**対応サイト:**
- TicketDive
- LivePocket
- tiget

---

## ②タイムテーブル画像の登録

### 概要
タイムテーブル画像をアップロードし、種別と形式を指定して登録します。

### 主要関数

#### `determine_timetable_image()`
アップロードされた画像を登録します。

**処理内容:**
1. 種別に応じたディレクトリ作成（`ライブ/`, `特典会/`, `ライブ特典会/` など）
2. 画像ファイルの保存（`raw.png`）
3. `project_info.json` のtimetablesセクション更新
4. 次ステップの対象設定

**種別オプション:**
| 種別 | 説明 | 作成フォルダ |
|------|------|-------------|
| ライブ | ライブのみのタイテ | `ライブ/` |
| 特典会 | 特典会のみのタイテ | `特典会/` |
| 両方(特典会別添え) | 同じ画像をライブ/特典会両方に登録 | `ライブ/`, `特典会/` |
| 両方(特典会併記) | ライブ+特典会が1枚に記載 | `ライブ特典会/` |
| その他 | 任意の種別名 | `{入力した種別名}/` |

**形式オプション:**
| 形式 | 説明 |
|------|------|
| 通常 | 各グループ枠内に時間が記載 |
| ライムライト式 | 枠外の時間軸から時間を読み取る |

#### `delete_uploaded_image(img_event_no, img_type)`
登録済み画像を削除します。

---

## ③タイムテーブル画像の切り取り

### 概要
アップロードした画像を加工し、読み取りに適した状態に準備します。

### (ⅰ) タイムテーブル領域の切り出し

`streamlit-cropper` を使用して、画像から必要な領域を切り出します。

**ポイント:**
- ステージ名の部分を含めることを推奨（自動読み取りに使用）
- 時間軸の部分を含めることを推奨（時間軸設定に使用）
- ライムライト式では時間軸が必須

**出力ファイル:**
- `{種別}/raw_cropped.png`

### (ⅱ) ステージごとの分割

#### 縦線検出による分割

**関数:** `detect_stageline(image)`

**処理フロー:**
1. グレースケール変換
2. Cannyエッジ検出
3. HoughLinesPによる直線検出（傾き85度以上のみ）
4. 近接線の統合
5. 検出された線で画像を分割

**パラメータ:**
| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `x_edge_threshold_1` | 130 | Cannyエッジ検出の低閾値 |
| `x_edge_threshold_2` | 285 | Cannyエッジ検出の高閾値 |
| `x_hough_threshold` | 100 | ハフ変換の投票閾値 |
| `x_hough_gap` | 1 | 線分の許容ギャップ（px） |
| `x_minlength_rate` | 0.01 | 最小線分長（画像高さ比） |
| `x_identify_interval` | 5 | 同一線とみなす横位置誤差（px） |

#### 均等幅での分割

**関数:** `get_image_eachstage_for_croppedimage_byevenly()`

**処理内容:**
- 画像幅をステージ数で均等分割
- 前後5%の重複領域を持たせる

**出力ファイル:**
- `{種別}/stage_0.png`, `stage_1.png`, ...

#### `determine_image_eachstage_without_nocheck()`
チェックボックスで採用した画像のみを保存します。

### (ⅲ) 基準時間の設定

**関数:** `save_time_pixel(time_start, top, height, total_duration)`

**保存内容:**
```json
{
  "time_pixel": {
    "time_start": "10:00",
    "start_pix": 50,
    "total_pix": 800,
    "total_duration": 600
  }
}
```

**変換関数:**
- `pix_to_time(pix)`: ピクセル位置 → 時刻
- `time_to_pix(tgt_time)`: 時刻 → ピクセル位置
- `time_length_to_pix(minutes)`: 分 → ピクセル幅

---

## ④タイムテーブル画像の読み取り

### 概要
GPT OCRを使用してタイムテーブル情報を抽出します。

### まとめて読み取り

**関数:** `get_timetabledata_together()`

**処理内容:**
1. 対象画像のチェックボックス確認
2. チケットURLからの出演者情報取得
3. ステージ名読み取り（オプション）
4. タイムテーブル読み取り
5. グループ名補正（オプション）

### ステージ名読み取り

**関数:** `get_stagelist(user_prompt)`

**処理内容:**
1. `gpt_ocr.getocr_fes_stagelist_structured()` でステージ名を抽出
2. 命名規則（数字/アルファベット/特になし）を判定
3. 接頭辞の付与（ライブA, 特典会1 など）

### タイムテーブル読み取り

**関数:** `get_timetabledata_allstages(mode, user_prompt, ticket_urls=None)`

**モード:**
| モード | 対応形式 | 使用プロンプト |
|--------|---------|---------------|
| `normal` | 通常形式 | `fes_timetable_singlestage.txt` |
| `notime` | ライムライト式 | `fes_timetable_singlestage_notime_live.txt` |
| `tokutenkai` | 特典会併記 | `fes_timetable_singlestage_liveandtokutenkai.txt` |

**並列処理:**
- 最大10スレッドで並列実行（`ThreadPoolExecutor`）

**出力ファイル:**
- `{種別}/stage_0.json`, `stage_1.json`, ...
- `{種別}/stage_0_timetable.png`, ... （読み取り結果の可視化画像）

### 横線検出（ライムライト式）

**関数:** `detect_timeline_onlyonestage(stage_no)`

**処理フロー:**
1. Cannyエッジ検出
2. HoughLinesPによる横線検出（傾き5度以下）
3. 近接線の統合
4. ピクセル位置 → 時刻変換
5. 時刻情報を画像に追記（`stage_X_addtime.png`）

**パラメータ:**
| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `y_edge_threshold_1` | 80 | Cannyエッジ検出の低閾値 |
| `y_edge_threshold_2` | 150 | Cannyエッジ検出の高閾値 |
| `y_hough_threshold` | 60 | ハフ変換の投票閾値 |
| `y_hough_gap` | 1 | 線分の許容ギャップ（px） |
| `y_minlength_rate` | 0.05 | 最小線分長（画像幅比） |
| `y_identify_interval` | 5 | 同一線とみなす縦位置誤差（px） |
| `y_ignoretime_threshold` | 5 | 無視する時間幅（分以下） |

### グループ名補正

**関数:** `idolname_correct_eachstage()`

**処理内容:**
1. 各グループ名に対して補正を実行
2. `idolname.get_name_by_levenshtein_and_vector()` で候補を検索
3. `グループ名_採用` フィールドに保存

**補正モード:**
- 全マスタから検索
- 確定済みグループ名一覧から検索（`correct_idolname_in_confirmed_list` オプション）

### 読み取り結果の編集

**関数:** `save_timetable_data_onlyonestage(stage_no)`

**処理内容:**
1. DataFrameからJSONに変換
2. ステージ名の更新
3. JSONファイルの保存
4. 可視化画像の再生成

---

## ⑤タイムテーブル画像の追加・変更

### 概要
タイムテーブルが更新された場合に、変更点を確認する機能です。

### 主要関数

#### `output_difference_image(new_image)`
新旧画像の差分を可視化します。

**処理内容:**
1. 新画像と既存画像（`raw.png`）を読み込み
2. `timetable_difference.output_difference()` で差分画像を生成
3. 差分ヒートマップを表示

---

## ⑥タイムテーブル情報の出力

### 概要
読み取り結果を構造化データとして出力します。

### 出力データ構造

#### ステージマスタ (`df_stage`)
| カラム | 説明 |
|--------|------|
| ステージID | 一意のID |
| ステージ名 | ステージ/ブースの名前 |
| 特典会フラグ | 特典会かどうか |

#### グループマスタ (`df_idolname`)
| カラム | 説明 |
|--------|------|
| グループID | 一意のID |
| グループ名_採用 | 補正後のグループ名 |

#### 出番データ (`df_live`)
| カラム | 説明 |
|--------|------|
| 出番ID | 一意のID |
| ライブ_from | 開始時刻 |
| ライブ_長さ(分) | 演目時間 |
| グループID | グループマスタへの参照 |
| ステージID | ステージマスタへの参照 |
| グループ名_raw | OCR読み取り結果 |
| グループ名 | 補正後の名前 |
| ステージ名 | ステージ/ブース名 |
| 備考 | 備考欄 |

### 主要関数

#### `determine_id_master()`
ID体系を確定し、JSONファイルに反映します。

**処理内容:**
1. ステージマスタの保存（`master_stage.csv`）
2. グループマスタの保存（`master_idolname.csv`）
3. 出番データの保存（`turn_id_data.csv`）
4. 各ステージのJSONにID情報を追記

#### `save_to_s3()`
プロジェクトデータをS3にアップロードします。

**処理内容:**
- `s3access.put_project_data()` を呼び出し
- プロジェクトフォルダ全体をアップロード
- `projects_master_s3.csv` も更新

#### `output_data_for_stella()`
Excel形式でデータを出力します。

**出力ファイル:**
- `{プロジェクト}/output.xlsx`

**シート構成:**
- 各イベント名をシート名として出力
- ステージマスタ、グループマスタ、出番データを配置

---

## ⑦マスタのアップデート

### 概要
新規登場したグループ名をマスタに追加します。

### 主要関数

#### `listup_new_idolname()`
新規グループ名をリストアップします。

**処理内容:**
1. 全イベントの `グループ名_採用` を収集
2. `idolname.detect_new_data()` で未登録のグループを検出
3. DataFrameとして表示

#### `update_master_idolname(df_new_idolname)`
選択したグループ名をマスタに追加します。

**処理内容:**
1. `idolname.add_new_data_file()` でローカルマスタを更新
   - グループ名のembedding生成
   - `idolname_embedding_data.csv` の更新
   - `idolname_latest.csv` の更新
2. S3にマスタファイルをアップロード
3. バージョン情報（`master_version_s3.json`）の更新
