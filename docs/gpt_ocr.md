# GPT OCR 詳細

本ドキュメントでは、GPT OCRによるタイムテーブル読み取り処理の詳細を説明します。

## 目次
- [使用モデル](#使用モデル)
- [処理フロー](#処理フロー)
- [プロンプトファイル一覧](#プロンプトファイル一覧)
- [チケットサイトスクレイパー](#チケットサイトスクレイパー)
- [グループ名補正](#グループ名補正)
- [API関数一覧](#api関数一覧)

---

## 使用モデル

### メインOCRモデル
- **モデル名**: `gpt-5`
- **設定場所**: `src/backend_functions/gpt_ocr.py` の `GPT_MODEL_NAME`
- **用途**: タイムテーブル画像からの情報抽出

### チケット出演者抽出モデル
- **モデル名**: `gpt-4o-mini`
- **設定場所**: `src/backend_functions/ticket_scraper.py` の `LLM_MODEL_NAME`
- **用途**: チケットサイトからの出演者リスト抽出（軽量・低コスト）

---

## 処理フロー

### 基本的な読み取りフロー

```
1. 画像のBase64エンコード
      ↓
2. プロンプトファイルの読み込み
      ↓
3. チケットURLから出演者情報取得（オプション）
      ↓
4. OpenAI API呼び出し（Vision API）
      ↓
5. レスポンスのJSON解析
      ↓
6. 結果の保存（stage_X.json）
```

### APIメソッドの種類

| メソッド | 説明 | 現在の使用状況 |
|---------|------|---------------|
| `getocr()` | 基本的なJSON出力 | 一部で使用 |
| `getocr_functioncalling()` | Function Calling形式 | 非推奨（コメントアウト） |
| `getocr_structured()` | Structured Output形式 | **推奨（現在のメイン）** |

---

## プロンプトファイル一覧

プロンプトファイルは `src/prompt_system/` ディレクトリに配置されています。

### fes_stagelist.txt
**用途**: ステージ名一覧の抽出

**入力**: タイムテーブル全体画像 + ステージ数

**出力形式**:
```json
{
    "ステージ名": ["STAGE-A", "STAGE-B", "STAGE-C"],
    "命名規則": "特になし"
}
```

**命名規則の値**:
- `特になし`: 自然言語のステージ名
- `数字`: 1, 2, 3, ...
- `アルファベット`: A, B, C, ...

**プロンプト要点**:
- 左から順にステージを出力
- 大半が数字/アルファベットの場合は例外があってもそのルールを採用

---

### fes_timetable_singlestage.txt
**用途**: 通常形式のタイムテーブル読み取り

**特徴**:
- 各グループ枠内に時間が記載されている形式
- ライブステージのみ（特典会情報なし）

**出力形式**:
```json
{
    "タイムテーブル": [
        {
            "グループ名": "アイドルグループA",
            "ライブステージ": {
                "from": "10:00",
                "to": "10:20"
            }
        }
    ]
}
```

**プロンプト要点**:
- 時間は5分刻みで出力
- グループ名の近くに記載された時間を採用
- 「終演後物販」「終演後特典会」は無視
- 同一グループの複数回出演は別エントリとして出力

---

### fes_timetable_singlestage_notime_live.txt
**用途**: ライムライト式（時間軸から時間を読み取る形式）のライブタイムテーブル

**特徴**:
- 枠外の時間軸から時間を推測する形式
- 画像に時刻情報が追記された状態で入力（`stage_X_addtime.png`）

**プロンプト要点**:
- グループ名の右側の「開始時刻-終了時刻」形式を採用
- グループ名と対応しない時刻は無視
- 時刻の左側に何も書かれていない場合は出力から除外

---

### fes_timetable_singlestage_notime_tokutenkai.txt
**用途**: ライムライト式の特典会タイムテーブル

**特徴**:
- 特典会は60分前後の長さが多い
- 「終演後特典会」がある場合は適宜時間を推測

**プロンプト要点**:
- 開始-終了の判断が難しい場合は60分前後を基準に判断
- 終演後の時間は他の演目から推測可能

---

### fes_timetable_singlestage_liveandtokutenkai.txt
**用途**: 特典会併記形式（ライブ+特典会が同一画像に記載）

**出力形式**:
```json
{
    "タイムテーブル": [
        {
            "グループ名": "アイドルグループA",
            "ライブステージ": {
                "from": "10:00",
                "to": "10:20"
            },
            "特典会": [
                {
                    "from": "10:30",
                    "to": "11:30",
                    "ブース": "A"
                }
            ]
        }
    ]
}
```

**プロンプト要点**:
- ライブステージは20-40分程度が多い
- 特典会は60分前後が多い
- 「特典会無し」の場合は特典会情報を空配列で出力
- ブースはアルファベット1文字の場合もあれば、場所名の場合もある
- 終演後特典会の時間が不明な場合は最後のライブ終了から60分で計算

---

### taiban.txt
**用途**: 対バン形式（単独ステージ）のタイムテーブル

---

### fes_info.txt
**用途**: フェス全体の情報抽出（イベント名、日程など）

---

## チケットサイトスクレイパー

### 概要

チケットサイトから出演者リストを取得し、OCRの精度向上に活用します。

**ファイル**: `src/backend_functions/ticket_scraper.py`

### 対応サイト

| サイト | URL例 |
|--------|-------|
| TicketDive | `https://ticketdive.com/event/xxx` |
| LivePocket | `https://livepocket.jp/e/xxx` |
| tiget | `https://tiget.net/events/xxx` |

### 処理フロー

```
1. URLからページ取得（requests）
      ↓
2. HTMLパース（BeautifulSoup）
      ↓
3. 不要要素の除去（script, style, nav, footer, header）
      ↓
4. テキスト抽出
      ↓
5. LLMで出演者リスト抽出（gpt-4o-mini）
```

### 主要関数

#### `get_performers_from_ticket_url(url: str) -> str | None`
メインの公開関数。結果は `@lru_cache(maxsize=128)` でキャッシュされます。

#### `_fetch_page_text(url: str) -> str | None`
- User-Agent偽装でページを取得
- エンコーディングを自動検出
- テキストは15,000文字で切り詰め

#### `_extract_performers_with_llm(page_text: str, domain: str) -> str | None`
- gpt-4o-miniで出演者を箇条書きで抽出
- 重複除去、修飾語の削除を指示

### OCRへの統合

出演者情報はプロンプトに追加されます：

```python
def _get_performers_text(ticket_urls) -> str:
    # ...
    return f"""
【出演者リスト（参考情報）】
以下はチケットサイトから得た出演者一覧情報です。
必ずしも正しいとは限らないので、この中に存在しない名前を出力しても構いませんし、
この中に存在する名前を全て出力しなくても構いません。
しかし一定の参考情報としてください。

{combined}
"""
```

---

## グループ名補正

### 概要

OCRで読み取ったグループ名をマスタデータと照合して補正します。

**ファイル**: `src/backend_functions/idolname.py`

### 補正アルゴリズム

#### `get_name_by_levenshtein_and_vector(name)`

```
1. 入力名と全マスタ名の編集距離を計算
      ↓
2. 編集距離率（距離 / 文字数）が閾値r以下の候補をリストアップ
      ↓
3-a. 候補が1つ → そのまま採用
3-b. 候補が複数 → FAISSベクトル検索で最近傍を採用
3-c. 候補なし → 全マスタからベクトル検索
```

### ベクトル検索

- **埋め込みモデル**: OpenAI Embeddings API
- **次元数**: 100次元（`text-embedding-3-small` + dimensions=100）
- **インデックス**: FAISS (Facebook AI Similarity Search)
- **マスタファイル**: `data/master/idolname_embedding_data.csv`

### 候補制限オプション

`correct_idolname_in_confirmed_list` オプションを有効にすると、
既に確定したグループ名の中からのみ候補を検索します。

```python
# 例: ライブのタイテで確定したグループ名を、
#     特典会のタイテでも候補として使用
idolname_confirmed_list = get_idolname_confirmed_list()
item['グループ名_採用'] = idolname.get_name_by_inlist(
    item["グループ名"],
    idolname_confirmed_list
)
```

---

## API関数一覧

### Structured Output形式（推奨）

現在メインで使用されているAPI呼び出し形式です。

| 関数 | 用途 | プロンプトファイル |
|------|------|-------------------|
| `getocr_fes_stagelist_structured()` | ステージ名一覧 | `fes_stagelist.txt` |
| `getocr_fes_timetable_structured()` | 通常形式タイテ | `fes_timetable_singlestage.txt` |
| `getocr_fes_timetable_notime_structured()` | ライムライト式 | `fes_timetable_singlestage_notime_*.txt` |
| `getocr_fes_withtokutenkai_timetable_structured()` | 特典会併記 | `fes_timetable_singlestage_liveandtokutenkai.txt` |

### 基本形式（レガシー）

```python
def getocr(image_path, prompt_user, prompt_system, ticket_urls=None):
    """
    基本的なJSON出力形式でのAPI呼び出し
    response_format={"type": "json_object"}を使用
    """
```

### Function Calling形式（非推奨）

```python
def getocr_functioncalling(image_path, prompt_user, prompt_system, tools, ticket_urls=None):
    """
    Function Callingを使用したAPI呼び出し
    現在はStructured Outputに移行
    """
```

### レスポンススキーマ

`response_format_live`:
```json
{
    "type": "json_schema",
    "json_schema": {
        "name": "live",
        "strict": true,
        "schema": {
            "type": "object",
            "properties": {
                "タイムテーブル": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "グループ名": {"type": "string"},
                            "ライブステージ": {
                                "type": "object",
                                "properties": {
                                    "from": {"type": "string"},
                                    "to": {"type": "string"}
                                },
                                "required": ["from", "to"]
                            }
                        },
                        "required": ["グループ名", "ライブステージ"]
                    }
                }
            },
            "required": ["タイムテーブル"]
        }
    }
}
```

---

## エラーハンドリング

### ステージ名読み取りのリトライ

`getocr_fes_stagelist_structured()` は最大5回リトライします：

```python
for i in range(5):
    try:
        response = getocr_structured(...)
        stage_list = json.loads(response.choices[0].message.content)["ステージ名"]
        if type(stage_list)==list and len(stage_list)==stage_num:
            return stage_list, rule
        else:
            time.sleep(1)  # 失敗時は1秒待機
    except:
        time.sleep(1)
else:
    raise TypeError  # 5回失敗で例外
```

### 出力補正

タイムテーブルJSONに「タイムテーブル」キーがない場合は空配列で初期化：

```python
if "タイムテーブル" not in return_json.keys():
    return_json["タイムテーブル"] = []
```
