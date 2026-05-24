# 画像種別・ステージ種別の永続化リファクタリング計画

## 目的

現状「特典会フラグ」が **画像の保存先キー名 (`timetables` の文字列キー)** から導出されており、以下の問題がある:

1. `event_type == "特典会"` の完全一致でしか特典会判定できず、「その他」系（例: `"縁日"`）は常にライブ扱いになる ([src/app.py:1133](../../src/app.py#L1133))。
2. 「その他」「その他(特典会併記)」を選んだ際のフォルダ名は `img_type_alternative` で任意指定でき、判定ロジックと結合している。
3. 同名フォルダの衝突を許容しないため、同一イベント内に複数の「特典会」系画像を登録できない。

この計画では、**画像の種別 (kind)** を文字列キーではなく明示的な属性として、画像分割時点で生成する **画像メタ + ステージマスタ** に永続化し、特典会判定をこの kind 属性から導出するようリファクタする。

---

## スコープ

設計のみ。実装は別タスクで行う。

---

## 用語

- **画像 (image)**: ユーザーがアップロードする 1 枚の元タイムテーブル画像。 1 つの画像は複数のステージを含みうる。
- **画像種別 (image kind)**: 画像が示す情報のカテゴリ。`live` / `tokutenkai` / `live_tokutenkai_heiki` の 3 種。
- **画像形式 (image format)**: タイムテーブルのレイアウト。`通常` / `ライムライト式` の 2 種。kind=`live_tokutenkai_heiki` の画像は format を持たない（併記タイテにライムライト式はあり得ないため）。
- **ステージ種別 (stage kind)**: 画像分割後の各ステージが「ライブステージ」か「特典会ステージ」かを示す bool あるいは enum。

---

## 現状の問題点詳細

### A. 画像種別の入力と保存

[src/app.py:648](../../src/app.py#L648) の `st.radio` で 6 択:

```
ライブ / 特典会 / 両方(特典会別添え) / 両方(特典会併記) / その他 / その他(特典会併記)
```

[src/workflow.py:131-155](../../src/workflow.py#L131-L155) で `img_type` を以下の `resolved_img_type` （= `timetables` のキー名 = フォルダ名）に変換:

| UI 入力 | `resolved_img_type` | format |
|---|---|---|
| ライブ | `"ライブ"` | 通常/ライムライト式 |
| 特典会 | `"特典会"` | 通常/ライムライト式 |
| 両方(特典会別添え) | `"ライブ"` と `"特典会"` の **2 つ** に同一画像を登録 | 通常/ライムライト式 |
| 両方(特典会併記) | `"ライブ特典会"` | `"特典会併記"` 固定 |
| その他 | `img_type_alternative` (任意文字列) | 通常/ライムライト式 |
| その他(特典会併記) | `img_type_alternative` (任意文字列) | `"特典会併記"` 固定 |

### B. 永続化先

[src/backend_functions/project_repository.py:219-223](../../src/backend_functions/project_repository.py#L219-L223):

```python
project_info_json["event_detail"][event_no]["timetables"][resolved_img_type] = {
    "format": img_format,
    "stage_num": 0,
    "stage_list": [],
}
```

`timetables` は **画像種別をキーとする辞書**。`stage_list[]` は `{stage_no, stage_name}` のみで、ステージ単位の種別情報はない。

### C. ステージマスタは ⑥ 工程まで作られない

`master_stage.csv` の生成は [src/backend_functions/output_builder.py:263](../../src/backend_functions/output_builder.py#L263) で、`特典会フラグ` 列は [src/app.py:1133](../../src/app.py#L1133), [1155](../../src/app.py#L1155), [1177](../../src/app.py#L1177) で組み立てられる:

```python
tokutenkai_flg = event_type == "特典会"    # ← 文字列リテラル一致
stage_master[id] = {"ステージ名": ..., "特典会フラグ": tokutenkai_flg}
```

特典会併記タイテのブースは [src/app.py:1177](../../src/app.py#L1177) で `True` 固定。「その他」系画像のステージは常に `False` 扱いになる。

---

## 新設計

### 1. 画像種別 (image kind) を 3 値の enum として正規化

```
KIND_LIVE                    = "live"
KIND_TOKUTENKAI              = "tokutenkai"
KIND_LIVE_TOKUTENKAI_HEIKI   = "live_tokutenkai_heiki"
```

### 2. UI: 種別 radio を以下の 7 択に再構成

```
・ライブ
・特典会
・ライブ特典会併記
・両方(特典会別添え)         # 1 画像の上部=ライブ・下部=特典会など、同一画像を2回登録する手間を省くための便宜的選択肢
・その他(ライブ)              # 任意フォルダ名で kind=live として登録
・その他(特典会)              # 任意フォルダ名で kind=tokutenkai として登録
・その他(ライブ特典会併記)    # 任意フォルダ名で kind=live_tokutenkai_heiki として登録（互換性ではなく可能性のため）
```

「形式 (format)」 radio は `通常 / ライムライト式` の 2 択のみ。kind=`live_tokutenkai_heiki` のとき format は disabled とし、内部的にも format フィールドを保持しない（**併記タイテにライムライト式はあり得ない**ため）。

これに伴い、これまで format に存在した `"特典会併記"` 値は廃止する。`format == "特典会併記"` で分岐していた箇所（[src/backend_functions/ocr_service.py:410](../../src/backend_functions/ocr_service.py#L410), [src/backend_functions/output_builder.py:53](../../src/backend_functions/output_builder.py#L53), [src/backend_functions/timetabledata.py:316](../../src/backend_functions/timetabledata.py#L316) 経由の `with_tokutenkai` 等）はすべて `kind == "live_tokutenkai_heiki"` ベースに置き換える。

### 3. 画像の識別子は配列インデックスベースに

`timetables` を「キー = 画像種別文字列」の辞書から、**画像エントリの配列** に変える。

```json
"timetables": [
    {
        "image_no": 0,
        "display_name": "ライブ",
        "dir_name": "ライブ",
        "kind": "live",
        "format": "通常",
        "stage_num": 11,
        "stage_list": [
            {"stage_no": 0, "stage_name": "STAGE-A", "kind": "live"},
            ...
        ]
    },
    {
        "image_no": 1,
        "display_name": "特典会",
        "dir_name": "特典会",
        "kind": "tokutenkai",
        "format": "通常",
        ...
    },
    {
        "image_no": 2,
        "display_name": "ライブ特典会",
        "dir_name": "ライブ特典会",
        "kind": "live_tokutenkai_heiki",
        "stage_num": 4,
        "stage_list": [
            {"stage_no": 0, "stage_name": "STAGE-A", "kind": "live_tokutenkai_heiki"},
            ...
        ]
    }
]
```

- `image_no`: 画像識別子。同一イベント内で一意の整数連番。**画像削除時も振り直さない**（欠番として残す）。「両方(特典会別添え)」を選んだ場合は内部的に kind=`live` と kind=`tokutenkai` の 2 エントリを別 `image_no` で生成する。
- `dir_name`: ディスク上のサブフォルダ名。`"ライブ"`, `"特典会"`, `"ライブ特典会"`, または「その他」系で指定されたカスタム名。
- `display_name`: UI 表示用ラベル。原則 `dir_name` と同値。
- `kind`: 3 値 enum。**これが特典会フラグの真の根拠**。
- `format`: `kind == "live_tokutenkai_heiki"` のときは保持しない（フィールド自体を省略）。
- 各 `stage_list[]` 要素にも `kind` を持たせる（次節参照）。

### 3.1. フォルダ名衝突時の挙動

新規画像登録時、同一イベント内で **既存の `dir_name` と一致するもの**（標準名 `"ライブ"`/`"特典会"`/`"ライブ特典会"` を含む）があれば、UI で**上書き確認ダイアログ**を表示する。

```
このイベントには既に「ライブ」の画像が登録されています。
上書きしますか？（既存の画像・分割結果・OCR結果は失われます）
[上書きする] [キャンセル]
```

- 確認 OK: 既存エントリの `raw.png` と `timetables[]` 内エントリを置き換える。`image_no` は **既存のものを再利用する**（OCR 済みステージ JSON 等を残すかは後述）。
- キャンセル: 登録を中止。

**重要**: 上書きが選ばれた場合、その画像配下の派生物（`stage_*.png`, `stage_*.json`, `raw_cropped.png` 等）は古いものが残ると不整合になるため、**フォルダ内をクリーンアップしてから新画像を保存する**。実装時はこのクリーンアップ漏れに注意。

現状の [src/backend_functions/project_repository.py:213-223](../../src/backend_functions/project_repository.py#L213-L223) は `os.makedirs(..., exist_ok=True)` で**黙って上書き**しており事故源となっているため、この変更は標準名「ライブ」「特典会」を含めた全種別に適用する。

### 4. ステージ単位の種別 (stage kind) を画像分割時点で確定

ステージ画像分割（③(ⅱ)）の時点で、`stage_list[i]` に以下を必ず付与:

```json
{
    "stage_no": 0,
    "stage_name": "STAGE-A",
    "kind": "live"   // 親画像の kind を継承
}
```

**継承ルール:**
| 親画像の kind | stage_list[i].kind |
|---|---|
| `live` | `live` |
| `tokutenkai` | `tokutenkai` |
| `live_tokutenkai_heiki` | `live_tokutenkai_heiki`（**「ライブとしては扱うがブースも併記されている」ことを保持**） |

現時点では「親画像 kind と完全一致」のため冗長だが、**将来「1 画像内に live と tokutenkai のステージが混在」する形式を許容できる余地** を確保するために stage 側にも明示的に持たせる。

### 5. 出力ビルド時の特典会フラグ導出（後方互換性なし・置き換え）

[src/app.py:1100-1190](../../src/app.py#L1100-L1190) の出力ビルドロジックを以下のように書き換える。**旧スキーマ（event_type 文字列リテラル一致）はサポートしない**：

```python
for image_entry in event_detail["timetables"]:
    for stage in image_entry["stage_list"]:
        if stage["kind"] == "tokutenkai":
            tokutenkai_flg = True
        elif stage["kind"] == "live_tokutenkai_heiki":
            # 親ステージとしてはライブ枠、ブースは派生時に kind=tokutenkai で別途登録
            tokutenkai_flg = False
        else:  # "live"
            tokutenkai_flg = False
        stage_master[next_id] = {
            "ステージ名": stage["stage_name"],
            "特典会フラグ": tokutenkai_flg,
        }
```

特典会併記画像の特典会ブースは、これまでどおり [src/backend_functions/timetabledata.py:307](../../src/backend_functions/timetabledata.py#L307) `devide_df_live_tokutenkai()` で OCR JSON から派生させ、ブース名のステージとして登録する（kind=`tokutenkai` 固定）。ただし、併記の判定は `image_entry["kind"] == "live_tokutenkai_heiki"` で行う（`format == "特典会併記"` は使わない）。

`master_stage.csv` のスキーマ自体は **据え置き** (`ステージID, ステージ名, 特典会フラグ`)。bool として出力する。kind を文字列で持ちたければ列追加の余地はあるが、初期段階では bool 維持で十分。

### 6. project_info.json 自動マイグレーション（将来削除予定）

既存プロジェクト読み込み時に旧スキーマ → 新スキーマへ自動変換するロジックを [src/backend_functions/project_repository.py:82](../../src/backend_functions/project_repository.py#L82) `get_project_json()` 内（または専用モジュール）に追加する。

| 旧キー | 旧 format | → 新 kind | 新 format |
|---|---|---|---|
| `"ライブ"` | 通常/ライムライト式 | `live` | そのまま |
| `"特典会"` | 通常/ライムライト式 | `tokutenkai` | そのまま |
| `"ライブ特典会"` | `特典会併記` | `live_tokutenkai_heiki` | （無し） |
| その他のカスタム名 | 通常/ライムライト式 | **`tokutenkai`** | そのまま |
| その他のカスタム名 | `特典会併記` | `live_tokutenkai_heiki` | （無し） |

旧カスタム名（例: `"縁日"`）は **一律 kind=`tokutenkai`** に変換する。本来ライブ枠だった画像が誤って特典会扱いになる可能性があるが、過去データであり、利用頻度も低いと想定されるため許容する。マイグレーション後にユーザーが個別に修正する必要があれば、画像登録 UI から既存画像の種別を変更できる小機能を提供する（実装フェーズで検討）。

各 stage_list[i] には親画像 kind を `kind` フィールドとして付与する。

**重要**: このマイグレーションコードは **全プロジェクトが新スキーマに移行した後は不要** となる。以下を満たすため、コードに `# TODO(post-migration): 全プロジェクト移行後に削除` コメントを付け、専用モジュール（例: `src/backend_functions/project_migration.py`）に隔離して将来削除しやすくする:

- マイグレータは独立モジュール
- 呼び出し箇所は `get_project_json()` 内の 1 箇所のみ
- マイグレータが触る旧キー / 旧 format 値の判定リストを定数で集約

将来「もう旧スキーマのプロジェクトは無い」と確認できたタイミングで、モジュールごと削除すれば良い状態にしておく。

---

## 影響範囲

| ファイル | 変更内容 |
|---|---|
| [src/app.py:648-657](../../src/app.py#L648-L657) | radio の選択肢変更、「その他(ライブ)」「その他(特典会)」分岐の追加 |
| [src/workflow.py:118-166](../../src/workflow.py#L118-L166) `register_image()` | `img_type` → `kind` のマッピング、`dir_name` 解決ロジック |
| [src/backend_functions/project_repository.py:50-58](../../src/backend_functions/project_repository.py#L50-L58) `get_stage_name_list/get_stage_name` | `timetables` のアクセス方法を配列ベースに変更 |
| [src/backend_functions/project_repository.py:28-40](../../src/backend_functions/project_repository.py#L28-L40) `get_event_type_list` | 戻り値の意味を再定義（種別文字列 → 画像エントリ） |
| [src/backend_functions/project_repository.py:199-224](../../src/backend_functions/project_repository.py#L199-L224) `register_timetable_image` | 新スキーマで永続化、`dir_name` 衝突検出ロジック追加、上書き時の派生物クリーンアップ |
| [src/backend_functions/project_repository.py:227-234](../../src/backend_functions/project_repository.py#L227-L234) `delete_timetable_image` | `image_no` 指定での削除（欠番許容） |
| [src/backend_functions/image_processing.py:459-462](../../src/backend_functions/image_processing.py#L459-L462) | stage_list 追加時に `kind` を継承付与 |
| [src/backend_functions/ocr_service.py:69,182,213-245](../../src/backend_functions/ocr_service.py#L69) | 画像参照を `image_no` ベースに変更 |
| [src/backend_functions/output_builder.py:36-56](../../src/backend_functions/output_builder.py#L36-L56) `determine_id_master` | event_type ループを画像エントリループに変更 |
| [src/app.py:1100-1190](../../src/app.py#L1100-L1190) `render_output_section` | 特典会フラグ導出を `stage["kind"]` ベースに |
| (新規) `src/backend_functions/project_migration.py` | 旧スキーマ → 新スキーマのマイグレータ |

加えて、既存の保存済みデータ（[data/projects/](../../data/projects/) 配下の全プロジェクト）の `project_info.json` は読み込み時に自動マイグレーションされ、次回保存時に新スキーマで書き戻される。

---

## 後方互換性

- ディスクのフォルダ構造 (`event_1/ライブ/raw.png` 等) は **変更しない**。`dir_name` フィールドで吸収。
- `master_stage.csv` のスキーマ（`特典会フラグ` bool）は **変更しない**。
- `output.xlsx` の形式は **変更しない**。
- stage_*.json は変更不要（既存ロジックがそのまま使える）。
- project_info.json の旧スキーマは **読み込み時の自動マイグレーションのみで対応**。判定ロジック（特典会フラグ導出など）は新スキーマのみを前提に書き換える（二重ロジック化はしない）。

---

## 確定済み論点（記録用）

1. **Q1: 「その他(ライブ特典会併記)」を UI に出すか** → **出す**。互換性ではなく可能性（標準名と被るカスタム併記画像の登録）のため。
2. **Q2: 既存プロジェクトの自動マイグレーション** → **読み込み時に自動書き換え**。マイグレータは将来削除予定のコードとして隔離。
3. **Q3: stage_list 内の kind を画像と別に持つ必要性** → **持つ**。現時点では親画像 kind と一致するため冗長だが、将来「1 画像内で kind が混在する形式」を許容できる余地として明示的に保持。
4. **Q4: format `"特典会併記"` の扱い** → **廃止**。kind=`live_tokutenkai_heiki` の画像は format フィールドを持たない。併記タイテにライムライト式はあり得ないため。
5. **Q5: 「両方(特典会別添え)」の UI 残存** → **残す**。1 画像の上部=ライブ、下部=特典会など同一画像を 2 回登録する手間を省く便宜的選択肢として有用。内部的には kind=`live` と kind=`tokutenkai` の 2 エントリを生成。
6. **Q6: フォルダ名衝突時の挙動** → **上書き確認ダイアログ**。種別を問わず（"ライブ"/"特典会"/"ライブ特典会"/カスタム名すべてに対し）、同一 `dir_name` の画像が既に登録されている場合は UI で確認を取る。OK なら派生物をクリーンアップしてから上書き、キャンセルなら登録中止。現状の暗黙上書きを廃止する。
7. **Q7: マイグレーション時のカスタム名 kind 推論** → **`通常`/`ライムライト式` は一律 `tokutenkai`、`特典会併記` は `live_tokutenkai_heiki`**。過去データでありデフォルト推論の誤りは許容。
8. **Q8: 画像識別子の発番** → **`image_no` 整数連番のみ**（`image_id` 文字列フィールドは不採用）。削除時の振り直しはしない（欠番として残す）。

---

## 残オープン論点

なし（実装着手可）。

実装中に判明した細部論点（例: 上書き確認ダイアログでクリーンアップする派生物の具体的な範囲、既存画像 kind 変更 UI の置き場所など）は、実装フェーズで個別判断する。

---

## 段階的実装案（実装フェーズで参照）

1. **Phase 1**: 設計確定 (本ドキュメント) + 残オープン論点のクローズ
2. **Phase 2**: マイグレータ実装 + 既存プロジェクトの kind 自動推論ロジック
3. **Phase 3**: project_repository を新スキーマで書き換え（旧スキーマはマイグレータが吸収するので二重化不要）
4. **Phase 4**: image_processing / ocr_service を image_id ベースに移行
5. **Phase 5**: app.py の UI 変更（radio 6→6 択再構成）
6. **Phase 6**: output_builder の 特典会フラグ導出を kind ベースに切替
7. **Phase 7**: 旧スキーマサポート除去
