# notificationData4.json 同時追記 実装計画

## 目的

ライブタイムテーブル（Stella JSON）を **一括 Push** する際、同じコミット / PR で
`data/timetableproj/notificationData4.json` の `notificationList` 先頭に「対応しました」
お知らせを追記できるようにする。アプリのトップページ表示用で、主に release=2 への切替時に使う。

## 対象範囲（確定事項）

- **一括 Push のみ対応**（`push_all_stella_json` / ⑦「プロジェクト全体を一括 Push」）。
  単一イベント Push (`push_stella_json`) は対象外。
- 追記の ON/OFF は **チェックボックス（デフォルト False）** で選択。画面は **⑦**。
- メッセージは **自動生成 → 手動編集可**（⑦のテキストエリア）。
- 「**エリア**」（横浜・お台場 等、都道府県とは別概念）は **①画面**で入力。
- `liveId` 重複時は **「取りやめ」「過去メッセージを削除して追加」「過去メッセージを残して追加」** を選択。

## 追記フォーマット

`notificationList`（配列）の **先頭要素**として以下を prepend する。

```json
{
  "icon": "mdi-alarm-plus",
  "liveId": [605, 606],
  "date": "6/23",
  "message":    "HERO SONIC (6/27-28 @横浜 アイドル) に対応しました。",
  "message_en": "HERO SONIC (6/27-28 IDLE) is supported."
}
```

| フィールド | 生成元 |
|-----------|--------|
| `icon` | 固定 `"mdi-alarm-plus"` |
| `liveId` | 一括 Push 対象 **全イベントの liveId 配列**（イベント1件でも配列）|
| `date` | **Push 当日（日本時間）** を `M/D`（ゼロ埋めなし） |
| `message` | `[liveName] ([日付] @[エリア] [ジャンルJP]) に対応しました。` |
| `message_en` | `[liveName] ([日付] [ジャンルEN]) is supported.` |

- `[日付]` = 対象イベントの公演日レンジ。`stella_metadata.date`(YYYYMMDD) の min/max から導出。
  - 単日 → `6/27` / 同月複数日 → `6/27-28` / 月跨ぎ → `6/27-7/1`
- `[liveName]` = `stella_project_meta.liveName`
- `[エリア]` = `stella_project_meta.area`（新規・①で入力。空なら ` @ `→` @ `そのまま or 省略は要検討、下記参照）
- ジャンルラベル: `genre` → 表示語のマップ（既存 `_STELLA_GENRE_OPTIONS`: 1=ロック, 2=アイドル）
  - `2`(アイドル) → JP `アイドル` / EN `IDLE`
  - `1`(ロック) → JP `バンド` / EN `BAND`（テンプレ既存表記に合わせる）

> 既存 `notificationList` の旧エントリには `liveId` フィールドが無い（手書き運用のため）。
> 重複検知は `liveId` を持つエントリのみ対象。旧エントリとは衝突しない（=安全）。

---

## 既存実装の足場（調査結果）

| 役割 | 場所 |
|------|------|
| 一括 Push オーケストレーション | [src/backend_functions/stella_push.py](../../src/backend_functions/stella_push.py) `push_all_stella_json` |
| Push に追加ファイルを同梱する既存パターン | `stella_push._stage_release_livelist`（liveList.json を同コミットに加える）|
| timetableproj リポ I/O | [src/backend_functions/github_ops.py](../../src/backend_functions/github_ops.py)（`LOCAL_REPO_PATH=data/timetableproj`, `clone_or_pull`/`commit_and_push`）|
| JSON 入出力ヘルパ | [src/backend_functions/stella_export.py](../../src/backend_functions/stella_export.py)（`update_live_list` 等。BOM=utf-8-sig）|
| プロジェクト単位メタ | [src/backend_functions/project_repository.py](../../src/backend_functions/project_repository.py) `get/set_stella_project_meta`（liveName/genre/release/pref）|
| ①画面 Stella 入力フォーム | [src/app.py](../../src/app.py) L944 付近 `render_*`（liveName/genre/release/pref/公演日）|
| ⑦一括 Push UI | [src/app.py](../../src/app.py) L3342 付近 `一括 Push` ＋ L3416 `_run_stella_bulk_push` |
| WF ラッパ | [src/workflow.py](../../src/workflow.py) L1386 `push_all_stella_json` |

`notificationData4.json` は timetableproj リポ直下の追跡ファイル。push_files に加えれば
既存の commit/PR フローでそのまま反映される。

---

## 変更点

### 1. ①画面：エリア入力の追加

- `project_repository._default_stella_project_meta()` に `"area": ""` を追加
  （`ensure_stella_project_meta` の setdefault で既存PJも後方互換補完）。
- [src/app.py](../../src/app.py) L944 の Stella 連携設定フォームに「エリア」テキスト入力を追加
  （都道府県セレクトの近く、横並び or 直下）。`key="stella_input_area"`、初期値 `project_meta.get("area","")`。
  help に「横浜・お台場など、お知らせ文に使う表示地名（都道府県とは別）」。
- `_save_stella_input_local()` の `project_meta_payload` に
  `"area": st.session_state.get("stella_input_area", "")` を追加。
  （`set_stella_project_meta` は merge update なので既存 save 経路でそのまま保存される）

### 2. お知らせ生成ロジック（純粋関数・新規）

`stella_export.py`（純粋・I/O薄）に追加：

- `build_notification_date_range(dates: list[str]) -> str`
  YYYYMMDD の list から `M/D` / `M/D-D` / `M/D-M/D` を生成。
- `GENRE_NOTIFICATION_LABELS = {1: ("バンド","BAND"), 2: ("アイドル","IDLE")}`
- `build_notification_messages(project_meta, event_dates, genre) -> tuple[str,str]`
  JP / EN メッセージ文字列を生成（エリア空時の扱いを含む）。
- `build_notification_entry(live_ids, push_date, message, message_en) -> dict`
  `icon` 固定・フィールド順を例に合わせて構築。
- `find_duplicate_notifications(notification_list, live_ids) -> list[int]`
  既存エントリ中 `liveId` が新規 live_ids と**1つでも交差**するもののインデックス list。
- `read_notification_data(path) -> dict` / `prepend_notification(data, entry, *, remove_indices=()) -> dict`
  `notificationList` の先頭追加（＋replace 時は指定インデックス除去）。
- `write_notification_data(path, data)` — **utf-8-sig**、`indent=2`, `ensure_ascii=False`。

> **書き出し方式（確定）**: **全面 JSON 再整形（`json.dump(indent=2, ensure_ascii=False)`）**。
> 既存ファイルの手書き桁揃え空白は失われ、初回追記時にファイル全体が一度差分化するが、
> 実装が単純で堅牢（追記・replace を同一経路で扱える）。読み書きは utf-8-sig を維持。

### 3. push_all_stella_json：notification 同梱

[src/backend_functions/stella_push.py](../../src/backend_functions/stella_push.py) `push_all_stella_json` に
`notification: NotificationPush | None = None` 引数を追加。

```python
@dataclass
class NotificationPush:
    enabled: bool
    message: str
    message_en: str
    date: str                     # "M/D"（UI で確定済み・編集後の値）
    live_ids: list[int]
    dedup_strategy: str = "none"  # "none" | "keep" | "replace"  ("cancel" は UI 側で None 化)
```

- `_stage_notification(notification, repo_path) -> list[str]` を追加（`_stage_release_livelist` と同型）:
  1. `notification is None` / `not enabled` → `[]`
  2. `read_notification_data(notif_path)`
  3. `dup_idx = find_duplicate_notifications(list, live_ids)`
     - 本関数到達時点で UI が dedup_strategy を確定済みである前提。
     - `replace` → `remove_indices=dup_idx`、`keep`/`none` → 除去なし。
  4. `prepend_notification(...)` → `write_notification_data(...)`
  5. `["notificationData4.json"]` を返す
- `push_files = [...records...] + extra_files(liveList) + notif_files` に連結。
- commit メッセージにお知らせ追記を併記（例: `+ notify(live605,606)`）。
- **ロールバック**: 既存どおり、PR モードは push 後に `reset_hard(before_hash)`、
  失敗時も `reset_hard`。notificationData4.json は内側リポ作業ツリーへの書込のみなので
  既存ロールバックで巻き戻る（本アプリ配下には残さない）。

> `notification` の確定は **push 成功後にローカルへ反映するものは無い**（version 系のような
> 確定書き戻しは不要）。お知らせは GitHub 側ファイルのみ。重複防止の状態も持たない
> （次回も notificationList を読んで毎回判定するため）。

### 4. WF ラッパ（workflow.py）

`push_all_stella_json(state, mode, release_override, notification=None)` に引数追加して
`stella_push.push_all_stella_json(..., notification=notification)` へ素通し。

### 5. ⑦UI：チェックボックス・編集欄・重複ダイアログ

[src/app.py](../../src/app.py) `_render_stella_bulk_push`（L3342付近）に追加。

1. **チェックボックス** `key="stella_notif_enabled"`、`value=False`（デフォルト OFF）。
   ラベル「notificationData4.json にお知らせを追記する（一括 Push と同時）」。
2. ON 時に編集 UI を展開:
   - 自動生成（`build_notification_messages` + 当日 JST `date` + 対象 live_ids）を初期値に
     `st.text_area("お知らせ文（日本語）", key=...)` / `st.text_area("お知らせ文（英語）", key=...)`
     ＋ `st.text_input("表示日付 (M/D)", value=今日JST)`。
   - 「自動文を再生成」ボタンで現在のメタから上書き再生成。
   - 対象 liveId（採番済みのみ）と、エリア未入力時の注意（①で入力を促す）を caption 表示。
3. **重複ハンドリング**（push ハンドラ内）:
   - `push_all_stella_json_pr` / 直接 Push 確定ハンドラを、お知らせ ON のとき
     **まず `clone_or_pull` 相当でリポ同期 → notificationData4.json 読込 → 重複判定** する
     段に拡張（`_output_wf.sync_stella_repo` 流用 or 専用の軽量 read）。
   - 重複あり: push せず `st.session_state["_notif_dup_pending"] = {...}` をセットし、
     `st.radio("既存お知らせと liveId が重複しています", ["取りやめ","過去を削除して追加","過去を残して追加"])`
     ＋「この内容で Push」ボタンを描画（直接 Push 確認ダイアログと同じ作法）。
   - 重複なし: そのまま `dedup_strategy="none"` で push。
   - 選択 → strategy 確定:
     - 取りやめ → `notification=None` で push（**タイテ Push は通常実行**、お知らせ追記のみ中止）【確定】
     - 過去を削除して追加 → `dedup_strategy="replace"`
     - 過去を残して追加 → `dedup_strategy="keep"`
4. `_run_stella_bulk_push(mode)` を `notification` を組み立てて WF へ渡すよう拡張。

> **重複チェックのタイミング**: push 関数内も `clone_or_pull` で最新化するが、ユーザー選択が
> push より前に必要なため、**ハンドラで先に1回 pull して判定 → 確認 → push**（push 内の
> pull は冪等で再度 origin に揃うだけ）。`stella_reserve` の Reserve-First と同じ
> 「先に最新を取得して判断」発想。

---

## ジャンル / エリアの文言仕様（確認ポイント）

- `area` が空のとき JP メッセージの ` @[エリア] ` をどうするか。
  **推奨: エリア空なら ` @` 部分を省く**（`(6/27-28 アイドル)`）。EN はもともとエリア無し。
- ジャンル `1`(ロック) の JP 表示語は `バンド`（テンプレ準拠）か `ロック` か。
  **推奨: テンプレ既存の `バンド/BAND` に合わせる。** アイドル運用が主のため影響軽微。

（いずれも自動生成の初期値であり、⑦で手編集可能なので運用で吸収できる。）

---

## テスト方針

`tests/backend_functions/` に追加（既存 `test_stella_push.py` / `test_stella_export.py` に倣う）:

- `build_notification_date_range`: 単日 / 同月レンジ / 月跨ぎ。
- `build_notification_messages`: エリア有無 × ジャンル(1/2) × 単複イベント。
- `find_duplicate_notifications`: liveId 交差あり / なし / 旧エントリ(liveId無し)を無視。
- `prepend_notification`: 先頭追加順序 / replace 時の指定除去。
- `push_all_stella_json(notification=...)`: github_ops を monkeypatch し
  push_files に `notificationData4.json` が含まれること、strategy 別の中身、
  `enabled=False`/`None` で不同梱、ロールバックで巻き戻ること。

---

## 影響範囲・非回帰

- 単一イベント Push・採番・liveList 再同期・既存一括 Push（お知らせ OFF）は無変更動作。
- `area` 追加は `_default_*` / `ensure_*` の後方互換補完で既存 PJ も安全。
- 失敗時ロールバックは既存 `reset_hard` 経路で notificationData4.json も巻き戻る。

## 実装順序

1. `project_repository` に `area` 追加 → ①UI・save 経路。
2. `stella_export` に生成/重複/読み書きヘルパ＋ユニットテスト。
3. `stella_push.push_all_stella_json` に `NotificationPush` 同梱＋`_stage_notification`＋テスト。
4. `workflow.py` ラッパ引数追加。
5. ⑦UI: チェックボックス・編集欄・重複ダイアログ・ハンドラ結線。
6. 結合確認（PR モードでブランチに notificationData4.json 差分が乗ること）。
