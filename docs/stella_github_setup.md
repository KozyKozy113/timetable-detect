# Stella GitHub 連携セットアップ手順

[stella_json_output_plan.md](plan/stella_json_output_plan.md) の Phase 6 (GitHub 連携) で必要な
ローカル環境セットアップと、PAT 受領後の clone smoke test 手順をまとめる。

---

## 1. 完了済セットアップ (リポにコミット済)

| ファイル | 内容 |
|---|---|
| `requirements.txt` | `GitPython` / `PyGithub` を追加 |
| `.gitignore` | `data/timetableproj/` を ignore に追加（ネスト git の clone 先） |
| `src/backend_functions/github_ops.py` | clone/pull/commit/push/PR 作成のラッパ実装 |

ローカル `.venv` には以下が既にインストール済:

- GitPython 3.1.43
- PyGithub 2.9.1
- PyNaCl 1.6.2 / cryptography 49.0.0 / pycparser 3.0 / cffi 2.0.0 (依存)

---

## 2. PAT 受領後の手順

### 2-1. `.env` への設定

リポオーナー (ys0512) から受領した PAT を `.env` に貼り付ける:

```ini
GITHUB_TOKEN=github_pat_xxxxxxxx...
GITHUB_USER_NAME=timetable-detect-bot       # commit author 表示名
GITHUB_USER_EMAIL=bot@example.com           # commit author email
```

- `.env` は `.gitignore` 済（リポには上がらない）
- `GITHUB_USER_NAME` / `GITHUB_USER_EMAIL` は任意の値で可。リポへの commit author として記録される
- 必要な PAT スコープ:
  - Classic PAT: `repo`
  - Fine-grained PAT: `ys0512/timetableproj` を対象に `Contents: Read and write` + `Pull requests: Read and write`

### 2-2. clone smoke test

`.env` 設定後、以下のワンライナーで認証チェック → clone → 中身確認まで一気に行える:

```powershell
cd c:\Users\kkoaz\Documents\projects\product\timetable-detect\timetable-detect
.venv\Scripts\python.exe -c "from dotenv import load_dotenv; load_dotenv(); import sys; sys.path.insert(0, 'src'); from backend_functions import github_ops; print('credentials_available:', github_ops.credentials_available()); print('cloning...'); h = github_ops.clone_or_pull(); print('HEAD hash:', h); import os; print('files:', os.listdir(github_ops.LOCAL_REPO_PATH)[:10])"
```

期待される出力:

```
credentials_available: True
cloning...
HEAD hash: <40 文字の hash>
files: ['.git', 'liveList.json', 'live547.json', ...]
```

### 2-3. pull の再実行確認

clone 済の状態でもう一度同じコマンドを実行する。今度は内部で `fetch` + `reset --hard origin/main` が走り、`HEAD hash` は同じものが返るはず。

```
credentials_available: True
cloning...                       ← clone と pull で同じメッセージ（関数は同じ）
HEAD hash: <同じ hash>
files: [...]
```

### 2-4. トラブルシュート

| 症状 | 原因 | 対処 |
|---|---|---|
| `GithubAuthError: GITHUB_TOKEN が未設定` | `.env` 未反映 | `load_dotenv()` の前に `.env` が存在するか確認 |
| `fatal: Authentication failed` | PAT が無効 / スコープ不足 | PAT を発行し直す。Fine-grained PAT は対象リポを `ys0512/timetableproj` に限定する |
| `GitCommandError: ... 'remote rejected'` | push 権限なし | PAT スコープに `Contents: Write` が無い |
| PR 作成時に `Resource not accessible by personal access token: 403` (直接 push は成功する) | PAT に PR 作成権限が無い (`Contents` のみ) | Fine-grained PAT に `Pull requests: Read and write` を追加 / Classic PAT は `repo`。当面は「Push (直接)」で回避可。push 済のフィーチャブランチは GitHub 上に残るため権限付与後に再実行で PR 化できる |
| `Permission denied: ...cffi...` | `.venv` 内 DLL がロック中 | Streamlit/Python プロセスを終了してから再実行 |
| `Unexpected UTF-8 BOM` を読み込み時に出した | 既存 JSON を `utf-8` で開いた | 全 JSON が BOM 付き運用 → `utf-8-sig` で開く（§2-5 参照） |

### 2-5. 運用リポの仕様メモ (smoke test で判明、2026-06)

- **文字コード**: 全 `liveList.json` / `live{id}.json` が **UTF-8 BOM 付き** (`ef bb bf`)
  - 読み書きは `encoding="utf-8-sig"` を使用すること
  - `stella_export.py` の `write_stella_json` / `update_live_list` も BOM 付き出力に対応済
- **liveList 件数 (smoke test 時点)**: 603 件、max `liveId` = 602 → 次回採番は 603 から
- **ファイル数**: 578 (マスタ系 `color.json` `delay_stage.json` 等も含む)
- **トップレベルキー**: `liveList.json` は `{"liveList": [...]}` のみ

---

## 3. 次のステップ (Phase 4 / 6 / 7-1 / 7-2 ⑥-D 実装)

clone smoke test が通ったら、計画書 [stella_json_output_plan.md](plan/stella_json_output_plan.md) の以下を順次実装する:

1. **Phase 3 残タスク**: `_default_stella_metadata` に `date` / `dow` 追加、`stella_project_meta` ブロック新設
2. **Phase 4**: `stella_reserve.py` 新規（Reserve-First 採番ロジック、bundleId 算出）
3. **Phase 7-1**: ①画面に Stella 入力ブロック + 採番ボタン追加
4. **Phase 7-2 ⑥-D**: Push UI 追加
5. **Phase 4-3**: `data/master/pref_master.json` 作成（47都道府県 + タイ + 台湾）
