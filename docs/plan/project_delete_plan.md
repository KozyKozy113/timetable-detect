# プロジェクト削除機能 実装計画

## 概要

特定プロジェクトのデータを、ローカル・S3 の両方から整合性を保って削除する機能を実装する。①画面の最下部に削除ボタンを設置し、確認ダイアログを経て以下 4 ステップを完了させる。

1. ローカル `data/projects/<PJ>/` 配下を再帰削除
2. ローカル `projects_master.csv` と `projects_master_s3.csv` から `<PJ>` 行を削除
3. S3 `projects/<PJ>/` プレフィックスを全削除
4. 更新後の `projects_master_s3.csv` を S3 `master/` に上書きアップロード

削除完了後はサイドバーのプロジェクト一覧から消え、`pj_name=None` の初期状態に戻る。

---

## 0. 削除順序の方針

**「ローカル → S3」の順** で削除する。理由:

- 途中失敗ケースには **ユーザーが意図的に中断したケース** も含まれる。S3 にデータが残っていれば、誤って消した場合でも「呼出」で再ダウンロードして復元できる。
- ローカルだけが消えた状態は、現状の `get_project_data()` の動作（S3 にあればローカルへ DL）と整合し、復元経路として既存仕様に乗っている。
- 逆順（S3 → ローカル）で途中失敗すると、S3 側の本体が消えた状態で復元手段を失う。

> ⚠️ **S3 master の整合トレードオフ**: ローカル先順では、ステップ 2 完了〜ステップ 4 完了の間にクラッシュすると、S3 `master/projects_master_s3.csv` には行が残ったままになる。別端末が起動すると一覧に表示されるが、`get_project_data()` で S3 本体プレフィックスが空のため空ダウンロードになり、ローカル `projects_master.csv` に空エントリだけが書かれる「ゴースト状態」になる。これは削除をリトライ（同じ pj_name で再度削除ボタン押下）すれば収束する（全工程冪等のため。後述「3. エラーハンドリング方針」参照）。

---

## 1. レイヤー別の追加実装

既存の create / load の対称形を踏襲し、4 レイヤーに 1 関数ずつ追加する。

### 1-1. S3 層 (`src/backend_functions/s3access.py`)

```python
def delete_project_from_s3(pj_name: str) -> None:
    """S3 の projects/<pj_name>/ プレフィックスを全削除"""
    my_bucket.objects.filter(Prefix=f"projects/{pj_name}/").delete()

def put_projects_master_s3() -> None:
    """ローカルの projects_master_s3.csv を S3 master/ に上書きアップロード"""
    upload_s3_file("master", "projects_master_s3.csv",
                   os.path.join(DATA_PATH, "master", "projects_master_s3.csv"))
```

- `put_projects_master_s3` は `put_project_data` 内の同等処理（s3access.py:192）と同じ呼び出し。副作用が大きいので既存関数は触らず、新規関数として独立させる。
- `boto3 resource API` の `Bucket.objects.filter(Prefix=...).delete()` は 1000 件ずつバッチ削除する。プロジェクト配下のオブジェクト数 (≪1000) を考慮するとこれで十分。

### 1-2. リポジトリ層 (`src/backend_functions/project_repository.py`)

```python
def delete_project_data(
    data_path: str,
    pj_name: str,
    project_master: pd.DataFrame,
    project_master_s3: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ローカルのプロジェクト本体ディレクトリと両マスタ行を削除する。
    Returns: (updated_project_master, updated_project_master_s3)
    """
    pj_dir = os.path.join(data_path, "projects", pj_name)
    if os.path.isdir(pj_dir):
        shutil.rmtree(pj_dir)
    if pj_name in project_master.index:
        project_master = project_master.drop(index=pj_name)
    if pj_name in project_master_s3.index:
        project_master_s3 = project_master_s3.drop(index=pj_name)
    project_master.to_csv(os.path.join(data_path, "master", "projects_master.csv"))
    project_master_s3.to_csv(os.path.join(data_path, "master", "projects_master_s3.csv"))
    return project_master, project_master_s3
```

- `create_project_data` (project_repository.py:379) の対称形にする。
- `shutil.rmtree` は Windows で read-only ファイルがあると失敗する可能性があるため、必要なら `onerror` で `os.chmod` 後リトライ（既存に同様パターンがなければ素直に rmtree のみ）。

### 1-3. ワークフロー層 (`src/workflow.py` の `ProjectWorkflow`)

```python
def delete_project(self, pj_name: str, state: AppState) -> WorkflowResult:
    """プロジェクトを削除する。
    順序: ローカル削除 → S3オブジェクト削除 → projects_master_s3.csv アップロード。
    """
    if pj_name not in state.project.project_master.index \
            and pj_name not in state.project.project_master_s3.index:
        return WorkflowResult(success=False, error="存在しないプロジェクトです")

    # 1) ローカル本体 + 両マスタCSV
    pm, pm_s3 = repo.delete_project_data(
        self._data_path, pj_name,
        state.project.project_master, state.project.project_master_s3,
    )
    state.project.project_master = pm
    state.project.project_master_s3 = pm_s3

    # 2) S3 オブジェクト削除
    s3access.delete_project_from_s3(pj_name)
    # 3) projects_master_s3.csv の再アップロード（S3 側の整合性回復）
    s3access.put_projects_master_s3()

    # 削除したのが選択中プロジェクトなら state をクリア
    if state.project.pj_name == pj_name:
        state.project.pj_name = None
        state.project.pj_path = None
        state.project.project_info_json = None
        state.project.event_type = None
        state.project.event_num = 1
        state.crop = CropState()
        state.ocr = OcrState()
        state.output = OutputState()

    return WorkflowResult(success=True)
```

- 順序は **「ローカル → S3」**。逆順だと、S3 削除後にローカル削除が失敗した場合、別端末/再起動時に `get_project_data` が S3 から復元できず迷子になるため。

### 1-4. UI 層 (`src/app.py`)

#### コールバック追加

```python
def request_delete_project():
    st.session_state["_pending_delete_confirm"] = True

def confirm_delete_project():
    pj_name = app_state.project.pj_name
    result = _project_wf.delete_project(pj_name, app_state)
    st.session_state.pop("_pending_delete_confirm", None)
    if not result.success:
        st.toast(result.error, icon="🚨")
    else:
        _sync_to_session(app_state)
        st.session_state.pop("exist_pj_name", None)
        st.toast(f"プロジェクト「{pj_name}」を削除しました", icon="🗑️")
        st.rerun()

def cancel_delete_project():
    st.session_state.pop("_pending_delete_confirm", None)
```

#### `render_project_setting()` の末尾に追加

```python
    st.divider()
    with st.expander("⚠️ プロジェクトの削除", expanded=False):
        st.warning(
            f"「{app_state.project.pj_name}」のローカル・S3 上の全データと "
            "プロジェクトマスタの該当行を完全に削除します。**この操作は取り消せません。**"
        )
        if not st.session_state.get("_pending_delete_confirm"):
            st.button(
                "このプロジェクトを削除",
                key="btn_request_delete_project",
                on_click=request_delete_project,
                type="secondary",
            )
        else:
            st.error(f"本当に「{app_state.project.pj_name}」を削除しますか?")
            confirm_text_key = "_delete_confirm_text"
            st.text_input(
                f"確認のため、プロジェクト名「{app_state.project.pj_name}」を入力してください",
                key=confirm_text_key,
            )
            cols = st.columns(2)
            with cols[0]:
                st.button(
                    "削除を実行する",
                    key="btn_confirm_delete_project",
                    on_click=confirm_delete_project,
                    type="primary",
                    disabled=(st.session_state.get(confirm_text_key) != app_state.project.pj_name),
                )
            with cols[1]:
                st.button(
                    "キャンセル",
                    key="btn_cancel_delete_project",
                    on_click=cancel_delete_project,
                )
```

**UI 設計のポイント**:
- ①画面の最下部に **expander で折りたたみ**、誤押下を防ぐ。
- 「プロジェクト名と完全一致するテキスト入力」で誤削除をさらに防止（GitHub 等のリポ削除 UI と同じパターン）。
- 確認状態は `st.session_state["_pending_delete_confirm"]` で管理。
- 削除後、`_sync_to_session` で session_state の `pj_name=None` 等が反映され、サイドバーは `app_state.project.pj_name is None` 表示に戻る。
- `pj_name_list` がモジュール先頭 (app.py:82) で一度だけ評価される現状仕様の都合上、削除後は `st.rerun()` でモジュール再評価させて一覧から消す。

---

## 2. 既存挙動への影響

- 既存の `make_project` / `set_project` / `update_timestamp` / `put_project_data` などには **一切手を入れない**。
- `pj_name_list` の再評価は `st.rerun()` 経由で自然に起きる。
- 編集モード中の削除を防ぐため、削除ボタンの `disabled` 条件として `bool(app_state.output.edits)` を追加するか、削除実行時に `_discard_all_pending_edits()` を呼ぶ。**未保存編集の破棄前に削除する誤操作の方が惨事**なので、`bool(app_state.output.edits)` のときは `st.info("編集中の作業コピーがあります。⑥で保存または破棄してから削除してください")` を出してボタンを `disabled=True` にする方針を推奨。

---

## 3. エラーハンドリング方針

ローカル削除と S3 削除の各ステップを try/except で囲み、途中失敗時には **どこまで進んだか** を表示する。

```python
try:
    pm, pm_s3 = repo.delete_project_data(...)
except Exception as e:
    return WorkflowResult(success=False, error=f"ローカル削除に失敗: {e}")
try:
    s3access.delete_project_from_s3(pj_name)
except Exception as e:
    return WorkflowResult(
        success=False,
        error=f"S3 削除に失敗（ローカルは削除済み・再度同名で削除リトライしてください）: {e}",
    )
try:
    s3access.put_projects_master_s3()
except Exception as e:
    return WorkflowResult(
        success=False,
        error=f"projects_master_s3.csv アップロードに失敗: {e}",
    )
```

リトライ時の冪等性:
- `delete_project_data` は存在チェックを内包しているので 2 回呼んでも安全。
- `delete_project_from_s3` も `.filter(Prefix=...).delete()` は対象が無くてもエラーにならない。
- `put_projects_master_s3` は常に最新を上書き。

**全工程が冪等**なので、エラー発生時に「もう一度同じ名前で削除ボタンを押す」だけで回復可能。

### 3-1. 冪等性の根拠（現状実装の検証結果）

| 関数 | 「対象がもう無い」状態での挙動 |
|---|---|
| `delete_project_data` | `os.path.isdir` / `in index` の存在チェックを内包しており、no-op |
| `delete_project_from_s3` | `boto3` の `Bucket.objects.filter(Prefix=...).delete()` は対象 0 件でも例外を出さず no-op |
| `put_projects_master_s3` | 常に「削除後のローカル写し」を上書き。S3 側にもう行が無くても結果は同じ |

### 3-2. 特殊ケースの確認

#### (a) ローカルにだけある（未 push）プロジェクトを削除

- `s3access.py:178` の `get_project_data` 条件 `pj_name in project_master_s3.index` を満たさないため、復元経路無し。
- ステップ 3 の S3 削除は no-op、ステップ 4 の S3 アップロードは「もともと S3 master に無い行を消す = no-op」。
- → 問題なく削除完了。

#### (b) S3 にだけある（他端末作成・このマシンで未呼出）プロジェクトを削除

- ローカル `projects_master.csv` には存在しないが、起動時 `get_master()` 経由でローカル `projects_master_s3.csv` には存在する。
- `delete_project_data` の `if pj_name in project_master.index` で master からの drop はスキップされる（ローカル本体ディレクトリも無いので `os.path.isdir` で skip）。
- S3 本体と S3 master の該当行は両方消える。
- → 「自分はこのマシンで一度も触っていないプロジェクト」も削除可能。**仕様として受け入れる**（権限分離は将来別タスク）。

---

## 4. 並行端末 race（既知の制約）

A 端末で削除している最中に、B 端末で別プロジェクト Y の `put_project_data(Y)` が走ると:

- B 端末のローカル `projects_master_s3.csv` 写しは「A 端末の削除前の状態」のため、X 行を含む。
- B の `put_project_data` が、X 行を含む写しで `master/projects_master_s3.csv` を上書きしてしまう → **S3 master 上で X が復活**。
- 一方、S3 `projects/X/` は A が消しているので、X を呼び出すと空フォルダになる。

これは現状の `put_project_data` 設計（自分の写しで他行も含めて丸ごと上書き）に由来する **既知の race** であり、削除機能特有ではない。

**運用回避**: 削除作業中は他端末で「S3 に保存」を実行しない。
**将来課題**: ETag / If-Match による楽観ロック導入は本計画のスコープ外。

---

## 5. 動作確認手順

1. テスト用に `テスト_削除` プロジェクトを作成 (`make_project`)、画像 1 枚登録 → `put_project_data` で S3 に push (一度 ⑦タブまで進めて `save_to_s3` を実行)
2. AWS マネジメントコンソールで `s3://idol-timetable/projects/テスト_削除/` にオブジェクトがあること、`master/projects_master_s3.csv` にエントリがあることを確認
3. ①画面の削除ボタンを押下 → 確認ダイアログでプロジェクト名を入力 → 「削除を実行する」
4. ローカル `data/projects/テスト_削除/` が消え、両 CSV から行が消え、サイドバー一覧から消えていることを確認
5. S3 `projects/テスト_削除/` が空になり、`master/projects_master_s3.csv` からも行が消えていることを確認
6. 別端末でアプリを起動し直し、`get_master()` 後にも当該プロジェクトが復活しないことを確認（最重要）

---

## 6. ファイル変更サマリ

| ファイル | 変更内容 |
|---|---|
| `src/backend_functions/s3access.py` | `delete_project_from_s3`, `put_projects_master_s3` を追加 |
| `src/backend_functions/project_repository.py` | `delete_project_data` を追加（要 `import shutil`） |
| `src/workflow.py` | `ProjectWorkflow.delete_project` を追加 |
| `src/app.py` | コールバック 3 つ + `render_project_setting()` 末尾に削除 expander |
| `docs/plan/project_delete_plan.md` | 本計画書（新規） |

---

## 7. スコープ外（明示）

- 「**複数プロジェクト一括削除**」「**削除済みプロジェクトの復元**（ゴミ箱）」は非対応。やる場合は S3 のバージョニング/ライフサイクル設定で別途。
- Stella 連携先 (`live{liveId}.json`) の自動削除は範囲外。Stella 側で手動削除。
