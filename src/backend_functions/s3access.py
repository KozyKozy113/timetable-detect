import os
import uuid
import json
import pandas as pd

import boto3
from botocore.exceptions import ClientError
DIR_PATH = os.path.dirname(__file__)
DATA_PATH = DIR_PATH +"/../../data"
BUCKET_NAME = 'idol-timetable'

s3 = boto3.resource('s3')
my_bucket = s3.Bucket(BUCKET_NAME)

def upload_directory_to_s3(local_directory, s3_prefix=""):
    """
    local_directory配下のファイルを再帰的に列挙し、
    s3_prefixを先頭としたKey名でS3へアップロードする。
    """
    for root, dirs, files in os.walk(local_directory):
        for file_name in files:
            # ローカルファイルの絶対パス
            local_path = os.path.join(root, file_name)

            # S3上でのファイルパスを生成
            # 例: root= /path/to/local_directory/subdir, file_name= file.txt
            #     relative_path = subdir/file.txt
            #     s3_key = my-prefix/subdir/file.txt
            relative_path = os.path.relpath(local_path, local_directory)
            s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")

            print(f"Uploading {local_path} to s3://{BUCKET_NAME}/{s3_key}")
            my_bucket.upload_file(local_path, s3_key)

def download_prefix_from_s3(s3_prefix, local_directory):
    """
    指定したS3バケットとプレフィックスにあるオブジェクトを、
    ローカルフォルダに一括ダウンロードする。
    """
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')

    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=s3_prefix):
        # オブジェクトが存在しない場合、Contentsキーが無いことがある
        if 'Contents' not in page:
            continue

        for obj in page['Contents']:
            key = obj['Key']
            # S3上でディレクトリプレースホルダの場合、キーが`xxx/`で終わることがあるのでスキップ
            if key.endswith('/'):
                continue
            filename = key.split("/")[-1]

            # ローカルに保存するためのパスを生成
            # プレフィックス部分を削除し、残りをファイルパスにする #最低限ファイル名は保持する
            relative_path = key[len(s3_prefix):].lstrip('/')
            if len(relative_path)<len(filename):
                relative_path = filename
            local_path = os.path.join(local_directory, relative_path)

            # ダウンロード先のディレクトリがなければ作成
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            print(f"Downloading s3://{BUCKET_NAME}/{key} to {local_path}")
            my_bucket.download_file(key, local_path)

def upload_s3_file(s3_prefix, s3_filename, local_file_path):
    """
    ローカルファイルをS3にアップロードする。
    s3://bucket_name/s3_prefix/s3_filename の形で保存される。

    :param bucket_name: アップロード先のS3バケット名
    :param s3_prefix: S3上の保存先プレフィックス (例: 'my-folder/subfolder')
                      (先頭や末尾の/の有無は任意。内部で調整する。)
    :param s3_filename: S3上に保存するときのファイル名 (例: 'uploaded_data.csv')
    :param local_file_path: ローカルのファイルパス (例: '/path/to/data.csv')
    """

    # s3_prefixの末尾に / がなければ付与
    if s3_prefix and not s3_prefix.endswith('/'):
        s3_prefix += '/'

    # 実際にS3にアップロードされるオブジェクトキーを組み立て
    object_key = f"{s3_prefix}{s3_filename}" if s3_prefix else s3_filename

    try:
        my_bucket.upload_file(local_file_path, object_key)
        print(f"Uploaded {local_file_path} to s3://{BUCKET_NAME}/{object_key}")
    except ClientError as e:
        print(f"Failed to upload {local_file_path} to s3://{BUCKET_NAME}/{object_key}: {e}")
        raise

def download_s3_object(object_key, local_dir, local_filename):
    """
    S3のオブジェクトを指定し、ローカルの指定フォルダ・ファイル名でダウンロードする関数。

    :param bucket_name: ダウンロード元のS3バケット名
    :param object_key: ダウンロードしたいオブジェクトのキー (例: 'folder/subfolder/data.csv')
    :param local_dir: ローカルの保存先ディレクトリ (例: '/path/to/save')
    :param local_filename: ローカルに保存する際のファイル名 (例: 'downloaded_data.csv')
    """
    
    # ローカルパスを生成
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)  # ディレクトリが無い場合は作成
    
    download_path = os.path.join(local_dir, local_filename)


    backup_path = None

    #  既に同名ファイルがあれば、一時ファイル名にリネームして退避
    if os.path.exists(download_path):
        # 同じディレクトリ下で重複しない一時ファイル名を作る
        backup_path = download_path + f".old_{uuid.uuid4()}"
        print(f"[Info] Found existing file. Renaming: {download_path} -> {backup_path}")
        os.rename(download_path, backup_path)
    try:
        my_bucket.download_file(object_key, download_path)
        print(f"Downloaded s3://{BUCKET_NAME}/{object_key} to {download_path}")
    except ClientError as e:
        # ダウンロード失敗時：
        # 退避ファイルがあれば元の名前に戻す（データを保全）
        if backup_path and os.path.exists(backup_path):
            print(f"[Error] Download failed. Restoring original file: {backup_path} -> {download_path}")
            os.rename(backup_path, download_path)
        raise e
    else:
        # 成功したら退避ファイルを削除
        if backup_path and os.path.exists(backup_path):
            print(f"[Info] Download succeeded. Removing backup file: {backup_path}")
            os.remove(backup_path)

def get_master():
    """
    マスタ類のバージョン情報ファイルをローカルとS3で参照し、
    差分のある場合に必要なマスタをダウンロードする。
    プロジェクトマスタもローカルとS3で参照できるようにしておき、
    必要になったタイミングで差分を検証してプロジェクトデータをダウンロードする。
    """
    download_s3_object("master/master_version_s3.json", os.path.join(DATA_PATH, "master"), "master_version_s3.json")
    json_path = os.path.join(DATA_PATH, "master/master_version.json")
    with open(json_path, 'r', encoding='utf-8') as f:
        master_version_local = json.load(f)
    json_path = os.path.join(DATA_PATH, "master/master_version_s3.json")
    with open(json_path, 'r', encoding='utf-8') as f:
        master_version_s3 = json.load(f)
    for key in master_version_s3:
        if key not in master_version_local or master_version_s3[key]>master_version_local[key]:
            download_s3_object(f"master/{key}", "master", key)
    download_s3_object("master/projects_master_s3.csv", os.path.join(DATA_PATH, "master"), "projects_master_s3.csv")

def get_project_data(pj_name):
    project_master = pd.read_csv(os.path.join(DATA_PATH, "master", "projects_master.csv"), index_col=0)
    project_master_s3 = pd.read_csv(os.path.join(DATA_PATH, "master", "projects_master_s3.csv"), index_col=0)
    if pj_name in project_master_s3.index and (pj_name not in project_master.index or project_master_s3.loc[pj_name,"updated_at"]>project_master.loc[pj_name,"updated_at"]):
        download_prefix_from_s3(f"projects/{pj_name}", os.path.join(DATA_PATH, pj_name))
        project_master.loc[pj_name] = project_master_s3.loc[pj_name]
        project_master.to_csv(os.path.join(DATA_PATH, "master", "projects_master.csv"))

def put_project_data(pj_name):
    #プロジェクトマスタの該当プロジェクト以外の行をアップデートしてしまうと嘘になるので注意
    project_master = pd.read_csv(os.path.join(DATA_PATH, "master", "projects_master.csv"), index_col=0)
    project_master_s3 = pd.read_csv(os.path.join(DATA_PATH, "master", "projects_master_s3.csv"), index_col=0)
    if pj_name not in project_master_s3.index or project_master.loc[pj_name,"updated_at"]>project_master_s3.loc[pj_name,"updated_at"] or project_master.loc[pj_name,"updated_at"]==project_master.loc[pj_name,"created_at"]:
        project_master_s3.loc[pj_name] = project_master.loc[pj_name]
        project_master_s3.to_csv(os.path.join(DATA_PATH, "master", "projects_master_s3.csv"))
        upload_directory_to_s3(os.path.join(DATA_PATH, "projects", pj_name), f"projects/{pj_name}")
        #プロジェクトマスタもアップロード
        upload_s3_file("master", "projects_master_s3.csv", os.path.join(DATA_PATH, "master", "projects_master_s3.csv"))