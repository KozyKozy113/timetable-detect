from dotenv import load_dotenv
load_dotenv()
import os
import glob

import urllib
import http.client, urllib.request, urllib.parse
import urllib.error, base64
import ast

import time
import datetime as dt
import json

def main():
    doc_folder = os.path.dirname(__file__) + "/../docs/"
    api_conn = getTextRead(doc_folder)
    files = glob.glob(doc_folder + "input/image_setlist/*.jpeg")
    for file in files:
        if not(os.path.exists(file.replace("input/image_setlist","output/ocr_result_raw").replace("jpeg","json"))):
            print(file.split("\\")[-1])
            api_conn.execute_runGetReadText(file)

class getTextRead:
    SUBSCRIPTION_KEY = os.getenv('COG_SERVICE_KEY')
    ENDPOINT = os.getenv('COG_SERVICE_ENDPOINT')

    # ホストを設定
    host = ENDPOINT.split("/")[2]
    # vision-v3.2のread機能のURLを設定
    text_recognition_url = (ENDPOINT + "vision/v3.2/read/analyze")

    # 読み取り用のヘッダー作成
    read_headers = {
        # サブスクリプションキーの設定
        "Ocp-Apim-Subscription-Key":SUBSCRIPTION_KEY,
        # bodyの形式を指定、json=URL/octet-stream=バイナリデータ
        "Content-Type":"application/octet-stream"
    }

    # 結果取得用のヘッダー作成
    result_headers = {
        # サブスクリプションキーの設定
        "Ocp-Apim-Subscription-Key":SUBSCRIPTION_KEY,
    }

    def __init__(self,doc_folder):
        self.doc_folder = doc_folder

   # Read APIを呼ぶ関数
    def call_read_api(self, body, params):
        # Read APIの呼び出し
        try:
            conn = http.client.HTTPSConnection(self.host)
            # 読み取りリクエスト
            conn.request(
                method = "POST",
                url = self.text_recognition_url + "?%s" % params,
                body = body,
                headers = self.read_headers,
            )

            # 読み取りレスポンス
            read_response = conn.getresponse()
            # print(dir(read_response))'begin', 'chunk_left', 'chunked', 'close', 'closed', 'code', 'debuglevel', 'detach', 'fileno', 'flush', 'fp', 'getcode', 'getheader', 'getheaders', 'geturl', 'headers', 'info', 'isatty', 'isclosed', 'length', 'msg', 'peek', 'read', 'read1', 'readable', 'readinto', 'readinto1', 'readline', 'readlines', 'reason', 'seek', 'seekable', 'status', 'tell', 'truncate', 'version', 'will_close', 'writable', 'write', 'writelines'
            print(read_response.headers)
            print(read_response.status)

            # レスポンスの中から読み取りのOperation-Location URLを取得
            OL_url = read_response.headers["Operation-Location"]

            conn.close()
            print("read_request:SUCCESS")

        except Exception as e:
            print("[ErrNo {0}]{1}".format(e.errno,e.strerror))

        return OL_url

    # OCR結果を取得する関数
    def call_get_read_result_api(self, file_name, OL_url):
        result_dict = {}
        # Read結果取得
        try:
            conn = http.client.HTTPSConnection(self.host)

            # 読み取り完了/失敗時にFalseになるフラグ
            poll = True
            while(poll):
                if (OL_url == None):
                    print(file_name + ":None Operation-Location")
                    break

                # 読み取り結果取得
                conn.request(
                    method = "GET",
                    url = OL_url,
                    headers = self.result_headers,
                )
                result_response = conn.getresponse()
                result_str = result_response.read().decode()
                result_dict = ast.literal_eval(result_str)

                if ("analyzeResult" in result_dict):
                    poll = False
                    print("get_result:SUCCESS")
                elif ("status" in result_dict and 
                    result_dict["status"] == "failed"):
                    poll = False
                    print("get_result:FAILD")
                else:
                    time.sleep(10)
            conn.close()

        except Exception as e:
            print("[ErrNo {0}] {1}".format(e.errno,e.strerror))

        return result_dict

    #OCRを実行して結果を保存する関数
    def execute_runGetReadText(self, file_name):
        tgt_name = file_name.split("\\")[-1]
        # body作成
        body = open(file_name,"rb").read()

        # パラメータの指定
        # 自然な読み取り順序で出力できるオプションを追加
        params = urllib.parse.urlencode({
            # Request parameters
            'readingOrder': 'natural',
        })

        # readAPIを呼んでOperation Location URLを取得
        ol_url = self.call_read_api(body, params)
        if (ol_url == None):
            print(file_name + ":None Operation-Location")
            return None
        print(ol_url)

        # 処理待ち10秒
        time.sleep(10)

        # Read結果取得
        result_dict = self.call_get_read_result_api(file_name, ol_url)

        # OCR結果を保存
        if result_dict != {}:
            output_json_file = self.doc_folder + "output/ocr_result_raw/{file_name}.json".format(file_name=tgt_name.split(".")[0])
            with open(output_json_file,"w",encoding = "utf8") as f:
                json.dump(result_dict,f, indent = 3, ensure_ascii = False)

if __name__ == "__main__":
    main()