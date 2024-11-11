import os
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
import faiss

openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    OpenAI.api_key = openai_api_key
client = OpenAI()

import pandas as pd
import numpy as np

import Levenshtein

BATCH_SIZE = 1000
DIMENTION_EMBEDDING = 100
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
DIR_PATH = os.path.dirname(__file__)
DATA_PATH = DIR_PATH +"/../../data"
data = pd.read_csv(os.path.join(DATA_PATH, "master/idolname_embedding_data.csv"))
key_name = "idol_group_name"

#ベクトル化する関数
def get_embedding(text, model=EMBEDDING_MODEL_NAME, dim=DIMENTION_EMBEDDING):
    response = client.embeddings.create(input=text, model=model, dimensions=dim)
    return response.data[0].embedding

#バッジデータをベクトル化する関数
def get_embedding_batch(input_text, dimention=DIMENTION_EMBEDDING, batch_size=BATCH_SIZE, model=EMBEDDING_MODEL_NAME):
    if type(input_text) == str:
        input_text = [input_text]
    embeddings = []
    for batch_start in range(0, len(input_text), batch_size):
        batch_end = batch_start + batch_size
        batch = input_text[batch_start:batch_end]
        try:
            response = client.embeddings.create(
                model=model,
                input=batch,
                dimensions=dimention
            )
            embeddings.extend([e.embedding for e in response.data])
        except Exception as e:
            print(f"Error: {e}")
            return None
    return embeddings

#ベクトルデータベースの作成
embeddings = data.drop(key_name,axis=1).values
d = len(embeddings[0])  # 次元数
index = faiss.IndexFlatL2(d)
index.add(embeddings)

#類似するデータのベクトル検索による出力（上位k個）
def find_similar_vector(text, k=3):
    embedding = np.array([get_embedding(text)]).astype('float32')
    distances, indices = index.search(embedding, k)
    return indices[0], distances[0]

#類似するデータのベクトル検索による出力（上位k個）（idxではなく実名で出力）
def find_similar_vector_returnidol(text, k=3):
    embedding = np.array([get_embedding(text)]).astype('float32')
    distances, indices = index.search(embedding, k)
    return [data.iloc[idx][key_name] for idx in indices[0]], distances[0]

#類似するデータのベクトル検索による出力（候補の中から出力）（上位k個）
def find_similar_vector_inlist(text, candidate_list, k=1):
    data_candidate = data[data.apply(lambda row:row[key_name] in candidate_list, axis=1)].reset_index(drop=True)
    embeddings = data_candidate.drop(key_name,axis=1).values
    d = len(embeddings[0])  # 次元数
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    embedding = np.array([get_embedding(text)]).astype('float32')
    distances, indices = index.search(embedding, k)
    return indices[0], distances[0]

#マスタにないデータを追加
def add_new_data(candidate_list):
    global data
    no_data_idol = []
    idol_group_list = list(data[key_name])
    for idol_name in candidate_list:
        if idol_name not in idol_group_list:
            no_data_idol.append(idol_name)
    add_embeddings = get_embedding_batch(no_data_idol)
    new_data = pd.concat((pd.DataFrame(no_data_idol,columns=["idol_group_name"]),pd.DataFrame(add_embeddings)),axis=1)
    new_data.columns = [str(col) for col in new_data.columns]
    data = pd.concat((data,new_data))

#類似するデータのベクトル検索による出力（候補の中から出力）（候補は元リストになくてもOK）（上位k個）（idxではなく実名で出力)
def find_similar_vector_inlist_returnidol(text, candidate_list, k=1):
    #元リストに存在していない候補があった場合には追加でembeddingする
    add_new_data(candidate_list)
    data_candidate = data[data.apply(lambda row:row[key_name] in candidate_list, axis=1)].reset_index(drop=True)
    embeddings = data_candidate.drop(key_name,axis=1).values
    d = len(embeddings[0])  # 次元数
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    embedding = np.array([get_embedding(text)]).astype('float32')
    distances, indices = index.search(embedding, k)
    if k==1:
        return data_candidate.iloc[indices[0][0]][key_name]
    else:
        return [data_candidate.iloc[idx][key_name] for idx in indices[0]], distances[0]

#類似するデータの編集距離順による出力（編集距離の小さいk個）（同率でk個を超える場合に全部出力するかフラグ）
def find_similar_levenshtein(text, k=3, same_output=False):
    idol_group_list = list(data[key_name])
    distances = [(group, Levenshtein.distance(text, group)) for group in idol_group_list]
    distances.sort(key=lambda x: x[1])# 距離が小さい順にソート
    if same_output:#同率は全部出力する
        threshold_distance = distances[k-1][1]  # k個目の距離を取得
        closest_groups = [group for group, distance in distances if distance <= threshold_distance]
        closest_distances = [distance for group, distance in distances if distance <= threshold_distance]
    else:#個数を優先し同率は並び順次第で足切りする
        closest_groups = [group for group, _ in distances[:k]]
        closest_distances = [distance for _, distance in distances[:k]]    
    return closest_groups, closest_distances

#類似するデータの編集距離順による出力（編集距離の閾値以下を出力）（複数候補がある場合に最も近いものだけを出力するフラグ）
def find_similar_levenshtein_distance(text, edit_distance=3, nearest_only=False):
    idol_group_list = list(data[key_name])
    distances = [(group, Levenshtein.distance(text, group)) for group in idol_group_list]
    distances.sort(key=lambda x: x[1])# 距離が小さい順にソート
    if nearest_only:
        threshold_distance = distances[0][1]
        if threshold_distance>edit_distance:
            return [],[]
        else:
            closest_groups = [group for group, distance in distances if distance <= threshold_distance]
            closest_distances = [distance for group, distance in distances if distance <= threshold_distance]
    else:
        closest_groups = [group for group, distance in distances if distance <= edit_distance]
        closest_distances = [distance for group, distance in distances if distance <= edit_distance]
    return closest_groups, closest_distances

#類似するデータの編集距離順による出力（編集距離（文字数比）の閾値以下を出力）（複数候補がある場合に最も近いものだけを出力するフラグ）
def find_similar_levenshtein_distance_rate(text, edit_distance_r=0.4, nearest_only=False):
    idol_group_list = list(data[key_name])
    distances = [(group, Levenshtein.distance(text, group)/(len(text)+len(group))*2) for group in idol_group_list]
    distances.sort(key=lambda x: x[1])# 距離が小さい順にソート
    if nearest_only:
        threshold_distance = distances[0][1]
        if threshold_distance>edit_distance_r:
            return [],[]
        else:
            closest_groups = [group for group, distance in distances if distance <= threshold_distance]
            closest_distances = [distance for group, distance in distances if distance <= threshold_distance]
    else:
        closest_groups = [group for group, distance in distances if distance <= edit_distance_r]
        closest_distances = [distance for group, distance in distances if distance <= edit_distance_r]
    return closest_groups, closest_distances

#類似するデータの編集距離順による出力（候補の中から出力）（idxではなく実名で出力）（編集距離（文字数比）の閾値以下を出力）（複数候補がある場合に最も近いものだけを出力するフラグ）
def find_similar_levenshtein_inlist_returnidol(text, candidate_list, edit_distance_r=0.4, nearest_only=False):
    distances = [(group, Levenshtein.distance(text, group)/(len(text)+len(group))*2) for group in candidate_list]
    distances.sort(key=lambda x: x[1])# 距離が小さい順にソート
    if nearest_only:
        threshold_distance = distances[0][1]
        if threshold_distance>edit_distance_r:
            return [],[]
        else:
            closest_groups = [group for group, distance in distances if distance <= threshold_distance]
            closest_distances = [distance for group, distance in distances if distance <= threshold_distance]
    else:
        closest_groups = [group for group, distance in distances if distance <= edit_distance_r]
        closest_distances = [distance for group, distance in distances if distance <= edit_distance_r]
    return closest_groups, closest_distances

#候補を適切な数出力する関数
def get_name_list_by_vector(text, search_num=1):#return (bool(完全一致があったか), 名前候補(リスト))
    if search_num == 1:
        indices, distances = find_similar_vector(text, 1)
        if distances[0]==0:
            return (True, data.iloc[indices[0]][key_name])
        else:
            return (False, data.iloc[indices[0]][key_name])
    else:
        indices, distances = find_similar_vector(text, search_num)
        dist_before = 0
        name_list = []
        for i, dist in zip(indices, distances):
            if dist==0:
                return (True, [data.iloc[i][key_name]])
            elif dist - dist_before > 0.5:
                break
            else:
                name_list.append(data.iloc[i][key_name])
                dist_before = dist
        return (False, name_list)
    
def get_name_by_levenshtein_and_vector(text, r=0.4):
    #①編集距離÷文字数がr以下の候補をリストアップ
    groups, _ = find_similar_levenshtein_distance_rate(text, edit_distance_r=r, nearest_only=True)
    if len(groups)==1:#②候補の中で最も距離が小さいものが1つだけならそれを採用
        return groups[0]
    elif len(groups)>1:#③候補の中で最も距離が小さいものが複数あるなら、AIでベクトル化して候補複数の中から距離最小のものを採用
        return find_similar_vector_inlist_returnidol(text, groups, k=1)
    else:#④候補が1つも無ければAIでベクトル化して全グループの中から距離最小のものを採用
        return find_similar_vector_returnidol(text, 1)[0][0]
    
def get_name_by_inlist(text, candidate_list, r=0.7):
    # print(text)
    #①編集距離÷文字数がr以下の候補をリストアップ
    groups, _ = find_similar_levenshtein_inlist_returnidol(text, candidate_list, edit_distance_r=r, nearest_only=True)
    if len(groups)==1:#②候補の中で最も距離が小さいものが1つだけならそれを採用
        # print(groups[0],"に決定")
        return groups[0]
    elif len(groups)>1:#③候補の中で最も距離が小さいものが複数あるなら、AIでベクトル化して候補複数の中から距離最小のものを採用
        # print(groups)
        return find_similar_vector_inlist_returnidol(text, groups, k=1)
    else:#④候補が1つも無ければAIでベクトル化して全グループの中から距離最小のものを採用
        # print("候補なし")
        return find_similar_vector_inlist_returnidol(text, candidate_list, 1)
