# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import pathlib

# ------
# ジェスチャデータをファイルからロードして、特徴を配列に変換する
# ------

recog_action_label = ['bloom', 'huwahuwa', 'throwup']   # 認識させたい動作, 正事例

# ジェスチャデータをファイルからロードする
def load_json_data(filePath):
    """
    jsonファイルからデータを読み込む関数

    Args:
        filePath: ファイル

    Returns:
        jsonデータ
    """

    with open(filePath) as f:
        df = json.load(f)
        return df

# ジェスチャデータ特徴を配列に変換する
def generate_labeldata(json_data):

    gesture_data = json_data['gestureData']
    gesture_feature = []

    for frame_data in gesture_data:
        gesture_feature.append(frame_data['triangleArea'])
        gesture_feature.append(frame_data['triangleNormal']['x'])
        gesture_feature.append(frame_data['triangleNormal']['y'])
        gesture_feature.append(frame_data['triangleNormal']['z'])
    
    return gesture_feature


# データパスからラベルの配列を生成する
def load_labels_fromDir(dirPath):
    labels = os.listdir(dirPath)
    return labels

# 学習にかける形に、ラベルと特徴を出力する
def generate_learning_datas(data_dir_path):

    label_list = load_labels_fromDir(data_dir_path)

    label_learning_datas = []
    feature_learning_datas = []

    for label in label_list:
        p_temp = pathlib.Path(data_dir_path + "/" + label).glob('*.json')
        for json_file in p_temp:

            # ラベルが正事例の場合はそのまま追加、そうでない場合はその他とする
            if label in recog_action_label:
                label_learning_datas.append(label)
            else:
                label_learning_datas.append("other")

            # 特徴データを配列に入れていく
            json_data = load_json_data(json_file)
            feature_data = generate_labeldata(json_data)
            feature_learning_datas.append(feature_data)

    return label_learning_datas, feature_learning_datas



