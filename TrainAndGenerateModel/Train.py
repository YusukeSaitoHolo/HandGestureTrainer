import LoadGestureData as loader


#学習データの読み込み・生成
datapath = "../GestureData"
label_data, feature_data = loader.generate_learning_datas(datapath)
print("feature data loaded. ")

import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.model_selection
import sklearn.metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

feature_train, feature_test = train_test_split(feature_data, test_size=0.25)
label_train, label_test = train_test_split(label_data, test_size=0.25)

print('-------random forest---------')
from sklearn.ensemble import RandomForestClassifier#ランダムフォレスト

classifier = RandomForestClassifier()
classifier.fit(feature_data, label_data)
pred = classifier.predict(feature_test) # テストデータを分類
print('score: {:.2%}'.format(classifier.score(feature_test, label_test)))
print(confusion_matrix(label_test, pred))   # 混同行列を出力

### データ保存

import pickle
with open('handgesture_v001.pickle', mode='wb') as fp:
    pickle.dump(classifier, fp)

# Pipelineに変換
from sklearn.preprocessing import Normalizer
normalizer = Normalizer()

from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(normalizer, classifier)
pipeline.fit(feature_data, label_data)

# ONNXモデルへコンバート
from skl2onnx import convert_sklearn

from winmltools.convert.common.data_types import FloatTensorType, StringTensorType, Int64TensorType
options = {id(pipeline): {'zipmap': False}}
rf_onnx = convert_sklearn(pipeline,
                            target_opset=7,
                            options=options,
                            name = 'RandomForestClassifier',
                            initial_types=[('input', FloatTensorType([1, 120]))],
                            )


# ONNXモデルを保存
from winmltools.utils import save_model
save_model(rf_onnx, 'handgesture_v001.onnx')