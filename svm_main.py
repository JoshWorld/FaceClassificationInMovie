
# 분류되어진 학습데이터를 가지고 SVM 모델을 이용해 학습 후
# 테스트 데이터 셋으로 해당 모델을 테스트 하는 모듈

import pickle
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
from keras.models import load_model
from keras.utils import CustomObjectScope
import tensorflow as tf
import cv2
import numpy as np


# 학습  (학습 데이터 경로)
# SVM 을 이용해 학습을 진행하고 학습된 모델과 학습데이터를 반환하는 함수
def train_svm(data_path):
    f = open(data_path, 'rb')  # 학습 데이터 불러오기
    groups = pickle.load(f)  # 학습 데이터 load

    data = []  # 학습 데이터
    Y = []  # 정답
    for group in groups:
        for item in group:
            try:
                data.append(item['embedding_vector'])
                Y.append(item['group_idx'])
            except:
                pass

    X = np.array(data).squeeze()
    y = np.array(Y)

    clf = SVC(kernel='linear', probability=True)
    clf.fit(X, y)  # 학습 진행
    return clf, groups  # 학습 된 모델과 학습데이터 반환


# 결과 예측 함수 (이미지 폴더경로, 학습된 모델)
# 테스트 이미지를 embedding vector 로 변경하고, 학습된 SVM 모델을 이용해 해당 이미지가
# 어떤 결과를 갖는지 예측하는 함수
def predict_image(image_dir_base, svm_model):
    with CustomObjectScope({'tf': tf}):
        model = load_model('models/nn4.small2.lrn.h5')  # image to embedding vector 모델

        image_dir_list = os.listdir(image_dir_base)
        predict_data = []  # 예측 결과 리스트
        for class_index, dir_name in enumerate(image_dir_list):
            image_list = os.listdir(image_dir_base + os.sep + dir_name)
            for file_name in image_list:
                img_path = image_dir_base + os.sep + dir_name + os.sep + file_name

                # image to embedding vector 를 위한 데이터 변환
                img = cv2.imread(img_path)
                img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_CUBIC)
                o_img = img.copy()
                img = img[..., ::-1]
                img = np.around(np.transpose(img, (0, 1, 2)) / 255.0, decimals=12)
                img = np.array([img])

                v = model.predict_on_batch(img)  # image to embedding vector

                predict = svm_model.predict([np.array(v).squeeze()])  # 해당 embedding vector 가 어떤 결과인지 예측
                predict_data.append({'predict_label': predict, 'real_label': class_index, 'image': o_img})

    return predict_data


# 데이터들을 이미지로 보여주는 함수
def show_data(train_groups, predict_data):
    fig = plt.figure(figsize=(10, 10))
    for i in range(1, len(predict_data)):
        ax = fig.add_subplot(5, 5, i)
        ax.set_title('p : ' + str(predict_data[i]['predict_label']).replace('[','').replace(']','') + ' / r :' + str(predict_data[i]['real_label']), fontdict={'fontsize': 15, 'fontweight': 'medium'})
        ax.imshow(cv2.cvtColor(predict_data[i]['image'], cv2.COLOR_BGR2RGB))
        ax.set_xticks([])
        ax.set_yticks([])

    plt.figure()
    plt.show()


if __name__ == '__main__':
    clf, groups = train_svm('face_data_blackpink.data')  # 분류된 데이터 셋으로 학습 진행
    p_data = predict_image('C:\\Users\\ADMIN\\Desktop\\FaceClassificationInMovie\\test_data\image', clf)
    show_data(groups, p_data)

