import pickle
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
from keras.models import load_model
from keras.utils import CustomObjectScope
import tensorflow as tf
import cv2
import numpy as np


def train_svm(data_path):
    f = open(data_path, 'rb')
    groups = pickle.load(f)

    data = []
    Y = []
    for group in groups:
        for item in group:
            try:
                Y.append(item['group_idx'])
                data.append(item['embedding_vector'])
            except:
                pass

    X = np.array(data).squeeze()
    y = np.array(Y)

    clf = SVC(kernel='linear', probability=True)
    clf.fit(X, y)
    return clf, groups


def predict_image(image_dir_base, svm_model):
    with CustomObjectScope({'tf': tf}):
        model = load_model('models/nn4.small2.lrn.h5')

        image_dir_list = os.listdir(image_dir_base)

        predict_data = []

        for class_index, dir_name in enumerate(image_dir_list):
            image_list = os.listdir(image_dir_base + os.sep + dir_name)
            for file_name in image_list:
                img_path = image_dir_base + os.sep + dir_name + os.sep + file_name

                img = cv2.imread(img_path)
                img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_CUBIC)
                o_img = img.copy()

                img = img[..., ::-1]
                img = np.around(np.transpose(img, (0, 1, 2)) / 255.0, decimals=12)
                img = np.array([img])
                v = model.predict_on_batch(img)

                predict = svm_model.predict([np.array(v).squeeze()])
                predict_data.append({'predict_label': predict, 'real_label': class_index, 'image': o_img})

    return predict_data


def show_data(train_groups, predict_data):

    fig, axes = plt.subplots(1, len(predict_data))

    for i in range(0, len(predict_data)):
        axes[i].set_title(str(predict_data[i]['predict_label']).replace('[','').replace(']','') + ' / ' + str(predict_data[i]['real_label']))
        axes[i].imshow(cv2.cvtColor(predict_data[i]['image'], cv2.COLOR_BGR2RGB))
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    plt.figure(figsize=(96, 96))
    plt.show()


if __name__ == '__main__':
    clf, groups = train_svm('face_data_blackpink.data')
    p_data = predict_image('C:\\Users\\ADMIN\\Desktop\\FaceClassificationInMovie\\test_data\image', clf)
    show_data(groups, p_data)

