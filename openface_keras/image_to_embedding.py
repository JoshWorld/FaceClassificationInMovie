from keras.models import load_model
from keras.utils import CustomObjectScope
import tensorflow as tf
import cv2
import numpy as np
import os

# https://github.com/iwantooxxoox/Keras-OpenFace/blob/master/Untitled.ipynb
# https://kangbk0120.github.io/articles/2018-01/face-net
# https://medium.com/@jongdae.lim/%EA%B8%B0%EA%B3%84-%ED%95%99%EC%8A%B5-machine-learning-%EC%9D%80-%EC%A6%90%EA%B2%81%EB%8B%A4-part-4-63ed781eee3c

def load_img(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_CUBIC)
    img = img[..., ::-1]
    img = np.around(np.transpose(img, (0, 1, 2)) / 255.0, decimals=12)
    img = np.array([img])
    return img


def train_data_iterator(features, labels, batch_size):
    while True:
        print('New epoch begins...')
        idxs = np.arange(0, len(features))
        np.random.shuffle(idxs)
        shuf_features = features[idxs]
        shuf_labels = labels[idxs]

        for batch_idx in range(0, len(features), batch_size):
            images_batch = shuf_features[batch_idx:batch_idx+batch_size]
            labels_batch = shuf_labels[batch_idx:batch_idx+batch_size]
            yield images_batch, labels_batch


def image_to_embedding(image_base_dir):
    with CustomObjectScope({'tf': tf}):

        model = load_model('model/nn4.small2.lrn.h5')
        image_dir_list = os.listdir(image_base_dir)
        features = []
        labels = []

        for class_index, dir_name in enumerate(image_dir_list):
            image_list = os.listdir(image_base_dir + os.sep + dir_name)
            if class_index == 0:
                continue
            c = 0

            for file_name in image_list:

                base_img = load_img('C:\\Users\\ADMIN\\PycharmProjects\\FaceClassificationInMovie\\openface_keras\\data\\hwang_jungmin\\hwang_jungmin_0001.jpg')
                b = model.predict_on_batch(base_img)

                image_path = image_base_dir + os.sep + dir_name + os.sep + file_name
                image = load_img(image_path)
                x = model.predict_on_batch(image)

                dist = np.linalg.norm(b - x)
                print(dist)

                if dist <= 1.1:
                    c = c + 1

                    print(c, image_path)

                features.append(x)
            break
                # if class_index == 0:
                #     labels.append([0,1])
                # else:
                #     labels.append([1,0])

        # from random import shuffle
        #
        # c = list(zip(features, labels))
        # shuffle(c)
        # features, labels = zip(*c)
        #
        # features = np.array(features, dtype=np.float32).squeeze()
        # labels = np.array(labels)

    return features, labels

image_to_embedding('data')