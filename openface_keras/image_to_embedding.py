from keras.models import load_model
from keras.utils import CustomObjectScope
import tensorflow as tf
import cv2
import numpy as np
import os


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
            for file_name in image_list:
                image_path = image_base_dir + os.sep + dir_name + os.sep + file_name
                image = load_img(image_path)
                x = model.predict_on_batch(image)
                features.append(x)

                if class_index == 0:
                    labels.append([0,1])
                else:
                    labels.append([1,0])

        from random import shuffle

        c = list(zip(features, labels))
        shuffle(c)
        features, labels = zip(*c)

        features = np.array(features, dtype=np.float32).squeeze()

        labels = np.array(labels)

        print('ll', labels[0])

    return features, labels

