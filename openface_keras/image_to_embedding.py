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


def load_data(image_dir_base, image_height, image_width, train_rate):
    image_dir_list = os.listdir(image_dir_base)

    features = []
    labels = []
    original_features = []

    for class_index, dir_name in enumerate(image_dir_list):
        image_list = os.listdir(image_dir_base + os.sep + dir_name)
        for file_name in image_list:
            image, original_image = load_image(image_dir_base + os.sep + dir_name + os.sep + file_name, image_height, image_width)
            features.append(image)
            labels.append(class_index)
            original_features.append(original_image)

    from random import shuffle

    c = list(zip(features, labels, original_features))
    shuffle(c)
    features, labels, original_features = zip(*c)

    features = np.array(features)
    labels = np.array(labels)

    train_features = features[0:int(train_rate * len(features))]
    train_labels = labels[0:int(train_rate * len(labels))]
    train_original_features = original_features[0:int(train_rate*len(original_features))]

    if train_rate == 1.0:
        set_figure(train_original_features, train_labels)

    test_features = features[int(train_rate * len(features)):]
    test_labels = labels[int(train_rate * len(labels)):]

    return train_features, train_labels, train_original_features, test_features, test_labels


def image_to_embedding(image_base_dir):
    with CustomObjectScope({'tf': tf}):

        model = load_model('model/nn4.small2.lrn.h5')

        image_dir_list = os.listdir(image_base_dir)

        features = []
        labels = []
        original_features = []

        for class_index, dir_name in enumerate(image_dir_list):
            image_list = os.listdir(image_base_dir + os.sep + dir_name)
            for file_name in image_list:
                image_path = image_base_dir + os.sep + dir_name + os.sep + file_name
                image = load_img(image_path)
                x = model.predict_on_batch(image)


                features.append(image)
                labels.append(class_index)
                original_features.append(original_image)

    
        y = model.predict_on_batch(img1)
        y2 = model.predict_on_batch(img2)

        dist = np.linalg.norm(y - y2)

        print(y)
        print('y2', y2)
        print(dist)

