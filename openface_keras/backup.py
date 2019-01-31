from keras.models import load_model
from keras.utils import CustomObjectScope
import tensorflow as tf
import cv2
import numpy as np


def load_img(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_CUBIC)
    img = img[..., ::-1]
    img = np.around(np.transpose(img, (0, 1, 2)) / 255.0, decimals=12)
    img = np.array([img])
    return img


with CustomObjectScope({'tf': tf}):

    model = load_model('model/nn4.small2.lrn.h5')

    img1 = load_img('C:\\Users\\ADMIN\\PycharmProjects\\FaceClassificationInMovie\\image_data\\test_data\\hwang\\6.JPG')
    img2 = load_img('data/lee_jungjae/lee_jungjae_0009.jpg')

    y = model.predict_on_batch(img1)
    y2 = model.predict_on_batch(img2)

    dist = np.linalg.norm(y - y2)

    print(y)
    print('y2', y2)
    print(dist)