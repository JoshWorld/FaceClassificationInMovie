from keras.models import load_model
from keras.utils import CustomObjectScope
import tensorflow as tf
import cv2
import numpy as np


def get_embedding_vector_func(image_path):
    with CustomObjectScope({'tf': tf}):
        model = load_model('models/nn4.small2.lrn.h5')
        img = cv2.imread(image_path)
        img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_CUBIC)
        o_img = img.copy()
        img = img[..., ::-1]
        img = np.around(np.transpose(img, (0, 1, 2)) / 255.0, decimals=12)
        img = np.array([img])
        y = model.predict_on_batch(img)
        return y, o_img
