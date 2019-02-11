from keras.models import load_model
from keras.utils import CustomObjectScope
import tensorflow as tf
import cv2
import numpy as np


def func(img):

    img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_CUBIC)
    img = img[..., ::-1]
    img = np.around(np.transpose(img, (0, 1, 2)) / 255.0, decimals=12)
    img = np.array([img])

    with CustomObjectScope({'tf': tf}):

        model = load_model('model/nn4.small2.lrn.h5')

        y = model.predict_on_batch(img)
        print(y)


