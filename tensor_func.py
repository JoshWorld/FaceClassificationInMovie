import cv2
import numpy as np
import os
from random import shuffle


def image_slicing(image, h, w, p):
    image = image.astype(np.float32)
    slicing_image_list = []
    for j in range(1, 10):
        for i in range(1, 17):
            c_begin = (i - 1) * 80
            c_end = i * 80
            r_begin = (j - 1) * 80
            r_end = j * 80
            image_frag = image[r_begin:r_end, c_begin:c_end, :]
            slicing_image_list.append(image_frag/255)

    slicing_image_list = np.array(slicing_image_list)
    return slicing_image_list


# image and label load func
def images_and_labels_load(dir_path):
    # features load
    feature_data = []
    path = os.listdir(dir_path)
    for filename in path:
        full_filename = os.path.join(dir_path, filename)
        img = cv2.imread(full_filename, cv2.IMREAD_GRAYSCALE) # read image gray scale
        img = img.astype(np.float32).ravel() / 255  # type casting and image ravel
        feature_data.append(img)

    # labels load
    f = open('label.txt', 'r')
    tmp = f.readlines()
    label_data = [int(i.replace('\n', '')) for i in tmp]
    feature_data = np.array(feature_data)
    label_data = np.array(label_data)

    # shuffle data
    c = list(zip(feature_data, label_data))
    shuffle(c)
    feature_data, label_data = zip(*c)
    feature_data = np.array(feature_data)
    label_data = np.array(label_data)

    # train, test data
    train_feature_data = feature_data[:int(len(feature_data)*0.8)]
    train_label_data = label_data[:int(len(label_data)*0.8)]
    test_feature_data = feature_data[int(len(feature_data) * 0.8):]
    test_label_data = label_data[int(len(label_data) * 0.8):]

    return train_feature_data, train_label_data, test_feature_data, test_label_data


# image and label load func for cnn
def cnn_images_and_labels_load(dir_path):
    # features load
    feature_data = []
    path = os.listdir(dir_path)
    for filename in path:
        full_filename = os.path.join(dir_path, filename)
        img = cv2.imread(full_filename)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)


        #img = (img - np.mean(img)) / np.std(img)
        feature_data.append(img/255)


    # labels load
    f = open('label.txt', 'r')
    tmp = f.readlines()
    label_data = [int(i.replace('\n', '')) for i in tmp]
    feature_data = np.array(feature_data)
    label_data = np.array(label_data)

    # shuffle data
    c = list(zip(feature_data, label_data))
    shuffle(c)
    feature_data, label_data = zip(*c)
    feature_data = np.array(feature_data)
    label_data = np.array(label_data)

    # train, test data
    train_feature_data = feature_data[:int(len(feature_data)*0.8)]
    train_label_data = label_data[:int(len(label_data)*0.8)]
    test_feature_data = feature_data[int(len(feature_data) * 0.8):]
    test_label_data = label_data[int(len(label_data) * 0.8):]

    return train_feature_data, train_label_data, test_feature_data, test_label_data


# batch train func
def train_data_iterator(feature_data, label_data, BATCH_SIZE):
    while True:
        idxs = np.arange(0, len(feature_data))
        np.random.shuffle(idxs)
        shuf_feature_data = feature_data[idxs]
        shuf_label_data = label_data[idxs]

        batch_size = BATCH_SIZE

        for batch_idx in range(0,len(feature_data), batch_size):
            images_batch = shuf_feature_data[batch_idx:batch_idx+batch_size]
            labels_batch = shuf_label_data[batch_idx:batch_idx + batch_size]
            yield images_batch, labels_batch
