import numpy as np
import os
import cv2

BATCH_SIZE = 50


def load_image(addr, img_height, img_width):
    img = cv2.imread(addr)
    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = (img - np.mean(img)) / np.std(img)
    return img


def load_data(image_dir_base):
    image_dir_list = os.listdir(image_dir_base)

    features = []
    labels = []

    for class_index, dir_name in enumerate(image_dir_list):
        image_list = os.listdir(image_dir_base + os.sep + dir_name)
        for file_name in image_list:
            image = load_image(image_dir_base + os.sep + dir_name + os.sep + file_name)
            features.append(image)
            labels.append(class_index)

    from random import shuffle

    c = list(zip(features, labels))
    shuffle(c)
    features, labels = zip(*c)

    features = np.array(features)
    labels = np.array(labels)

    train_features = features[0:int(0.8 * len(features))]
    train_labels = labels[0:int(0.8 * len(labels))]
    test_features = features[int(0.8 * len(features)):]
    test_labels = labels[int(0.8 * len(labels)):]

    print(train_labels)

    return train_features, train_labels, test_features, test_labels


def train_data_iterator(features, labels):
    while True:
        print('New epoch begins...')
        idxs = np.arange(0, len(features))
        np.random.shuffle(idxs)
        shuf_features = features[idxs]
        shuf_labels = labels[idxs]
        batch_size = BATCH_SIZE

        for batch_idx in range(0, len(features), batch_size):
            images_batch = shuf_features[batch_idx:batch_idx+batch_size]
            labels_batch = shuf_labels[batch_idx:batch_idx+batch_size]
            yield images_batch, labels_batch
