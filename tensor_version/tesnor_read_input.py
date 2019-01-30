import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib

BATCH_SIZE = 10


def load_image(addr, img_height, img_width):
    img = cv2.imread(addr)
    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_img = img.copy()
    img = img.astype(np.float32)
    img = (img - np.mean(img)) / np.std(img)
    return img, original_img


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


def set_figure(images, titles, cols=10):

    plt.figure(1)
    matplotlib.use('TkAgg')

    columns = 10
    rows = 10
    fig = plt.figure(figsize=(10,10))

    for i in range(1, len(titles) + 1):
        img = images[i-1]
        fig.add_subplot(rows, columns, i)
        plt.xlabel('real : {}'.format(str(titles[i-1])))
        plt.xticks([])

        plt.yticks([])
        plt.subplots_adjust(wspace=0, hspace=0.7)

        plt.imshow(img)

        if i == 100:
            break
    plt.show()


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
