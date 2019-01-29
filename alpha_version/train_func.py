import cv2
import numpy as np
import os
from random import shuffle


def load_image(path):
    img = cv2.imread(path)
    original_img = img.copy()
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (80, 80), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32)
    #img = (img - np.mean(img)) / np.std(img)
    img = img/255
    return img, original_img


# face train function
def set_data(root_dir):

    features = []
    labels = []
    original_images = []

    dir_list = os.listdir(root_dir)

    for cls_index, dir_name in enumerate(dir_list):
        image_dir = os.listdir(root_dir + os.sep + dir_name)

        for image_file in image_dir:

            image, original_image = load_image(root_dir + os.sep + dir_name + os.sep + image_file)
            features.append(image)
            labels.append(cls_index)
            original_images.append(original_image)

    features = np.array(features)
    labels = np.array(labels)

    # shuffle labels
    c = list(zip(features, labels, original_images))
    shuffle(c)
    features, labels, original_images = zip(*c)

    features = np.array(features)
    labels = np.array(labels)

    return features, labels, original_images


def train_data_iterator(feature_data, label_data, original_img, BATCH_SIZE):
    while True:
        idxs = np.arange(0, len(feature_data))
        np.random.shuffle(idxs)
        shuf_feature_data = feature_data[idxs]
        shuf_label_data = label_data[idxs]
        shuf_original_img = np.array(original_img)[idxs]


        batch_size = BATCH_SIZE

        for batch_idx in range(0,len(feature_data), batch_size):
            images_batch = shuf_feature_data[batch_idx:batch_idx+batch_size]
            labels_batch = shuf_label_data[batch_idx:batch_idx + batch_size]
            original_images_batch = shuf_original_img[batch_idx:batch_idx + batch_size]

            # for label, item in zip(labels_batch, original_images_batch):
            #     cv2.imshow(str(label), item)
            #     cv2.waitKey(0)

            yield images_batch, labels_batch
