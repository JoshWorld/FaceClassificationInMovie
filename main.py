#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import numpy as np
import tensorflow as tf
import cv2
from utils import label_map_util
from keras.models import load_model
from keras.utils import CustomObjectScope


def calc_vector_distance(v1, v2):
    dist = np.linalg.norm(v1 - v2)
    return dist


def calc_min_distance(v1_list, v2_list):
    match_list = []
    for i in range(1, len(v1_list)):
        min_dist = calc_vector_distance(A[0], B[0])
        min_idx = 0
        for j in range(1, len(v2_list)):
            dist = calc_vector_distance(A[i], B[j])
            if dist < min_dist:
                min_dist = dist
                min_idx = j
        match_list.append({'match_index':(i, min_idx),'min_dist':min_dist})

    # Normalization
    x = np.array([item['min_dist'] for item in match_list])
    y = x / sum(x)

    for item, norm_dist in zip(match_list, y):
        item['norm_min_dist'] = norm_dist
    return match_list


PATH_TO_CKPT = 'models/face_detection_graph.pb'
PATH_TO_LABELS = 'labels/face_label_map.pbtxt'

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

cap = cv2.VideoCapture("C:\\Users\\ADMIN\\PycharmProjects\\FaceClassificationInMovie\\test_video\\sample01.mp4")

detection_graph = tf.Graph()

with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


with detection_graph.as_default():

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(graph=detection_graph, config=config) as sess:
        with CustomObjectScope({'tf': tf}):
            model = load_model('models/nn4.small2.lrn.h5')

            frame_index = 0
            frame_list = []

            while True:
                ret, image = cap.read()
                face_list = []

                if ret == 0:
                    break

                image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
                                                                    feed_dict={image_tensor: image_np_expanded})

                for score_val, box_val, class_val in zip(np.squeeze(scores), np.squeeze(boxes), np.squeeze(classes)):

                    if score_val > 0.4 and class_val == 1:
                        h = image.shape[0]
                        w = image.shape[1]

                        y_min = int(h * box_val[0])
                        x_min = int(w * box_val[1])

                        y_max = int(h * box_val[2])
                        x_max = int(w * box_val[3])

                        center_x = int(x_min + ((x_max - x_min) / 2))
                        center_y = int(y_min + ((y_max - y_min) / 2))

                        cv2.circle(image, (center_x, center_y), 10, (0, 0, 255), -1)

                        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                        cv2.imshow('t', image)
                        cv2.waitKey(1)

                        crop_img = image.copy()[y_min:y_max, x_min:x_max]

                        t = cv2.resize(crop_img, (96, 96), interpolation=cv2.INTER_CUBIC)
                        t = t[..., ::-1]
                        t = np.around(np.transpose(t, (0, 1, 2)) / 255.0, decimals=12)
                        t = np.array([t])

                        embedding_vector = model.predict_on_batch(t)

                        face_dict = {'score_val': score_val,
                                     'min': np.array([x_min, y_min]),
                                     'max': np.array([x_max, y_max]),
                                     'center': np.array([center_x, center_y]),
                                     'embedding_vector': embedding_vector}

                        face_list.append(face_dict)

                frame_list.append(face_list)

                # compare pre_frame and current_frame
                for face_a in frame_list[frame_index]:
                    for face_b in frame_list[frame_index-1]:
                        face_a['center']

                frame_index = frame_index + 1




            cap.release()
