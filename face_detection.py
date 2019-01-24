#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import numpy as np
import tensorflow as tf
import cv2
from utils import label_map_util

PATH_TO_CKPT = 'models/face_detection_graph.pb'
PATH_TO_LABELS = 'labels/face_label_map.pbtxt'

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

cap = cv2.VideoCapture("C:\\Users\\ADMIN\\PycharmProjects\\FaceClassificationInMovie\\test_video\\sinsegae2.mp4")

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
        c = 0
        while True:
            ret, image = cap.read()
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

                if score_val > 0.7 and class_val == 1:
                    h = image.shape[0]
                    w = image.shape[1]

                    y_min = int(h * box_val[0])
                    x_min = int(w * box_val[1])

                    y_max = int(h * box_val[2])
                    x_max = int(w * box_val[3])

                    #cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                    crop_img = image.copy()[y_min:y_max, x_min:x_max]
                    cv2.imwrite('sinsegae2_face/frame{}.jpg'.format(c),crop_img)
                    c = c + 1


        cap.release()

