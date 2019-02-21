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


def get_min_idx(tmp_list):
    min_idx = 0
    min = tmp_list[0]
    for idx, item in enumerate(tmp_list):
        if item < min:
            min_idx = idx
            min = item
    return min_idx, min


def calc_min_distance(v1_list, v2_list):
    match_list = []
    for i in range(0, len(v1_list)):
        min_dist = calc_vector_distance(v1_list[0], v2_list[0])
        min_idx = 0
        for j in range(0, len(v2_list)):
            dist = calc_vector_distance(v1_list[i], v2_list[j])
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

cap = cv2.VideoCapture("C:\\Users\\ADMIN\\PycharmProjects\\FaceClassificationInMovie\\test_video\\blackpink01.mp4")

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
            frame_list = []  # [ {'center':(123,21), 'embedding':[123,123,123]} ]
            face_group = []  # [ {'label':'object0', 'center':(123,21), 'embedding':[123,123,123]}, {'label':'object1', 'center':(124,15), 'embedding':[120,100,140]} ]

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

                    if score_val > 0.25 and class_val == 1:
                        h = image.shape[0]
                        w = image.shape[1]

                        y_min = int(h * box_val[0])
                        x_min = int(w * box_val[1])

                        y_max = int(h * box_val[2])
                        x_max = int(w * box_val[3])

                        center_x = int(x_min + ((x_max - x_min) / 2))
                        center_y = int(y_min + ((y_max - y_min) / 2))



                        crop_img = image.copy()[y_min:y_max, x_min:x_max]

                        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)


                        t = cv2.resize(crop_img, (96, 96), interpolation=cv2.INTER_CUBIC)
                        t = t[..., ::-1]
                        t = np.around(np.transpose(t, (0, 1, 2)) / 179.0, decimals=12)
                        t = np.array([t])

                        embedding_vector = model.predict_on_batch(t)

                        face_dict = {'score_val': score_val,
                                     'min': np.array([x_min, y_min]),
                                     'max': np.array([x_max, y_max]),
                                     'center': np.array([center_x, center_y]),
                                     'embedding_vector': embedding_vector}

                        face_list.append(face_dict)

                if len(face_group) == 0:
                    for i in face_list:
                        face_group.append([i])

                else:

                    for face in face_list:
                        tmp_dis_group = []
                        for group in face_group:
                            dis = 0
                            for item in group:

                                dis = dis + calc_vector_distance(item['embedding_vector'], face['embedding_vector'])
                                print("dis",calc_vector_distance(item['embedding_vector'], face['embedding_vector']))

                            m_dis = dis/len(group)
                            print("mdis",m_dis)
                            tmp_dis_group.append(m_dis)
                        print('')


                        min_idx, min_value = get_min_idx(tmp_dis_group)


                        if min_value < 0.9:
                            if min_idx == 1:
                                cv2.circle(image, (face['center'][0],face['center'][1]), 10, (0, 0, 255),-1)
                                cv2.rectangle(image, tuple(face['min']), tuple(face['max']), (0, 0, 255), 2)

                            else:
                                cv2.circle(image, (face['center'][0], face['center'][1]), 10, (255, 0, 0), -1)
                                cv2.rectangle(image, tuple(face['min']), tuple(face['max']), (255, 0, 0), 2)

                            face_group[min_idx].append(face)
                        else:
                            face_group.append([face])

                    print('face_group',len(face_group))


                # frame_list.append(face_list)
                #
                # # compare pre_frame and current_frame
                # if frame_index != 0:
                #     current_face_positions = np.array([item['center'] for item in frame_list[frame_index]])
                #     pre_face_positions = np.array([item['center'] for item in frame_list[frame_index-1]])
                #     result = calc_min_distance(pre_face_positions, current_face_positions)
                #     print(result)
                #
                #     for item in result:
                #         if item['match_index'][0] == 1:
                #             cv2.circle(image, (current_face_positions[item['match_index'][1]][0], current_face_positions[item['match_index'][1]][1]), 10, (0, 0, 255), -1)
                #         else:
                #
                #             cv2.circle(image, (current_face_positions[item['match_index'][1]][0],current_face_positions[item['match_index'][1]][1]), 10, (255, 0, 0),-1)
                #

                cv2.imshow('t', image)
                cv2.imwrite('blackpink_error.jpg',image)
                cv2.waitKey(0)

                frame_index = frame_index + 1


            cap.release()
