import tensorflow as tf
import cv2
from utils import label_map_util
from keras.models import load_model
from keras.utils import CustomObjectScope
import numpy as np


def max_value(my_list, key):
    max = 0
    for item in my_list:
        if max < item[key]:
            max = item[key]
    return [False for _ in range(max+1)]


def calc_vector_distance(v1, v2):
    dist = np.linalg.norm(v1 - v2)
    return dist


PATH_TO_CKPT = 'models/face_detection_graph.pb'
PATH_TO_LABELS = 'labels/face_label_map.pbtxt'

NUM_CLASSES = 2

PERSON_DETECTION_RATE = 0.2
NEW_FACE_RATE = 1.5
E_DISTANCE_RATE = 0.999
P_DISTANCE_RATE = 0.001


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

cap = cv2.VideoCapture("C:\\Users\\ADMIN\\PycharmProjects\\FaceClassificationInMovie\\test_video\\blackpink01.mp4")

ret, frame = cap.read()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_apply_distance.mp4',fourcc, 20.0, (frame.shape[1],frame.shape[0]))

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

                    if score_val > PERSON_DETECTION_RATE and class_val == 1:
                        h = image.shape[0]
                        w = image.shape[1]

                        y_min = int(h * box_val[0])
                        x_min = int(w * box_val[1])

                        y_max = int(h * box_val[2])
                        x_max = int(w * box_val[3])

                        center_x = int(x_min + ((x_max - x_min) / 2))
                        center_y = int(y_min + ((y_max - y_min) / 2))

                        crop_img = image.copy()[y_min:y_max, x_min:x_max]

                        t = cv2.resize(crop_img, (96, 96), interpolation=cv2.INTER_CUBIC)
                        t = t[..., ::-1]
                        t = np.around(np.transpose(t, (0, 1, 2)) / 255, decimals=12)
                        t = np.array([t])

                        embedding_vector = model.predict_on_batch(t)

                        face_dict = {'min': np.array([x_min, y_min]),
                                     'max': np.array([x_max, y_max]),
                                     'center': np.array([center_x, center_y]),
                                     'embedding_vector': embedding_vector}

                        face_list.append(face_dict)

                if len(face_group) == 0:
                    for i in face_list:
                        face_group.append([i])
                else:

                    face_info = []  # [ [ face와 group간 비교한 distance 결과] , [], [] ]
                    for face_idx, face in enumerate(face_list):
                        for group_idx, group in enumerate(face_group):
                            e_dis = 0  # embedding vector distance
                            p_dis = 0  # position vector distance
                            for item in group:
                                e_dis = e_dis + calc_vector_distance(item['embedding_vector'], face['embedding_vector'])
                                p_dis = p_dis + calc_vector_distance(item['center'], face['center'])
                            e_m_dis = e_dis / len(group)
                            p_m_dis = p_dis / len(group)

                            face_info.append({'face_idx': face_idx, 'group_idx': group_idx,
                                              'e_distance': e_m_dis*E_DISTANCE_RATE, 'p_distance': p_m_dis*P_DISTANCE_RATE, 'sum': e_m_dis*E_DISTANCE_RATE + p_m_dis*P_DISTANCE_RATE,
                                              'center': face['center'], 'min': face['min'], 'max': face['max']})

                    m = np.array([item['sum'] for item in face_info])
                    m = (m - np.mean(m)) / np.std(m)
                    for mean, f in zip(m, face_info):
                        f['result'] = mean

                    sort_face = sorted(face_info, key=lambda k: k['result'])

                    group_check = max_value(sort_face, 'group_idx')
                    face_check = max_value(sort_face, 'face_idx')
                    color_list = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (0, 153, 255)]

                    for item in sort_face:
                        if (face_check[item['face_idx']] == False) and (group_check[item['group_idx']] == False):

                            if item['e_distance'] < NEW_FACE_RATE:

                                cv2.circle(image, (item['center'][0], item['center'][1]), 10, color_list[item['group_idx']], -1)
                                cv2.rectangle(image, tuple(item['min']), tuple(item['max']), color_list[item['group_idx']], 2)
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                cv2.putText(image, "result: " + str(item['result']) + " / group_idx : " + str(item['group_idx']), tuple(item['max']), font, 0.5, color_list[item['group_idx']], 1, cv2.LINE_AA)

                                face_check[item['face_idx']] = True
                                group_check[item['group_idx']] = True

                            else:
                                face_group.append([item])

                cv2.imshow('test', image)
                out.write(image)
                cv2.waitKey(1)

                frame_index = frame_index + 1

            cap.release()
            out.release()
