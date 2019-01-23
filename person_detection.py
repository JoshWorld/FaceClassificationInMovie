import numpy as np

import tensorflow as tf
import cv2
from utils import label_map_util
from utils import visualization_utils as vis_util


cap = cv2.VideoCapture('C:\\Users\\ADMIN\PycharmProjects\\FaceClassificationInMovie\\test_video\\sinsegae2.mp4')  # Change only if you have more than one webcams

PATH_TO_CKPT = 'models/object_detection_graph.pb'
PATH_TO_LABELS = 'labels/mscoco_label_map.pbtxt'

NUM_CLASSES = 1

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

c = 0
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            ret, image_np = cap.read()

            image_np_expanded = np.expand_dims(image_np, axis=0)

            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')

            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            for score_val, box_val, class_val in zip(np.squeeze(scores), np.squeeze(boxes), np.squeeze(classes)):
                if score_val > 0.8 and class_val == 1:  # score > 0.9 and person
                    h = image_np.shape[0]
                    w = image_np.shape[1]

                    y_min = int(h*box_val[0])
                    x_min = int(w* box_val[1])

                    y_max = int(h*box_val[2])
                    x_max = int(w*box_val[3])

                    cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                    crop_img = image_np.copy()[y_min:y_max,x_min:x_max]

                    gray = cv2.cvtColor(crop_img.copy(), cv2.COLOR_BGR2GRAY)
                    t = np.max(cv2.convertScaleAbs(cv2.Laplacian(gray, 3)))
                    if t >= 150:
                        cv2.imwrite("alpha_version/frame/frame%d.jpg" % c,crop_img)
                        c = c + 1

                    #cv2.imshow('t',image_np[y_min:y_max,x_min:x_max])
                    #cv2.waitKey(0)
                    #print(score_val,box_val, class_val)

            # Visualization of the results of a detection.
            # vis_util.visualize_boxes_and_labels_on_image_array(
            #     image_np,
            #     np.squeeze(boxes),
            #     np.squeeze(classes).astype(np.int32),
            #     np.squeeze(scores),
            #     category_index,
            #     use_normalized_coordinates=True,
            #     line_thickness=6,
            #     min_score_thresh=0.8)

            # Display output
            cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
