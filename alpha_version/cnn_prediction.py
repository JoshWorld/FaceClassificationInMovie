import tensorflow as tf
from alpha_version import tensor_func
import cv2
import numpy as np

# load tensorflow model

sess = tf.Session()
saver = tf.train.import_meta_graph('model/model.ckpt.meta')
saver.restore(sess, 'model/model.ckpt')

graph = tf.get_default_graph()
images_batch = graph.get_tensor_by_name("images_batch:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")

class_prediction = graph.get_tensor_by_name('class_prediction:0')

# open video
video = cv2.VideoCapture('C:\\Users\ADMIN\\PycharmProjects\\FaceClassificationInMovie\\test_video\\sinsegae2.mp4')




while video.isOpened():
    ret, frame = video.read()
    if ret is True:
        slicing_img = tensor_func.image_slicing(frame, frame.shape[0], frame.shape[1], 80)
        pred_label = sess.run([class_prediction], feed_dict={images_batch: slicing_img, keep_prob:1.0})
        print(pred_label)

        # red mask
        red_image = np.zeros((80, 80, 3), np.uint8)
        red_image[:] = (0, 0, 255)

        # overlap red mask
        c = 0
        for j in range(1, 10):
            for i in range(1, 17):
                c_begin = (i - 1) * 80
                c_end = i * 80
                r_begin = (j - 1) * 80
                r_end = j * 80
                if pred_label[0][c] == 1:
                    frame[r_begin:r_end, c_begin:c_end, :] = red_image
                c = c + 1

        cv2.imshow('test',frame)
        cv2.waitKey(1)
    else:
        break

