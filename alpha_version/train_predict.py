import tensorflow as tf
import cv2
from alpha_version import train_func
# load tensorflow model

sess = tf.Session()
saver = tf.train.import_meta_graph('model/model.ckpt.meta')
saver.restore(sess, 'model/model.ckpt')

graph = tf.get_default_graph()
images_batch = graph.get_tensor_by_name("images_batch:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")

class_prediction = graph.get_tensor_by_name('class_prediction:0')

x_data, y_data, o = train_func.set_data('test_data')
pred_label = sess.run([class_prediction], feed_dict={images_batch: x_data, keep_prob:1.0})
print(pred_label, y_data)

for i in o :
    cv2.imshow('a',i)
    cv2.waitKey(0)

#
# [1, 1, 1, 1, 1, 0]
# [1, 0, 1, 1, 1, 0]