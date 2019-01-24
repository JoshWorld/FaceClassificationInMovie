import tensorflow as tf
import cv2

# load tensorflow model

sess = tf.Session()
saver = tf.train.import_meta_graph('model/model.ckpt.meta')
saver.restore(sess, 'model/model.ckpt')

graph = tf.get_default_graph()
images_batch = graph.get_tensor_by_name("images_batch:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")

class_prediction = graph.get_tensor_by_name('class_prediction:0')



pred_label = sess.run([class_prediction], feed_dict={images_batch: slicing_img, keep_prob:1.0})
print(pred_label)

