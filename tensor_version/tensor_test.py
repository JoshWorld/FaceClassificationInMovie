import tensorflow as tf
import numpy as np

# [[1. 0.]
#  [0. 1.]
#  [1. 0.]
#  [0. 1.]]


input = np.array([0, 1, 0, 1])
one_hot = tf.one_hot(input, 2)
squeeze = tf.squeeze(one_hot)

sess = tf.Session()
b = sess.run(squeeze)
print(b)
