import tensorflow as tf
from alpha_version import train_func
import cv2
import os

BATCH_SIZE = 10
NUM_CLASS = 2  # ouput / 2->0,1 / 4 -> 0,1,2,3
NUM_CHANNEL = 3  # R G B

IMG_HEIGHT = 80
IMG_WEIGHT = 80

MODEL_SAVE_DIR = 'model'

x_train, y_train, original_images = train_func.set_data('train_data')  # image load for cnn

images_batch = tf.placeholder(dtype=tf.float32, shape=[None, IMG_HEIGHT, IMG_WEIGHT, NUM_CHANNEL], name="images_batch")
labels_batch = tf.placeholder(dtype=tf.int32, shape=[None, ], name="labels_batch")


def con_layer(input_tensor, filter_size, in_channels, out_channels, layer_name):
    with tf.variable_scope(layer_name):
        filt = tf.get_variable(name="filter", shape=[filter_size, filter_size, in_channels, out_channels], dtype=tf.float32)
        bias = tf.get_variable(name='bias', shape=[out_channels], dtype=tf.float32)
        pre_act = tf.nn.conv2d(input_tensor, filt, [1,1,1,1], padding='SAME') + bias
        activated = tf.nn.relu(pre_act)
        return activated


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def dense_layer(input_tensor, input_dim, output_dim, layer_name, act=True):
    with tf.variable_scope(layer_name):
        weights = tf.get_variable(name="weight", shape=[input_dim, output_dim])
        biases = tf.get_variable(name='bias', shape=[output_dim])
        preact = tf.matmul(input_tensor, weights) + biases
        if act:
            return tf.nn.relu(preact)
        return preact


input_shape = images_batch.get_shape()[1:]

# cnn layer
conv1 = con_layer(images_batch, 5, input_shape[2], 32, 'con_layer1') # (input_tensor, filter_size, in_channels, out_channels, layer_name)
h_pool1 = max_pool_2x2(conv1)
conv2 = con_layer(h_pool1, 5, 32, 64, 'con_layer2')
h_pool2 = max_pool_2x2(conv2)
fc_size = input_shape[0]//(2**2) * input_shape[1]//(2**2) * 64

h_pool2_flat = tf.reshape(h_pool2, [-1, fc_size])
fc1 = dense_layer(h_pool2_flat, fc_size, 1024, 'dense1')

# drop out layer
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
h_fc1_drop = tf.nn.dropout(fc1, keep_prob)

# prediction and accuracy
y_pred = dense_layer(h_fc1_drop, 1024, NUM_CLASS, 'dense2', act=False)
y_pred = tf.identity(y_pred, "ypred")

class_prediction = tf.argmax(y_pred, 1, output_type=tf.int32, name='class_prediction')
correct_prediction = tf.equal(class_prediction, labels_batch)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=labels_batch)
loss_mean = tf.reduce_mean(loss)
train_op = tf.train.AdamOptimizer(0.001).minimize(loss_mean)

# For model save
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

iter_ = train_func.train_data_iterator(x_train, y_train, original_images, BATCH_SIZE)
for step in range(100):
    images_batch_val, labels_batch_val = next(iter_)
    accuracy_, _, loss_val = sess.run([accuracy, train_op, loss_mean],feed_dict={images_batch:images_batch_val,
                                                                                 labels_batch:labels_batch_val,
                                                                                 keep_prob: 0.5 })
    print('Iteration {}: ACC={}, LOSS={}'.format(step, accuracy_, loss_val))


print('Training Finished....')


x_test, y_test, original_img_list = train_func.set_data('C:\\Users\\ADMIN\\PycharmProjects\\FaceClassificationInMovie\\alpha_version\\test_data')  # image load for cnn

loss_, loss_val, accuracy_, class_prediction_ = sess.run([loss, loss_mean, accuracy, class_prediction],
                                 feed_dict={images_batch: x_test,
                                            labels_batch: y_test,
                                            keep_prob: 1.0})

print(loss_)
print('ACC = {}, LOSS = {}, pred_label = {}, real_label = {}'.format(accuracy_, loss_val, class_prediction_, y_test))

for real, pred, item in zip(y_test, class_prediction_, original_img_list):
    cv2.imshow(str(pred) + " real : " + str(real), item)
    cv2.waitKey(0)

#save_path = saver.save(sess, MODEL_SAVE_DIR + os.sep + 'model.ckpt')
#print('Model saved in file : {}'.format(save_path))

# hwang = 2
# lee_train = 1
# choi = 0

# for pre_label, image_ in zip(y_test, original_img_list):
#     cv2.imshow(str(pre_label), image_)
#     cv2.waitKey(0)
#
