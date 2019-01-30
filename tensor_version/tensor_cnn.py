from tensor_version.tesnor_read_input import *
import tensorflow as tf

# set train_data
IMG_HEIGHT = 80
IMG_WIDTH = 80
NUM_CHANNEL = 3
NUM_CLASS = 2

# input tensor
images_batch = tf.placeholder(dtype=tf.float32,
                              shape=[None, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNEL])
labels_batch = tf.placeholder(dtype=tf.int32, shape=[None, 2])  # one hot
keep_prob = tf.placeholder(tf.float32)


# cnn layer function
def con_layer(input_tensor, filter_size, in_channels, out_channels, layer_name):
    with tf.variable_scope(layer_name):
        filt = tf.get_variable(name="filter", shape=[filter_size, filter_size, in_channels, out_channels], dtype=tf.float32)
        bias = tf.get_variable(name='bias', shape=[out_channels], dtype=tf.float32)
        pre_activate = tf.nn.conv2d(input_tensor, filt, [1, 1, 1, 1], padding='SAME') + bias
        activated = tf.nn.relu(pre_activate)
        return activated


# max pool function
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# dense layer function
def dense_layer(input_tensor, input_dim, output_dim, layer_name, act=True):
    with tf.variable_scope(layer_name):
        weights = tf.get_variable(name="weight", shape=[input_dim, output_dim])
        biases = tf.get_variable(name='bias', shape=[output_dim])
        preactivate = tf.matmul(input_tensor, weights) + biases
        if act:
            return tf.nn.relu(preactivate)
        return preactivate


input_tensor_shape = images_batch.get_shape()[1:]

# graph
conv1 = con_layer(images_batch, 5, input_tensor_shape[2], 32, 'con_layer1')
h_pool1 = max_pool_2x2(conv1)
conv2 = con_layer(h_pool1, 5, 32, 64, 'con_layer2')
h_pool2 = max_pool_2x2(conv2)

fc_size = input_tensor_shape[0]//4*input_tensor_shape[1]//4*64
h_pool2_flat = tf.reshape(h_pool2, [-1, fc_size])

fc1 = dense_layer(h_pool2_flat, fc_size, 1024, 'dense1')
h_fc1_drop = tf.nn.dropout(fc1, keep_prob)

y_pred = dense_layer(h_fc1_drop, 1024, NUM_CLASS, 'dense2', act=False)

class_prediction = tf.argmax(y_pred, 1, output_type=tf.int32)
correct_prediction = tf.equal(class_prediction,
                              tf.argmax(labels_batch, 1, output_type=tf.int32))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=labels_batch)
loss_mean = tf.reduce_mean(loss)
train_op = tf.train.AdamOptimizer().minimize(loss_mean)

# training
train_features, train_labels, _, _, _ = load_data('train_data', IMG_HEIGHT, IMG_WIDTH, 1.0)
test_features, test_labels, _, _, _ = load_data('test_data', IMG_HEIGHT, IMG_WIDTH, 1.0)

train_labels_one_hot = tf.squeeze(tf.one_hot(train_labels, 2))
test_labels_one_hot = tf.squeeze(tf.one_hot(test_labels, 2))

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    iter_ = train_data_iterator(train_features, train_labels_one_hot.eval(session=sess))
    for step in range(1000):
        images_batch_val, labels_batch_val = next(iter_)
        accuracy_, _, loss_val = sess.run([accuracy, train_op, loss_mean],
                                          feed_dict={
                                              images_batch: images_batch_val,
                                              labels_batch: labels_batch_val,
                                              keep_prob: 0.5
                                          })
        print('Iteration {}: ACC={}, LOSS={}'.format(step, accuracy_, loss_val))

    print('Test begins….')

    class_prediction_val, loss_val, accuracy_ = sess.run([class_prediction, loss_mean, accuracy], feed_dict={
                            images_batch: test_features,
                            labels_batch: test_labels_one_hot.eval(session=sess),
                            keep_prob: 1.0
                            })

    print(class_prediction_val, test_labels, accuracy_, loss_val)


# print('Test begins….')
# TEST_BSIZE = 50
# for i in range(int(len(test_features)/TEST_BSIZE)):
#     images_batch_val = test_features[i*TEST_BSIZE:(i+1)*TEST_BSIZE]
#     labels_batch_val = test_labels_one_hot.eval(session=sess)[i*TEST_BSIZE:(i+1)*TEST_BSIZE]
#
#     loss_val, accuracy_ = sess.run([loss_mean, accuracy], feed_dict={
#                         images_batch: images_batch_val,
#                         labels_batch: labels_batch_val,
#                         keep_prob: 1.0
#                         })
#     print('ACC = {}, LOSS = {}'.format(accuracy_, loss_val))
