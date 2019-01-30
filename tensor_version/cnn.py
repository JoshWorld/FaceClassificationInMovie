from tensor_version.read_input import *
import tensorflow as tf

images_batch = tf.placeholder(dtype=tf.float32,
                              shape=[None, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNEL])
labels_batch = tf.placeholder(dtype=tf.int32, shape=[None, ])
keep_prob = tf.placeholder(tf.float32)


def con_layer(input_tensor, filter_size, in_channels, out_channels, layer_name):
    with tf.variable_scope(layer_name):
        filt = tf.get_variable(name="filter", shape=[filter_size, filter_size, in_channels, out_channels], dtype=tf.float32)
        bias = tf.get_variable(name='bias', shape=[out_channels], dtype=tf.float32)
        pre_activate = tf.nn.conv2d(input_tensor, filt, [1, 1, 1, 1], padding='SAME') + bias
        activated = tf.nn.relu(pre_activate)
        return activated


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def dense_layer(input_tensor, input_dim, output_dim, layer_name, act=True):
    with tf.variable_scope(layer_name):
        weights = tf.get_variable(name="weight", shape=[input_dim, output_dim])
        biases = tf.get_variable(name='bias', shape=[output_dim])
        preactivate = tf.matmul(input_tensor, weights) + biases
        if act:
            return tf.nn.relu(preactivate)
        return preactivate


input_tensor_shape = images_batch.get_shape()[1:]
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
correct_prediction = tf.equal(class_prediction, labels_batch)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=labels_batch)
loss_mean = tf.reduce_mean(loss)
train_op = tf.train.AdamOptimizer().minimize(loss_mean)

# To visualize default graph in tensorboard
LOG_DIR = 'tmp/logs'
writer = tf.summary.FileWriter(LOG_DIR)
writer.add_graph(tf.get_default_graph())
writer.flush()

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

CONTINUE_TRAIN = False
if CONTINUE_TRAIN is True:
    states = tf.train.get_checkpoint_state(LOG_DIR)
    if states is not None:
        saver.restore(sess, LOG_DIR + os.sep + "model.ckpt")

iter_ = train_data_iterator()
for step in range(70):
    # get a batch of data
    images_batch_val, labels_batch_val = next(iter_)
    accuracy_, _, loss_val = sess.run([accuracy, train_op, loss_mean],
                    feed_dict={
                        images_batch:images_batch_val,
                        labels_batch:labels_batch_val,
                        keep_prob: 0.5
                    })
    print('Iteration {}: ACC={}, LOSS={}'.format(step, accuracy_, loss_val))


print('Test begins….')
TEST_BSIZE = 50
for i in range(int(len(test_features)/TEST_BSIZE)):
    images_batch_val = test_features[i*TEST_BSIZE:(i+1)*TEST_BSIZE]
    labels_batch_val = test_labels[i*TEST_BSIZE:(i+1)*TEST_BSIZE]

    loss_val, accuracy_ = sess.run([loss_mean, accuracy], feed_dict={
                        images_batch:images_batch_val,
                        labels_batch:labels_batch_val,
                        keep_prob: 1.0
                        })
    print('ACC = {}, LOSS = {}'.format(accuracy_, loss_val))

# Add ops to save and restore all the variables.
save_path = saver.save(sess, LOG_DIR + os.sep + 'model.ckpt')
print('Model saved in file: {}'.format(save_path))
