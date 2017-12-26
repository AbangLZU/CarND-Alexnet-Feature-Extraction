import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
from sklearn.utils import shuffle
import time

nb_classes = 43
batch_size = 64
epochs = 20

# Load traffic signs data.
with open('train.p', 'rb') as f:
    data = pickle.load(f)

# Split data into training and validation sets.
X_train, X_val, y_train, y_val = train_test_split(data['features'], data['labels'], test_size=0.33, random_state=0)


# Define placeholders and resize operation.
input_x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
input_y = tf.placeholder(tf.int64, shape=None)
resized = tf.image.resize_images(input_x, [227, 227])


# pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
fc7 = tf.stop_gradient(fc7)
# dd the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix
fc8W = tf.Variable(tf.truncated_normal(shape, mean=0.01))
fc8b = tf.Variable(tf.truncated_normal([nb_classes]))
out = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
probs = tf.nn.softmax(out)


# Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
one_hot = tf.one_hot(input_y, nb_classes)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot, logits=out)
loss_op = tf.reduce_mean(cross_entropy)
opt = tf.train.AdamOptimizer()
train_op = opt.minimize(loss_op, var_list=[fc8W, fc8b])
init_op = tf.global_variables_initializer()
preds = tf.arg_max(out, 1)
accuracy_op = tf.reduce_mean(tf.cast(tf.equal(preds, input_y), tf.float32))


# Train and evaluate the feature extraction model.
def eval(x, y, sess):
    total_acc = 0
    total_loss = 0
    for offset in range(0, x.shape[0], batch_size):
        end = offset + batch_size
        X_batch = x[offset:end]
        y_batch = y[offset:end]

        l, acc = sess.run([loss_op, accuracy_op], feed_dict={input_x: X_batch, input_y: y_batch})
        total_loss += (l * X_batch.shape[0])
        total_acc += (acc * X_batch.shape[0])

    return total_loss / x.shape[0], total_acc / y.shape[0]


with tf.Session() as sess:
    sess.run(init_op)

    for i in range(epochs):
        # training
        X_train, y_train = shuffle(X_train, y_train)
        t0 = time.time()
        for offset in range(0, X_train.shape[0], batch_size):
            end = offset + batch_size
            sess.run(train_op, feed_dict={input_x: X_train[offset:end], input_y: y_train[offset:end]})

        val_loss, val_acc = eval(X_val, y_val, sess)
        print("Epoch", i + 1)
        print("Time: %.3f seconds" % (time.time() - t0))
        print("Validation Loss =", val_loss)
        print("Validation Accuracy =", val_acc)
        print("")

