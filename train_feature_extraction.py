import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet

nb_classes = 43

# Load traffic signs data.
with open('train.p', 'rb') as f:
    data = pickle.load(f)

# Split data into training and validation sets.
X_train, X_val, y_train, y_val = train_test_split(data['features'], data['labels'], test_size=0.33, random_state=0)


# Define placeholders and resize operation.
input_x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
input_y = tf.placeholder(tf.int64, shape=[None])
resized = tf.image.resize_images(input_x, [227, 227])


# pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)

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
loss = tf.nn.softmax_cross_entropy_with_logits(labels=input_y, logits=out)
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)

preds = tf.arg_max(out, 1)
accuracy_op = tf.reduce_mean(tf.cast(tf.equal(preds, input_y), tf.float32))


# TODO: Train and evaluate the feature extraction model.
def train():
    pass


def eval():
    pass
