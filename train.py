# read codes and labels from file
import csv
import numpy as np
import tensorflow as tf
from tensorflow_vgg import utils
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from tensorflow_vgg import vgg16

with open('labels') as f:
    reader = csv.reader(f, delimiter='\n')
    labels = np.array([each for each in reader if len(each) > 0]).squeeze()
with open('codes') as f:
    codes = np.fromfile(f, dtype=np.float32)
    codes = codes.reshape((len(labels), -1))
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
lb.fit(labels)

labels_vecs = lb.transform(labels)

#接下来就是抽取数据，因为不同类型的花的数据数量并不是完全一样的，而且 labels 数组中的数据也还没有被打乱，
#所以最合适的方法是使用 StratifiedShuffleSplit 方法来进行分层随机划分。假设我们使用训练集：验证集：测试集 = 8:1:1，那么代码如下：
from sklearn.model_selection import StratifiedShuffleSplit

ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

train_idx, val_idx = next(ss.split(codes, labels))

half_val_len = int(len(val_idx)/2)
val_idx, test_idx = val_idx[:half_val_len], val_idx[half_val_len:]

train_x, train_y = codes[train_idx], labels_vecs[train_idx]
val_x, val_y = codes[val_idx], labels_vecs[val_idx]
test_x, test_y = codes[test_idx], labels_vecs[test_idx]

print("Train shapes (x, y):", train_x.shape, train_y.shape)
print("Validation shapes (x, y):", val_x.shape, val_y.shape)
print("Test shapes (x, y):", test_x.shape, test_y.shape)

#Train shapes (x, y): (2936, 4096) (2936, 5)
#Validation shapes (x, y): (367, 4096) (367, 5)
#Test shapes (x, y): (367, 4096) (367, 5)
# train Network

# dimensions of the input data
inputs_ = tf.placeholder(tf.float32, shape=[None, codes.shape[1]])
# Dimensions of tag data
labels_ = tf.placeholder(tf.int64, shape=[None, labels_vecs.shape[1]])

# Join a 256-dimensional fully connected layer
fc = tf.contrib.layers.fully_connected(inputs_, 256)

# Add a 5-dimensional fully connected layer
logits = tf.contrib.layers.fully_connected(fc, labels_vecs.shape[1], activation_fn=None)

# Calculate the cross entropy value
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels_, logits=logits)

# Calculated loss function
cost = tf.reduce_mean(cross_entropy)

# Use the most widely used AdamOptimizer optimizer
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Get the final predicted distribution
predicted = tf.nn.softmax(logits)

# Calculation accuracy
correct_pred = tf.equal(tf.argmax(predicted, 1), tf.argmax(labels_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In order to facilitate the division of data into batches to reduce the use of memory
def get_batches(x, y, n_batches=10):
    """ This is a generator function that divides the data into small chunks according to the size of n_batches. """
    batch_size = len(x) // n_batches

    for ii in range(0, n_batches * batch_size, batch_size):
        if ii != (n_batches - 1) * batch_size:
            X, Y = x[ii: ii + batch_size], y[ii: ii + batch_size]
        else:
            X, Y = x[ii:], y[ii:]
        yield X, Y


# epochs
epochs = 20
# Frequency of statistical training effects
iteration = 0
# Model saver
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for x, y in get_batches(train_x, train_y):
            feed = {inputs_: x,
                    labels_: y}
            # train
            loss, _ = sess.run([cost, optimizer], feed_dict=feed)
            print("Epoch: {}/{}".format(e + 1, epochs),
                  "Iteration: {}".format(iteration),
                  "Training loss: {:.5f}".format(loss))
            iteration += 1

            if iteration % 5 == 0:
                feed = {inputs_: val_x,
                        labels_: val_y}
                val_acc = sess.run(accuracy, feed_dict=feed)
                # Output verification machine to verify training progress
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Validation Acc: {:.4f}".format(val_acc))
    # save model
    saver.save(sess, "checkpoints/flowers.ckpt")


