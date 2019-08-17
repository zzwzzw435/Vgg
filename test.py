import numpy as np
import tensorflow as tf
from tensorflow_vgg import utils
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from tensorflow_vgg import vgg16
import csv


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

# Train shapes (x, y): (2936, 4096) (2936, 5)
# Validation shapes (x, y): (367, 4096) (367, 5)
# Test shapes (x, y): (367, 4096) (367, 5)


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


test_img_path = 'flower_photos/dandelion/126012913_edf771c564_n.jpg'

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))

    feed = {inputs_: test_x,
            labels_: test_y}
    test_acc = sess.run(accuracy, feed_dict=feed)
    print("Test accuracy: {:.4f}".format(test_acc))

with tf.Session() as sess:
    input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16.Vgg16()
    vgg.build(input_)
with tf.Session() as sess:
    img = utils.load_image(test_img_path)
    img = img.reshape((1, 224, 224, 3))

    feed_dict = {input_: img}
    code = sess.run(vgg.relu6, feed_dict=feed_dict)


with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))

    feed = {inputs_: code}
    prediction = sess.run(predicted, feed_dict=feed).squeeze()

plt.barh(np.arange(5), prediction)
_ = plt.yticks(np.arange(5), lb.classes_)
plt.show()
