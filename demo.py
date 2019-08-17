import os
import numpy as np
import tensorflow as tf
from tensorflow_vgg import vgg16
from tensorflow_vgg import utils

data_dir = 'flower_photos/'
contents = os.listdir(data_dir)
classes = [each for each in contents if os.path.isdir(data_dir + each)]

#Using vgg16 to calculate the eigenvalue
batch_size = 10
# use codes_list store eigenvalue
codes_list = []
# Use labels to store the category of flowers
labels = []
# Batch array is used to temporarily store image data
batch = []
codes = None

with tf.Session() as sess:
    # Building a VGG16 model object
    vgg = vgg16.Vgg16()
    input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
    with tf.name_scope("content_vgg"):
        # Load VGG16 model
        vgg.build(input_)

    # Calculate the eigenvalues with VGG16 for each different flower
    for each in classes:
        print("Starting {} images".format(each))
        class_path = data_dir + each
        files = os.listdir(class_path)
        for ii, file in enumerate(files, 1):
            # Load the image and put it in the batch array
            img = utils.load_image(os.path.join(class_path, file))
            batch.append(img.reshape((1, 224, 224, 3)))
            labels.append(each)

            # If the number of pictures reaches batch_size, start the specific operation.
            if ii % batch_size == 0 or ii == len(files):
                images = np.concatenate(batch)

                feed_dict = {input_: images}
                # cal eigenvalue
                codes_batch = sess.run(vgg.relu6, feed_dict=feed_dict)

                # put results
                if codes is None:
                    codes = codes_batch
                else:
                    codes = np.concatenate((codes, codes_batch))

                # reset batch,and start another
                batch = []
                print('{} images processed'.format(ii))

# The codes array, the labels array, store the eigenvalues and categories of all the flowers, respectively.
with open('codes', 'w') as f:
    codes.tofile(f)
import csv
with open('labels', 'w') as f:
 writer = csv.writer(f, delimiter='\n')
 writer.writerow(labels)


