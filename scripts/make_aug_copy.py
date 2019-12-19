# Copyright 2019 Uizard Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl import app
from absl import flags
from shutil import copyfile
from libml import data, utils
from third_party.random_eraser_tf import get_random_eraser

import tensorflow as tf
import numpy as np
import os
import glob
import tqdm
import random

flags.DEFINE_string("aug_dataset", "", "Name of dataset to perform augmentation on.")
flags.DEFINE_integer("aug_copy", 1, "Number of augmented copies to make of the unlabeled data.")
flags.DEFINE_enum('augment', "cifar10", enum_values=["cifar10", "color", "cutout", "svhn", "stl10"], help="Types of augmentations that can be performed on the data.")
flags.DEFINE_integer("unlabel_size", 10000, "Size of unlabeled dataset.")
FLAGS = flags.FLAGS
DATA_DIR = os.environ['ML_DATA']

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_info(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={'image': tf.FixedLenFeature([], tf.string),
                  'label': tf.FixedLenFeature([], tf.int64)})
    return tf.cast(tf.image.decode_png(features['image']), tf.float32), features['label']

def reshape(x):
    return tf.reshape(x, [32, 32, 3])

def augment_cifar10(image, label):
    return data.augment_shift(data.augment_mirror(reshape(image)), 4), label

def augment_cutout(image, label):
    eraser = get_random_eraser(p=0.3)
    return eraser(image), label

def augment_color(image, label):
    return data.augment_shift(data.augment_color_func(data.augment_mirror(image)), 4), label

def augment_stl10(image, label):
    return data.augment_shift(data.augment_mirror(image), 12), label

def augment_svhn(image, label):
    return data.augment_shift(image, 4), label

def save_as_tfrecord(images, labels, filename):
    assert len(images) == len(labels)
    # filename = os.path.join(DATA_DIR + "/SSL", filename)
    print('Saving tfrecord with ' + str(FLAGS.aug_copy) + ' augmentations performed: ', filename)
    with tf.python_io.TFRecordWriter(filename) as writer:
        for x in tqdm.trange(len(images), desc='Building Records'):
            feat = dict(image=_bytes_feature(images[x]),
                        label=_int64_feature(labels[x]))
            record = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(record.SerializeToString())
    print('Saved:', filename)

def _encode_png(images):
    raw = []
    with tf.Session() as sess, tf.device('cpu:0'):
        image_x = tf.placeholder(tf.uint8, [None, None, None], 'image_x')
        to_png = tf.image.encode_png(image_x)
        for x in tqdm.trange(len(images), desc='Creating PNG Encoding', leave=False):
            raw.append(sess.run(to_png, feed_dict={image_x: images[x]}))
    return raw

def main(argv):
    # Supported augmentations are contained in the following dict.
    augment_dict = {"cifar10": augment_cifar10, "color": augment_color, "cutout": augment_cutout, "svhn": augment_svhn, "stl10": augment_stl10}

    # TODO: Make script keep a copy of unaugmented images in the dataset, investigate how to apply autoaugment policies

    assert FLAGS.aug_dataset, "Please enter the name of the dataset to generate unlabeled augmentations."
    assert FLAGS.aug_copy, "Please provide a number of augmented copies of the unlabeled data to produce."
    assert FLAGS.augment, "Please provide a type of augmentation to perform, supported types are " + " ".join(augment_dict.keys())
    assert FLAGS.unlabel_size, "Please give the size of the labeled and unlabeled dataset."

    tfrecords = [os.path.join(DATA_DIR + "/SSL", FLAGS.aug_dataset + "-unlabel.tfrecord")]
    for index, tfrecord in enumerate(tfrecords):
        assert os.path.isfile(tfrecord), tfrecord + \
            " not found. Please provide a dataset that has been initialized."
        # Create TFRecordDataset using TFRecord containing labeled and unlabeled images.
        filenames = sorted(sum([glob.glob(tfrecord)], []))
        if not filenames:
            raise ValueError('Dataset using ' + tfrecord + ' not found.')
        dataset = tf.data.TFRecordDataset(filenames)

        # Initialize dataset, create FLAGS.aug_copy number of copies, and apply augmentation.
        dataset = dataset.map(get_info, 4).repeat(FLAGS.aug_copy).map(augment_dict[FLAGS.augment], 4)
        data = dataset.make_one_shot_iterator().get_next()

        images = []
        labels = []

        with tf.Session() as sess:
            if tfrecord[-16:-9] is "unlabel":
                num_of_images = int(FLAGS.aug_dataset.split("@")[1]) * FLAGS.aug_copy
            else:
                num_of_images = FLAGS.unlabel_size * FLAGS.aug_copy
            loop = tqdm.trange(num_of_images, desc="Performing " + FLAGS.augment + " augmentation")
            try:
                while True:
                    img, label = sess.run(data)
                    images.append(img)
                    labels.append(label)
                    loop.update()
            except:
                pass
            loop.close()

        # Encode augmented images for saving as tfrecords.
        imgs_labels = [(images[i], labels[i]) for i in range(0, len(images))]
        random.seed(1)
        random.shuffle(imgs_labels)
        images = [img_label[0] for img_label in imgs_labels]
        labels = [img_label[1] for img_label in imgs_labels]
        images = _encode_png(images)
        filesplit = tfrecord.split('.')
        filename = filesplit[0] + "_aug" + str(FLAGS.aug_copy) + '.' + '.'.join(filesplit[1:])
        save_as_tfrecord(images, labels, filename)

    # Copy the test file to the new name
    copyfile(os.path.join(DATA_DIR + "/", FLAGS.aug_dataset.split('.')[0] + "-test.tfrecord"), \
        os.path.join(DATA_DIR + "/", FLAGS.aug_dataset.split('.')[0] + "_aug" + str(FLAGS.aug_copy) + "-test.tfrecord"))


if __name__ == '__main__':
    utils.setup_tf()
    app.run(main)
