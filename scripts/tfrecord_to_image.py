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
from urllib import request
from PIL import Image

import tensorflow as tf
import numpy as np
import os
import glob
import tqdm
import tarfile
import tempfile
import scipy.io
import random

flags.DEFINE_string("tfrecord", "", "Name of dataset to perform augmentation on.")
flags.DEFINE_string("save_path", "", "Place to save images")
FLAGS = flags.FLAGS
DATA_DIR = os.environ['ML_DATA']

def get_info(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={'image': tf.FixedLenFeature([], tf.string),
                  'label': tf.FixedLenFeature([], tf.int64)})
    return tf.cast(tf.image.decode_png(features['image']), tf.float32), features['label']

def main(argv):
    assert os.path.isfile(FLAGS.tfrecord), FLAGS.tfrecord + \
            " not found. Please provide a dataset that has been initialized."
    # def unflatten(images):
    #     return np.transpose(images.reshape((images.shape[0], 3, 32, 32)),
    #                         [0, 2, 3, 1])

    # with tempfile.NamedTemporaryFile() as f:
    #     request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz", f.name)
    #     tar = tarfile.open(fileobj=f)
    #     train_data_batches, train_data_labels = [], []
    #     for batch in range(1, 6):
    #         data_dict = scipy.io.loadmat(tar.extractfile(
    #             'cifar-10-batches-mat/data_batch_{}.mat'.format(batch)))
    #         train_data_batches.append(data_dict['data'])
    #         train_data_labels.append(data_dict['labels'].flatten())
    #     train_set = {'images': np.concatenate(train_data_batches, axis=0),
    #                  'labels': np.concatenate(train_data_labels, axis=0)}
    #     data_dict = scipy.io.loadmat(tar.extractfile(
    #         'cifar-10-batches-mat/test_batch.mat'))
    #     test_set = {'images': data_dict['data'],
    #                 'labels': data_dict['labels'].flatten()}

    #     train_set['images'] = unflatten(train_set['images'])
        
    #     for index in tqdm.tqdm(range(train_set['images'].shape[0])):
    #         im = Image.fromarray(np.asarray(train_set['images'][index, :, :, :]), mode='RGB')
    #         im.save(FLAGS.save_path + str(index).zfill(len(str(train_set['images'].shape[0]))) + 
    #             "_" + str(train_set['labels'][index]) + ".png")
    dataset = tf.data.TFRecordDataset(sorted(sum([glob.glob(FLAGS.tfrecord)], []))).map(get_info, 4)
    data = dataset.make_one_shot_iterator().get_next()
    images = []
    labels = []

    with tf.Session() as sess:
        try:
            while True:
                img, label = sess.run(data)
                img = np.asarray(img, dtype=np.intc)
                images.append(img)
                labels.append(np.asarray(label, dtype=np.intc))
        except:
            pass
    print(images[0].shape)
    for index in tqdm.trange(len(labels)):
        import matplotlib.pyplot as plt
        path = FLAGS.save_path + str(index).zfill(len(str(len(labels)))) + "_" + str(labels[index]) + ".png"
        plt.imsave(path, images[index])
        #im = Image.fromarray(images[index], mode='RGB')
        #im.save(FLAGS.save_path + str(index).zfill(len(str(len(labels)))) + "_" + str(labels[index]) + ".png")


if __name__ == '__main__':
    utils.setup_tf()
    app.run(main)