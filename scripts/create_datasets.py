#!/usr/bin/env python

# Copyright 2018 Google LLC (original)
# Copyright 2019 Uizard Technologies (small modifications)
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


"""Script to download all datasets and create .tfrecord files.
"""

from urllib import request
from absl import app, flags
from easydict import EasyDict
from libml.data import DATA_DIR, augment_shift, augment_color, augment_mirror
from libml import utils
from tensorflow.python.client import device_lib
from keras.preprocessing import image
from tensorflow.python.client import device_lib
from third_party.random_eraser import get_random_eraser

import collections
import gzip
import os
import sys
import tarfile
import tempfile
import math
import numpy as np
import scipy.io
import tensorflow as tf
import tqdm
import glob
import matplotlib.pyplot as plt
import argparse
import json


FLAGS = flags.FLAGS


URLS = {
    'svhn': 'http://ufldl.stanford.edu/housenumbers/{}_32x32.mat',
    'cifar10': 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz',
    'cifar100': 'https://www.cs.toronto.edu/~kriz/cifar-100-matlab.tar.gz',
    'stl10': 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz',
}

# Encode an array of images as PNGs.
def _encode_png(images):
    raw = []
    with tf.Session() as sess, tf.device('cpu:0'):
        image_x = tf.placeholder(tf.uint8, [None, None, None], 'image_x')
        to_png = tf.image.encode_png(image_x)
        for x in tqdm.trange(images.shape[0], desc='PNG Encoding', leave=False):
            raw.append(sess.run(to_png, feed_dict={image_x: images[x]}))
    return raw

# Encode and Class-Balance an array of images as PNGs.
# To class-balance, data is randomly undersampled or oversampled
# by data augmentation.
def _encode_and_aug_png(images, labels, num_aug_per_class, class_ids, imgs_per_class):
    raw = []
    new_labels = []

    aug_datagen = image.ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    with tf.Session() as sess, tf.device('cpu:0'):
        image_x = tf.placeholder(tf.uint8, [None, None, None], 'image_x')
        to_png = tf.image.encode_png(image_x)

        # Iterate through each class and perform class-balancing.
        # num_of_augs dictates the number of augmentations to be performed
        # if the class is to be oversampled.
        for classname, num_of_augs in enumerate(tqdm.tqdm(num_aug_per_class)):
            if num_of_augs == 0:
                continue

            img_num = 0

            # Oversample the class until the required number of images has been met.
            while img_num < imgs_per_class:
                try:
                    img_id = class_ids[classname].pop(0)
                except:
                    break
                raw.append(sess.run(to_png, feed_dict={image_x: images[img_id]}))
                new_labels.append(classname)
                img_num += 1

                # Data augmentation is used to oversample if the num_of_augs > 1.
                if num_of_augs > 1:
                    for _ in range(0, num_of_augs - 1):
                        if img_num < imgs_per_class:
                            aug_img = aug_datagen.random_transform(images[img_id], seed=1)
                            raw.append(sess.run(to_png, feed_dict={image_x: aug_img}))
                            new_labels.append(classname)
                            img_num += 1
                        else:
                            break

    assert len(raw) == len(new_labels)

    return raw, new_labels


def _load_svhn():
    splits = collections.OrderedDict()
    for split in ['train', 'test', 'extra']:
        with tempfile.NamedTemporaryFile() as f:
            request.urlretrieve(URLS['svhn'].format(split), f.name)
            data_dict = scipy.io.loadmat(f.name)
        dataset = {}
        dataset['images'] = np.transpose(data_dict['X'], [3, 0, 1, 2])
        dataset['images'] = _encode_png(dataset['images'])
        dataset['labels'] = data_dict['y'].reshape((-1))
        # SVHN raw data uses labels from 1 to 10; use 0 to 9 instead.
        dataset['labels'] -= 1
        splits[split] = dataset
    return splits


def _load_stl10():
    def unflatten(images):
        return np.transpose(images.reshape((-1, 3, 96, 96)),
                            [0, 3, 2, 1])

    with tempfile.NamedTemporaryFile() as f:
        if os.path.exists('stl10/stl10_binary.tar.gz'):
            f = open('stl10/stl10_binary.tar.gz', 'rb')
        else:
            request.urlretrieve(URLS['stl10'], f.name)
        tar = tarfile.open(fileobj=f)
        train_X = tar.extractfile('stl10_binary/train_X.bin')
        train_y = tar.extractfile('stl10_binary/train_y.bin')

        test_X = tar.extractfile('stl10_binary/test_X.bin')
        test_y = tar.extractfile('stl10_binary/test_y.bin')

        unlabeled_X = tar.extractfile('stl10_binary/unlabeled_X.bin')

        train_set = {'images': np.frombuffer(train_X.read(), dtype=np.uint8),
                     'labels': np.frombuffer(train_y.read(), dtype=np.uint8) - 1}

        test_set = {'images': np.frombuffer(test_X.read(), dtype=np.uint8),
                    'labels': np.frombuffer(test_y.read(), dtype=np.uint8) - 1}

        _imgs = np.frombuffer(unlabeled_X.read(), dtype=np.uint8)
        unlabeled_set = {'images': _imgs,
                         'labels': np.zeros(100000, dtype=np.uint8)}

        fold_indices = tar.extractfile('stl10_binary/fold_indices.txt').read()

    train_set['images'] = _encode_png(unflatten(train_set['images']))
    test_set['images'] = _encode_png(unflatten(test_set['images']))
    unlabeled_set['images'] = _encode_png(unflatten(unlabeled_set['images']))
    return dict(train=train_set, test=test_set, unlabeled=unlabeled_set,
                files=[EasyDict(filename="stl10_fold_indices.txt", data=fold_indices)])


def _load_cifar10():
    def unflatten(images):
        return np.transpose(images.reshape((images.shape[0], 3, 32, 32)),
                            [0, 2, 3, 1])

    with tempfile.NamedTemporaryFile() as f:
        request.urlretrieve(URLS['cifar10'], f.name)
        tar = tarfile.open(fileobj=f)
        train_data_batches, train_data_labels = [], []
        for batch in range(1, 6):
            data_dict = scipy.io.loadmat(tar.extractfile(
                'cifar-10-batches-mat/data_batch_{}.mat'.format(batch)))
            train_data_batches.append(data_dict['data'])
            train_data_labels.append(data_dict['labels'].flatten())
        train_set = {'images': np.concatenate(train_data_batches, axis=0),
                     'labels': np.concatenate(train_data_labels, axis=0)}
        data_dict = scipy.io.loadmat(tar.extractfile(
            'cifar-10-batches-mat/test_batch.mat'))
        test_set = {'images': data_dict['data'],
                    'labels': data_dict['labels'].flatten()}
    train_set['images'] = _encode_png(unflatten(train_set['images']))
    test_set['images'] = _encode_png(unflatten(test_set['images']))
    return dict(train=train_set, test=test_set)


def _load_cifar100():
    def unflatten(images):
        return np.transpose(images.reshape((images.shape[0], 3, 32, 32)),
                            [0, 2, 3, 1])

    with tempfile.NamedTemporaryFile() as f:
        request.urlretrieve(URLS['cifar100'], f.name)
        tar = tarfile.open(fileobj=f)
        data_dict = scipy.io.loadmat(tar.extractfile('cifar-100-matlab/train.mat'))
        train_set = {'images': data_dict['data'],
                     'labels': data_dict['fine_labels'].flatten()}
        data_dict = scipy.io.loadmat(tar.extractfile('cifar-100-matlab/test.mat'))
        test_set = {'images': data_dict['data'],
                    'labels': data_dict['fine_labels'].flatten()}
    train_set['images'] = _encode_png(unflatten(train_set['images']))
    test_set['images'] = _encode_png(unflatten(test_set['images']))
    return dict(train=train_set, test=test_set)

# Load a custom dataset from a local directory.
def _load_custom():
    assert args.traindir and args.testdir, "A training and test image directory must be provided for custom datasets."

    if args.traindir[-1] is not "/":
        args.traindir += "/"

    if args.testdir[-1] is not "/":
        args.testdir += "/"

    # Read images and labels from local directories.
    # The labels are extracted through each image's filename.
    # Ex: 24628_dog.png -> "dog"
    # Ex: 73158_cat.png -> "cat"

    train_data_paths = glob.glob(args.traindir + "*.jpg")
    test_data_paths = glob.glob(args.testdir + "*.jpg")

    train_data = np.asarray([plt.imread(path) for path in train_data_paths])
    class_names = list(set([path.split('_')[-1][:-4] for path in train_data_paths]))
    class_names.sort()
    train_labels = [class_names.index(path.split('_')[-1][:-4]) for path in train_data_paths]

    test_data = np.asarray([plt.imread(path) for path in test_data_paths])
    test_labels = [class_names.index(path.split('_')[-1][:-4]) for path in test_data_paths]

    # If cutout augmentation is specified, here we apply it each image in the training set.
    if args.cutout:
        assert args.cutout_prob, "Please specify a cutout parameter."
        cutout_prob = float(args.cutout_prob)
        eraser = get_random_eraser(p=cutout_prob)
        if "train" in args.cutout or args.cutout == "all":
            train_data = np.asarray([eraser(train_data[i]) for i in tqdm.trange(train_data.shape[0], \
                                desc="Performing cutout on training images with probability of " + str(cutout_prob))])

    train_set = {'images': train_data,
                 'labels': train_labels}
    test_set = {'images': test_data,
                 'labels': test_labels}

    if args.train_balance:
        # If the training set is to be balanced, we calculate how many imgs_per_class
        # are needed and how many of each class is currently present.
        imgs_per_class = int(math.ceil(int(args.train_balance) / len(class_names)))
        train_class_count = [train_labels.count(i) for i in range(len(class_names))]

        print("Balancing training set to " + str(imgs_per_class) + " ...")
        class_ids = {class_num: [] for class_num in range(len(class_names))}

        for i, label in enumerate(train_set['labels']):
            class_ids[label].append(i)

        # We shuffle the dataset to sample evenly.
        np.random.seed(1)
        for i in range(len(class_names)):
            np.random.shuffle(class_ids[i])

        num_aug_per_class = [max(1, int(math.ceil(imgs_per_class / class_count))) for class_count in train_class_count]

        # The class-balancing operation occurs in _encode_and_aug_png, which returns
        # the class-balanced set of images and labels.
        print("Number of augmentations needed: ", num_aug_per_class)
        train_set['images'], train_set['labels'] = _encode_and_aug_png(train_set['images'], train_set['labels'], num_aug_per_class, class_ids, imgs_per_class)

    else:
        train_set['images'] = _encode_png(train_set['images'])


    test_set['images'] = _encode_png(test_set['images'])

    _save_class_mappings(class_names)

    return dict(train=train_set, test=test_set)

# Load a custom dataset with unlabeled images from a local directories.
def _load_custom_extra():
    assert args.traindir and args.testdir and args.unlabeldir,  \
    "A training, unlabeled, and test image directory must be provided for custom-extra datasets."

    if args.traindir[-1] is not "/":
        args.traindir += "/"

    if args.testdir[-1] is not "/":
        args.testdir += "/"

    if args.unlabeldir[-1] is not "/":
        args.unlabeldir += "/"

    train_data_paths = glob.glob(args.traindir + "*.jpg")
    test_data_paths = glob.glob(args.testdir + "*.jpg")
    unlabel_data_paths = glob.glob(args.unlabeldir + "*.jpg")

    # Read images and labels from local directories.
    # The labels are extracted through each image's filename.
    # Ex: 24628_dog.png -> "dog"
    # Ex: 73158_cat.png -> "cat"

    train_data = np.asarray([plt.imread(path) for path in train_data_paths])
    unlabel_data = np.asarray([plt.imread(path) for path in unlabel_data_paths])
    class_names = list(set([path.split('_')[-1][:-4] for path in train_data_paths]))
    class_names.sort()
    train_labels = [class_names.index(path.split('_')[-1][:-4]) for path in train_data_paths]
    unlabel_labels = [0 for path in unlabel_data_paths]

    test_data = np.asarray([plt.imread(path) for path in test_data_paths])
    test_labels = [class_names.index(path.split('_')[-1][:-4]) for path in test_data_paths]

    train_set = {'images': train_data,
                 'labels': train_labels}
    test_set = {'images': test_data,
                 'labels': test_labels}

    if args.cutout:
        # If cutout augmentation is specified, here we apply it each image in the training set.
        assert args.cutout_prob, "Please specify a cutout parameter."

        cutout_prob = float(args.cutout_prob)
        eraser = get_random_eraser(p=cutout_prob)

        if "train" in args.cutout or args.cutout == "all":
            train_data = np.asarray([eraser(train_data[i]) for i in tqdm.trange(train_data.shape[0], desc="Performing cutout on training images with probability of " + str(cutout_prob))])
        if "unlabel" in args.cutout or args.cutout == "all":
            unlabel_data = np.asarray([eraser(unlabel_data[i]) for i in tqdm.trange(unlabel_data.shape[0], desc="Performing cutout on unlabeled images with probability of " + str(cutout_prob))])

    if args.train_balance:
        # If the training set is to be balanced, we calculate how many imgs_per_class
        # are needed and how many of each class is currently present.
        imgs_per_class = int(math.ceil(int(args.train_balance) / len(class_names)))
        train_class_count = [train_labels.count(i) for i in range(len(class_names))]
        print("Balancing training set to " + str(imgs_per_class) + " ...")

        class_ids = {class_num: [] for class_num in range(len(class_names))}

        for i, label in enumerate(train_set['labels']):
            class_ids[label].append(i)

        # We shuffle the dataset to sample evenly.
        np.random.seed(1)
        for i in range(len(class_names)):
            np.random.shuffle(class_ids[i])

        num_aug_per_class = [max(1, int(math.ceil(imgs_per_class / class_count))) for class_count in train_class_count]

        # The class-balancing operation occurs in _encode_and_aug_png, which returns
        # the class-balanced set of images and labels.
        print("Number of augmentations needed: ", num_aug_per_class)
        train_set['images'], train_set['labels'] = _encode_and_aug_png(train_set['images'], train_set['labels'], num_aug_per_class, class_ids, imgs_per_class)

    else:
        train_set['images'] = _encode_png(train_set['images'])

    print("Training Images: ", len(train_set['images']))

    if args.unlabel_balance:
        unlabel_labels = [class_names.index(path.split('_')[-1][:-4]) for path in unlabel_data_paths]

        # If the unlabeled set is to be balanced, we calculate how many imgs_per_class
        # are needed and how many of each class is currently present.
        imgs_per_class = int(math.ceil(int(args.unlabel_balance) / len(class_names)))
        unlabel_class_count = [unlabel_labels.count(i) for i in range(len(class_names))]

        print("Balancing unlabeled set to " + str(imgs_per_class) + " ...")
        class_ids = {class_num: [] for class_num in range(len(class_names))}

        for i, label in enumerate(unlabel_labels):
            class_ids[label].append(i)

        # We shuffle the dataset to sample evenly.
        np.random.seed(1)
        for i in range(len(class_names)):
            np.random.shuffle(class_ids[i])

        if args.prediction_balance:
            # The class balancing is performed on the predicted distribution of the unlabeled set,
            # given by predictions.npy. This is a more realistic scenario of class-balancing for unlabeled
            # sets, as one doesn't know the ground truth distribution of an unlabeled.

            print("Creating class balance from predictions on the unlabeled set.")

            unlabel_preds = np.load('predictions.npy')
            unlabel_pred_class_count = [collections.Counter(unlabel_preds)[i] for i in range(len(class_names))]
            num_aug_per_class = [max(1, int(math.ceil(imgs_per_class / class_count))) if class_count > 0 else 0 for class_count in unlabel_pred_class_count]
        else:
            num_aug_per_class = [max(1, int(math.ceil(imgs_per_class / class_count))) if class_count > 0 else 0 for class_count in unlabel_class_count]

        if args.pseudolabel:
            # This assigns the unlabeled set labels from an external file,
            # here used from predictions.npy
            print("Reading pseduo labels to unlabel_labels...")
            unlabel_labels = np.load('predictions.npy')
            unlabel_labels = [label for label in unlabel_labels]

        # Perform class-balancing
        print("Number of augmentations needed: ", num_aug_per_class, [max(1, int(math.ceil(imgs_per_class / class_count))) if class_count > 0 else 0 for class_count in unlabel_class_count])
        unlabel_data, unlabel_labels = _encode_and_aug_png(unlabel_data, unlabel_labels, num_aug_per_class, class_ids, imgs_per_class)
        print("Unlabeled Images after Balancing: ", len(unlabel_data))

        train_set['images'] = np.concatenate((train_set['images'], unlabel_data))
        train_set['labels'] = train_set['labels'] + unlabel_labels

    else:
        train_set['images'] = np.concatenate((train_set['images'], _encode_png(unlabel_data)))

        if args.pseudolabel:
            print("Reading pseduo labels to unlabel_labels...")
            unlabel_labels = np.load('predictions.npy')
            unlabel_labels = [label for label in unlabel_labels]

        train_set['labels'] = train_set['labels'] + unlabel_labels

    test_set['images'] = _encode_png(test_set['images'])

    print("Training and Unlabeled Images: ", len(train_set['images']))
    print("Test Images: ", len(test_set['images']))

    _save_class_mappings(class_names)

    return dict(train=train_set, test=test_set)

def _save_class_mappings(class_names):
    class_mapping = {index: name for index, name in enumerate(class_names)}
    with open(args.name +'_class_mappings.json', 'w') as f:
        json.dump(class_mapping, f)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _save_as_tfrecord(data, filename):
    assert len(data['images']) == len(data['labels'])
    filename = os.path.join(DATA_DIR, filename + '.tfrecord')
    print('Saving dataset:', filename)
    with tf.python_io.TFRecordWriter(filename) as writer:
        for x in tqdm.trange(len(data['images']), desc='Building records'):
            feat = dict(image=_bytes_feature(data['images'][x]),
                        label=_int64_feature(data['labels'][x]))
            record = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(record.SerializeToString())
    print('Saved:', filename)


def _is_installed(name, checksums):
    for subset, checksum in checksums.items():
        filename = os.path.join(DATA_DIR, '%s-%s.tfrecord' % (name, subset))
        if not os.path.exists(filename):
            return False
    return True


def _save_files(files, *args, **kwargs):
    del args, kwargs
    for folder in frozenset(os.path.dirname(x) for x in files):
        os.makedirs(os.path.join(DATA_DIR, folder), exist_ok=True)
    for filename, contents in files.items():
        with open(os.path.join(DATA_DIR, filename), 'w') as f:
            f.write(contents)


def _is_installed_folder(name, folder):
    return os.path.exists(os.path.join(DATA_DIR, name, folder))


CONFIGS = dict(
    cifar10=dict(loader=_load_cifar10,
                 checksums=dict(train=None, test=None)),
    cifar100=dict(loader=_load_cifar100,
                  checksums=dict(train=None, test=None)),
    svhn=dict(loader=_load_svhn,
              checksums=dict(train=None, test=None, extra=None)),
    stl10=dict(loader=_load_stl10,
               checksums=dict(train=None, test=None)),
    custom=dict(loader=_load_custom,
                checksums=dict(train=None, test=None)),
    custom_extra=dict(loader=_load_custom_extra,
                checksums=dict(train=None, test=None))
)

if __name__ == '__main__':
    utils.setup_tf()

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", help="Name for custom dataset")
    parser.add_argument("-tr", "--traindir", help="Directory for Custom Training Images")
    parser.add_argument("-ts", "--testdir", help="Directory for Custom Test Images")
    parser.add_argument("-u", "--unlabeldir", help="Directory for Custom Unlabeled Images")
    parser.add_argument("-trbal", "--train_balance", help="Balance training set with this many images total.")
    parser.add_argument("-unlbal", "--unlabel_balance", help="Balance unlabeled set with this many images total.")
    parser.add_argument("-predbal", "--prediction_balance", help="Balance unlabeled set using predictions from the model.")
    parser.add_argument("-ps", "--pseudolabel", help="Create unlabeled labels from predictions file.")
    parser.add_argument("-cut", "--cutout", help="Apply random erasing (cutout) on images.")
    parser.add_argument("-cutp", "--cutout_prob", help="Percentage of images to apply cutout to.")

    args, leftovers = parser.parse_known_args()

    if len(sys.argv[1:]):
        subset = set(sys.argv[1:])
    else:
        subset = set(CONFIGS.keys())
    try:
        os.makedirs(DATA_DIR)
    except OSError:
        pass
    for name, config in CONFIGS.items():
        if name not in subset:
            continue
        if (name == "custom" or name == "custom_extra") and args.name:
            name = args.name
        elif (name == "custom" or name == "custom_extra") and not args.name:
            print("Please provide a name for the custom dataset, using -n or --name.")
            continue

        if 'is_installed' in config:
            print('is_installed', 'is in config')
            if config['is_installed']():
                print('Skipping already installed:', name)
                print("Skip 1")
                continue

        elif _is_installed(name, config['checksums']):
            print('Skipping already installed:', name)
            print("Skip 2")
            continue

        print('Preparing', name)
        datas = config['loader']()
        saver = config.get('saver', _save_as_tfrecord)
        for sub_name, data in datas.items():
            if sub_name == 'readme':
                filename = os.path.join(DATA_DIR, '%s-%s.txt' % (name, sub_name))
                with open(filename, 'w') as f:
                    f.write(data)
            elif sub_name == 'files':
                for file_and_data in data:
                    path = os.path.join(DATA_DIR, file_and_data.filename)
                    open(path, "wb").write(file_and_data.data)
            else:
                saver(data, '%s-%s' % (name, sub_name))
