# Copyright 2019 Google LLC (original)
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

"""Fully supervised training.
"""

import functools
import os

from absl import app
from absl import flags
from easydict import EasyDict
from fully_supervised.lib.data import DATASETS, DataSetFS
from fully_supervised.lib.train import ClassifyFullySupervised
from libml import utils
from libml.models import MultiModel
from libml import data
import tensorflow as tf

FLAGS = flags.FLAGS


class FSBaseline(ClassifyFullySupervised, MultiModel):

    def model(self, lr, wd, ema, **kwargs):
        hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
        x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')
        l_in = tf.placeholder(tf.int32, [None], 'labels')
        wd *= lr
        l = tf.one_hot(l_in, self.nclass)

        x, l = self.augment(x_in, l, **kwargs)
        classifier = functools.partial(self.classifier, **kwargs)
        logits = classifier(x, training=True)

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=l, logits=logits)
        loss = tf.reduce_mean(loss)
        tf.summary.scalar('losses/xe', loss)

        ema = tf.train.ExponentialMovingAverage(decay=ema)
        ema_op = ema.apply(utils.model_vars())
        ema_getter = functools.partial(utils.getter_ema, ema)
        post_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) + [ema_op]
        post_ops.extend([tf.assign(v, v * (1 - wd)) for v in utils.model_vars('classify') if 'kernel' in v.name])

        train_op = tf.train.AdamOptimizer(lr).minimize(loss, colocate_gradients_with_ops=True)
        with tf.control_dependencies([train_op]):
            train_op = tf.group(*post_ops)

        # Tuning op: only retrain batch norm.
        skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        classifier(x_in, training=True)
        train_bn = tf.group(*[v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                              if v not in skip_ops])

        return EasyDict(
            x=x_in, label=l_in, train_op=train_op, tune_op=train_bn,
            classify_raw=tf.nn.softmax(classifier(x_in, training=False)),  # No EMA, for debugging.
            classify_op=tf.nn.softmax(classifier(x_in, getter=ema_getter, training=False)),
            eval_loss_op=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=classifier(x_in, getter=ema_getter, training=False), 
                labels=tf.one_hot(l_in, self.nclass))))


def main(argv):
    del argv  # Unused.

    assert FLAGS.dataset in DATASETS.keys() or FLAGS.custom_dataset, "Please specify a dataset which is in data.py or use --custom_dataset."

    if not FLAGS.custom_dataset:
        dataset = DATASETS[FLAGS.dataset]()
    else:
        print("Preparing to train the " + FLAGS.dataset + " dataset.")
        valid_size = [int(size) for size in FLAGS.valid_size]

        augmentation = data.augment_cifar10

        # Do not name your dataset using a "-", otherwise the following line will not work for a custom dataset.
        DATASETS.update([DataSetFS.creator(FLAGS.dataset.split("-")[0], [FLAGS.train_record], [FLAGS.test_record], valid,
                                           augmentation, nclass=FLAGS.nclass, height=FLAGS.img_size, width=FLAGS.img_size)
                                           for valid in valid_size])
        dataset = DATASETS[FLAGS.dataset]()

    log_width = utils.ilog2(dataset.width)
    model = FSBaseline(
        os.path.join(FLAGS.train_dir, dataset.name),
        dataset,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        arch=FLAGS.arch,
        batch=FLAGS.batch,
        nclass=dataset.nclass,
        ema=FLAGS.ema,
        smoothing=FLAGS.smoothing,

        scales=FLAGS.scales or (log_width - 2),
        filters=FLAGS.filters,
        repeat=FLAGS.repeat)
    model.train(FLAGS.train_kimg << 10, FLAGS.report_kimg << 10)


if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_float('wd', 0.02, 'Weight decay.')
    flags.DEFINE_float('ema', 0.999, 'Exponential moving average of params.')
    flags.DEFINE_float('smoothing', 0.001, 'Label smoothing.')
    flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    flags.DEFINE_bool('custom_dataset', True, 'True if using a custom dataset.')
    flags.DEFINE_integer('nclass', 42, 'Number of classes present in custom dataset.')
    flags.DEFINE_integer('img_size', 32, 'Size of Images in custom dataset')
    flags.DEFINE_string('train_record', 'sketch-train.tfrecord', 'Name of training tfrecord.')
    flags.DEFINE_string('test_record', 'sketch-test.tfrecord', 'Name of test tfrecord.')
    flags.DEFINE_spaceseplist('valid_size', ['1'], 'List of different validation sizes.')
    flags.DEFINE_string('augment', 'custom', 'Type of augmentation to use, as defined in libml.data.py')
    FLAGS.set_default('dataset', 'cifar10')
    FLAGS.set_default('batch', 64)
    FLAGS.set_default('lr', 0.002)
    FLAGS.set_default('train_kimg', 1 << 16)
    app.run(main)
