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
"""mixup: Beyond Empirical Risk Minimization.

Adaption to SSL of MixUp: https://arxiv.org/abs/1710.09412
"""
import functools
import itertools
import os

from absl import app
from absl import flags
from easydict import EasyDict
from libml import data, utils, models
from libml.data_pair import DATASETS, stack_augment
from libml.data import DataSet, augment_cifar10, augment_custom
import tensorflow as tf

FLAGS = flags.FLAGS


class Mixup(models.MultiModel):

    def augment(self, x, l, beta, **kwargs):
        del kwargs
        mix = tf.distributions.Beta(beta, beta).sample([tf.shape(x)[0], 1, 1, 1])
        mix = tf.maximum(mix, 1 - mix)
        xmix = x * mix + x[::-1] * (1 - mix)
        lmix = l * mix[:, :, 0, 0] + l[::-1] * (1 - mix[:, :, 0, 0])
        return xmix, lmix

    def model(self, lr, wd, ema, **kwargs):
        hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
        x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')
        y_in = tf.placeholder(tf.float32, [None] + hwc, 'y')
        l_in = tf.placeholder(tf.int32, [None], 'labels')
        wd *= lr
        classifier = functools.partial(self.classifier, **kwargs)

        def get_logits(x):
            logits = classifier(x, training=True)
            return logits

        x, labels_x = self.augment(x_in, tf.one_hot(l_in, self.nclass), **kwargs)
        logits_x = get_logits(x)
        post_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        y, labels_y = self.augment(y_in, tf.nn.softmax(get_logits(y_in)), **kwargs)
        labels_y = tf.stop_gradient(labels_y)
        logits_y = get_logits(y)

        loss_xe = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_x, logits=logits_x)
        loss_xe = tf.reduce_mean(loss_xe)
        loss_xeu = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_y, logits=logits_y)
        loss_xeu = tf.reduce_mean(loss_xeu)
        tf.summary.scalar('losses/xe', loss_xe)
        tf.summary.scalar('losses/xeu', loss_xeu)

        ema = tf.train.ExponentialMovingAverage(decay=ema)
        ema_op = ema.apply(utils.model_vars())
        ema_getter = functools.partial(utils.getter_ema, ema)
        post_ops.append(ema_op)
        post_ops.extend([tf.assign(v, v * (1 - wd)) for v in utils.model_vars('classify') if 'kernel' in v.name])

        train_op = tf.train.AdamOptimizer(lr).minimize(loss_xe + loss_xeu, colocate_gradients_with_ops=True)
        with tf.control_dependencies([train_op]):
            train_op = tf.group(*post_ops)

        # Tuning op: only retrain batch norm.
        skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        classifier(x_in, training=True)
        train_bn = tf.group(*[v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                              if v not in skip_ops])

        return EasyDict(
            x=x_in, y=y_in, label=l_in, train_op=train_op, tune_op=train_bn,
            classify_raw=tf.nn.softmax(classifier(x_in, training=False)),  # No EMA, for debugging.
            classify_op=tf.nn.softmax(classifier(x_in, getter=ema_getter, training=False)))

def get_dataset():
    assert FLAGS.dataset in DATASETS.keys() or FLAGS.custom_dataset, "Please enter a valid dataset name or use the --custom_dataset flag."

    if FLAGS.dataset in DATASETS.keys():
        dataset = DATASETS[FLAGS.dataset]()
    else:
        label_size = [int(size) for size in FLAGS.label_size]
        valid_size = [int(size) for size in FLAGS.valid_size]

        if FLAGS.augment == "cifar10":
            augmentation = augment_cifar10
        else:
            augmentation = augment_custom

        DATASETS.update([DataSet.creator(FLAGS.dataset.split(".")[0], seed, label, valid, [augmentation, stack_augment(augmentation)], nclass=FLAGS.nclass, height=FLAGS.img_size, width=FLAGS.img_size)
                         for seed, label, valid in
                         itertools.product(range(2), label_size, valid_size)])
        dataset = DATASETS[FLAGS.dataset]()

    return dataset

def main(argv):
    del argv  # Unused.
    dataset = get_dataset()

    log_width = utils.ilog2(dataset.width)
    model = Mixup(
        os.path.join(FLAGS.train_dir, dataset.name),
        dataset,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        arch=FLAGS.arch,
        batch=FLAGS.batch,
        nclass=dataset.nclass,
        ema=FLAGS.ema,
        beta=FLAGS.beta,

        scales=FLAGS.scales or (log_width - 2),
        filters=FLAGS.filters,
        repeat=FLAGS.repeat)
    model.train(FLAGS.train_kimg << 10, FLAGS.report_kimg << 10)


if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_float('wd', 0.02, 'Weight decay.')
    flags.DEFINE_float('ema', 0.999, 'Exponential moving average of params.')
    flags.DEFINE_float('beta', 0.5, 'Mixup beta distribution.')
    flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    flags.DEFINE_bool('custom_dataset', True, 'True if using a custom dataset.')
    flags.DEFINE_integer('nclass', 42, 'Number of classes present in custom dataset.')
    flags.DEFINE_integer('img_size', 32, 'Size of Images in custom dataset')
    flags.DEFINE_spaceseplist('label_size', ['250', '1000', '2000', '124994'], 'List of different labeled data sizes.')
    flags.DEFINE_spaceseplist('valid_size', ['1', '500'], 'List of different validation sizes.')
    flags.DEFINE_string('augment', 'custom', 'Type of augmentation to use, as defined in libml.data.py')
    FLAGS.set_default('dataset', 'cifar10.3@250-5000')
    FLAGS.set_default('batch', 64)
    FLAGS.set_default('lr', 0.002)
    FLAGS.set_default('train_kimg', 1 << 16)
    app.run(main)
