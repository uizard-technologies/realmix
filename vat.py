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
"""Virtual adversarial training:a regularization method for supervised and semi-supervised learning.

Application to SSL of https://arxiv.org/abs/1704.03976
"""

import functools
import itertools
import os

from absl import app
from absl import flags
from easydict import EasyDict
from libml import utils, data, layers, models
from libml.data_pair import DATASETS, stack_augment
from libml.data import DataSet, augment_cifar10, augment_custom
import tensorflow as tf
from third_party import vat_utils

FLAGS = flags.FLAGS


class VAT(models.MultiModel):

    def model(self, lr, wd, ema, warmup_pos, vat, vat_eps, entmin_weight, **kwargs):
        hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
        x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')
        y_in = tf.placeholder(tf.float32, [None] + hwc, 'y')
        l_in = tf.placeholder(tf.int32, [None], 'labels')
        wd *= lr
        warmup = tf.clip_by_value(tf.to_float(self.step) / (warmup_pos * (FLAGS.train_kimg << 10)), 0, 1)

        classifier = functools.partial(self.classifier, **kwargs)
        l = tf.one_hot(l_in, self.nclass)
        logits_x = classifier(x_in, training=True)
        post_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Take only first call to update batch norm.
        logits_y = classifier(y_in, training=True)
        delta_y = vat_utils.generate_perturbation(y_in, logits_y, lambda x: classifier(x, training=True), vat_eps)
        logits_student = classifier(y_in + delta_y, training=True)
        logits_teacher = tf.stop_gradient(logits_y)
        loss_vat = layers.kl_divergence_from_logits(logits_student, logits_teacher)
        loss_vat = tf.reduce_mean(loss_vat)
        loss_entmin = tf.reduce_mean(tf.distributions.Categorical(logits=logits_y).entropy())

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=l, logits=logits_x)
        loss = tf.reduce_mean(loss)
        tf.summary.scalar('losses/xe', loss)
        tf.summary.scalar('losses/vat', loss_vat)
        tf.summary.scalar('losses/entmin', loss_entmin)

        ema = tf.train.ExponentialMovingAverage(decay=ema)
        ema_op = ema.apply(utils.model_vars())
        ema_getter = functools.partial(utils.getter_ema, ema)
        post_ops.append(ema_op)
        post_ops.extend([tf.assign(v, v * (1 - wd)) for v in utils.model_vars('classify') if 'kernel' in v.name])

        train_op = tf.train.AdamOptimizer(lr).minimize(loss + loss_vat * warmup * vat + entmin_weight * loss_entmin,
                                                       colocate_gradients_with_ops=True)
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


def main(argv):
    del argv  # Unused.
    if FLAGS.dataset in DATASETS.keys():
        dataset = DATASETS[FLAGS.dataset]()
    elif FLAGS.dataset not in DATASETS.keys() and FLAGS.custom_dataset:
        print("Preparing to train the " + FLAGS.dataset + " dataset.")
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

    log_width = utils.ilog2(dataset.width)
    model = VAT(
        os.path.join(FLAGS.train_dir, dataset.name),
        dataset,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        arch=FLAGS.arch,
        warmup_pos=FLAGS.warmup_pos,
        batch=FLAGS.batch,
        nclass=dataset.nclass,
        ema=FLAGS.ema,
        smoothing=FLAGS.smoothing,
        vat=FLAGS.vat,
        vat_eps=FLAGS.vat_eps,
        entmin_weight=FLAGS.entmin_weight,

        scales=FLAGS.scales or (log_width - 2),
        filters=FLAGS.filters,
        repeat=FLAGS.repeat)
    model.train(FLAGS.train_kimg << 10, FLAGS.report_kimg << 10)


if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_float('wd', 0.02, 'Weight decay.')
    flags.DEFINE_float('vat', 0.3, 'VAT weight.')
    flags.DEFINE_float('vat_eps', 6, 'VAT perturbation size.')
    flags.DEFINE_float('entmin_weight', 0.06, 'Entropy minimization weight.')
    flags.DEFINE_float('warmup_pos', 0.4, 'Relative position at which constraint loss warmup ends.')
    flags.DEFINE_float('ema', 0.999, 'Exponential moving average of params.')
    flags.DEFINE_float('smoothing', 0.1, 'Label smoothing.')
    flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    flags.DEFINE_bool('custom_dataset', False, 'True if using a custom dataset.')
    flags.DEFINE_integer('nclass', 6, 'Number of classes present in custom dataset.')
    flags.DEFINE_integer('img_size', 32, 'Size of Images in custom dataset')
    flags.DEFINE_spaceseplist('label_size', ['250', '1000', '2000'], 'List of different labeled data sizes.')
    flags.DEFINE_spaceseplist('valid_size', ['1', '500'], 'List of different validation sizes.')
    flags.DEFINE_string('augment', 'custom', 'Type of augmentation to use, as defined in libml.data.py')
    FLAGS.set_default('dataset', 'cifar10.3@250-5000')
    FLAGS.set_default('batch', 64)
    FLAGS.set_default('lr', 0.002)
    FLAGS.set_default('train_kimg', 1 << 16)
    app.run(main)
