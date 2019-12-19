# Copyright 2019 Uizard Technologies
# Significantly inspired by:
# MixMatch - A Holistic Approach to Semi-Supervised Learning, Berthelot et al. (2019) - Google LLC
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

"""RealMix: Towards Realistic Deep Semi-Supervised Learning Algorithms"""


from absl import app, flags
from easydict import EasyDict
from libml import layers, utils, models
from libml.data_pair import DATASETS, stack_augment
from libml.data import DataSet, augment_cifar10, augment_color, augment_cutout, augment_stl10, augment_svhn, memoize, default_parse, dataset
from libml.layers import MixMode
from tqdm import trange

import functools
import itertools
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import numpy as np

FLAGS = flags.FLAGS


class RealMix(models.MultiModel):

    def augment(self, x, l, beta, **kwargs):
        assert 0, 'Do not call.'

    def guess_label(self, y, classifier, T, **kwargs):
        del kwargs
        logits_y = [classifier(yi, training=True) for yi in y]
        logits_y = tf.concat(logits_y, 0)
        # Compute predicted probability distribution py.
        p_model_y = tf.reshape(tf.nn.softmax(logits_y), [len(y), -1, self.nclass])
        p_model_y = tf.reduce_mean(p_model_y, axis=0)
        # Compute the target distribution.
        p_target = tf.pow(p_model_y, 1. / T)
        p_target /= tf.reduce_sum(p_target, axis=1, keep_dims=True)
        return EasyDict(p_target=p_target, p_model=p_model_y)

    def get_tsa_threshold(self, schedule, global_step, num_train_steps, start, end):
        # Originally written in google-research/uda/image/main.py

        # Returns the current TSA (Training Signal Annealing) thresholds given the
        # schedule, current training step, total training steps, start threshold,
        # and end threshold.

        # Typical values are as follows:
        # schedule = "linear_schedule", "exp_schedule", "log_schedule"
        # global_step = self.step
        # num_train_step = FLAGS.train_kimg << 10, or FLAGS.train_kimg * FLAGS.epochs
        # start = 1. / FLAGS.nclass
        # end = 1

        step_ratio = tf.to_float(global_step) / tf.to_float(num_train_steps)
        if schedule == "linear_schedule":
            coeff = step_ratio
        elif schedule == "exp_schedule":
            scale = 5
            # [exp(-5), exp(0)] = [1e-2, 1]
            coeff = tf.exp((step_ratio - 1) * scale)
        elif schedule == "log_schedule":
            scale = 5
            # [1 - exp(0), 1 - exp(-5)] = [0, 0.99]
            coeff = 1 - tf.exp((-step_ratio) * scale)
        return coeff * (end - start) + start


    def anneal_sup_loss(self, sup_logits, sup_labels, sup_loss, global_step):
        # Adapted from google-research/uda/image/main.py

        # This is a version of TSA (Training Signal Annealing) that has been
        # adapted for use with RealMix. Specifically, it can deal with ground
        # truth values between 0 and 1 as created by MixUp.

        # The start value for TSA.
        tsa_start = 1. / FLAGS.nclass

        # Probability thresh above which loss for a sup image is not computed.
        eff_train_prob_threshold = self.get_tsa_threshold(
            FLAGS.tsa, global_step, FLAGS.train_kimg * FLAGS.epochs,
            tsa_start, end=1)

        # Calculate probabilities of each class for each image.
        sup_probs = tf.nn.softmax(sup_logits, axis=-1)

        # Mask the predicted probabilities to only ground truth classes.
        ground_truth_class_threshold = tf.greater(sup_labels, tf.constant(0.0, tf.float32))
        ground_truth_class_mask = tf.cast(ground_truth_class_threshold, tf.float32)
        correct_label_probs = sup_probs * ground_truth_class_mask

        # Calculate TSA threshold for each ground truth probability.
        # This is necessary since MixUp generates values between 0 and 1.
        eff_train_prob_threshold = sup_logits * eff_train_prob_threshold

        ones = tf.ones(tf.shape(correct_label_probs), tf.float32)
        pos_tensor = tf.multiply(ones, FLAGS.nclass + 1)
        neg_tensor = tf.multiply(ones, -1)

        # Loss for an image is kept if all its ground truth thresholds
        # are not met. A temporary mask is created to find which images
        # contain unmet thresholds.
        imgs_to_train_mask = tf.where(tf.less(correct_label_probs, eff_train_prob_threshold), \
                                pos_tensor, neg_tensor)
        imgs_to_train_mask = tf.reduce_mean(imgs_to_train_mask, axis=1)

        ones = tf.ones(tf.shape(imgs_to_train_mask), tf.float32)
        zeros = tf.zeros(tf.shape(imgs_to_train_mask), tf.float32)

        loss_mask = tf.where(tf.greater(imgs_to_train_mask, zeros), ones, zeros)
        loss_mask = tf.stop_gradient(loss_mask)

        # Mask the supervised loss and return the average.
        sup_loss = sup_loss * loss_mask
        avg_sup_loss = (tf.reduce_sum(sup_loss) /
                        tf.maximum(tf.reduce_sum(loss_mask), 1))

        return avg_sup_loss

    def confidence_mask_unsup(self, logits_y, labels_y, loss_l2u):
        # Adapted from google-research/uda/image/main.py

        # This function masks the unsupervised predictions that are below
        # a set confidence threshold. # Note the following will only work
        # using MSE loss and not KL-divergence.

        # Calculate largest predicted probability for each image.
        unsup_prob = tf.nn.softmax(logits_y, axis=-1)
        largest_prob = tf.reduce_max(unsup_prob, axis=-1)

        # Mask the loss for images that don't contain a predicted
        # probability above the threshold.
        loss_mask = tf.cast(tf.greater(largest_prob, FLAGS.percent_mask), tf.float32)
        tf.summary.scalar('losses/high_prob_ratio', tf.reduce_mean(loss_mask))
        loss_mask = tf.stop_gradient(loss_mask)
        loss_l2u = loss_l2u * tf.expand_dims(loss_mask, axis=-1)

        # Return the average unsupervised loss.
        avg_unsup_loss = (tf.reduce_sum(loss_l2u) /
                        tf.maximum(tf.reduce_sum(loss_mask) * FLAGS.nclass, 1))
        return avg_unsup_loss

    def percent_confidence_mask_unsup(self, logits_y, labels_y, loss_l2u):
        # Adapted from google-research/uda/image/main.py

        # This function masks the unsupervised predictions that are below
        # a set confidence threshold. # Note the following will only work
        # using MSE loss and not KL-divergence.

        # Calculate largest predicted probability for each image.
        unsup_prob = tf.nn.softmax(logits_y, axis=-1)
        largest_prob = tf.reduce_max(unsup_prob, axis=-1)

        # Get the indices of the bottom x% of probabilities and mask those out.
        # In other words, get the probability of the image with the x%*#numofsamples
        # lowest probability and use that as the mask.

        # Calculate the current confidence_mask value using the specified schedule:
        sorted_probs = tf.sort(largest_prob, axis=-1, direction='ASCENDING')
        sort_index = tf.math.multiply(tf.to_float(tf.shape(sorted_probs)[0]), FLAGS.percent_mask)
        curr_confidence_mask = tf.slice(sorted_probs, [tf.to_int64(sort_index)], [1])

        # Mask the loss for images that don't contain a predicted
        # probability above the threshold.
        loss_mask = tf.cast(tf.greater(largest_prob, curr_confidence_mask), tf.float32)
        tf.summary.scalar('losses/high_prob_ratio', tf.reduce_mean(loss_mask)) # The ratio of unl images above the thresh
        tf.summary.scalar('losses/percent_confidence_mask', tf.reshape(curr_confidence_mask,[]))
        loss_mask = tf.stop_gradient(loss_mask)
        loss_l2u = loss_l2u * tf.expand_dims(loss_mask, axis=-1)

        # Return the average unsupervised loss.
        avg_unsup_loss = (tf.reduce_sum(loss_l2u) /
                        tf.maximum(tf.reduce_sum(loss_mask) * FLAGS.nclass, 1))
        return avg_unsup_loss

    def model(self, batch, lr, wd, ema, beta, w_match, warmup_kimg=1024, nu=2, mixmode='xxy.yxy', **kwargs):
        hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]

        # Create placeholders for the labeled images, unlabeled images,
        # and the ground truth supervised labels respectively.
        x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')
        y_in = tf.placeholder(tf.float32, [None, nu] + hwc, 'y')
        l_in = tf.placeholder(tf.int32, [None], 'labels')
        wd *= lr
        w_match *= tf.clip_by_value(tf.cast(self.step, tf.float32) / (warmup_kimg << 10), 0, 1)
        augment = MixMode(mixmode)
        classifier = functools.partial(self.classifier, **kwargs)

        y = tf.reshape(tf.transpose(y_in, [1, 0, 2, 3, 4]), [-1] + hwc)
        guess = self.guess_label(tf.split(y, nu), classifier, T=0.5, **kwargs)
        ly = tf.stop_gradient(guess.p_target)
        lx = tf.one_hot(l_in, self.nclass)

        # Create MixUp examples.
        xy, labels_xy = augment([x_in] + tf.split(y, nu), [lx] + [ly] * nu, [beta, beta])
        x, y = xy[0], xy[1:]
        labels_x, labels_y = labels_xy[0], tf.concat(labels_xy[1:], 0)
        del xy, labels_xy

        # Create batches that represent both labeled and unlabeled batches.
        # For more, see google-research/mixmatch/issues/5.
        batches = layers.interleave([x] + y, batch)
        skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        logits = [classifier(batches[0], training=True)]
        post_ops = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if v not in skip_ops]
        for batchi in batches[1:]:
            logits.append(classifier(batchi, training=True))
        logits = layers.interleave(logits, batch)
        logits_x = logits[0]
        logits_y = tf.concat(logits[1:], 0)

        # Calculate supervised and unsupervised losses.
        loss_xe = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_x, logits=logits_x)
        if FLAGS.tsa != "none":
            print("Using training signal annealing...")
            loss_xe = self.anneal_sup_loss(logits_x, labels_x, loss_xe, self.step)
        else:
            loss_xe = tf.reduce_mean(loss_xe)

        loss_l2u = tf.square(labels_y - tf.nn.softmax(logits_y))


        if FLAGS.percent_mask > 0:
            print("Using percent-based confidence masking...")
            loss_l2u = self.percent_confidence_mask_unsup(logits_y, labels_y, loss_l2u)
        else:
            loss_l2u = tf.reduce_mean(loss_l2u)

        # Calculate largest predicted probability for each image.
        unsup_prob = tf.nn.softmax(logits_y, axis=-1)
        tf.summary.scalar('losses/min_unsup_prob', tf.reduce_min(tf.reduce_max(unsup_prob, axis=-1)))
        tf.summary.scalar('losses/mean_unsup_prob', tf.reduce_mean(tf.reduce_max(unsup_prob, axis=-1)))
        tf.summary.scalar('losses/max_unsup_prob', tf.reduce_max(tf.reduce_max(unsup_prob, axis=-1)))

        # Print losses to tensorboard.
        tf.summary.scalar('losses/xe', loss_xe)
        tf.summary.scalar('losses/l2u', loss_l2u)
        tf.summary.scalar('losses/overall', loss_xe + w_match * loss_l2u)

        # Applying EMA weights to model. Conceptualized by Tarvainen & Valpola, 2017
        # See https://arxiv.org/abs/1703.01780 for more.
        ema = tf.train.ExponentialMovingAverage(decay=ema)
        ema_op = ema.apply(utils.model_vars())
        ema_getter = functools.partial(utils.getter_ema, ema)
        post_ops.append(ema_op)
        post_ops.extend([tf.assign(v, v * (1 - wd)) for v in utils.model_vars('classify') if 'kernel' in v.name])

        train_op = tf.train.AdamOptimizer(lr).minimize(loss_xe + w_match * loss_l2u, colocate_gradients_with_ops=True)
        with tf.control_dependencies([train_op]):
            train_op = tf.group(*post_ops)

        # Tuning op: only retrain batch norm.
        skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        classifier(batches[0], training=True)
        train_bn = tf.group(*[v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                              if v not in skip_ops])

        return EasyDict(
            x=x_in, y=y_in, label=l_in, train_op=train_op, tune_op=train_bn,
            classify_raw=tf.nn.softmax(classifier(x_in, training=False)),  # No EMA, for debugging.
            classify_op=tf.nn.softmax(classifier(x_in, getter=ema_getter, training=False)),
            eval_loss_op=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=classifier(x_in, getter=ema_getter, training=False), 
                labels=tf.one_hot(l_in, self.nclass))))


def get_dataset():
    assert FLAGS.dataset in DATASETS.keys() or FLAGS.custom_dataset, "Please enter a valid dataset name or use the --custom_dataset flag."

    # CIFAR10, CIFAR100, STL10, and SVHN are the pre-configured datasets
    # with each dataset's default augmentation.
    if FLAGS.dataset in DATASETS.keys():
        dataset = DATASETS[FLAGS.dataset]()

    # If the dataset has not been pre-configured, create it.
    else:
        label_size = [int(size) for size in FLAGS.label_size]
        valid_size = [int(size) for size in FLAGS.valid_size]

        augment_dict = {"cifar10": augment_cifar10, "cutout": augment_cutout, "svhn": augment_svhn, "stl10": augment_stl10, "color": augment_color}
        augmentation = augment_dict[FLAGS.augment]

        DATASETS.update([DataSet.creator(FLAGS.dataset.split(".")[0], seed, label, valid, [augmentation, stack_augment(augmentation)], \
            nclass=FLAGS.nclass, height=FLAGS.img_size, width=FLAGS.img_size, do_memoize=FLAGS.memoize)
                         for seed, label, valid in
                         itertools.product(range(2), label_size, valid_size)])
        dataset = DATASETS[FLAGS.dataset]()

    return dataset


def main(argv):
    del argv

    # Num of augmentations to perform on each image and measure consistency loss.
    # Performance does not significantly increase with more augmentations.
    assert FLAGS.nu == 2

    dataset = get_dataset()

    log_width = utils.ilog2(dataset.width)
    model = RealMix(
        os.path.join(FLAGS.train_dir, dataset.name),
        dataset,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        arch=FLAGS.arch,
        batch=FLAGS.batch,
        nclass=dataset.nclass,
        ema=FLAGS.ema,
        beta=FLAGS.beta,
        w_match=FLAGS.w_match,
        scales=FLAGS.scales or (log_width - 2),
        filters=FLAGS.filters,
        repeat=FLAGS.repeat,
        tsa=FLAGS.tsa,
        ood_mask=FLAGS.percent_mask,
        augmentation=FLAGS.augment)

    # if FLAGS.perform_inference:
    #     print("Performing inference...")
    #     assert FLAGS.inference_dir and FLAGS.inference_ckpt
    #     inference_dir = FLAGS.inference_dir
    #     inference_ckpt = FLAGS.inference_ckpt

    #     # images = model.session.run(memoize(default_parse(dataset([inference_dir]))).prefetch(10))

    #     if inference_dir[-1] != "/":
    #         inference_dir += "/"
    #     inference_img_paths = [path for path in glob.glob(inference_dir + "*.png")]
    #     images = np.asarray([plt.imread(img_path) for img_path in inference_img_paths])
    #     images = images * (2.0 / 255) - 1.0
    #     model.eval_mode(ckpt=inference_ckpt)
    #     # batch = FLAGS.batch
    #     feed_extra = None
    #     logits = [model.session.run(model.ops.classify_op, feed_dict={
    #         model.ops.x: images[0:10], **(feed_extra or {})})]

    #     print(np.asarray(logits).shape)
    #     print(logits)
    #     for i in range(10):
    #         print(np.amax(logits, axis=-1)[:, i], inference_img_paths[i])

    print("Preparing to train the %s dataset with %d classes, img_size of %d, %s augmentation, %s tsa schedule, %f weight decay, and learning rate of %f using RealMix" \
            % (FLAGS.dataset, FLAGS.nclass, FLAGS.img_size, FLAGS.augment, FLAGS.tsa, FLAGS.wd, FLAGS.lr))
    model.train(FLAGS.train_kimg << 10, FLAGS.report_kimg << 10)


if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_float('wd', 0.02, 'Weight decay.')
    flags.DEFINE_float('ema', 0.999, 'Exponential moving average of params.')
    flags.DEFINE_float('beta', 0.75, 'Mixup beta distribution.')
    flags.DEFINE_float('w_match', 75, 'Weight for distribution matching loss.')
    flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    flags.DEFINE_bool('custom_dataset', True, 'True if using a custom dataset.')
    flags.DEFINE_integer('nclass', 10, 'Number of classes present in custom dataset.')
    flags.DEFINE_integer('img_size', 32, 'Size of Images in custom dataset')
    flags.DEFINE_spaceseplist('label_size', ['250', '500', '1000', '2000', '4000'], 'List of different labeled data sizes.')
    flags.DEFINE_spaceseplist('valid_size', ['1', '500'], 'List of different validation sizes.')
    flags.DEFINE_enum('tsa', "none", enum_values=["none", "linear_schedule", "log_schedule", "exp_schedule"], help="anneal schedule of training signal annealing. ""tsa='' means not using TSA. See the paper for other schedules.")
    flags.DEFINE_float('percent_mask', -1, 'Confidence value above which the loss for an unsupervised example is masked.')
    flags.DEFINE_enum('augment', 'cifar10', enum_values=["cifar10", "color", "cutout", "svhn", "stl10"], help='Type of augmentation to use, as defined in libml.data.py')
    flags.DEFINE_bool('perform_inference', False, 'True if performing inference on a set of images.')
    flags.DEFINE_string('inference_dir', '', 'Directory of images to perform inference on.')
    flags.DEFINE_string('inference_ckpt', '', 'Checkpoint to perform inference with.')
    flags.DEFINE_bool('memoize', True, 'True if the dataset can be modified in memory.')
    FLAGS.set_default('dataset', 'cifar10.3@250-5000')
    FLAGS.set_default('batch', 64)
    FLAGS.set_default('lr', 0.002)
    FLAGS.set_default('train_kimg', 1 << 16)
    app.run(main)
