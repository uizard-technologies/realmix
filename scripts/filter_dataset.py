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

"""Script to create SSL splits from a dataset.
"""

from collections import defaultdict
import json
import os

from absl import app
from absl import flags
from libml import utils
from libml.data import DATA_DIR
import numpy as np
import tensorflow as tf
from tqdm import trange, tqdm
from typing import List, Any

flags.DEFINE_string("class_filter", "",
"Comma-delimited list of class numbers from labeled "
"dataset to use during training. Defaults to all classes.")
flags.DEFINE_string("filter_dataset", "", "Dataset to filter.")
flags.DEFINE_string("save_name", "", "Name to save filtered dataset as.")
FLAGS = flags.FLAGS

def get_class(serialized_example):
    return tf.parse_single_example(serialized_example, features={'label': tf.FixedLenFeature([], tf.int64)})['label']


def main(argv):
    assert FLAGS.class_filter and FLAGS.filter_dataset and FLAGS.save_name, "Please provide all 3 arguments."
    input_file = FLAGS.filter_dataset
    save_path = "/".join(input_file.split("/")[:-1]) + "/" + FLAGS.save_name
    count = 0
    id_class = []
    class_id = defaultdict(list)

    dataset = tf.data.TFRecordDataset(input_file)
    dataset = dataset.map(get_class, 4).batch(1 << 10)
    it = dataset.make_one_shot_iterator().get_next()

    classes_to_keep = list(map(int, FLAGS.class_filter.split(",")))
    all_classes = set()
    try:
        # Store each image in a dict by its class number and image id.
        with tf.Session() as session, tqdm(leave=False) as t:
            while 1:
                old_count = count
                for i in session.run(it):
                    all_classes.add(i)
                    if i in classes_to_keep:
                        id_class.append(count)
                        class_id[i].append(count)
                    count += 1

                t.update(count - old_count)
    except tf.errors.OutOfRangeError:
        pass
    print('%d records found' % count)

    nclass = len(class_id)

    print(all_classes, classes_to_keep)

    class_data = [[] for _ in range(nclass)]
    label_writes = 0
    print("Saving to: ", save_path)
    with tf.python_io.TFRecordWriter(save_path) as writer_label:
        pos, loop = 0, trange(count, desc='Writing records')
        for record in tf.python_io.tf_record_iterator(input_file):
            if pos in id_class:
                writer_label.write(record)
                label_writes += 1
            pos += 1
            loop.update()
        loop.close()

    print(label_writes)


if __name__ == '__main__':
    utils.setup_tf()
    app.run(main)
