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

from tensorflow.contrib.framework.python.framework import checkpoint_utils
from absl import app
from absl import flags
import json
from libml import utils

flags.DEFINE_string("path", "", help="Path to checkpoint to analyze.")
flags.DEFINE_string("json", "", help="Name of json file to save variables.")
FLAGS = flags.FLAGS


def main(argv):
    var_list = checkpoint_utils.list_variables(FLAGS.path)
    with open(FLAGS.json + ".json", 'w', encoding="utf-8") as f:
        json.dump(var_list, f, ensure_ascii=False, indent=4)
    for v in var_list:
        print(v)

if __name__ == '__main__':
    utils.setup_tf()
    app.run(main)