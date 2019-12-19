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

import numpy as np
import tensorflow as tf
from absl import flags, app

# Function to perform cutout (random-erasing) augmentation on an image.
# Implemented from yu4u/cutout-random-erasing and adapted to work in a tensorflow graph

def get_random_eraser(p=0.5, width = 32, height = 32, channels = 32, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h = height
        img_w = width
        img_c = channels
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        c = np.random.uniform(v_l, v_h)
        input_img_var = tf.Variable(input_img, name="input_image", expected_shape=[height, width, 3])
        assign_x = tf.assign(input_img_var[top:top + h, left:left + w, :], c)

        with tf.Session() as sess, tf.device('cpu:0'):
            sess.run([assign_x])
            sess.run(tf.global_variables_initializer())
            return input_img_var.eval()

    return eraser
