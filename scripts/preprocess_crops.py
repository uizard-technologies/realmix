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

import os
import argparse
import sys
from tqdm import tqdm

"""Script to convert images from a directory to a square image of specified dimensions.
"""

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--targetdir", help="Directory containing crops to process")
parser.add_argument("-s", "--savedir", help="Directory to save processed crops")
parser.add_argument("-dim", "--dimension", help="Processed Image Size (ex: 32x32)")
parser.add_argument("-f", "--force", help="Use if targetdir and savedir are the same.")
args, leftovers = parser.parse_known_args()

assert args.targetdir and args.savedir and args.dimension, \
    "Please provide a target directory, save directory, and dimension for cropped images."

if args.targetdir == args.savedir:
    assert args.force, \
    "Target directory cannot be the same as the save directory. Use the --force flag to do so."

crops = os.listdir(path=args.targetdir)

if args.targetdir[-1] is not "/":
    args.targetdir += "/"

if args.savedir[-1] is not "/":
    args.savedir += "/"

for index, crop in enumerate(tqdm(crops, desc="Processing crops...")):
    crop_to_convert = args.targetdir + str(crop)
    dim = args.dimension + "x" + args.dimension

    # Terminal command to convert one image:
    # convert *.jpg -resize 32x32 -gravity center -background "rgb(0,0,0)" -extent 32x32 -set filename:base "%[basename]" ../all_processed_32x32/"%[filename:base].jpg"

    os.system("convert " + crop_to_convert + " -resize " + dim + " -gravity center -background \"rgb(0,0,0)\" -extent " + dim + " -set filename:base \"%[basename]\" " + args.savedir + "\"%[filename:base].jpg\"")
