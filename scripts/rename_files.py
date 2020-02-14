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
import glob
import sys
import argparse
import tqdm

"""
- Script to rename images in a directory to the format used by this repository.
- Images will take on the classname of the directory they are in, and renamed starting with 000.
- Any directories containing the name "all" will be skipped, useful if you want to keep a backup of images.
"""
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--targetdir", help="Directory containing files to rename")
parser.add_argument("-i", "--index", help="Starting number from which to rename files.")
parser.add_argument("-f", "--filetype", help="File type of images - png or jpg.")
args, leftovers = parser.parse_known_args()

start_num = 0

if args.index:
    start_num = int(args.index)

assert args.targetdir

if not args.filetype or args.filetype == "png":
    filetype = ".png"
else:
    filetype = ".jpg"

if args.targetdir[-1] is not "/":
    args.targetdir += "/"

files = [os.path.abspath(f) for f in glob.glob(args.targetdir+"**/*"+filetype, recursive=True)]

files.sort()

class_names = set()

print("Total image count: ", len(files))

for index, filename in enumerate(tqdm.tqdm(files)):

    if "all" in os.path.dirname(filename).split("/")[-1]:
        print("Skipping directory with all files in it.")
        continue

    filepath_arr = filename.split("/")[:-1]
    filepath = "/".join(filepath_arr)
    class_name = os.path.dirname(filename).split("/")[-1].replace("-", "")
    file_num = index+start_num

    # New filename will be listed with an index first, followed by
    # its classname provided by its directory location.
    # Ex: 4931_dog.png, located in directory "dog".
    new_filename = filepath + "/" + str(file_num).zfill(len(str(len(files) + start_num))) + "_" + class_name + filetype
    class_names.add(class_name)

    # NOTE: File is renamed in place.
    os.rename(filename, new_filename)

# print(len(class_names), " classes in this dataset: ",class_names)
