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
import os
import time
import glob
import keras
from keras import applications
from keras.preprocessing import image
from classification_models.resnet import ResNet18, preprocess_input
from skimage.io import imread
from skimage.transform import resize
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras import backend
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-tr", "--traindir", help="Directory for Custom Training Images")
parser.add_argument("-ts", "--testdir", help="Directory for Custom Test Images")
parser.add_argument("-e", "--epochs", help="Number of epochs")
parser.add_argument("-s", "--save", help="Name to save model weights as:")
parser.add_argument("-l", "--load", help="File to load model weights from:")
args, leftovers = parser.parse_known_args()

img_size = 32
batch_size = 64

if args.epochs:
    epochs = int(args.epochs)
else:
    epochs = 250

assert args.traindir
assert args.testdir
assert args.save

# Add a / to the paths
if args.traindir[-1] is not "/":
	args.traindir += "/"

if args.testdir[-1] is not "/":
	args.testdir += "/"

train_path_list = glob.glob(args.traindir + '*.jpg')
test_path_list = glob.glob(args.testdir + '*.jpg')

train_img_data = []
train_img_labels = []
test_img_data = []
test_img_labels = []

# Read train and test images and labels
for path in train_path_list:
    classname = path.split('_')[-1][:-4]
    x = imread(path)
    x = resize(x, (img_size, img_size)) * 255
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    train_img_data.append(x)
    train_img_labels.append(classname)


for path in test_path_list:
    classname = path.split('_')[-1][:-4]
    x = imread(path)
    x = resize(x, (img_size, img_size)) * 255
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    test_img_data.append(x)
    test_img_labels.append(classname)

train_img_data = np.asarray(train_img_data)
if train_img_data.shape[0] == 0:
    print("No training images found on provided path: ", args.traindir)
    sys.exit(0)
train_img_data = np.rollaxis(train_img_data,1,0)[0]

test_img_data = np.asarray(test_img_data)
if test_img_data.shape[0] == 0:
    print("No test images found on provided path: ", args.testdir)
    sys.exit(0)
test_img_data = np.rollaxis(test_img_data,1,0)[0]

classes = sorted(list(set(train_img_labels+test_img_labels)))
train_img_labels = np.asarray([classes.index(classname) for classname in train_img_labels])
test_img_labels = np.asarray([classes.index(classname) for classname in test_img_labels])
num_classes = len(classes)

# convert class labels to on-hot encoding
Y_train = np_utils.to_categorical(train_img_labels, num_classes)
Y_test = np_utils.to_categorical(test_img_labels, num_classes)

# Shuffle the dataset
X_train, y_train = shuffle(train_img_data,Y_train, random_state=2)
X_test, y_test = shuffle(test_img_data,Y_test, random_state=2)

# Create data augmentation generators
train_datagen = image.ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = image.ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

it_train = train_datagen.flow(X_train, y_train)
it_test = test_datagen.flow(X_test, y_test)

print("Training Input Shape: ", X_train.shape, y_train.shape, " Test Input Shape: ", X_test.shape, y_test.shape)

base_model = ResNet18((img_size, img_size, 3), weights='imagenet', classes=num_classes, include_top=False)
avg_pool = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(num_classes, activation='softmax')(avg_pool)
model = Model([base_model.input], output)

if args.load:
    model.load_weights(args.load)

model.compile(loss='categorical_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])
# model.summary()

t=time.time()

hist = model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=X_train.shape[0] // batch_size,
                        validation_data=(X_test, y_test),
                        epochs=epochs, verbose=1)

print('Training time: %s' % (t - time.time()))

(loss, accuracy) =  model.evaluate(X_test, y_test, batch_size=64, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

model.save(args.save + '.h5')

###########################################################################################################################
# # Custom_resnet_model_1
# # Training the classifier alone
# image_input = Input(shape=(img_size, img_size, 3))
#
# model = ResNet50(input_tensor=image_input, include_top=True, weights='imagenet')
#
# last_layer = model.get_layer('avg_pool').output
# out = Dense(num_classes, activation='softmax', name='output_layer')(last_layer)
# custom_resnet_model = Model(inputs=image_input,outputs= out)
# custom_resnet_model.summary()
#
# for layer in custom_resnet_model.layers[:-1]:
# 	layer.trainable = False
#
# # custom_resnet_model.layers[-1].trainable
#
# custom_resnet_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#
# t=time.time()
# # hist = custom_resnet_model.fit(X_train, y_train, batch_size=64, epochs=12, verbose=1, validation_data=(X_test, y_test))
# print(X_train.shape)
# hist = custom_resnet_model.fit_generator(it_train, epochs=12, verbose=1, steps_per_epoch=66,validation_data=(X_test, y_test))
# print('Training time: %s' % (t - time.time()))
# (loss, accuracy) = custom_resnet_model.evaluate(X_test, y_test, batch_size=64, verbose=1)
#
# print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

# ###########################################################################################################################
#
# Fine tune the resnet 50
# image_input = Input(shape=(img_size, img_size, 3))
# model = ResNet18(input_tensor=image_input, weights='imagenet',include_top=False)
# model.summary()
# last_layer = model.output
# # add a global spatial average pooling layer
# x = GlobalAveragePooling2D()(last_layer)
# # add fully-connected & dropout layers
# x = Dense(512, activation='relu',name='fc-1')(x)
# x = Dropout(0.5)(x)
# x = Dense(256, activation='relu',name='fc-2')(x)
# x = Dropout(0.5)(x)
# # a softmax layer for 4 classes
# out = Dense(num_classes, activation='softmax',name='output_layer')(x)
#
# # this is the model we will train
# custom_resnet_model2 = Model(inputs=model.input, outputs=out)
#
# custom_resnet_model2.summary()
#
# for layer in custom_resnet_model2.layers[:-6]:
# 	layer.trainable = False
#
# custom_resnet_model2.layers[-1].trainable
#
# custom_resnet_model2.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#
# t=time.time()
# hist = custom_resnet_model2.fit(X_train, y_train, batch_size=32, epochs=12, verbose=1, validation_data=(X_test, y_test))
# print('Training time: %s' % (t - time.time()))
# (loss, accuracy) = custom_resnet_model2.evaluate(X_test, y_test, batch_size=10, verbose=1)
#
# print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
#
############################################################################################
import matplotlib.pyplot as plt
# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(12)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.savefig(args.save + '.jpg')
