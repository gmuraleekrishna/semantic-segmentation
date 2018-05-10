# Importing the Keras libraries and packages
from keras.models import Sequential
from keras import losses, optimizers
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import utils
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import color, exposure, transform, io

from model import Model
from helpers import get_images_and_masks, convert_to_labels
K.set_image_data_format('channels_last')

img_w = 480
img_h = 352
classes = 12
n_train = 399
n_val = 46
seed = 1
TRAIN_IMAGE_DIR = 'kitti/train/images/'
TRAIN_MASK_DIR = 'kitti/train/labels/'

VAL_IMAGE_DIR = 'kitti/test/images/'
VAL_MASK_DIR = 'kitti/test/labels/'
BATCH_SIZE = 4

print("Model compiled")
model = Model(no_of_classes=classes, height=img_h, width=img_w) 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


print("Generating dataset")
images, masks = get_images_and_masks(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, img_h, img_w)
print("Images imported")
print("Creating labels")
labels = convert_to_labels(masks)
print("Labels created")
image_datagen = ImageDataGenerator()
mask_datagen = ImageDataGenerator()

image_generator = image_datagen.flow(images, seed=seed, batch_size=BATCH_SIZE)
mask_generator = image_datagen.flow(labels, seed=seed, batch_size=BATCH_SIZE)

tensor_board_callback = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)
model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

train_generator = zip(image_generator, mask_generator)
model.fit(images, labels, batch_size=32, epochs=20, verbose=1, shuffle=True, validation_split=0.2, callbacks=[model_checkpoint, tensor_board_callback])
# model.fit_generator(train_generator, epochs=200, steps_per_epoch=n_train//BATCH_SIZE, shuffle=True, callbacks=[model_checkpoint, tensor_board_callback], validation_split=0.2)
model.save('fcn.h5')

