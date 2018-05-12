# Importing the Keras libraries and packages
from keras.models import Sequential
from keras import losses, optimizers
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import utils
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import os
import platform
if platform.system() is 'Windows':
    import win_unicode_console
    win_unicode_console.enable()

from segnet_basic import SegNetBasic
from segnet import SegNet
from helpers import get_images_and_masks, convert_to_labels, get_model_memory_usage
K.set_image_data_format('channels_last')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
K.tensorflow_backend.set_session(sess)

img_w = 480
img_h = 352
n_classes = 12
n_train = 399
n_val = 46
seed = 1
TRAIN_IMAGE_DIR = 'kitti/train/images/'
TRAIN_MASK_DIR = 'kitti/train/labels/'

VAL_IMAGE_DIR = 'kitti/test/images/'
VAL_MASK_DIR = 'kitti/test/labels/'
BATCH_SIZE = 2

model = SegNet(no_of_classes=n_classes, height=img_h, width=img_w)
optimizer = Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
utils.print_summary(model)
print('Model size: ' + str(get_model_memory_usage(BATCH_SIZE, model)) + ' GB')
print("Model compiled")

print("Generating dataset")
images, masks = get_images_and_masks(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, img_h, img_w, load_from_file=True)
print("Images imported")
print("Creating labels")
labels = convert_to_labels(masks, load_from_file=True)
print("Labels created")
image_datagen = ImageDataGenerator()
mask_datagen = ImageDataGenerator()

image_generator = image_datagen.flow(images, seed=seed, batch_size=BATCH_SIZE)
mask_generator = mask_datagen.flow(labels, seed=seed, batch_size=BATCH_SIZE)

tensor_board_callback = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)
model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

train_generator = zip(image_generator, mask_generator)
print("Training")
model.fit(images, labels, batch_size=BATCH_SIZE, epochs=20, verbose=1, validation_split=0.2, callbacks=[model_checkpoint, tensor_board_callback])
model.save('fcn.h5')

