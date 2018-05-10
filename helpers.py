import numpy as np
import os
from PIL import Image
import numpy as np
import tensorflow as tf

def load_train_data():
    imgs_train = np.load('train_imgs.npy')
    imgs_mask_train = np.load('train_mask.npy')
    return imgs_train, imgs_mask_train


def load_test_data():
    imgs_test = np.load('test_imgs.npy')
    imgs_id = np.load('test_maks.npy')
    return imgs_test, imgs_id


def preprocess(imgs, img_h, img_w):
    imgs_p = np.ndarray((imgs.shape[0], img_h, img_w), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = transform.resize(imgs[i], (img_h, img_w), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

palette = {
    (128, 128, 128): 1,
    (128, 0, 0): 2,
    (128, 64, 128): 3,
    (0, 0, 192): 4,
    (64, 64, 128): 5,
    (128, 128, 0): 6,
    (192, 192, 128): 7,
    (64, 0, 128): 8,
    (192, 128, 128): 9,
    (64, 64, 0): 10,
    (0, 128, 192): 11,
    (0, 0, 0): 0,
}

classes = {
    'sky': 1,
    'building': 2,
    'road': 3,
    'sidewalk': 4,
    'fence': 5,
    'vegetation': 6,
    'pole': 7,
    'car': 8,
    'sign': 9,
    'pedestrian': 10,
    'cyclist': 11
}


def convert_to_labels(masks):
    labels = np.zeros((masks.shape[0], masks.shape[1], masks.shape[2], 1), dtype=np.uint8)
    count = 0
    for image in masks:
        count += 1
        percentage = int(100*count/masks.shape[0])
        s = str(percentage) + '%'  
        print('{0}\r'.format(s), end='') 
        label = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
        for c, i in palette.items():
            m = np.all(image == np.array(c).reshape(1, 1, 3), axis=2)
            label[m] = i
        np.append(labels, label)
    return labels


def get_images_and_masks(image_folder, mask_folder, height, width):
    image_file_names = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f)) and f.endswith(".png")]
    number_of_images = len(image_file_names)
    images = np.zeros((number_of_images, height, width, 3))
    masks = np.zeros((number_of_images, height, width, 3))
    count = 0
    for image_file_name in image_file_names:
        percentage = int(100*count/number_of_images)
        s = str(percentage) + '%'                        # string for output
        print('{0}\r'.format(s), end='') 
        image = Image.open(os.path.join(image_folder, image_file_name))
        mask = Image.open(os.path.join(mask_folder, image_file_name))
        im = np.asarray(image)
        msk = np.asarray(mask)
        image.close()
        mask.close()
        im = np.resize(im, (height, width, 3))
        msk = np.resize(msk, (height, width, 3))
        images[count] = im
        masks[count] = msk
        count += 1
    print("\n")
    return images, masks
