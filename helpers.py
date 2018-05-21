import numpy as np
import os
from PIL import Image
import numpy as np
from keras import backend as K
from keras import utils


def load_train_data():
    imgs_train = np.load('train_imgs.npy')
    imgs_mask_train = np.load('train_mask.npy')
    return imgs_train, imgs_mask_train


def load_test_data():
    imgs_test = np.load('test_imgs.npy')
    imgs_id = np.load('test_maks.npy')
    return imgs_test, imgskitti__id


# kitti_palette = {
#     (128, 128, 128): 1,
#     (128, 0, 0): 2,
#     (128, 64, 128): 3,
#     (0, 0, 192): 4,
#     (64, 64, 128): 5,
#     (128, 128, 0): 6,
#     (192, 192, 128): 7,
#     (64, 0, 128): 8,
#     (192, 128, 128): 9,
#     (64, 64, 0): 10,
#     (0, 128, 192): 11,
#     (0, 0, 0): 0,
# }

kitti_palette = {
    0: (0, 0, 0),
    1: (128, 128, 128),
    2: (128, 0, 0),
    3: (128, 64, 128),
    4: (0, 0, 192),
    5: (64, 64, 128),
    6: (128, 128, 0),
    7: (192, 192, 128),
    8: (64, 0, 128),
    9: (192, 128, 128),
    10: (64, 64, 0),
    11: (0, 128, 192)
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

rwth_kitti_palette = {
    0: (0, 0, 0),
    1: (255, 153,  0),
    2: (0, 255, 0),
    3: (255, 0,  0),
    4: (255, 0, 255),
    5: (153, 153, 153),
    6: (0, 255, 255),
    7: (255, 0, 153),
    8: (0, 0, 255),
    9: (153, 0, 255),
    10: (0, 153, 255),
    11: (255, 255, 153)
}


def convert_to_labels(masks, load_from_file=False, data_set='kitti'):
    labels = np.zeros((masks.shape[0], masks.shape[1], masks.shape[2], 12), dtype=np.uint8)

    if(load_from_file and os.path.exists('labels.npy')):
        labels = np.load('labels.npy')
    else:
        if(data_set == "kitti"):
            selected_palette = kitti_palette
        elif(data_set == "rwth"):
            selected_palette = rwth_kitti_palette
        count = 0
        for image in masks:
            percentage = int(100*count/masks.shape[0])
            s = str(percentage) + '%'  
            print('{0}\r'.format(s), end='') 
            background = np.all(image == np.array(selected_palette[0]).reshape(1, 1, 3), axis=2).astype('uint8')
            sky = np.all(image == np.array(selected_palette[1]).reshape(1, 1, 3), axis=2).astype('uint8')
            building = np.all(image == np.array(selected_palette[2]).reshape(1, 1, 3), axis=2).astype('uint8')
            road = np.all(image == np.array(selected_palette[3]).reshape(1, 1, 3), axis=2).astype('uint8')
            sidewalk = np.all(image == np.array(selected_palette[4]).reshape(1, 1, 3), axis=2).astype('uint8')
            fence = np.all(image == np.array(selected_palette[5]).reshape(1, 1, 3), axis=2).astype('uint8')
            vegetation = np.all(image == np.array(selected_palette[6]).reshape(1, 1, 3), axis=2).astype('uint8')
            pole = np.all(image == np.array(selected_palette[7]).reshape(1, 1, 3), axis=2).astype('uint8')
            car = np.all(image == np.array(selected_palette[8]).reshape(1, 1, 3), axis=2).astype('uint8')
            sign = np.all(image == np.array(selected_palette[9]).reshape(1, 1, 3), axis=2).astype('uint8')
            pedestrian = np.all(image == np.array(selected_palette[10]).reshape(1, 1, 3), axis=2).astype('uint8')
            cyclist = np.all(image == np.array(selected_palette[11]).reshape(1, 1, 3), axis=2).astype('uint8')
            categorical_labels = np.dstack([background, sky, building, road, sidewalk, fence, vegetation, pole, car, sign, pedestrian, cyclist])
            labels[count] = categorical_labels.astype(np.uint8)
            count += 1
        np.save('labels.npy', labels)
    return labels


def get_images_and_masks(image_folder, mask_folder, height, width, load_from_file=False):
    if(load_from_file and os.path.exists('images.npy') and os.path.exists('masks.npy')):
        images = np.load('images.npy')
        masks = np.load('masks.npy')
    else:
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
        np.save('images.npy', images)
        np.save('masks.npy', masks)
    return images, masks


def get_model_memory_usage(batch_size, model):
    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes

def normalized(rgb):
    #return rgb/255.0
    norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)

    b=rgb[:,:,0]
    g=rgb[:,:,1]
    r=rgb[:,:,2]

    norm[:,:,0]=cv2.equalizeHist(b)
    norm[:,:,1]=cv2.equalizeHist(g)
    norm[:,:,2]=cv2.equalizeHist(r)

    return norm