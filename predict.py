from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import RMSprop
import PIL.Image as Image
import numpy as np
import random
import cv2
from vgg_16_segnet_basic import VGG16SegNetBasic
np.set_printoptions(threshold=np.nan)

model = VGG16SegNetBasic(no_of_classes=12, height=224, width=224)
optimizer = RMSprop(lr=0.001)
model.load_weights('weights.h5')
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
test_image = image.load_img('scene.png', target_size=(224, 224, 3))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
prob = model.predict(test_image)[0]


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

num_classes = 12
# print(prob[:, :, 0])
color_image = np.zeros((224, 224, 3), dtype='uint8')
prediction = np.argmax(prob, axis=2).astype('uint8')
inds = np.unravel_index(np.argmax(prob, axis=None), prob.shape)
color_image[inds] = (255, 0, 0)
# print(prediction)
# color_image = np.array([kitti_palette[k] for k in prediction.ravel()]).reshape(prediction.shape + (3,))
# color_image = np.array(color_image, dtype='uint8')
with open('segmented.png', 'wb') as out_file:
    Image.fromarray(color_image).save(out_file)
# print(np.any(result[:, :, 10] > 5))
# seg_img = cv2.resize(seg_img, (224, 224))
# cv2.imwrite('segmented.png', seg_img)
