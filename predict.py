from keras.models import load_model
from keras.preprocessing import image
import numpy as np

model = load_model('model.h5')
test_image = image.load_img('image.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
if result[0][0] == 1:
    print('dog')
else:
    print('cat')
