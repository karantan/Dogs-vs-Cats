import numpy

from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from src.utils import get_logger

logger = get_logger()


IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224


vgg_mean = numpy.array(
    [123.68, 116.779, 103.939], dtype=numpy.float32).reshape((3, 1, 1))
model = load_model(
    'models/dogs_cats_v1.h5', custom_objects={'vgg_mean': vgg_mean})


def predict(image_path: str, model):
    image_data = load_img(image_path, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))
    image_data = img_to_array(image_data)
    return model.predict(numpy.array([image_data]))
