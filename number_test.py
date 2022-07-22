import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

model = tf.keras.models.load_model('my_mnist.h5')


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, [28, 28])
    image /= 255.0  # normalize to [0,1] range
    image = tf.reshape(image, [28, 28, 1])
    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


filepath = 'hh.jpg'
test_my_img = load_and_preprocess_image(filepath)
test_my_img = (np.expand_dims(test_my_img, 0))
my_result = model.predict(test_my_img)
print('自己的图片预测值 =', np.argmax(my_result))

