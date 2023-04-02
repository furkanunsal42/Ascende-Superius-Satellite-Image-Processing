import tensorflow as tf
from keras import Model, Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.losses import MeanSquaredError

import tensorflow_datasets as tfds

import numpy as np
import cv2

dataset_name = "Mnist"

image_size = 32


def preprocess_image(data):
    # center crop image
    height = tf.shape(data["image"])[0]
    width = tf.shape(data["image"])[1]
    crop_size = tf.minimum(height, width)
    image = tf.image.crop_to_bounding_box(
        data["image"],
        (height - crop_size) // 2,
        (width - crop_size) // 2,
        crop_size,
        crop_size,
    )
    # resize and clip
    # for image downsampling it is important to turn on antialiasing
    image = tf.image.resize(image, size=[image_size, image_size], antialias=True)
    return tf.clip_by_value(image / 255.0, 0.0, 1.0)


def prepare_dataset():
    return (
        tfds.load(dataset_name, split="train", shuffle_files=True)
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

data_train = prepare_dataset()

for example in data_train.take(10):
    image, label = example['image'], example['label']
    print(image.shape)
    cv2.imshow("window", image)
    cv2.waitKey(1000)


def get_network(image_size):
    pass