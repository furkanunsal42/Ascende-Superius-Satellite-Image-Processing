import cv2 as cv
import keras.losses
import numpy as np
from keras import layers
from keras.models import Model

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class Colorer:
    input_filepath = None
    target_filepath = None
    data_amount = 1024
    epoch = 1
    image_size = (1024, 1024)
    slide = 0
    batch_size = 16

    def __init__(self, image_size):
        self.image_size = image_size

        input = layers.Input(shape=(self.image_size[0], self.image_size[1], 1))

        # Encoder
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(input)
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", strides=(2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", strides=(2, 2))(x)
        x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
        x = layers.Conv2D(256, (3, 3), activation="relu", padding="same", strides=(2, 2))(x)
        x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)

        # Decoder
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)

        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)

        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)

        x = layers.Conv2D(2, (3, 3), activation="tanh", padding="same")(x)

        self.model = Model(input, x)
        self.model.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())
        self.model.summary()

    def read_data(self):
        self.x = []
        self.y = []

        for i in range(self.slide * self.data_amount, self.data_amount * (self.slide + 1)):
            input_image = cv.cvtColor(cv.imread(self.input_filepath + "image_" + str(i) + ".png"), cv.COLOR_BGR2GRAY)
            input_image = cv.resize(input_image, (self.image_size[0], self.image_size[1]), interpolation=cv.INTER_AREA)

            target_image = cv.cvtColor(cv.imread(self.target_filepath + "image_" + str(i) + ".png"), cv.COLOR_BGR2LAB)
            target_image = cv.resize(target_image, (self.image_size[0], self.image_size[1]), interpolation=cv.INTER_AREA)

            self.x.append(target_image[:, :, 0])
            self.y.append(target_image[:, :, 1:])

        self.x = np.array(self.x, np.float32)
        self.y = np.array(self.y, np.float32)

        self.x = self.x / 255.0
        self.y = self.y / 255.0

    def train(self, target_slide, save=False, read=-1):
        if read >= 0:
            self.model.load_weights("model" + str(read) + ".h5")
        for i in range(self.slide, target_slide):
            self.read_data()
            self.model.fit(self.x, self.y, batch_size=self.batch_size, epochs=self.epoch)
            if save:
                self.model.save("model" + str(self.slide) + ".h5")
            self.slide = i+1

IMAGE_SIZE = 1024
colorer = Colorer((IMAGE_SIZE, IMAGE_SIZE))
colorer.input_filepath = "../../newyork_monochrome_images/"
colorer.target_filepath = "../../newyork_original_images/"
colorer.data_amount = 64
colorer.batch_size = 1
colorer.slide = 0

def test_model():
    test = cv.resize(cv.cvtColor(cv.imread("../../newyork_monochrome_images/image_0.png"), cv.COLOR_BGR2GRAY), (IMAGE_SIZE, IMAGE_SIZE))
    L = np.reshape(test, (IMAGE_SIZE, IMAGE_SIZE, 1))
    test = np.reshape(test, (1, IMAGE_SIZE, IMAGE_SIZE, 1))
    test = np.asarray(test, np.float32) / 255.0

    original = cv.resize(cv.imread("../../newyork_original_images/image_0.png"), (IMAGE_SIZE, IMAGE_SIZE))

    lab = cv.cvtColor(original, cv.COLOR_BGR2LAB)
    test = np.reshape(lab[:, :, 0], (1, IMAGE_SIZE, IMAGE_SIZE, 1)) / 255.0

    #original_lab = cv.cvtColor(original, cv.COLOR_BGR2LAB) / 128.0
    colorer.model.load_weights("model9.h5")
    result = colorer.model.predict(test)
    print(result[:, :, 1])
    result = np.reshape(result, (IMAGE_SIZE, IMAGE_SIZE, 2))
    final_lab = np.concatenate([lab[:, :, 0, np.newaxcoloring_model.pyis], result[:, :, 0, np.newaxis], result[:, :, 1, np.newaxis]], 2)
    final_lab = cv.cvtColor(final_lab, cv.COLOR_LAB2BGR)

    input = np.reshape(test, (IMAGE_SIZE, IMAGE_SIZE))
    cv.imshow("result", final_lab)
    cv.imshow("original", original)
    cv.imshow("input", input)
    cv.imshow("A target", lab[:, :, 1])
    cv.imshow("A result", final_lab[:, :, 1])
    cv.waitKey(0)
    cv.destroyAllWindows()

colorer.slide = 10
colorer.train(10, save=True, read=9)
test_model()
