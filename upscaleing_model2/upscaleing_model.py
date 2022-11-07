import cv2 as cv
import keras.losses
import numpy as np
from keras import layers
from keras.models import Model
from keras.optimizers import Adam
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Upscaler:
    input_filepath = None
    target_filepath = None
    data_amount = 1024
    epoch = 1
    image_size = (512, 512)
    upsampling = 2
    slide = 0
    batch_size = 16

    def __init__(self, image_size):
        self.image_size = image_size

        input = layers.Input(shape=(self.image_size[0], self.image_size[1], 1))

        # Encoder
        x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(input)
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        x = layers.Conv2D(1, (3, 3), activation="linear", padding="same")(x)

        self.model = Model(input, x)
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss=keras.losses.MeanSquaredError())
        self.model.summary()

    def read_data(self):
        self.x = []
        self.y = []

        for i in range(self.slide * self.data_amount, self.data_amount * (self.slide + 1)):
            self.input_image = cv.cvtColor(cv.imread(self.input_filepath + "image_" + str(i) + ".png"), cv.COLOR_BGR2GRAY)
            lowered_size = (int(self.image_size[0] / self.upsampling), int(self.image_size[1] / self.upsampling))
            self.input_image = cv.resize(self.input_image, lowered_size, interpolation=cv.INTER_AREA)
            self.input_image = cv.resize(self.input_image, (self.image_size[0], self.image_size[1]))

            self.target_image = cv.imread(self.target_filepath + "image_" + str(i) + ".png")
            self.target_image = cv.resize(self.target_image, (self.image_size[0], self.image_size[1]), interpolation=cv.INTER_AREA)

            self.x.append(self.input_image)
            self.y.append(self.target_image)

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


    def display(self, test_filename, original_filename = None):
        test = cv.resize(cv.cvtColor(cv.imread(test_filename), cv.COLOR_BGR2GRAY),
                         (int(IMAGE_SIZE / 2), int(IMAGE_SIZE / 2)))
        test = cv.resize(test, (IMAGE_SIZE, IMAGE_SIZE))
        test = np.reshape(test, (1, IMAGE_SIZE, IMAGE_SIZE, 1))
        test = np.asarray(test, np.float32) / 255.0

        result = upscaler.model.predict(test)
        result = np.reshape(result, (IMAGE_SIZE, IMAGE_SIZE))

        original = None
        if original_filename is not None:
            original = cv.resize(cv.imread(original_filename), (IMAGE_SIZE, IMAGE_SIZE))
            original = np.reshape(original, (IMAGE_SIZE, IMAGE_SIZE, 3))
            original = np.asarray(original, np.float32) / 255.0

        input = np.reshape(test, (IMAGE_SIZE, IMAGE_SIZE))
        show_window = True
        while show_window:
            cv.imshow("result", cv.resize(result, (1024, 1024)))
            if original_filename is not None:
                cv.imshow("original", original)
            cv.imshow("input", cv.resize(input, (1024, 1024)))
            show_window = cv.waitKey(1) == -1


IMAGE_SIZE = 256
upscaler = Upscaler((IMAGE_SIZE, IMAGE_SIZE))
upscaler.input_filepath = "../newyork_monochrome_images/"
upscaler.target_filepath = "../newyork_monochrome_images/"
upscaler.data_amount = 1024
upscaler.upsampling = 2
upscaler.batch_size = 8
upscaler.slide = 5

#upscaler.train(6, save=False, read=4)

upscaler.model.load_weights("model4.h5")
upscaler.display("../newyork_monochrome_images/image_0.png")