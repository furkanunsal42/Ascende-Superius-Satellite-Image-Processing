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

        input = layers.Input(shape=(self.image_size[0], self.image_size[1], 4))

        # Encoder
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(input)
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(input)

        x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)

        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)

        x = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

        self.model = Model(input, x)
        self.model.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())
        self.model.summary()

    def read_data(self):
        self.x = []
        self.y = []

        for i in range(self.slide * self.data_amount, self.data_amount * (self.slide + 1)):
            input_image = cv.imread(self.input_filepath + "image_" + str(i) + ".png")
            input_image = cv.resize(input_image, (self.image_size[0], self.image_size[1]), interpolation=cv.INTER_AREA)

            high_res_mono_input = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)   # 0.5m resolution
            high_res_mono_input = np.reshape(high_res_mono_input, [self.image_size[0], self.image_size[1], 1])

            low_res_color_input = cv.resize(input_image, (int(self.image_size[0] / 20), int(self.image_size[1] / 20)), interpolation=cv.INTER_AREA)  # 10.0m resolution
            low_res_color_input = cv.resize(low_res_color_input, (self.image_size[0], self.image_size[1]), interpolation=cv.INTER_AREA)

            data_x = np.concatenate([high_res_mono_input, low_res_color_input[:, :, 0:]], 2)
            data_y = input_image

            self.x.append(data_x)
            self.y.append(data_y)

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

    def test_model(self, testing_model_index=None):
        input_image = cv.imread(self.input_filepath + "image_" + str(0) + ".png")
        input_image = cv.resize(input_image, (self.image_size[0], self.image_size[1]), interpolation=cv.INTER_AREA)

        high_res_mono_input = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)  # 0.5m resolution
        high_res_mono_input = np.reshape(high_res_mono_input, [self.image_size[0], self.image_size[1], 1])

        low_res_color_input = cv.resize(input_image, (int(self.image_size[0] / 20), int(self.image_size[1] / 20)),
                                        interpolation=cv.INTER_AREA)  # 10.0m resolution
        low_res_color_input = cv.resize(low_res_color_input, (self.image_size[0], self.image_size[1]),
                                        interpolation=cv.INTER_AREA)

        data_x = np.concatenate([high_res_mono_input, low_res_color_input[:, :, 0:]], 2)
        data_x = np.asarray(data_x, np.float32) / 255.0
        data_x = np.reshape(data_x, [1, self.image_size[0], self.image_size[1], 4])

        if testing_model_index is not None:
            colorer.model.load_weights("model" + str(testing_model_index) + ".h5")

        result = colorer.model.predict(data_x)
        result = np.reshape(result, (self.image_size[0], self.image_size[1], 3))
        model_input = np.reshape(data_x, [self.image_size[0], self.image_size[1], 4])

        cv.imshow("result", result)
        cv.imshow("original", input_image)
        cv.imshow("input_mono", model_input[:, :, 0])
        cv.imshow("input_color", model_input[:, :, 1:])
        cv.waitKey(0)
        cv.destroyAllWindows()


def manual_dressing():
    input_image = cv.imread("../newyork_original_images/image_8000.png")
    input_image = cv.resize(input_image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv.INTER_AREA)

    high_res_mono_input = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)  # 0.5m resolution
    high_res_mono_input = np.reshape(high_res_mono_input, [IMAGE_SIZE, IMAGE_SIZE, 1])

    low_res_color_input = cv.resize(input_image, (int(IMAGE_SIZE / 20), int(IMAGE_SIZE / 20)),
                                    interpolation=cv.INTER_AREA)  # 10.0m resolution
    low_res_color_input = cv.resize(low_res_color_input, (IMAGE_SIZE, IMAGE_SIZE),
                                    interpolation=cv.INTER_AREA)


    color_lab = cv.cvtColor(low_res_color_input, cv.COLOR_BGR2LAB)
    result_lab = np.concatenate([high_res_mono_input, color_lab[:, :, 1:]], 2)
    result = cv.cvtColor(result_lab, cv.COLOR_LAB2BGR)
    cv.imshow("result", result)
    cv.imshow("color_input", low_res_color_input)
    cv.imshow("monospectral", high_res_mono_input)
    cv.imshow("original", input_image)
    cv.waitKey(0)


IMAGE_SIZE = 1024
colorer = Colorer((IMAGE_SIZE, IMAGE_SIZE))
colorer.input_filepath = "../newyork_original_images/"
colorer.target_filepath = "../newyork_original_images/"
colorer.data_amount = 64
colorer.batch_size = 1
colorer.slide = 0

manual_dressing()

colorer.slide = 10
colorer.train(10, save=True, read=9)
colorer.test_model()
