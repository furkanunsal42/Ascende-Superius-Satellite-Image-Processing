import cv2 as cv
import keras.losses
import numpy as np

from keras import layers
from keras.models import Model

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

input = layers.Input(shape=(1024, 1024, 1))

# Encoder
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

model = Model(input, x)
model.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())
model.summary()

input_filepath = "../newyork_low_noisy_monochrome_images/"
target_filepath = "../newyork_monochrome_images/"
input_filepath = "../newyork_monochrome_images/"

data_amount = 512
slide = 22

while True:
    x = []
    y = []

    for i in range(slide * data_amount, data_amount * (slide + 1)):
        x.append(cv.cvtColor(cv.imread(input_filepath + "image_" + str(i) + ".png"), cv.COLOR_BGR2GRAY))
        y.append(cv.cvtColor(cv.imread(target_filepath + "image_" + str(i) + ".png"), cv.COLOR_BGR2GRAY))

    x = np.array(x, np.float32)
    y = np.array(y, np.float32)

    x = x / 255.0
    y = y / 255.0

    print(x.shape, y.shape, x[0].shape)
    a = np.reshape(np.array(x[0]), (1, 1024, 1024, 1))

    model.load_weights("model" + str(slide-1) + ".h5")
    model.fit(x, y, batch_size=2, epochs=1)

    result = model.predict(a)
    result = np.reshape(result, (1024, 1024, 1))

    model.save("model" + str(slide) + ".h5")

    cv.imshow("result", result)
    cv.imshow("input", np.reshape(np.array(x[0]), (1024, 1024, 1)))
    cv.waitKey(0)
    cv.destroyAllWindows()

    slide += 1
