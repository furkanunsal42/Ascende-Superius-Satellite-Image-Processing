import cv2
import keras.losses
import numpy as np

from keras import layers
from keras.models import Model
from keras.optimizers import Adam
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def generate_denoiser():
    input = layers.Input(shape=(1024, 1024, 1))

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

    model = Model(input, x, name="denoiser")
    model.compile(optimizer=Adam(learning_rate=0.001), loss=keras.losses.BinaryCrossentropy())
    #model.summary()
    return model


def generate_upscaler():
    input = layers.Input(shape=(512, 512, 1))

    # Encoder
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(input)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

    model = Model(input, x, name="upscaler")
    model.compile(optimizer=Adam(learning_rate=0.001), loss=keras.losses.MeanSquaredError())
    #model.summary()
    return model


def generate_colorer():
    input = layers.Input(shape=(1024, 1024, 1))

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

    model = Model(input, x, name="colorer")
    model.compile(optimizer=Adam(learning_rate=0.001), loss=keras.losses.MeanSquaredError())
    #model.summary()
    return model


def test_model(model, test_input, input_shape, output_shape=(1, 1024, 1024, 1), original=None):
    test = np.reshape(test_input, input_shape)

    result = model.predict(test, verbose=0)
    if output_shape[3] == 1:
        result = np.reshape(result, (output_shape[1], output_shape[2]))
    else:
        result = np.reshape(result, (output_shape[1], output_shape[2], output_shape[3]))

    if original is not None:
        if output_shape[3] == 1:
            original = np.reshape(original, (output_shape[1], output_shape[2]))
        else:
            original = np.reshape(original, (output_shape[1], output_shape[2], output_shape[3]))

    test = np.reshape(test, (input_shape[1], input_shape[2]))
    if input_shape[1] != output_shape[1]:
        test = cv2.resize(test, (output_shape[1], output_shape[2]))

    display = True

    cv2.imshow("input", test)
    cv2.imshow("result", result)
    cv2.imshow("original", original)

    while display:
        display = cv2.waitKey(1) == -1
    cv2.destroyAllWindows()
    return result


d_model = generate_denoiser()
u_model = generate_upscaler()
c_model = generate_colorer()

d_model.load_weights("denoising_model.h5")
u_model.load_weights("upscaleing_model.h5")
c_model.load_weights("coloring_model.h5")

test_image = cv2.cvtColor(cv2.imread("newyork_low_noisy_monochrome_images/image_0.png"), cv2.COLOR_BGR2GRAY)
test_image = np.reshape(test_image, (1, 1024, 1024, 1))
test_image = np.asarray(test_image, np.float32) / 255.0

original_image = cv2.cvtColor(cv2.imread("newyork_monochrome_images/image_0.png"), cv2.COLOR_BGR2GRAY)
original_image = np.reshape(original_image, (1, 1024, 1024, 1))
original_image = np.asarray(original_image, np.float32) / 255.0

colored_original = cv2.imread("newyork_original_images/image_0.png")
colored_original = np.reshape(colored_original, (1, 1024, 1024, 3))
colored_original = np.asarray(colored_original, np.float32) / 255.0

print("press any key while windows are in focus to see the next model")
output = test_model(model=d_model, test_input=test_image, input_shape=(1, 1024, 1024, 1), original=original_image)
output = test_model(model=u_model, test_input=cv2.resize(output, (512, 512)), input_shape=(1, 512, 512, 1), original=original_image)
test_model(c_model, output, (1, 1024, 1024, 1), (1, 1024, 1024, 3), original=colored_original)