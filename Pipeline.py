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
    input = layers.Input(shape=(256, 256, 1))

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

    model = Model(input, x)
    model.compile(optimizer="adam", loss=keras.losses.BinaryCrossentropy())
    #model.summary()
    return model


def generate_upscaler():
    input = layers.Input(shape=(256, 256, 1))

    # Encoder
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(input)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(1, (3, 3), activation="linear", padding="same")(x)

    model = Model(input, x)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=keras.losses.MeanSquaredError())
    #model.summary()
    return model


def generate_colorer():
    input = layers.Input(shape=(256, 256, 1))

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

    model = Model(input, x)
    model.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())
    #model.summary()
    return model


def generate_full_model():
    d_model = generate_denoiser()
    u_model = generate_upscaler()
    c_model = generate_colorer()

    full_model_Input = layers.Input((256, 256, 1))
    full_layers = d_model(full_model_Input)
    full_layers = u_model(full_layers)
    full_layers = c_model(full_layers)
    full_model = Model(full_model_Input, full_layers)
    full_model.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())

    return full_model


def test_model(model, test_input, input_shape, output_shape=(1, 256, 256, 1), original=None):
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

    cv2.imshow("input", cv2.resize(test, (256, 256)))
    cv2.imshow("result", cv2.resize(result, (256, 256)))
    cv2.imshow("original", cv2.resize(original, (256, 256)))

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

test_image = cv2.resize(cv2.resize(cv2.cvtColor(cv2.imread("newyork_low_noisy_monochrome_images/image_0.png"), cv2.COLOR_BGR2GRAY), (128, 128)), (256, 256))
test_image = np.reshape(test_image, (1, 256, 256, 1))
test_image = np.asarray(test_image, np.float32) / 255.0

original_image = cv2.resize(cv2.cvtColor(cv2.imread("newyork_monochrome_images/image_0.png"), cv2.COLOR_BGR2GRAY), (256, 256))
original_image = np.reshape(original_image, (1, 256, 256, 1))
original_image = np.asarray(original_image, np.float32) / 255.0

colored_original = cv2.resize(cv2.imread("newyork_original_images/image_0.png"), (256, 256))
colored_original = np.reshape(colored_original, (1, 256, 256, 3))
colored_original = np.asarray(colored_original, np.float32) / 255.0

#print("press any key while windows are in focus to see the next model")
#output = test_model(model=d_model, test_input=test_image, input_shape=(1, 256, 256, 1), original=original_image)
#output = test_model(model=u_model, test_input=cv2.resize(cv2.resize(output, (128, 128)), (256, 256)), input_shape=(1, 256, 256, 1), original=original_image)
#test_model(c_model, output, (1, 256, 256, 1), (1, 256, 256, 3), original=colored_original)

full_model_Input = layers.Input((256, 256, 1))
full_layers = d_model(full_model_Input)
full_layers = u_model(full_layers)
full_layers = c_model(full_layers)
full_model = Model(full_model_Input, full_layers)
full_model.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())

result = full_model.predict(test_image)
result = np.reshape(result, (256, 256, 3))
cv2.imshow("final", result)
cv2.waitKey(0)

#full_model.save_weights("full_model256.h5")
