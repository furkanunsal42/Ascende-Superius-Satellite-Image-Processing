from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.losses import MeanSquaredError, BinaryCrossentropy, MeanAbsoluteError
import numpy as np
import cv2
import os
import time
import keyboard

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def generate_full_model():
    input = layers.Input(shape=(256, 256, 1))

    # Encoder
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(input)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(input)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

    full_model = Model(input, x)
    full_model.compile(optimizer=Adam(learning_rate=0.0001), loss=MeanAbsoluteError())
    full_model.summary()

    return full_model


def generate_inputs(input_directory, target_input_directory, training_data_amount=1024, slide=0):
    inputs = []
    targets = []
    for i in range(slide*training_data_amount, (slide+1)*training_data_amount):
        file_name = "image_"+str(i)+".png"
        file_name = target_input_directory + file_name
        target_image = cv2.resize(cv2.imread(file_name), (256, 256))
        targets.append(target_image)

        file_name = "image_" + str(i) + ".png"
        file_name = input_directory + file_name
        input_image = cv2.resize(cv2.resize(cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2GRAY), (256, 256)), (256, 256))
        inputs.append(input_image)

    inputs = np.asarray(np.array(inputs), np.float32) / 255.0
    targets = np.asarray(np.array(targets), np.float32) / 255.0

    inputs = np.reshape(inputs, (training_data_amount, 256, 256, 1))
    targets = np.reshape(targets, (training_data_amount, 256, 256, 3))

    return inputs, targets


def test_full_model(test_input, original=None):
    test = np.reshape(test_input, (1, 256, 256, 1))
    result = full_model.predict(test, verbose=0)
    result = np.reshape(result, (256, 256, 3))
    test = np.reshape(test, (256, 256))

    if original is not None:
        original = np.reshape(original, (256, 256, 3))

    display = True
    initial_time = time.time()

    if original is not None:
        cv2.imshow("original", cv2.resize(original, (256, 256)))
    cv2.imshow("test", cv2.resize(test, (256, 256)))
    cv2.imshow("result", cv2.resize(result, (256, 256)))

    while display: # and time.time() - initial_time < 10:
        display = cv2.waitKey(1) == -1
    cv2.destroyAllWindows()


IMAGE_AMOUNT = 1024
BATCH_SIZE = 8

full_model = generate_full_model()
full_model.load_weights("full_model256_1.h5")

x, y = generate_inputs("newyork_low_noisy_monochrome_images/", "newyork_original_images/", 1, slide=0)
test_full_model(x[0], y[0])
"""
for i in range(2, 10):
    #data generation
    print("data generation has begun slide:" + str(i))
    x, y = generate_inputs("newyork_low_noisy_monochrome_images/", "newyork_original_images/", IMAGE_AMOUNT, slide=i)

    print("initial state")
    test_full_model(x[0], y[0])

    #training loop
    print("classical training for full_model slide:" + str(i))
    full_model.fit(x, y, batch_size=BATCH_SIZE, epochs=1)
    test_full_model(x[0], y[0])

    full_model.save_weights("full_model256_" + str(i) + ".h5")
"""