from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.losses import MeanSquaredError, BinaryCrossentropy
import numpy as np
import cv2
import os
import time
import keyboard

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#model

upscaler_input = layers.Input(shape=(256, 256, 1))
upscaler = layers.Conv2D(128, kernel_size=[3, 3], strides=(1, 1), padding="same", activation="relu")(upscaler_input)
upscaler = layers.Conv2D(64, kernel_size=[3, 3], strides=(1, 1), padding="same", activation="relu")(upscaler)
upscaler = layers.Conv2D(32, kernel_size=[3, 3], strides=(1, 1), padding="same", activation="relu")(upscaler)
upscaler = layers.Conv2D(16, kernel_size=[3, 3], strides=(1, 1), padding="same", activation="relu")(upscaler)
upscaler = layers.Conv2D(1, kernel_size=[3, 3], strides=(1, 1), padding="same", activation="sigmoid")(upscaler)
upscaler_model = Model(upscaler_input, upscaler)
upscaler_model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
#upscaler_model.summary()

detective_input_one = layers.Input((256, 256, 1))
detective_input_two = layers.Input((256, 256, 1))
detective_model_one = layers.Conv2D(32, kernel_size=[3, 3], strides=(1, 1), padding="same", activation="relu")(detective_input_one)
detective_model_two = layers.Conv2D(32, kernel_size=[3, 3], strides=(1, 1), padding="same", activation="relu")(detective_input_two)
detective_model = layers.concatenate([detective_model_one, detective_model_two])
detective_model = layers.Conv2D(64, kernel_size=[3, 3], strides=(2, 2), padding="same", activation="relu")(detective_model)
detective_model = layers.Conv2D(32, kernel_size=[3, 3], strides=(2, 2), padding="same", activation="relu")(detective_model)
detective_model = layers.Conv2D(16, kernel_size=[3, 3], strides=(2, 2), padding="same", activation="relu")(detective_model)
detective_model = layers.Conv2D(1, kernel_size=[3, 3], strides=(2, 2), padding="same", activation="sigmoid")(detective_model)
detective_model = layers.Flatten()(detective_model)
detective_model = layers.Dense(2, activation="sigmoid")(detective_model)
detective_model = Model([detective_input_one, detective_input_two], detective_model)
detective_model.compile(optimizer=Adam(learning_rate=0.001), loss=BinaryCrossentropy(), metrics=["accuracy"])
detective_model.summary()

detective_model.trainable = False

full_model_input = layers.Input((256, 256, 1))
fake_image = upscaler_model(full_model_input)
full_detective_input = layers.Input((256, 256, 1))
det_out = detective_model([[fake_image, full_detective_input]])
full_model = Model([full_model_input, full_detective_input], det_out)

full_model.compile(optimizer=Adam(learning_rate=0.001), loss=BinaryCrossentropy(), metrics=["accuracy"])
full_model.summary()


def generate_inputs(input_directory, training_data_amount=1024, slide=0):
    inputs = []
    targets = []
    for i in range(slide*training_data_amount, (slide+1)*training_data_amount):
        file_name = "image_"+str(i)+".png"
        file_name = input_directory + file_name
        target_image = cv2.resize(cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2GRAY), (256, 256))
        targets.append(target_image)

        input_image = cv2.resize(cv2.resize(np.copy(target_image), (128, 128)), (256, 256))
        inputs.append(input_image)

    inputs = np.asarray(np.array(inputs), np.float32) / 255.0
    targets = np.asarray(np.array(targets), np.float32) / 255.0

    inputs = np.reshape(inputs, (training_data_amount, 256, 256, 1))
    targets = np.reshape(targets, (training_data_amount, 256, 256, 1))

    return inputs, targets


def test_upscaler_model(test_input, original=None):
    test = np.reshape(test_input, (1, 256, 256, 1))
    result = upscaler_model.predict(test, verbose=0)
    result = np.reshape(result, (256, 256))
    test = np.reshape(test, (256, 256))

    if original is not None:
        original = np.reshape(original, (256, 256))

    display = True
    initial_time = time.time()

    if original is not None:
        cv2.imshow("original", original)
    cv2.imshow("test", test)
    cv2.imshow("result", result)

    while display and time.time() - initial_time < 3:
        display = cv2.waitKey(1) == -1
    cv2.destroyAllWindows()


def test_detective_model(test_input, switch=False, nontrain=False):
    for image in test_input:
        original = np.reshape(image, (1, 256, 256, 1))
        test = upscaler_model.predict(original, verbose=0)

        if switch:
            result = detective_model.predict([original, test], verbose=0)
        else:
            result = detective_model.predict([test, original], verbose=0)

        print(result)
        test = np.reshape(test, (256, 256))

        display = True
        initial_time = time.time()


        if switch:
            original_result = result[0, 0]
            fake_result = result[0, 1]
        else:
            fake_result = result[0, 0]
            original_result = result[0, 1]

        test = np.reshape(test, (256, 256))
        original = np.reshape(original, (256, 256))

        cv2.imshow(str(fake_result), test)
        cv2.imshow(str(original_result), original)

        while display:# and time.time() - initial_time < 3:
            display = cv2.waitKey(1) == -1
        cv2.destroyAllWindows()


def generate_fakes(originals, training_data_amount=1024):
    originals = np.reshape(originals, (training_data_amount, 256, 256, 1))
    fakes = upscaler_model.predict(originals, verbose=0)
    return fakes


def generate_labeled_detective_data(original, fake, old_data=None, old_labels=None, training_data_amount=1024):
    original = np.reshape(original, (training_data_amount, 1, 256, 256, 1))
    fake = np.reshape(fake, (training_data_amount, 1, 256, 256, 1))

    nested_array_first_half = np.concatenate([original[:int(len(original)/2)], fake[:int(len(fake)/2)]], axis=1)
    nested_array_second_half = np.concatenate([fake[int(len(fake)/2):], original[int(len(original)/2):]], axis=1)
    data = np.concatenate([nested_array_first_half, nested_array_second_half])
    labels_first_half = np.concatenate([np.reshape(np.ones((int(len(original)/2))), (int(training_data_amount/2), 1, 1)), np.reshape(np.zeros(int((len(fake)/2))), (int(training_data_amount/2), 1, 1))], axis=1)
    labels_second_half = np.concatenate([np.reshape(np.zeros((int(len(fake)/2))), (int(training_data_amount/2), 1, 1)), np.reshape(np.ones(int((len(original)/2))), (int(training_data_amount/2), 1, 1))], axis=1)
    labels = np.concatenate([labels_first_half, labels_second_half])
    labels = np.reshape(labels, (training_data_amount, 2))

    p = np.random.permutation(len(data))
    data = data[p]
    labels = labels[p]

    if old_data is not None and old_labels is not None:
        data = np.concatenate([data, old_data[:int(len(old_data)/2)]])
        labels = np.concatenate([labels, old_labels[:int(len(old_labels)/2)]])

        p = np.random.permutation(len(labels))
        data = data[p]
        labels = labels[p]

        if len(data) > 1024:
            data = data[:1024]
            labels = labels[:1024]

    return data, labels


IMAGE_AMOUNT = 128
BATCH_SIZE = 8

#full_model.load_weights("gan_model_gan1.h5")

detective_x = None
detective_y = None

x, y = generate_inputs("newyork_monochrome_images/", 1, slide=0)
#upscaler_model.load_weights("upscaler_initial.h5")
test_upscaler_model(x[0], y[0])

for i in range(0, 30):
    #data generation
    print("data generation has begun slide:" + str(i))
    x, y = generate_inputs("newyork_monochrome_images/", IMAGE_AMOUNT, slide=i)
    fakes = generate_fakes(y, IMAGE_AMOUNT)
    detective_x, detective_y = generate_labeled_detective_data(y, fakes, old_data=detective_x, old_labels=detective_y, training_data_amount=IMAGE_AMOUNT)

    full_model_y = np.concatenate([np.ones((IMAGE_AMOUNT, 1), np.float32), np.zeros((IMAGE_AMOUNT, 1), np.float32)], axis=1)

    print("initial state")
    test_upscaler_model(x[0], y[0])

    #training loop
    print("classical training for upscaler slide:" + str(i))
    #upscaler_model.fit(x, y, batch_size=8, epochs=1)
    #test_upscaler_model(x[0], y[0])

    print("classical training for detective slide:" + str(i))
    #test_detective_model(y[0:8], False)
    #test_detective_model(y[0:8], True)

    for j in range(2):
        d_loss, d_accuracy = detective_model.evaluate([detective_x[:, 0], detective_x[:, 1]], detective_y, batch_size=8)
        if d_accuracy > 0.95 or (d_accuracy > 0.80 and j > 3):
            break
        detective_model.fit([detective_x[:, 0], detective_x[:, 1]], detective_y, batch_size=8, epochs=2)

    test_detective_model(y[0:1], False)
    test_detective_model(y[0:1], True)

    print("gan training for upscaler slide:" + str(i))

    for j in range(2 ):
        d_loss, d_accuracy = full_model.evaluate([x, y], full_model_y, batch_size=8)
        if d_accuracy > 0.70 or (d_accuracy > 0.60 and j > 2):
            break
        full_model.fit([x, y], full_model_y, batch_size=8, epochs=1)

    test_detective_model(y[0:1], False)
    test_detective_model(y[0:1], True)

    full_model.save_weights("gan_model_gan" + str(i) + ".h5")
    #test_upscaler_model(x[0], y[0])
