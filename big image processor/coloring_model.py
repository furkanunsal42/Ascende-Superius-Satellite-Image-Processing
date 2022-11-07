from keras import layers
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import MeanAbsoluteError
import numpy as np
import cv2
import os
import time
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def create_gaussian_noise(img, mean=0, var=0.1):
    row, col, ch = img.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    return np.asarray(gauss, dtype=np.uint8)


def create_salt_and_pepper_noise(img, ratio=0.5, amount=0.004, strength=255):
    row, col, ch = img.shape
    s_vs_p = ratio
    out = np.zeros(img.shape)
    # Salt mode
    num_salt = np.ceil(amount * img.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    out[coords] = strength

    # Pepper mode
    num_pepper = np.ceil(amount* img.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    out[coords] = 0
    return np.asarray(out, np.uint8)


def create_poisson_noise(img, vals=2):
    noisy = np.random.poisson(np.asarray(img / 255 * vals, np.float32)) / vals * 255
    return cv2.subtract(np.asarray(noisy, dtype=np.uint8), img)


def add_noises(img, noise_list):
    result = img
    for noise in noise_list:
        result = cv2.add(result, noise)
    return result


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


def get_image_part(img, x_start, y_start, width=1024, height=1024):
    img_cropped = img[x_start:x_start + width, y_start:y_start + height]
    return img_cropped


def process_image(img, model, image_size=1024):
    print("Generating images...")
    index = 0
    all_parts = None
    for x in range(0, img.shape[0]-image_size, image_size):
        for y in range(0, img.shape[1] - image_size, image_size):

            new_image = get_image_part(img, x, y, image_size, image_size)[:, :, 1]
            new_image = np.reshape(new_image, (new_image.shape[0], new_image.shape[1], 1))

            new_image = np.reshape(new_image, (new_image.shape[0], new_image.shape[1]))
            new_image = cv2.resize(new_image, (256, 256))
            new_image = np.asarray(new_image, np.float32) / 255.0

            if all_parts is None:
                all_parts = np.reshape(new_image, (1, 256, 256, 1))
            else:
                all_parts = np.concatenate([all_parts, np.reshape(new_image, (1, 256, 256, 1))])
            print("\rimage_"+str(index) + " was read", end='')
            index += 1
    print()

    all_parts = model.predict(all_parts)

    horizontal_image_count = math.ceil((img.shape[1]-image_size) / image_size)

    rows = []
    for i in range(0, index, horizontal_image_count):
        row = np.concatenate(all_parts[i:i+horizontal_image_count, :, :, :], axis=1)
        rows.append(row)

    rows = np.array(rows)
    constructed_image = np.concatenate(rows, axis=0)
    return constructed_image


file_name = input("İşlenecek fotoğrafın adını giriniz. Eğer Input.png'yi kullanmak istiyorsanız sadece Enter'a basınız.")
if file_name == "":
    file_name = "Input.png"

full_model = generate_full_model()
full_model.load_weights("full_model256_2.h5")

image = cv2.imread(file_name)

mode = input("algoritmayı \"ön-izlenim\" veya \"yüksek-çözünürlük\" modunda çalıştırabilirsiniz. \"ön-izlenim\" için 0, \"yüksek-çözünürlük\" için 1 giriniz.")
while mode != "0" and mode != "1":
    mode = input("algoritmayı \"ön-izlenim\" veya \"yüksek-çözünürlük\" modunda çalıştırabilirsiniz. \"ön-izlenim\" için 0, \"yüksek-çözünürlük\" için 1 giriniz.")

slide = None
if mode == "0":
    slide = 1024
elif mode == "1":
    slide = 256

result = process_image(image, model=full_model, image_size=slide) * 255
print(result.shape)
cv2.imwrite("result.png", result)
