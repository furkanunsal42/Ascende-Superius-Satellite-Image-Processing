import cv2 as cv
import numpy as np


def get_image_part(img, x_start, y_start, width=1024, height=1024):
    img_cropped = img[x_start:x_start + width, y_start:y_start + height]
    return img_cropped


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
    return cv.subtract(np.asarray(noisy, dtype=np.uint8), img)


def add_noises(img, noise_list):
    result = img
    for noise in noise_list:
        result = cv.add(result, noise)
    return result


def generate_images(img, displacement_x, displacement_y, filepath="images/", filename="image", format="png", width=1024, height=1024):
    print("Generating images...")
    index = 0
    for x in range(0, img.shape[0]-width, displacement_x):
        for y in range(0, img.shape[1]-height, displacement_y):
            new_image = get_image_part(img, x, y, width, height)

            """
            noises = []
            noises.append(create_gaussian_noise(new_image, 0, 0.1)) #0.2 high   0.1 low
            noises.append(create_salt_and_pepper_noise(new_image))
            noises.append(create_poisson_noise(new_image, 20))       #2 high     20 low
            new_image = add_noises(new_image, noises)
            """

            name = filepath+filename+"_"+str(index)+"."+format
            cv.imwrite(name, new_image[:, :, 1])
            print(name + " is created")
            index += 1


image = cv.imread("New_York_crop.png")
cropped = get_image_part(image, 0, 0)

generate_images(image, 100, 100, filepath="newyork_monochrome_images/")


"""
noises = []
noises.append(create_gaussian_noise(cropped, 0, 0.1))
noises.append(create_salt_and_pepper_noise(cropped))
noises.append(create_poisson_noise(cropped, 20))

noisy_cropped = add_noises(cropped, noises)

while True:
    cv.imshow("satellite image original green", cropped[:, :, 1])
    cv.imshow("satellite image noisy_image", noisy_cropped[:, :, 1])

    cv.waitKey(1)
"""