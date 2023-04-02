#!/usr/bin/env python

# https://www.youtube.com/watch?v=EujccFRio7o

"""
Unable to upload lots of images to Github.
Create your own set of images by downloading from Google.

"""

from keras.layers import Conv2D, UpSampling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import tensorflow as tf
import cv2

path_x = '../../newyork_original_images/'
path_y = '../../newyork_original_images/'

data_count = 1024
image_size = 256
"""
#Normalize images - divide by 255
train_datagen = ImageDataGenerator(rescale=1. / 255)

x = []
y = []
for i in range(data_count):
    input_image = cv2.cvtColor(cv2.imread(path_x + "image_" + str(i) + ".png"), cv2.COLOR_BGR2GRAY)
    input_image = cv2.resize(input_image, (image_size, image_size),
                                 interpolation=cv2.INTER_AREA)

    target_image = cv2.cvtColor(cv2.imread(path_y + "image_" + str(i) + ".png"), cv2.COLOR_BGR2LAB)
    target_image = cv2.resize(target_image, (image_size, image_size),
                                  interpolation=cv2.INTER_AREA)

    x.append(target_image[:, :, 0] / 255)
    y.append(target_image[:, :, 1:] / 128)  # A and B values range from -127 to 128,

x = np.array(x, np.float32)
y = np.array(y, np.float32)

print(x.shape)
print(y.shape)
"""
#Convert from RGB to Lab
"""
by iterating on each image, we convert the RGB to Lab. 
Think of LAB image as a grey image in L channel and all color info stored in A and B channels. 
The input to the network will be the L channel, so we assign L channel to X vector. 
And assign A and B to Y.

"""

#Encoder

model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2, input_shape=(image_size, image_size, 1)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3,3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(Conv2D(256, (3,3), activation='relu', padding='same', strides=2))
model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
model.add(Conv2D(256, (3,3), activation='relu', padding='same'))

#Decoder
#Decoder
#Note: For the last layer we use tanh instead of Relu.
#This is because we are colorizing the image in this layer using 2 filters, A and B.
#A and B values range between -1 and 1 so tanh (or hyperbolic tangent) is used
#as it also has the range between -1 and 1.
#Other functions go from 0 to 1.
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(Conv2D(16, (3,3), activation='relu', padding='same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.summary()


#model.fit(x,y,validation_split=0.1, epochs=100, batch_size=16)

#model.save('colorize_autoencoder.model.h5')


###########################################################
#Load saved model and test on images.
#colorize_autoencoder300.model is trained on 300 epochs
#

tf.keras.models.load_model(
    'colorize_autoencoder.model.h5',
    custom_objects=None,
    compile=True)

img1_color=[]

img1=img_to_array(load_img('../../newyork_monochrome_images/image_0.png'))
img1 = resize(img1 ,(image_size, image_size))
img1_color.append(img1)

img1_color = np.array(img1_color, dtype=float)
img1_color = rgb2lab(1.0/255*img1_color)[:,:,:,0]
img1_color = img1_color.reshape(img1_color.shape+(1,))

output1 = model.predict(img1_color)
output1 = output1*128

result = np.zeros((256, 256, 3))
result[:,:,0] = img1_color[0][:,:,0]
result[:,:,1:] = output1[0]
imsave("result.png", lab2rgb(result))
