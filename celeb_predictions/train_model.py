# create and train CNN

import tensorflow as tf
import keras 
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import os
import cv2
import numpy as np

num_classes = 196

'''
convolutional layers - good use for detecting features in images, outputs a feature map 
max pooling - reduces the dimensionality of each feature map but keeps the most important information (max in this case)
flatten - flattens the input (after being run through previous layers) to one dimension, important transition layer to input 
          into the fully connected layers
dense - a fully connected layer used for classification tasks, takes the input run through the conv layers and classifies the 
        image into one of the classes (computationally expensive)

MODEL DESIGN: 
    relu activation on each layer to eliminate negative values  

    - 2 convolutional layers with 64 filters each, kernel size of 3x3, relu activation function
    - max pooling layer with pool size of 2x2 and stride of 2x2
    - 2 convolutional layers with 128 filters each, kernel size of 3x3, relu activation function
    - max pooling layer with pool size of 2x2 and stride of 2x2
    - 3 convolutional layers with 256 filters each, kernel size of 3x3, relu activation function
    - max pooling layer with pool size of 2x2 and stride of 2x2
    - 3 convolutional layers with 512 filters each, kernel size of 3x3, relu activation function
    - max pooling layer with pool size of 2x2 and stride of 2x2
    - 3 convolutional layers with 512 filters each, kernel size of 3x3, relu activation function
    - max pooling layer with pool size of 2x2 and stride of 2x2
    - flatten layer
    - 3 dense layers with 256, 128, and num_classes nodes respectively, relu activation function
'''

# create the model
def vgg_face(): 
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=(224,224,3)))
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

    model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),name='vgg16'))

    model.add(Flatten(name='flatten'))
    model.add(Dense(256, activation='relu', name='fc1'))
    model.add(Dense(128, activation='relu', name='fc2'))
    model.add(Dense(num_classes, activation='softmax', name='output'))

    return model

def load_img(path): 
    img = cv2.imread(path)
    # normalize pixel values to be between 0 and 1
    img = (img / 255.).astype(np.float32)
    # resize image to 224x224
    img = cv2.resize(img, (224,224))
    # preprocess image for vgg model
    # print(img.shape)
    img = img.reshape((img.shape[0], img.shape[1], img.shape[2]))
    img = keras.applications.vgg16.preprocess_input(img)
    # convert from BGR to RGB
    return img[...,::-1]

model = vgg_face()
# print(model.summary())

# load in the pre-trained weights using the vgg model up to the last layer (fc2)
model.load_weights('vgg16_weights.h5' , by_name = True, skip_mismatch = True) 

# this is a functional, rather than sequential network - they are more flexible 
# look at "descriptor.png" to see the model architecture
functional_model = keras.models.Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

# generate embeddings for each image in our dataset based on the pre-trained weights 
PATH = './celeb_predictions/trial_images/'
images = os.listdir(PATH)
print(images)

img_embeddings = []
for image in images:
    img =load_img(PATH + image)
    # print(img.size)
    embedding = functional_model.predict(np.expand_dims(img, axis=0))[0]
    img_embeddings.append(embedding)






