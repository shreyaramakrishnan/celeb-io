# import keras 
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
import os
import cv2
import numpy as np
import face_crop as fc
import sklearn.preprocessing as sk
import sklearn.svm as svm
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle

# TODO: figure out how to get the embeddings so that this can be used as a pretrained model
# TODO: test on celebrity images 

num_classes = 196

INPUT_DIR = './celeb_predictions/data/basic_input_spoof/'
OUTPUT_DIR = './celeb_predictions/data/img_output/'

'''
convolutional - good use for detecting features in images, outputs a feature map 
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
    img = img.reshape((img.shape[0], img.shape[1], img.shape[2]))
    img = keras.applications.vgg16.preprocess_input(img)
    # convert from BGR to RGB
    return img[...,::-1]

def get_functional_model():
    model = vgg_face()
    # load in the pre-trained weights using the vgg model up to the last layer (fc2)
    model.load_weights('vgg16_weights.h5' , by_name = True, skip_mismatch = True) 
      # this is a functional, rather than sequential network - they are more flexible 
    # look at "descriptor.png" to see the model architecture
    functional_model = keras.models.Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    return functional_model


def train_model(input=INPUT_DIR,output=OUTPUT_DIR,save=False,show_pred=False):
  # model = vgg_face()

  # load in the pre-trained weights using the vgg model up to the last layer (fc2)
  # model.load_weights('vgg16_weights.h5' , by_name = True, skip_mismatch = True) 

  # # this is a functional, rather than sequential network - they are more flexible 
  # # look at "descriptor.png" to see the model architecture
  # functional_model = keras.models.Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

  # functional_model.save('functional_model')

  functional_model = get_functional_model()

  # preprocess the data by calling face_crop.py 

  #fc.dir_face_crop(input, output)

  # generate embeddings for each image in our dataset based on the pre-trained weights 
  PATH = output
  images = os.listdir(PATH)
  # print(images)

  img_embeddings = []
  targets = []
  for image in images:
      img = load_img(PATH + image)
      targets.append(fc.get_label(PATH + image))
      embedding = functional_model.predict(np.expand_dims(img, axis=0))[0]
      img_embeddings.append(embedding)

  # plot the embeddings

  # create train and test datasets using the embeddings we generated 
  x_train = []
  y_train = []
  x_test = []
  y_test = []
  x_test_images = []
  x_train_images = []

  # following the 80/20 rule for train/test split
  for i in range(len(img_embeddings)):
      if i % 5 == 0: 
          x_test.append(img_embeddings[i])
          x_test_images.append(images[i])
          y_test.append(targets[i])
      else: 
          x_train.append(img_embeddings[i])
          x_train_images.append(images[i])
          y_train.append(targets[i])
  print("printing y_train & y_test")
  print(y_train, y_test)

  # encode the labels using label encoder 
  le = sk.LabelEncoder()
  train_labels = le.fit_transform(y_train)
  test_labels = le.transform(y_test)
  print("printing training & testing ")
  print(train_labels, test_labels)
  np.save('classes.npy', le.classes_)

  # standardize the feature encodings 
  scaler = sk.StandardScaler()
  x_train_std = scaler.fit_transform(x_train)
  x_test_std = scaler.transform(x_test)

  # reduce the dimensionality of the feature encodings using PCA
  pca = PCA(n_components=128)
  x_train_pca = pca.fit_transform(x_train_std)
  x_test_pca = pca.transform(x_test_std)

  # create and train an SVM classifier 
  clf = svm.SVC(kernel='rbf', C=10, gamma=0.001)
  clf.fit(x_train_pca, train_labels)

  encoded_predictions = clf.predict(x_test_pca)
  decoded_predictions = le.inverse_transform(encoded_predictions)
  print("Predictions: ", decoded_predictions)

  accuracy = accuracy_score(test_labels, encoded_predictions)
  print("Accuracy: ", accuracy)

  if save:
    # save the model 
    filename = 'svm_model.pkl'
    pickle.dump(clf, open(filename, 'wb'))


  if show_pred:
    # visualize predictions 

    for i in range(30, 50): 
        
        example_image = cv2.imread(PATH + x_test_images[i])
        example_prediction = encoded_predictions[i]
        example_identity =  decoded_predictions[i]

        cv2.imshow(f'Identified as {example_identity}', example_image)
        cv2.waitKey(0)

  return clf

  # save the model 
  # filename = 'svm_model.pkl'
  # pickle.dump(clf, open(filename, 'wb'))

  # visualize predictions 

  # for i in range(30, 50): 
      
  #     example_image = cv2.imread(PATH + x_test_images[i])
  #     example_prediction = encoded_predictions[i]
  #     example_identity =  decoded_predictions[i]

  #     cv2.imshow(f'Identified as {example_identity}', example_image)
  #     cv2.waitKey(0)

train_model()