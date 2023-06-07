#----------------------------------------------------------------------------
# Created By  : Audrey Craig, Lianne Kniest, Shreya Ramakrishnan 
# Created Date: June 2023
# ---------------------------------------------------------------------------
""" 
    face_crop.py is file that pre-processes input images and crops them to a face.
    It is imported as a library but can also be used in script format, see bottom
    for changes necessary.
    Script usage: python ./celec_predictions/face_crop.py
""" 
# ---------------------------------------------------------------------------
import cv2
import glob
import sys
import os
from PIL import Image
from pathlib import Path
import numpy as np

IM_EXTENSION = ".jpg"
OUTPUT_SIZE = 224  # square output, size for processed faces to be saved

# Loading the required haar-cascade xml classifier file
haar_cascade = cv2.CascadeClassifier('./celeb_predictions/haarcascade_frontalface_default.xml')


# returns label, None if this isn't a valid format
def get_label(file):
    try:
      img_name = os.path.basename(file)
      names = img_name.split("_")
      return names[0]
    except Exception as e:
      print(e)
      # no underscore
      print("NO LABEL, this isn't a file")
      return None
  
# return bounding box coords of detected face
def frame_face_crop(frame, scale=1.3):
   faces_rect = haar_cascade.detectMultiScale(frame, scaleFactor=scale, minNeighbors=5)
   return faces_rect

# take a directory of input images, detect faces and crop, and save to output directory
# larger scale = slower but better detection
def dir_face_crop(input_dir, output_dir, scale=1.7):
    for dir in Path(input_dir).iterdir():
        # print(dir)
        if os.path.isdir(dir):
            # print("is dir")
            os.chdir(dir)
            for item in os.listdir("."):
                # print(item)
                item = os.path.abspath(item)
                filename, extension  = os.path.splitext(item)
                if extension == IM_EXTENSION:
                    filename = os.path.basename(filename)
                    filename = os.path.basename(dir) + "_" + filename
                    new_file = "{}".format(filename)
                    new_file += IM_EXTENSION
                    output_file_name = output_dir + "/" + new_file

                    img = cv2.imread(item)
                    #  TODO: if there are more than 1 faces in the photo, discard photo
                    # basically, faces_rect will return a list of coordinates. Ignore this iteration
                    # if there are multiple faces detected (list len > 0)
                    faces_rect = haar_cascade.detectMultiScale(img, scaleFactor=scale, minNeighbors=5)
                    if len(faces_rect) == 1:
                      # resize each face detected in images
                      for (x, y, w, h) in faces_rect:
                          face_cropped = img[y:y+h, x:x+w]
                          face_resized_img = cv2.resize(img[y:y+h, x:x+w], (OUTPUT_SIZE, OUTPUT_SIZE), interpolation = cv2.INTER_AREA)

                          cv2.imwrite(output_file_name, face_resized_img)

            os.chdir("..")


#create a function to grab each image, detect the face, crop the face, save the face image
def face_crop(img, scale=1.7):
  face_cropped = None
  faces_rect = haar_cascade.detectMultiScale(img, scaleFactor=scale, minNeighbors=5)
  if len(faces_rect) > 1:
     return None
  
  # resize each face detected in images
  for (x, y, w, h) in faces_rect:
      # face_cropped = img[y:y+h, x:x+w]
      face_cropped = cv2.resize(img[y:y+h, x:x+w], (OUTPUT_SIZE, OUTPUT_SIZE), interpolation = cv2.INTER_AREA)
      
  return face_cropped


def normalize_image(img):
    # normalize pixel values to be between 0 and 1
    img = (img / 255.).astype(np.float32)
    # resize image to 224x224
    img = cv2.resize(img, (224,224))
    # preprocess image for vgg model
    img = img.reshape((img.shape[0], img.shape[1], img.shape[2]))
    # img = keras.applications.vgg16.preprocess_input(img)
    # convert from BGR to RGB
    return img[...,::-1]


# for "main method" running, comment out the lines below and change the directories as needed.
# an absolute path is necessary to the directories.

# input = "C:/Users/liann/Documents/UW/celeb-io/celeb_predictions/data/basic_input_spoof"
# output = "C:/Users/liann/Documents/UW/celeb-io/celeb_predictions/data/img_output"

# try:
#   dir_face_crop(input, output, 1.5)
#   for file in os.listdir(output):
#       print(get_label(file))
#       # get_label(file)
# except Exception as e:
#   print("EXCEPTION")
#   print(e)
