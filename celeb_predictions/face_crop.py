# this is the face preprocessing file
import cv2
import glob
import sys
import os
from PIL import Image
from pathlib import Path

IM_EXTENSION = ".jpg"
OUTPUT_SIZE = 224

haar_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# returns label, None if this isn't a valid format
def get_label(file):
    try:
      return os.path.basename(file)[:file.index("_")]
    except Exception as e:
      print(e)
      # no underscore
      print("NO LABEL, this isn't a file")
      return None
    
# normalize frame face into our dimension standards
def process_frame_face():
   
   pass

# return bounding box coords of detected face
def frame_face_crop(frame, scale=1.3):
   faces_rect = haar_cascade.detectMultiScale(frame, scaleFactor=scale, minNeighbors=5)
   return faces_rect

# 224x224x3
def dir_face_crop(input_dir, output_dir, scale=1.7):
    
    # Loading the required haar-cascade xml classifier file
    haar_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

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
                    new_file = "{}.png".format(filename)
                    output_file_name = output_dir + "/" + new_file

                    img = cv2.imread(item)
                    faces_rect = haar_cascade.detectMultiScale(img, scaleFactor=scale, minNeighbors=5)
                    
                    # resize each face detected in images
                    for (x, y, w, h) in faces_rect:
                        face_cropped = img[y:y+h, x:x+w]
                        # THIS IS A GOOD LINE
                        face_resized_img = cv2.resize(img[y:y+h, x:x+w], (OUTPUT_SIZE, OUTPUT_SIZE), interpolation = cv2.INTER_AREA)
                        # print(face_resized_img.shape)
                        
                        #save cropped face images
                        # new_img_name = img_name.replace('.jpg', '')

                        # this is the good line
                        # print(output_file_name)
                        cv2.imwrite(output_file_name, face_resized_img)


                    # with Image.open(item) as im:
                    #   print(new_file)
                    #   # im.save(output_dir, new_file)
                    #   im.save(new_file)
            os.chdir("..")
    

#create a function to grab each image, detect the face, crop the face, save the face image
def face_crop(path, scale):
    
    #grabs all image directory paths
    img_list = glob.glob(path + '/*.jpg')
    
    #face cascade from OpenCV
    # haar_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')

    # Loading the required haar-cascade xml classifier file
    haar_cascade = cv2.CascadeClassifier('Haarcascade_frontalface_default.xml')
    
    for img_name in img_list:
        img = cv2.imread(img_name)
        faces_rect = haar_cascade.detectMultiScale(img, scaleFactor=scale, minNeighbors=5)
        
        # resize each face detected in images
        for (x, y, w, h) in faces_rect:
            face_cropped = img[y:y+h, x:x+w]
            face_resized_img = cv2.resize(img[y:y+h, x:x+w], (175,175), interpolation = cv2.INTER_AREA)
            
            #save cropped face images
            new_img_name = img_name.replace('.jpg', '')
            cv2.imwrite(new_img_name + '.jpg', face_resized_img)



# try:
#   # dir_face_crop("/homes/iws/ljkniest/celeb-io/celeb_predictions/data/basic_input_spoof", "/homes/iws/ljkniest/celeb-io/celeb_predictions/data/basic_output_spoof", 1.5)
#   for file in os.listdir("/homes/iws/ljkniest/celeb-io/celeb_predictions/data/basic_output_spoof"):
#       print(get_label(file))
#       # get_label(file)
# except Exception as e:
#   print("EXCEPTION")
#   print(e)
