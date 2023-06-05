#----------------------------------------------------------------------------
# Created By  : Audrey Craig, Lianne Kniest, Shreya Ramakrishnan 
# Created Date: June 2023
# version ='0.5'
# ---------------------------------------------------------------------------
""" 
    Celeb_predict.py is the "main method" script for the celeb.io project.
    At a given interval, a face is detected and a new prediction is generated.
    To use, change the constant parameters below.
    Usage: python celeb_predict.py
    To quit this program, use either q or esc.
""" 
# ---------------------------------------------------------------------------

import cv2
import pickle  # used for loading a saved model
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
import sklearn.preprocessing as sk
from sklearn.decomposition import PCA
import face_crop as fc 
import train_model as tm

# celeb.io parameters
FRAME_DELAY = 50  # changes the number of frames processed before a new prediction is generated 
MODEL = './svm_model.pkl'  # put the model to be imported here, None if a new model is to be trained.

# optional: update these if training a new model.
INPUT_DIR = './celeb_predictions/data/img_input/'  # image input directory
OUTPUT_DIR = './celeb_predictions/data/img_output/' # processed image output dir, will be used to train model

# load in the train model
clf = None
if MODEL is None:
  clf = tm.train_model(input=INPUT_DIR,output=OUTPUT_DIR,save=False,show_pred=False)
else:
  print("---start import model---")
  with open(MODEL, 'rb') as f:
      clf = pickle.load(f)
  print("---end import model---")

# get webcam spooled
vid = cv2.VideoCapture(0)

label = "Unknown"
frame_count = 0
face_model =  tm.get_functional_model()  # get face model from saved file
le = sk.LabelEncoder()
le.classes_ = np.load('classes.npy')

while(True):
    # Capture the video frame by frame
    ret, frame = vid.read()
    frame_count += 1

    # grab face from photo (draws line only, does not crop)
    coords = fc.frame_face_crop(frame)
    for (x, y, w, h) in coords:
          cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # add current prediction label to screen
    cv2.putText(frame, 
            label, 
            (50, 50), 
            cv2.FONT_HERSHEY_DUPLEX, 1, 
            (0, 255, 255), 
            2, 
            cv2.LINE_4)

    # check to see if we need to update our prediction
    if (frame_count > FRAME_DELAY):

      # grab detected, cropped face from camera
      cropped_face = fc.face_crop(frame)

      if cropped_face is None:
         print("face not detected")
         label = "Unknown"
      else:
         print("face detected")

      try:
        # should error out if cropped face is None leading to no change of prediction
        predict_input = fc.normalize_image(cropped_face)

      
        embedding = face_model.predict(np.expand_dims(predict_input, axis=0))
        print("EMBEDDING:",str(embedding))
        
        pred = clf.predict(embedding)
        print("PREDICTION:", pred)

        label = str(le.inverse_transform(pred))
        print("LABEL: ", label)

        classes = le.inverse_transform(clf.classes_)
        print("classes ", classes)
      except Exception as e:
        # print("exception.")
        # print(e)
        pass

      frame_count = 0
    
    # Display the resulting video frame
    cv2.imshow('frame', frame)
    if (frame_count %10 == 1):
      print("Prediction: " + str(label))

   # Wait for Esc key to stop
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
    # q will also escape
    if cv2.waitKey(1) == ord('q'):
      break

# Close the window
vid.release()
  
# De-allocate any associated memory usage
cv2.destroyAllWindows() 

