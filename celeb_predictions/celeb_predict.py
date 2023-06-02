# this will handle intake of photos and spitting out of a prediction.
# "main method"

import cv2
import face_crop as fc
import pickle
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
import sklearn.preprocessing as sk
from sklearn.decomposition import PCA
import train_model as tm

FRAME_DELAY = 50


# call preprocessing file

# # load in the train model
print("---start import model---")
with open('./svm_model.pkl', 'rb') as f:
    clf = pickle.load(f)
print("---end import model---")
print(clf.classes_)

# train model
# clf = tm.train_model()


# get webcam spooled
# define a video capture object
vid = cv2.VideoCapture(0)

label = "Unknown"
picture = None
frame_count = 0
# face_model = keras.models.load_model("./functional_model")
face_model =  tm.get_functional_model()
# cropped_face = None
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

    # check to see if we need to update our prediction
    if (frame_count > FRAME_DELAY):
       # do prediction

      cropped_face = fc.face_crop(frame)
      # predict_input = cropped_face
      if not cropped_face is None:
         print("face detected")
      else:
         print("face not detected")
         label = "Unknown"
         
      try:
        predict_input = fc.normalize_image(cropped_face)
      # picture = predict_input

      
        embedding = face_model.predict(np.expand_dims(predict_input, axis=0))
        print("EMBEDDING:",str(embedding))

        # clf = clf.fit(my_test, clf.classes_)

        # le = sk.LabelEncoder()
        
        pred = clf.predict(embedding)
        print("PREDICTION:", pred)

        label = le.inverse_transform(pred)
        print("LABEL: ", label)

        classes = le.inverse_transform(clf.classes_)
        print("classes ", classes)
      except Exception as e:
        # print("exception.")
        # print(e)
        pass

      frame_count = 0
    

    # Display the resulting video frame

    # try:
    #    cv2.imshow('frame', cropped_face)
    # except:
    #    pass
    cv2.imshow('frame', frame)
    # cv2.imshow('frame', picture)
    if (frame_count %10 == 1):
      print("Prediction: " + str(label))


    # display prediction

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


# cv2.createButton("Save my prediction", save_pred, None,cv2.QT_PUSH_BUTTON, 1)
