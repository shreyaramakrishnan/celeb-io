# this will handle intake of photos and spitting out of a prediction.
# "main method"

import cv2
import face_crop as fc

FRAME_DELAY = 50

# method to save prediction
def save_pred(*args):
    pass


# call preprocessing file

# train model

# get webcam spooled
# define a video capture object
vid = cv2.VideoCapture(0)

label = "Unknown"
picture = None
frame_count = 0
while(True):
    # Capture the video frame by frame
    ret, frame = vid.read()
    frame_count += 1

    # grab face from photo
    coords = fc.frame_face_crop(frame)
    for (x, y, w, h) in coords:
          cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # check to see if we need to update our prediction
    if (frame_count > FRAME_DELAY):
       # do prediction
      predict_input = frame
      print("update")
      # picture = get_picture(prediction) # get closest neighbor photo
      # label = fc.get_label(prediction) # get label
       # reset frame count
      frame_count = 0

    # Display the resulting video frame
    cv2.imshow('frame', frame)


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
