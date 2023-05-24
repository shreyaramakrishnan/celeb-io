# this will handle intake of photos and spitting out of a prediction.
# "main method"

import cv2

# call preprocessing file

# train model

# get webcam spooled
# define a video capture object
vid = cv2.VideoCapture(0)

while(True):
    # Capture the video frame by frame
    ret, frame = vid.read()

    # if (time % 10 == 0):
    #     grab frame
    #     do predict
    #     update predict
    #     show frame
    # else:
    #     show old predict, show current frame

    # Display the resulting frame
    cv2.imshow('frame', frame)

   # Wait for Esc key to stop
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Close the window
vid.release()
  
# De-allocate any associated memory usage
cv2.destroyAllWindows() 



# get webcam photo and upload