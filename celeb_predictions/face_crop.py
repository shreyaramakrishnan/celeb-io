# this is the face preprocessing file
import cv2
import glob

#create a function to grab each image, detect the face, crop the face, save the face image
def face_crop(path, scale):
    
    #grabs all image directory paths
    img_list = glob.glob(path + '/*.jpg')
    
    #face cascade from OpenCV
    # haar_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')

    # Loading the required haar-cascade xml classifier file
    haar_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    
    for img_name in img_list:
        img = cv2.imread(img_name)
        faces_rect = haar_cascade.detectMultiScale(img, scaleFactor=scale, minNeighbors=5)
        num = 0
        
        # resize each face detected in images
        for (x, y, w, h) in faces_rect:
            face_cropped = img[y:y+h, x:x+w]
            face_resized_img = cv2.resize(img[y:y+h, x:x+w], (175,175), interpolation = cv2.INTER_AREA)
            num += 1
            #save cropped face images
            new_img_name = img_name.replace('.jpg', '') + str(num)
            print("new image name" + new_img_name)
            cv2.imwrite(new_img_name + '.jpg', face_resized_img)



print("finding and cropping faces...")
face_crop('/homes/iws/aecraig/celeb-io/celeb_predictions/trial_images', 1.3)
