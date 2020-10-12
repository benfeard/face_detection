'''
https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/
'''

# plot photo with detected faces using opencv cascade classifier
from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle


# load the pre-trained model (this is from the OpenCV Github project)
# https://github.com/opencv/opencv/tree/master/data/haarcascades
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')


# load the photograph
pixels = imread('test1.jpg')

def face_detect(pixels, scaleFactor = 1.1, minNeighbors = 3):
    # perform face detection
    bboxes = classifier.detectMultiScale(pixels, scaleFactor, minNeighbors)


    # print bounding box for each detected face
    for box in bboxes:
        print(box)
        # extract
        x, y, width, height = box
        x2, y2 = x + width, y + height
        # draw a rectangle over the pixels
        rectangle(pixels, (x, y), (x2, y2), (0,0,255), 1)
        
        # show the image
        imshow('face detection', pixels)
        # keep the window open until a key is pressed
        waitKey(0)
        # close the window
        destroyAllWindows()

# load the first photo
pixels = imread('test1.jpg')
face_detect(pixels)

# load the second photograph
pixels2 = imread('test2.jpg')
face_detect(pixels2)

# decrease scale to 1.05 or 105%
# increase robustness of detection and required confidence with
# 8 candidate rectangles to find a face
face_detect(pixels2, 1.05, 8)



