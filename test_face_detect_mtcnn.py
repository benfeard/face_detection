'''
by Benfeard Williams
on Mon Oct 19 2020

pytest
'''

#import pytest
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN

# count each face found 
def count_faces(result_list):
    num_faces = len(result_list)
    print("I found %d faces in the photo" %(num_faces))
    return num_faces
    
# find the faces
def detect_faces(image_name):
    pixels = pyplot.imread(image_name)
    # create the detector, using default weights
    detector = MTCNN()
    
    # detect faces in the image
    faces = detector.detect_faces(pixels)
    
    # return result list
    return faces

#####################################################
# test to see how many faces are detectd in test1.jpg
# inputs = jpg 
# outputs = num_faces, box coordinates
#####################################################

def test_count_faces():

    #input
    image = 'test1.jpg'
    
    #test function
    assert count_faces(detect_faces(image)) == 2
    
 
#def test_raises_exception_on_non_image_arguments():
#    with pytest.raises(TypeError):
#        detect_faces(9)