#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 20:37:39 2020

@author: benfeard

source:
https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/


usage: python face_detect_mtcnn.py test1.jpg
"""

# Now use a Multi-Task Cascaded Convolutional Neural Network or MTCNN
# see Kaipeng Zhang et al. 2016 
# “Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks.”
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
import sys

def draw_image_with_boxes(filename, result_list):
    # load the image
    data = pyplot.imread(filename)
    # plot the image
    pyplot.imshow(data)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box
    for result in result_list:
        # get coordinates
        x, y, width, height = result['box']
        # create the shape
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        # draw the box
        ax.add_patch(rect)
        # draw the dots
        for key, value in result['keypoints'].items():
            # create and draw dot
            dot = Circle(value, radius=2, color='red')
            ax.add_patch(dot)
    # show the plot
    pyplot.show()
    
# draw each face separately
def draw_faces(filename, result_list):
	# load the image
	data = pyplot.imread(filename)
	# plot each face as a subplot
	for i in range(len(result_list)):
		# get coordinates
		x1, y1, width, height = result_list[i]['box']
		x2, y2 = x1 + width, y1 + height
		# define subplot
		pyplot.subplot(1, len(result_list), i+1)
		pyplot.axis('off')
		# plot face
		pyplot.imshow(data[y1:y2, x1:x2])
	# show the plot
	pyplot.show()
   
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
    
# load image from file
#filename = 'test1.jpg'
#filename = 'test2.jpg'
#filename = 'headshot.jpg'
#filename = 'anime_faces.jpg'
#filename = 'sunglasses.jpeg'
filename = sys.argv[-1]

# find the faces
faces = detect_faces(filename)

# number of faces found
count_faces(faces)

# draw on the original images
#draw_image_with_boxes(filename, faces)

# display faces from the original image
draw_faces(filename, faces)
