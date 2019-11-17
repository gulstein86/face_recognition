# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 22:42:49 2018

@author: Suraj Shukla
"""

import face_recognition
import cv2
from PIL import Image
import glob
import numpy as np
#import os
#import csv

vAR_ReadAllImages = 'D:\\1 DataScience Code and Data\\FaceRecognition\\Images\\*'
# Get a reference to webcam #0 (the default one)
# Load a sample picture and learn how to recognize it.


import sys
argument = sys.argv[1]
#argument = 'D:\\1 DataScience Code and Data\\FaceRecognition\\Images\\Megan.jpg'
## Read training set image files

filelist = glob.glob(vAR_ReadAllImages)
filelist[0]

## Creating numpy array of all images
def gET_ImagesAndLabels():
    
    #creating empty face list
    faceSamples=[]
    name =[]
    IDs = []
    Encoding =[]
    #GO through all image paths and load the gender and the images
    for filename in filelist:
        
        # extract the face from the training image sample
        faces =face_recognition.load_image_file(filename)
        faceSamples.append(faces)
       
        ID = filename.split('\\')[-1].split('.')[0]
        n = list(filter(lambda x: x.isalpha(), ID))
        n = ''.join(n)
        name.append(n)
        IDs.append(ID)
        Encoding.append(face_recognition.face_encodings(faces)[0])
        
        #If a face is there then append that in the list as well as Id of it
    return faceSamples, name,IDs, Encoding

faces, names, FileNames, Encoding = gET_ImagesAndLabels()
#facesI = np.array(faces)


############################################################################
unknown_image = face_recognition.load_image_file(argument)
unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
# Create arrays of known face encodings and their names
known_face_encodings = Encoding
known_face_names = names

results = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding)

#face_distances = face_recognition.face_distance(known_face_encodings, unknown_face_encoding)

#for i, face_distance in enumerate(face_distances):
#    print("The test image has a distance of {:.2} from known image #{}".format(face_distance, i))
#    print("- With a normal cutoff of 0.6, would the test image match the known image? {}".format(face_distance < 0.6))
#    print("- With a very strict cutoff of 0.5, would the test image match the known image? {}".format(face_distance < 0.5))
#    print()
face_distances = face_recognition.face_distance(known_face_encodings, unknown_face_encoding)
if True in results:
                name = known_face_names[np.argmin(face_distances)]
else:
     name ="Unknown"        


cv2.imshow(name,unknown_image)


face_recognition.face_locations(unknown_image)