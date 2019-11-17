# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 22:42:49 2018

@author: Suraj Shukla
"""

import face_recognition
import cv2
#from PIL import Image
import glob
import numpy as np
import os
#import csv
from win10toast import ToastNotifier
vAR_cascadePath ='D:\\FaceRecognition\\haarcascades\\haarcascade_frontalface_default.xml'
vAR_ReadAllImages = 'D:\\FaceRecognition\\Images\\*'
vAR_PathPrj = 'D:\\FaceRecognition'

detector= cv2.CascadeClassifier(vAR_cascadePath);

os.chdir(vAR_PathPrj)
## Read training set image files

filelist = glob.glob(vAR_ReadAllImages)
filelist[0]

def gET_ImagesAndLabels():
    
    #creating empty face list
    faceSamples=[]
    name =[]
    IDs = []
    Encoding =[]
    #GO through all image paths and load the gender and the images
    for filename in filelist:
        
        # extract the face from the training image sample
        if len(face_recognition.face_encodings(face_recognition.load_image_file(filename))) > 0:
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

type(faces)

import pickle

with open("faces.txt", "wb") as fp:   #Pickling
    pickle.dump(faces, fp)

with open("names.txt", "wb") as fp:   #Pickling
    pickle.dump(names, fp)
    
with open("Encoding.txt", "wb") as fp:   #Pickling
    pickle.dump(Encoding, fp)
    
