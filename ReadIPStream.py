# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 22:42:49 2018

@author: Suraj Shukla
"""

import face_recognition
import cv2
#from PIL import Image
#import glob
import numpy as np
import os
#import csv
from win10toast import ToastNotifier
import pyttsx3
engine = pyttsx3.init()

vAR_cascadePath ='D:\\FaceRecognition\\haarcascades\\haarcascade_frontalface_default.xml'
vAR_ReadAllImages = 'D:\\FaceRecognition\\FaceRecognition\\Images\\*'
vAR_PathPrj = 'D:\\FaceRecognition'

detector= cv2.CascadeClassifier(vAR_cascadePath);

os.chdir(vAR_PathPrj)
## Read training set image files
import pickle


    
with open("faces.txt", "rb") as fp:   
    faces = pickle.load(fp)

with open("names.txt", "rb") as fp:   
    names = pickle.load(fp)
    
with open("Encoding.txt", "rb") as fp:  
    Encoding = pickle.load(fp)
############################################################################


# Create arrays of known face encodings and their names
known_face_encodings = Encoding
known_face_names = names

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = 1
video_capture = cv2.VideoCapture('http://192.168.1.3:8080/frame.mjpg')
#cv2.VideoCapture('http://192.168.43.131:8888/frame.mjpg')
faceCascade = cv2.CascadeClassifier(vAR_cascadePath);


while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

   
    rgb_small_frame = frame[:, :, ::-1]
    
    if process_this_frame ==4:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=1)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        toaster = ToastNotifier()
        process_this_frame = 0
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                name = known_face_names[np.argmin(face_distances)]
                
            face_names.append(name)
            

    process_this_frame = process_this_frame + 1
    

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 1
        right *= 1
        bottom *= 1
        left *= 1
        i = 0
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        if name == 'Daphanee' and i ==0:
                    
            engine.say("Hello Daphanee, Please go back to your desk")
            engine.setProperty('rate',120)  #120 words per minute
            engine.setProperty('volume',0.9) 
            engine.runAndWait()
        if name == 'Richard' and i ==0:
                    
            toaster.show_toast("Preferred Customer Notification",
                               "Richard is in the branch (Net Worth RM XXXX)", 
                               duration=3)
            engine.say("Preferred Customer Richard is in the branch. Net worth X Million ringgit")
            engine.setProperty('rate',120)  #120 words per minute
            engine.setProperty('volume',0.9) 
            engine.runAndWait()
            i = i+1
    # Display the resulting image
    cv2.imshow('Video', frame)
    #print(process_this_frame)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()