from tkinter import Y
import cv2
import os

from cv2 import imwrite

cap = cv2.VideoCapture(0)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

frontal_face_model = 'haarcascade_frontalface.xml'
haar_model = cv2.CascadeClassifier(os.path.join(BASE_DIR, 'classifiers', 'haar', frontal_face_model))

count = 0

name = str(input("Enter your name:")).lower()

path = os.path.join(BASE_DIR, 'images', 'collected_data', name)

isExistingPath = os.path.exists(path)

if isExistingPath:
    print('Name exists already! Try again...')
    name = str(input("Enter your name:")).lower()   
else:
    os.makedirs(path)
    
while True:
    ret, video = video.read()
    
    faces = haar_model.detectMultiScale(video, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        picture_name = os.path.join(BASE_DIR, 'images', 'collected_data', name, count + '.jpg')
        cv2.imwrite(picture_name, video[y:y + h, x:x + w])
        count += 1
    
    cv2.imshow('data collecting...', video)

    if (count > 300):
        break
    
    key = cv2.waitKey(30) & 0xff
    if key == 27:
        break