import cv2
import os
from util import numberic_reorder_dir, draw_rectangle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

frontal_face_model = 'haarcascade_frontalface.xml'
haar_model = cv2.CascadeClassifier(os.path.join(BASE_DIR, 'classifiers', 'haar', frontal_face_model))

count = 0

name = str(input("Enter your name:")).lower()

path = os.path.join(BASE_DIR, 'images', 'collected_data', name)

isExistingPath = os.path.exists(path)

if isExistingPath:
    sorted_dir = numberic_reorder_dir(path)
    isEmpty = len(sorted_dir) == 0
    print(sorted_dir)
    
    if not isEmpty:
        last_item = int(sorted_dir[-1].split('.')[0])
        starting_number = last_item + 1
else:
    os.makedirs(path)
    

cap = cv2.VideoCapture(0)

while True:
    ret, video = cap.read()
    
    faces = haar_model.detectMultiScale(video, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        picture_name = os.path.join(BASE_DIR, 'images', 'collected_data', name, str(count) + '.jpg')
        cv2.imwrite(picture_name, video[y:y + h, x:x + w])
        count += 1
        draw_rectangle(video, x, y, w, h)
    
    cv2.imshow('data collecting...', video)

    if (count > 300):
        break
    
    key = cv2.waitKey(30) & 0xff
    if key == 27:
        break