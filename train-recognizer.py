import os
import numpy as np
from PIL import Image
import cv2
import pickle

def label_faces(label_name, labels = {}):
    if label_name in labels:
        pass
    else:
        ids = list(labels.values())
        if len(ids) == 0:
            labels[label_name] = 0
        else:
            ids.sort()
            new_id = ids[-1] + 1
            labels[label_name] = new_id
    return labels

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(BASE_DIR, 'images', 'collected_data')

frontal_face_model = 'haarcascade_frontalface.xml'
haar_model = cv2.CascadeClassifier(os.path.join(BASE_DIR, 'classifiers', 'haar', frontal_face_model))

recognizer = cv2.face.LBPHFaceRecognizer_create()

labels = {}
x_train = []
y_train = []

for root, dirs, files in os.walk(img_dir):
    for file in files:
        lowercase_file = file.lower()
        if lowercase_file.endswith('png') or lowercase_file.endswith('jpg') or lowercase_file.endswith('jpeg'):
            path = os.path.join(root, file)
            label_name = os.path.basename(os.path.dirname(path)).lower()

            # assign a number to each unique label
            labels = label_faces(label_name, labels)
            current_id = labels[label_name]

            # convert as gray image
            pil_image = Image.open(path).convert('L')

            # prepare image
            image_size = (550, 550)
            resized_image = pil_image.resize(image_size, resample=Image.Resampling.LANCZOS)
            image_array = np.array(resized_image, np.uint8)
            
            faces = haar_model.detectMultiScale(image_array, scaleFactor=1.2, minNeighbors=5)
            print(label_name, file)
            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_train.append(current_id)

with open(os.path.join(BASE_DIR, 'recognizer', 'labels.pickle'), 'wb') as f:
    pickle.dump(labels, f)

recognizer.train(x_train, np.array(y_train))
recognizer.save(os.path.join(BASE_DIR, 'recognizer', 'trained_model.yml'))
