import cv2
import os
import pickle
from util import draw_rectangle

def load_labels(filename):
    with open(filename, 'rb') as f:
        label_dict = pickle.load(f)
        return {v:k for k, v in label_dict.items()}


BASE_DIR = os.path.dirname(__file__)

haar_frontal_face = 'haarcascade_frontalface.xml'
haar_smile = 'haarcascade_smile.xml'
haar_eye = 'haarcascade_eye.xml'

labels = load_labels(os.path.join(BASE_DIR, 'recognizer', 'labels.pickle'))

face_model = cv2.CascadeClassifier(os.path.join(BASE_DIR, 'classifiers', 'haar', haar_frontal_face))
# smile_model = cv2.CascadeClassifier(os.path.join(BASE_DIR, 'classifiers', 'haar', haar_smile))
# eye_model = cv2.CascadeClassifier(os.path.join(BASE_DIR, 'classifiers', 'haar', haar_eye))


recognizer = cv2.face.LBPHFaceRecognizer_create() # could be a trained model
recognizer.read(os.path.join(BASE_DIR, 'recognizer', 'trained_model.yml'))

cap = cv2.VideoCapture('test_vid.mp4')

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    
    try:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_model.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_color = img[y:y+h, x:x+w]
            roi_gray = img_gray[y:y+h, x:x+w]

            # Recognizer who it is
            id, accuracy = recognizer.predict(roi_gray)
            print(accuracy)
            if (accuracy > 40 and accuracy < 100):
                font = cv2.FONT_HERSHEY_COMPLEX
                name = labels[id]
                color = (255, 255, 255)
                cv2.putText(img, name, (x, y - 5), font, 1, color, 2, cv2.LINE_AA)
                cv2.putText(img, str(accuracy).split('.')[0], (x + w, y - 5), font, 1, color, 2, cv2.LINE_AA)
                # cv2.putText(img, accuracy, (x + w, y - 5), font, 1, color, 2, cv2.LINE_AA)
                draw_rectangle(img, x, y, w, h, (0, 0, 255))

        cv2.imshow('mamica', img)

        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break
    except :
        print('video has ended')
        break
    
cap.release()
cv2.destroyAllWindows()
