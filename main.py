import os
import cv2
from imutils import paths, resize
import numpy as np

root_path = os.path.dirname(__file__)
proto_path = fr'{root_path}/classifiers/caffe/deploy.prototxt'

model_path = fr'{root_path}/classifiers/caffe/res10_300x300_ssd_iter_140000.caffemodel'

detector = cv2.dnn.readNet(proto_path, model_path, 'caffe')

# embedder = cv2.dnn.readNet(fr'{root_path}/output/embeddings.pickle')

all_training_images = list(paths.list_images(fr'{root_path}/images/train_images'))

found_embeddings = []
found_labels = []
MINMUM_DETECTION_THRESHHOLD = 0.5

total_faces = 0

for (k, image_path) in enumerate(all_training_images):
    # print(f'Processing image {k + 1}/{len(all_training_images)}')
    
    # use the name of the folder as label for the person
    name = image_path.split(os.path.sep)[-2]
    
    image = cv2.imread(image_path)
    image = resize(image, width=600)
    (h, w) = image.shape[:2]
    
    # Create a blob
    try:
        image_blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
        detector.setInput(image_blob)
        detections = detector.forward()
    except:
        print('error has occurred with loading the image...')
        continue
    
    face_found = len(detections > 0)
    if face_found:
        index = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, index, 2]
        
    # Filter the weak detections
    strong_detection = confidence > MINMUM_DETECTION_THRESHHOLD
    