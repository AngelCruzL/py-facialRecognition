import os

import cv2
import numpy as np

data_path = os.path.join(os.getcwd(), 'data')
people_list = os.listdir(data_path)

labels = []
face_samples = []
label = 0

for name_dir in people_list:
    person_path = os.path.join(data_path, name_dir)

    for filename in os.listdir(person_path):
        filename_path = os.path.join(person_path, filename)
        labels.append(label)
        face_samples.append(cv2.imread(os.path.join(filename_path), 0))

    label = label + 1

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(face_samples, np.array(labels))
face_recognizer.write('LBPHFace_model.xml')
print('Modelo almacenado')
