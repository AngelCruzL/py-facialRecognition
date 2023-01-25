import os

import cv2
import numpy as np

data_path = os.path.join(os.getcwd(), 'data')
people_list = os.listdir(data_path)
# print(people_list)

labels = []
face_samples = []
label = 0

for name_dir in people_list:
    person_path = os.path.join(data_path, name_dir)
    # print('Leyendo las imágenes')

    for filename in os.listdir(person_path):
        # print('Rostros: ', name_dir + '/' + filename)
        filename_path = os.path.join(person_path, filename)
        labels.append(label)
        face_samples.append(cv2.imread(os.path.join(filename_path), 0))
        # image = cv2.imread(filename_path, 0)
        # cv2.imshow('image', image)
        # cv2.waitKey(10)

    label = label + 1

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(face_samples, np.array(labels))
face_recognizer.write('LBPHFace_model.xml')
print('Modelo almacenado')
