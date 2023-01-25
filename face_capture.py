import os

import cv2
import imutils

person_name = input('Ingrese el nombre completo de la persona: ')

data_path = os.path.join(os.getcwd(), 'data')
person_path = os.path.join(data_path, person_name)

if not os.path.exists(person_path):
    os.mkdir(person_path)
    print('Carpeta creada: ', person_path)

# capture = cv2.VideoCapture('assets/videos/test-2.mp4')
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
count = 0

while True:
    ret, frame = capture.read()

    if ret == False:
        break

    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aux_frame = frame.copy()

    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        rostro = aux_frame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(person_path + f'/rostro_{count}.jpg', rostro)
        count = count + 1

    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)
    if k == 27 or count > 300:
        break

capture.release()
cv2.destroyAllWindows()
