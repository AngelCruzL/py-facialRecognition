import os

import cv2

data_path = os.path.join(os.getcwd(), 'data')
people_list = os.listdir(data_path)
# print(people_list)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('LBPHFace_model.xml')

# capture = cv2.VideoCapture('assets/videos/test-3.mp4')
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = capture.read()

    if ret == False:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aux_frame = gray.copy()

    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = aux_frame[y:y+h, x:x+w]
        face = cv2.resize(face, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(face)

        cv2.putText(frame, '{}'.format(result), (x, y-5),
                    1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

        if result[1] < 70:
            cv2.putText(frame, '{}'.format(
                people_list[result[0]]), (x, y-25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Desconocido', (x, y-20), 2,
                        0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

capture.release()
cv2.destroyAllWindows()
