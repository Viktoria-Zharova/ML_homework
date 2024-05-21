import cv2
import os

haar_file = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

datasets = 'datasets'
sub_data = 'Person'
path = os.path.join(datasets, sub_data)

if not os.path.isdir(path):
    os.makedirs(path)

(width, height) = (130, 100)

webcam = cv2.VideoCapture(0)

count = 1
while count <= 30:
    ret, im = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite('%s/%s.png' % (path, count), face_resize)
        count += 1

    cv2.imshow('Collect Data', im)
    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()
