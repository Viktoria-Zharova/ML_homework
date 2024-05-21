import cv2
import numpy as np
import os

datasets = 'datasets'
sub_data = 'Person'
path = os.path.join(datasets, sub_data)

(width, height) = (130, 100)

(images, labels, names, id) = ([], [], {}, 0)

for subdir, dirs, files in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1

(images, labels) = [np.array(lst) for lst in [images, labels]]

model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

model.save('face_recognizer.yml')
