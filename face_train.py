import os
import cv2 as cv
import numpy as np

people=['Subhopriyo','Bedantika']
DIR = r'Resources/Train'

haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR,person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path=os.path.join(path,img)

            image=cv.imread(img_path)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)
            
print('Traning Model----------------------->')
create_train()

features = np.array(features,dtype='object')
labels = np.array(labels)

face_recognizer_model = cv.face.LBPHFaceRecognizer_create()

face_recognizer_model.train(features,labels)
face_recognizer_model.save('face_recognizer.yml')
# np.save('features.npy',features)
# np.save('labels.py',labels)

print('Model Trained and Saved------------->')