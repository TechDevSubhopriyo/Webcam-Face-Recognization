import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people=['Subhopriyo','Bedantika']

face_recognizer_model = cv.face.LBPHFaceRecognizer_create()
face_recognizer_model.read('face_recognizer.yml')

cap = cv.VideoCapture(0)

while True:
    _, img = cap.read()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces_rect = haar_cascade.detectMultiScale(gray,1.1,4)

    for (x,y,w,h) in faces_rect:
        faces_roi = gray[y:y+h, x:x+w]

        label, confidence = face_recognizer_model.predict(faces_roi)
        # print(f'Face Detected of {str(people[label])} with Confidence of {confidence}')
        #if(confidence>100):
        cv.rectangle(img, (x,y), (x+w, y+h) , (0,255,0), 2)
        cv.putText(img, str(people[label]), (x,y-10), cv.FONT_HERSHEY_SIMPLEX,1, (0,255,0),2)
    
    cv.putText(img, 'Face Recognization is ON, ESC to close', (0,20), cv.FONT_HERSHEY_SIMPLEX,0.8, (0,0,0),2)
    cv.imshow('Face Recognization', img)
    k =cv.waitKey(10) & 0xFF
    if k==27:
        break

cap.release()

#cv.imshow("Detected", img)

#cv.waitKey(0)