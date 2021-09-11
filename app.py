from flask import Flask, render_template, Response
from camera import Video
import time
import cv2
haar_cascade = cv2.CascadeClassifier('haar_face.xml')

people=['Subhopriyo','Bedantika']

face_recognizer_model = cv2.face.LBPHFaceRecognizer_create()
face_recognizer_model.read('face_recognizer.yml')

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/register')
def register():
    return render_template('register.html')


def generate(camera):
    while True:
        start = time.time()
        frame=camera.getFrame()
        img = frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces_rect = haar_cascade.detectMultiScale(gray,1.1,4)

        for (x,y,w,h) in faces_rect:
            faces_roi = gray[y:y+h, x:x+w]

            label, confidence = face_recognizer_model.predict(faces_roi)
            cv2.rectangle(img, (x,y), (x+w, y+h) , (0,255,0), 2)
            cv2.putText(img, str(people[label]), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0),2)
            # if confidence>50:
            #     cv2.putText(img, str(people[label]), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0),2)
            # else:
            #     cv2.putText(img, 'Unknown', (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0),2)

        ret, jpg = cv2.imencode('.jpg',img)
        img=jpg.tobytes()
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')
        end = time.time()
        fps = 1/(end-start)
        fps = "{:.2f}".format(fps)


@app.route('/video')
def video():
    return Response(generate(Video()),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)