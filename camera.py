import cv2

class Video(object):
    def __init__(self):
        self.video=cv2.VideoCapture(0, cv2.CAP_DSHOW)
    def __del__(self):
        self.video.release()
    def getFrame(self):
        ret ,frame = self.video.read()
        return frame

