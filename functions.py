import numpy as np
import cv2 as cv
from PIL import ImageGrab

class Fake3D:

    resolution = ImageGrab.grab().size

    def __init__(self):
        self.w = self.resolution[0]
        self.h = self.resolution[1]
    
    def face_track(self):
        face_cascade = cv.CascadeClassifier('cascade/data/haarcascade_frontalface_alt2.xml')
        cap = cv.VideoCapture(0)

        # Capture frame by frame
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

        for (x, y, w, h) in faces:
            return (x, y, w, h)
            