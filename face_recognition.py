import numpy as np
import cv2 as cv
from PIL import ImageGrab

face_cascade = cv.CascadeClassifier('cascade/data/haarcascade_frontalface_alt2.xml')

cap = cv.VideoCapture(0)

while True:
    # Webcam capture
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)


    for (x, y, w, h) in faces:
        # Position
        print(x, y, w, h)

        # Draw rectangle where face is detected
        color = (255, 0, 0)
        stroke = 2
        width = x + w
        height = y + h
        cv.rectangle(frame, (x,y), (width, height), color, stroke)

    # Screen capture
    scr = ImageGrab.grab()
    scr_np = np.array(scr)

    screen = cv.cvtColor(scr_np, cv.COLOR_BGR2GRAY)
    print(screen)
    canny_scr = cv.Canny(screen, 125, 125)

    cv.imshow('Screen', canny_scr)

    # Display resulting frame
    if cv.waitKey(20) & 0xFF == ord('q'):
        cv.destroyAllWindows()
        break