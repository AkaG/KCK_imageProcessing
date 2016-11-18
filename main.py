import numpy as np
import sys
import cv2

def main():
    facePath = "haarcascades/haarcascade_frontalface_default.xml"
    smilePath = "haarcascades/haarcascade_smile.xml"

    faceCascade = cv2.CascadeClassifier(facePath)
    smileCascade = cv2.CascadeClassifier(smilePath)

    cap = cv2.VideoCapture(2)

    sF = 1.05

    while True:

        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor= sF,
            minNeighbors=8,
            minSize=(55, 55),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            smile = smileCascade.detectMultiScale(
                roi_gray,
                scaleFactor= 1.7,
                minNeighbors=22,
                minSize=(25, 25),
                )

            for (x, y, w, h) in smile:
                cv2.rectangle(roi_color, (x, y), (x+w, y+h), (255, 0, 0), 1)

        cv2.imshow('Smile Detector', frame)
        c = cv2.waitKey(7)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()