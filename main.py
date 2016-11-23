import numpy as np
import sys
import cv2

def overlayImage(imgToOverlay, imgOverlying, x_offset, y_offset):
    rgbaFrame = cv2.cvtColor(imgToOverlay, cv2.COLOR_BGR2RGBA)
    for c in range(0,3):
        rgbaFrame[y_offset:y_offset + imgOverlying.shape[0], x_offset:x_offset + imgOverlying.shape[1], c] =\
            imgOverlying[:, :, c] * (imgOverlying[:, :, 3] / 255.0) +\
            rgbaFrame[y_offset:y_offset + imgOverlying.shape[0], x_offset:x_offset+ imgOverlying.shape[1], c] * (1.0 - imgOverlying[:, :, 3] / 255.0)

    return cv2.cvtColor(rgbaFrame, cv2.COLOR_RGBA2BGR)

def main():
    facePath = "haarcascades/haarcascade_frontalface_default.xml"
    smilePath = "haarcascades/haarcascade_smile.xml"
    eyePath = "haarcascades/haarcascade_eye.xml"

    faceCascade = cv2.CascadeClassifier(facePath)
    smileCascade = cv2.CascadeClassifier(smilePath)
    eye_cascade = cv2.CascadeClassifier(eyePath)

    cap = cv2.VideoCapture(2)

    googles = cv2.imread("resources/glasses.png", cv2.IMREAD_UNCHANGED)

    while True:

        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor= 1.05,
            minNeighbors=8,
            minSize=(55, 55)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            roi_gray = gray[y:y+h, x:x+w]
            half_roi_gray = gray[y:y+h, x:x+int(w/2)]
            roi_color = frame[y:y+h, x:x+w]

            smile = smileCascade.detectMultiScale(
                roi_gray,
                scaleFactor= 1.7,
                minNeighbors=22,
                minSize=(25, 25)
                )
            for (sx, sy, sw, sh) in smile:
                cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (255, 0, 0), 1)

            eyes = eye_cascade.detectMultiScale(
                half_roi_gray,
                scaleFactor=1.3,
                minNeighbors=4,
                minSize=(55, 55)
                )
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                frame = overlayImage(frame, googles, x+ex-20, y+ey-20)

        cv2.imshow('Smile Detector', frame)
        c = cv2.waitKey(7)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()