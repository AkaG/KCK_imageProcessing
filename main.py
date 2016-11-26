import cv2


def overlay_image(img_to_overlay, img_overlying, x_offset, y_offset, face_width, face_height):

    scale_x = (face_width / 254.0)
    scale_y = (face_height / 254.0)

    resized_image = cv2.resize(img_overlying, (0, 0), fx=scale_x, fy=scale_y)
    rgba_frame = cv2.cvtColor(img_to_overlay, cv2.COLOR_BGR2RGBA)

    for c in range(0, 3):
        rgba_frame[y_offset:y_offset + resized_image.shape[0], x_offset:x_offset + resized_image.shape[1], c] = \
            resized_image[:, :, c] * (resized_image[:, :, 3] / 255.0) +\
            rgba_frame[y_offset:y_offset + resized_image.shape[0], x_offset:x_offset + resized_image.shape[1], c] * \
            (1.0 - resized_image[:, :, 3] / 255.0)

    return cv2.cvtColor(rgba_frame, cv2.COLOR_RGBA2BGR)


def main():
    facePath = "haarcascades/haarcascade_frontalface_default.xml"
    smilePath = "haarcascades/haarcascade_smile.xml"
    eyePath = "haarcascades/haarcascade_eye.xml"

    faceCascade = cv2.CascadeClassifier(facePath)
    smileCascade = cv2.CascadeClassifier(smilePath)
    eye_cascade = cv2.CascadeClassifier(eyePath)

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    googles = cv2.imread("resources/glasses.png", cv2.IMREAD_UNCHANGED)

    while True:

        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=8,
            minSize=(55, 55),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            roi_gray = gray[y:y+h, x:x+w]
            half_roi_gray = gray[y:y+h, x:x+int(w/2)]
            roi_color = frame[y:y+h, x:x+w]

            smile = smileCascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.7,
                minNeighbors=22,
                minSize=(25, 25)
                )

            for (sx, sy, sw, sh) in smile:
                cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (255, 0, 0), 1)

                eyes = eye_cascade.detectMultiScale(
                    half_roi_gray,
                    scaleFactor=1.02,
                    minNeighbors=20,
                    minSize=(45, 45)
                    )

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                    frame = overlay_image(frame, googles, x+ex-30, y+ey-10, w, h)

        cv2.imshow('Smile Detector', frame)
        c = cv2.waitKey(7)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
