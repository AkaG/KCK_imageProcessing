import cv2


def overlay_image(img_to_overlay, img_overlying, x_offset, y_offset, face_width, face_height):

    scale_x = (face_width / 254.0)
    scale_y = (face_height / 254.0)

    resized_image = cv2.resize(img_overlying, (0, 0), fx=scale_x, fy=scale_y)
    rgba_frame = cv2.cvtColor(img_to_overlay, cv2.COLOR_BGR2RGBA)

    x_offset -= (25 * scale_x)

    for c in range(0, 3):
        rgba_frame[y_offset:y_offset + resized_image.shape[0], x_offset:x_offset + resized_image.shape[1], c] = \
            resized_image[:, :, c] * (resized_image[:, :, 3] / 255.0) +\
            rgba_frame[y_offset:y_offset + resized_image.shape[0], x_offset:x_offset + resized_image.shape[1], c] * \
            (1.0 - resized_image[:, :, 3] / 255.0)

    return cv2.cvtColor(rgba_frame, cv2.COLOR_RGBA2BGR)


def main():
    face_path = "haarcascades/haarcascade_frontalface_default.xml"
    smile_path = "haarcascades/haarcascade_smile.xml"
    eye_path = "haarcascades/haarcascade_eye.xml"

    face_cascade = cv2.CascadeClassifier(face_path)
    smile_cascade = cv2.CascadeClassifier(smile_path)
    eye_cascade = cv2.CascadeClassifier(eye_path)

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    googles = cv2.imread("resources/glasses.png", cv2.IMREAD_UNCHANGED)

    animation = False
    animate_y = 0

    while True:

        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=8,
            minSize=(55, 55)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            roi_gray = gray[y:y+h, x:x+w]
            half_roi_gray = gray[y:y+h, x:x+int(w/2)]
            roi_color = frame[y:y+h, x:x+w]

            smile = smile_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.4,
                minNeighbors=22,
                minSize=(25, 25)
                )

            if len(smile) == 0 and len(faces) == 1:
                animation = False
                animate_y = 0

            for index, elem in enumerate(smile):
                (sx, sy, sw, sh) = elem
                cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (255, 0, 0), 1)

                eyes = eye_cascade.detectMultiScale(
                    half_roi_gray,
                    scaleFactor=1.02,
                    minNeighbors=20,
                    minSize=(45, 45)
                    )

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'Smile!', (x + int(w / 2) - 50, y), font, 1, (255, 255, 255), 2)

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

                    if not animation:
                        frame = overlay_image(frame, googles, x+ex, animate_y, w, h)
                        animate_y += 20

                        if animate_y >= y+ey-10:
                            animation = True

                    else:
                        frame = overlay_image(frame, googles, x+ex, y+ey-10, w, h)

        cv2.imshow('Smile Detector', frame)
        c = cv2.waitKey(7)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
