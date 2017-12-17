import cv2
from pygame import mixer


def overlay_image(img_to_overlay, img_overlying, x_offset, y_offset, face_width, face_height):
    scale_x = (face_width / 254.0)
    scale_y = (face_height / 254.0)

    x_offset += int(20 / scale_x) - 10
    y_offset += int(10 / scale_y) - 5

    resized_image = cv2.resize(img_overlying, (0, 0), fx=scale_x, fy=scale_y)
    rgba_frame = cv2.cvtColor(img_to_overlay, cv2.COLOR_BGR2RGBA)

    for c in range(0, 3):
        rgba_frame[y_offset:y_offset + resized_image.shape[0], x_offset:x_offset + resized_image.shape[1], c] = \
            resized_image[:, :, c] * (resized_image[:, :, 3] / 255.0) + \
            rgba_frame[y_offset:y_offset + resized_image.shape[0], x_offset:x_offset + resized_image.shape[1], c] * \
            (1.0 - resized_image[:, :, 3] / 255.0)

    return cv2.cvtColor(rgba_frame, cv2.COLOR_RGBA2GRAY)

wait_cycles = 10


def stop_playing_music_wait():
    global wait_cycles
    if wait_cycles == 0:
        return True
    else:
        wait_cycles -= 1
        return False


def main():
    face_path = "haarcascades/haarcascade_frontalface_default.xml"
    eye_path = "haarcascades/haarcascade_eye.xml"
    music_path = "resources/thug_life.mp3"

    face_cascade = cv2.CascadeClassifier(face_path)
    eye_cascade = cv2.CascadeClassifier(eye_path)

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    googles = cv2.imread("resources/glasses_with_cigarette.png", cv2.IMREAD_UNCHANGED)

    global wait_cycles

    mixer.init()
    mixer.music.load(music_path)
    play_music, is_music_playing = False, False

    animation = False
    animate_y = 0

    while True:
        try:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=8,
                minSize=(55, 55)
            )

            if len(faces) == 0:
                play_music = False

            if play_music and not is_music_playing:
                mixer.music.play(-1)
                is_music_playing, wait_cycles = True, 10
            elif not play_music and stop_playing_music_wait():
                mixer.music.stop()
                is_music_playing = False

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                half_roi_gray = gray[y:y + h, x:x + int(w / 2)]
                roi_color = frame[y:y + h, x:x + w]

                eyes = eye_cascade.detectMultiScale(
                    half_roi_gray,
                    scaleFactor=1.02,
                    minNeighbors=20,
                    minSize=(45, 45)
                )

                if len(eyes) == 0 and len(faces) == 1:
                    animation, play_music = False, False
                    animate_y = 0
                else:
                    play_music = True

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                    if not animation:
                        frame = overlay_image(frame, googles, x + ex - 30, animate_y, w, h)
                        animate_y += 20

                        if animate_y >= y + ey - 10:
                            animation = True
                    else:
                        frame = overlay_image(frame, googles, x + ex - 30, y + ey - 10, w, h)

        except Exception as e:
            print(e)
        cv2.imshow('Face Detector', frame)
        c = cv2.waitKey(7)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
