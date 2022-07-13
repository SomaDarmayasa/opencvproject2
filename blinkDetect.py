# Importing libraries
import cv2
import numpy as np
# menggunakan algoritma LBPH dari library opencv
recognition = cv2.face.LBPHFaceRecognizer_create()
# membaca file model train yang sudah dilakukan
recognition.read('train/trainer70.yml')
# Face and eye cascade classifiers from xml files
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')

font = cv2.FONT_HERSHEY_SIMPLEX
id = 0


nama = ['Tidak Diketahui', 'Soma', 'Elon', 'Chris', 'Soma Cerah']


first_read = True
# Video Capturing by using webcam
cam = cv2.VideoCapture(0)
cam.set(3, 680)  # lebar windows
cam.set(4, 480)  # tinggi win

weightMin = 0.1*cam.get(3)
heightMin = 0.1*cam.get(4)
retV, frame = cam.read()
while retV:
    # this will keep the web-cam running and capturing the frame for every loop
    retV, frame = cam.read()
    # Convert the rgb frame to gray
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Applying bilateral filters to remove impurities
    abuAbu = cv2.bilateralFilter(abuAbu, 5, 1, 1)
    # to detect face
    wajah = faceCascade.detectMultiScale(
        abuAbu, 1.2, 5, minSize=(int(weightMin), int(heightMin)))
    if len(wajah) > 0:
        for (x, y, w, h) in wajah:
            frame = cv2.rectangle(
                frame, (x, y), (x + w, y + h), (1, 190, 200), 2)
            # face detector
            roi_face = abuAbu[y:y + h, x:x + w]
            # frame
            roi_face_clr = frame[y:y + h, x:x + w]
            # to detect eyes
            eyes = eyeCascade.detectMultiScale(
                roi_face, 1.3, 5, minSize=(50, 50))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_face_clr, (ex, ey),
                              (ex+ew, ey+eh), (255, 153, 255), 2)
                if len(eyes) >= 2:
                    if first_read:
                        cv2.putText(frame, "Mata terdeteksi, kedipkan mata untuk pengenalan wajah", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                                    1, (255, 0, 0), 2)
                    else:
                        cv2.putText(frame, "Mata terbuka", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                                    1, (255, 255, 255), 2)
                else:
                    if first_read:
                        cv2.putText(frame, "Mata tidak terdeteksi", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                                    1, (255, 0, 255), 2)
                    else:
                        cv2.putText(frame, "Blink Detected.....!!!!", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                                    1, (0, 0, 0), 2)
                        id, confidance = recognition.predict(
                            abuAbu[y:y+h, x:x+w])

                        if (confidance < 60):  # bagusnya <60
                            id = nama[id]
                        else:
                            id = nama[0]

        cv2.putText(frame, str(id), (x+5, y-5), font, 1, (255, 255, 255), 2)
        # cv2.imshow('frame', frame)
        cv2.waitKey(10)
        print("Blink Detected.....!!!!")
    else:
        cv2.putText(frame, "Tidak Terdeteksi Wajah.", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                    1, (0, 255, 255), 2)
    cv2.imshow('Camera', frame)
    # press q to Quit and S to start
    # ord(ch) returns the ascii of ch
    k = cv2.waitKey(1)
    if k == 27 or k == ord('q'):
        break
    else:
        first_read = False

# release the web-cam
cam.release()
# close the window
cv2.destroyAllWindows()
