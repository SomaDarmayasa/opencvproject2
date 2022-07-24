import cv2
# menghitung jarak facial landmark dengan rasio mata
from scipy.spatial import distance as dist
import numpy as np
import dlib  # detect landmark
import imutils
from imutils import face_utils  # mengubah koordinat(x,y) menjadi numpy
from datetime import datetime

# menggunakan algoritma LBPH dari library opencv
recognition = cv2.face.LBPHFaceRecognizer_create()
# membaca file model train yang sudah dilakukan
recognition.read('train/trainer70.yml')
# Face and eye cascade classifiers from xml files
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')


""" 
dlib describes eye with 6 points.
when you blink, the EAR value will change from 0.3 to  near 0.05
"""


# batas rasio ukuran mata terbuka jika dibawah 0.2 maka mata tertutup
Eye_AR_Thresh = 0.2


# berapa frame yang ditampilkan pada saat berkedip
Eye_AR_Consec_frames = 3


# inisialisasi counter frame untuk menambah jumlah kedipan
counter = 0
total = 0

#
font = cv2.FONT_HERSHEY_COMPLEX
id = " "  # set default id dengan string kosong

nama = ['Tidak Diketahui', 'Soma', 'Krisna', 'Risma']
# menampung nama yang akan di recognisi ke dalam suatu list array


# membuat definisi markattandance dengan parameter requestnya adalah nama
def markAttendance(nama):
    # dengan membuka file Absen.csv setelah itu diread dan diwrite dengan string r+ as f
    with open("Absen.csv", 'r+') as f:
        namesDatalist = f.readlines()  # memebaca baris filenya dengan fungsi readlines
        namelist = []  # inisialisasi awal namelist dengan list kosong
        for line in namesDatalist:  # loop line pada nameDataList
            entry = line.split(',')  # split line dengan string koma
            # pada namelist ditambahkan parameter request nama ke index list pertama yaitu 0
            namelist.append(entry[0])
        if nama not in namelist:  # jika nama tidak terdapat dalam namelist
            now = datetime.now()  # membuat variable now yang berisi datetime
            # ubah date time menjadi string hours,menutes, second
            dtString = now.strftime('%H:%M:%S')
            # tulis string nama dan  datetimestring tadi ke dalam baris file excel
            f.writelines(f'\n{nama},{dtString}')


def eye_aspect_ratio(eye):
    # menghitung jarak euclidean distance diantara dua himpunan eye landmark koordinat(x,y)
    A = dist.euclidean(eye[1], eye[5])  # horizontal
    B = dist.euclidean(eye[2], eye[4])  # horizontal

    # menghitung jarak euclidean distance diantara dua himpunan eye landmark koordinat(x,y)
    C = dist.euclidean(eye[0], eye[3])  # vertical

    # menghitung aspek rasio mata
    ear = (A+B) / (2*C)

    # return the eye aspect ratio
    return ear


# inisialisasi dlib face detector dengan (HOG-based)
# membuat model facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cam = cv2.VideoCapture(0)
weightMin = 0.1*cam.get(3)  # weightmin
heightMin = 0.1*cam.get(4)  # heightmin

while True:
    ret, frame = cam.read()
    if frame is None:
        break

    # ambil frame/gambar dari video capture, selanjutnya resize dan convert kedalam bentuk grayscale
    frame = imutils.resize(frame, width=500)
    abuabu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(abuabu, 0)

    wajah = faceCascade.detectMultiScale(
        abuabu, 1.2, 5, minSize=(int(weightMin), int(heightMin)))

    for(x, y, w, h) in wajah:

        # loop over the face detections
        for face in rects:
            (x1, y1) = (face.left(), face.top())
            (x2, y2) = (face.right(), face.bottom())

            # tentukan facial landmarks(x,y) untuk wilayah wajah
            # convert facial landmark koordinat(x,y) kedalam bentuk numpy array
            shape = predictor(abuabu, face)
            shape = face_utils.shape_to_np(shape)

            # mencari left dan right eyes setelah itu hitung dengan EAR
            # left eye berisi 37-42 poin(numpy start from 0)
            # mengubah koordinat left dan right eye menjadi numpy array
            leftEye = shape[36:42]
            rightEye = shape[42:48]

            # extract koordinat left dan right eye
            # selanjutnya gunakan koordinat dari fungsi eye_aspect_ratio
            # untuk menghitung rasio kedua mata baik left maupun right eye
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # hitung rata2 aspek rasio keseluruhan untuk kedua mata
            EAR = (leftEAR + rightEAR) / 2

            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if EAR < Eye_AR_Thresh:
                counter += 1
                id, confidance = recognition.predict(
                    abuabu[y:y+h, x:x+w])
                """menggunakan variabel id berdasarkan id yg 
                        direkam dan confidance untuk mempredict dari file train"""

                if (confidance < 70):  # jika nilai confidence kurang dari 60 ,bagusnya <60
                    # maka digunakan variable nama sesuai id yang telah dibuat
                    id = nama[id]
                else:  # selain itu
                    # maka digunakan variable nama dengan index pertama adalah 0
                    id = nama[0]

            # otherwise, the eye aspect ratio is not below the blink
            # threshold
            else:
                # if the eyes were closed for a sufficient number of
                # then increment the total number of blinks
                if counter > Eye_AR_Consec_frames:
                    total += 1

                # reset the eye frame counter
                counter = 0

            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            # membuat semacam garis di area mata
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # draw the total number of blinks on the frame along with
            # the computed eye aspect ratio for the frame
            cv2.putText(frame, "Blinks: {}".format(total),
                        (10, 20), font, 0.55, (0, 0, 255), 1)
            cv2.putText(frame, "EAR: {:.2f}".format(EAR),
                        (10, 50), font, 0.55, (0, 0, 255), 1)
            cv2.putText(frame, str(id), (x+5, y-5),
                        font, 1, (255, 255, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("frame", frame)
    if cv2.waitKey(30) == ord('q'):
        break
markAttendance(id)
cam.release()
cv2.destroyAllWindows()
