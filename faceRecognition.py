import cv2  # import library opencv

# menggunakan algoritma LBPH dari library opencv
recognition = cv2.face.LBPHFaceRecognizer_create()
# membaca file model train yang sudah dilakukan
recognition.read('train/trainer70.yml')
# melakukan load classifier haarcascade_facefrontal_default.xml
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
EyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")

font = cv2.FONT_HERSHEY_SIMPLEX
id = 0


nama = ['Tidak Diketahui', 'Soma', 'Elon', 'Chris', 'Soma Cerah']

cam = cv2.VideoCapture(0)  # membuka kamera

cam.set(3, 680)  # lebar windows
cam.set(4, 480)  # tinggi win

weightMin = 0.1*cam.get(3)
heightMin = 0.1*cam.get(4)


while True:
    revT, frame = cam.read()  # mengambil frame dari camera dan ditampilkan
    # variabel untuk membuat gambar bg menjadi gray
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = faceCascade.detectMultiScale(
        abuAbu,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(weightMin), int(heightMin))
    )
    
    
    for (x,y,w,h) in wajah:
        cv2.rectangle(frame,(x,y), (x+w,y+h),(0,255,0),2)
        
        id, confidance = recognition.predict(abuAbu[y:y+h, x:x+w])
        
        if (confidance < 60): #bagusnya <60
            id = nama[id] 
        else:
            id = nama[0]
            
        
        cv2.putText(frame, str(id),(x+5, y-5), font, 1,(255,255,255),2)
        # cv2.putText(frame, str(confidance),(x+5, y+h-5), font, 1,(255,255,0),2)
        
    cv2.imshow('Camera', frame)
    
    k = cv2.waitKey(10) & 0xff
    if k == 27 or k == ord('q'):
        break

print("\n [INFO] Program Keluar")
cam.release()
cv2.destroyAllWindows

    
    
    