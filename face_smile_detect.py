# face_smile_detect.py
import cv2

# Haar Cascade XML dosyalarının yolları
face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
smile_cascade_path = cv2.data.haarcascades + "haarcascade_smile.xml"

# Cascade sınıflandırıcılarını yükle
face_cascade = cv2.CascadeClassifier(face_cascade_path)
smile_cascade = cv2.CascadeClassifier(smile_cascade_path)

# Kamerayı başlat
cap = cv2.VideoCapture(0)

while True:
    # Kameradan bir kare al
    ret, frame = cap.read()
    if not ret:
        print("Kamera erişilemiyor!")
        break

    # Görüntüyü gri tonlara dönüştür
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüzleri tespit et
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Her bir yüz için:
    for (x, y, w, h) in faces:
        # Yüzü kutucukla göster
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Yüz bölgesinde gülümseme tespiti
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)

        for (sx, sy, sw, sh) in smiles:
            # Gülümsemeyi kutucukla göster
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)

    # Görüntüyü ekranda göster
    cv2.imshow("Yüz ve Gülümseme Tespiti", frame)

    # Çıkış için 'q' tuşuna basılabilir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı ve pencereyi kapat
cap.release()
cv2.destroyAllWindows()
