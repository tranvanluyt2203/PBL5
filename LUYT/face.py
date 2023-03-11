import cv2
from PIL import Image
# Tải file XML Cascade Classifier
face_cascade = cv2.CascadeClassifier('LUYT/haarcascade_frontalface_default.xml')

# Đọc ảnh và chuyển đổi sang grayscale
img = cv2.imread('LUYT/image/User.1.1.jpg')
image=Image.open('LUYT/image/User.1.1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Nhận diện khuôn mặt trong ảnh
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Vẽ khung bao quanh khuôn mặt

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cropped_image = image.crop((x, y, x + w, y + h))
print(faces)
# Hiển thị ảnh đã nhận diện khuôn mặt
cv2.imshow('img', img)
cropped_image.save("LUYT/drop.jpg")
cv2.waitKey()
cv2.destroyAllWindows()