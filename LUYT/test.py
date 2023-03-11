import cv2
import face_recognition
#BRG->RGB
img = face_recognition.load_image_file("image/Luyt.jpg")
image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_check = face_recognition.load_image_file("image/Luyt.jpg")
image_check = cv2.cvtColor(img_check,cv2.COLOR_BGR2RGB)

#xac định vị trí khuôn mặc
faceLoc = face_recognition.face_locations(img)[0]   #y1,x2,y2,x1

# mã hóa hình ảnh
encode = face_recognition.face_encodings(img)[0]
y1,x2,y2,x1= faceLoc 
cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),2)

cv2.imshow("Image",img)
cv2.waitKey()



# cho image check
#xac định vị trí khuôn mặc
faceLoc_check = face_recognition.face_locations(img_check)[0]   #y1,x2,y2,x1

# mã hóa hình ảnh
encode_check = face_recognition.face_encodings(img_check)[0]
y1,x2,y2,x1=faceLoc_check 
cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),2)

cv2.imshow("Image_check",img_check)
cv2.waitKey()

#kết quả so sánh true or false
results = face_recognition.compare_faces([encode],encode_check)
print(results)

#Sai số các ảnh

face_dis = face_recognition.face_distance([encode],encode_check)
print(face_dis)
#ghi kết quả so sánh trên ảnh
cv2.putText(img_check,f"{results}{round(face_dis[0],2)}",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)