import cv2
import face_recognition
import numpy as np
import os

#Load ảnh từ image
path="image"
images=[]
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f"{path}/{cl}") #chay tung buc anh
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0]) # lấy tên bức ảnh

def Mahoa(images) :
    encodeList = []
    for img in images :
        img=cv2.cvtColor(img, cv2.COLOR_BRG2RGB)
        encode = face_recognition.face_encodeings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnow = Mahoa(images)

cap = cv2.VideoCapture(0) # hoặc để đường link nhận dạng qua video

while True:
    ret, frame = cap.read()
    frames = cv2.resize(frame,(0,0),None,fx=0.5,fy=0.5)
    frames = cv2.cvtColor(frames, cv2.COLOR_BRG2RGB)

    # xác định vị trí khuôn mặt trên cam và encode hình ảnh trên cam
    faceCurFrame = face_recognition.face_locations(frames) # lấy từng khuôn mặt và vị trí khuôn mặt hiện tại
    encodeCurFrame = face_recognition.face_encodings(frames)
    
    for encodeFace, faceLoc in zip(encodeCurFrame,faceCurFrame) : # lấy từng khuôn mặt hiện tại theo cặp
        matches = face_recognition.compare_faces(encodeListKnow,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnow,encodeFace)
        matchIndex = np.argmin(faceDis) # đẩy về vị trí của faceDis  nhỏ nhất

        if faceDis[matchIndex] < 0.50 : 
            name = classNames[matchIndex].upper()
        else : 
            name = "Unknown face"
        
        #In tên lên frame
        
        y1,x2,y2,x1=faceLoc 
        y1,x2,y2,x1= y1*2,x2*2,y2*2,x1*2
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,name,(x2,y2),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        
        
    cv2.imshow("Frame",frame)
    if cv2.waitKey(1) == ord("q") :  # độ trễ 1/1000s, bấm q thoát
        break
cap.release() #giải phóng camera
cv2.destroyAllWindows() #thoát tất cả các cửa sổ
