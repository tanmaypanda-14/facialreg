import cv2
import numpy as np

#to classiy face we use cascades
face_cascade=cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

#To capture video
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #To identify face using cascades
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)

    for(x,y,w,h) in faces:
        print(x,    y,  w,    h)
        #To capture face we use the cascade co-ordinates
        roi=gray[y:y+h,  x:x+w]
        roi2=frame[y:y+h,  x:x+w]

        #To save image as jpeg
        img_save="faceimage.jpeg"

        #To write image
        cv2.imwrite(img_save,roi2)

        #To identify face while in live camera
        color=(0,255,0)
        stroke=2
        width=x+w
        height=y+h
        cv2.rectangle(frame,(x,y),(width,height),color,stroke)

    #To show gui of the live camera
    cv2.imshow('frame', rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()