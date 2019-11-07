import cv2
import numpy as np

img_eye_1 = cv2.imread('nchallah.png')
img_eye_2 = cv2.imread('eye.png')
img_eye_3 = cv2.imread('aayn.png')
Face_Cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_Cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
i=0
images = [img_eye_1, img_eye_2, img_eye_3]
j = 0

while True :
    ret , frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = Face_Cascade.detectMultiScale(gray,1.3, 5)
    for(x,y,w,h) in faces : 
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        i += 1
        Face_Name = "face" + str(i) + ".jpg"
        #cv2.imwrite(Face_Name, roi_color)
        eyes = eye_Cascade.detectMultiScale(roi_gray)
        for(ex,ey,ew,eh) in eyes:
            if(cv2.waitKey(1) == 49) :
                j = 0
            elif(cv2.waitKey(1) == 50) :
                j = 1
            elif(cv2.waitKey(1) == 51) :
                j = 2
            eye = cv2.resize(images[j], (ew, eh), interpolation=cv2.INTER_AREA)
            roi_color[ey:ey + eh, ex:ex + ew] = eye
            Eye_Name = "Eyes" + str(i) + ".jpg"
            #cv2.imwrite(Eye_Name, roi_color)
    cv2.imshow("Test", frame)
    if(cv2.waitKey(1) == 27):
        break
cap.release()
cv2.destroyAllWindows()
