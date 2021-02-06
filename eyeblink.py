# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 20:08:18 2021

@author: RAJAT HORE
"""
import cv2
f_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
e_cascade=cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
f_read=True
cap=cv2.VideoCapture(0)
ret,img=cap.read()
while(ret):
    ret,img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #to convert to gray image(single channel)because haar cascade works on gray images
    gray=cv2.bilateralFilter(gray,5,1,1) #to remove the impurities in the image so that haar cascade classifier works more properly 
    faces=f_cascade.detectMultiScale(gray,1.3,5,minSize=(200,200)) #the frame which is captured is scaled to different scales to detect face.Minsize of scale is 200*200 
    if(len(faces)>0):
        for (x,y,w,h) in faces:
            img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            roi_face=gray[y:y+h,x:x+w]
            roi_face_clr=img[y:y+h,x:x+w]
            eyes=e_cascade.detectMultiScale(roi_face,1.3,5,minSize=(50,50))
            if(len(eyes)>=2):
                if(f_read):
                    cv2.putText(img,"press s to start",(100,100),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),2)
                else:
                    print("------------------")
            else:
                if(f_read):
                    cv2.putText(img,"no eyes detected",(100,100),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),2)
                else:
                    print("You loose")
                    first_read=True
    else:
        cv2.putText(img,"no face detected",(100,100),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),2)
    cv2.imshow('img',img)
    a=cv2.waitKey(1)
    if(a==ord('q')):
        break
    elif(a==ord('s') and first_read):
        first_read=False
cap.release()
cv2.destroyALLWindows()
    