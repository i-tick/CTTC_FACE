# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 18:54:10 2020

@author: hp
"""


import cv2
import urllib
import numpy as np
import urllib.request as ur
classifier = cv2.CascadeClassifier('C:/Python38/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
#classifier = cv2.CascadeClassifier(face_data)Aitik

url = "http://26.40.251.196:8080/shot.jpg"
data=[]

while len(data)!=100:
    img_of_url = ur.urlopen(url)
    
    frame = np.array(bytearray(img_of_url.read()),np.uint8)
    frame = cv2.imdecode(frame,-1)
    
    faces = classifier.detectMultiScale(frame,1.5,5)
    
    if len(faces)>0:
        for x,y,w,h in faces:
            face_frame = frame[y:y+h,x:x+w].copy()
            cv2.imshow("only_face",face_frame)
            
            if len(data)<=100:
                print(len(data)+1,"/100")
                data.append(face_frame)
            else:
                break
    cv2.imshow("capture",frame)
    if cv2.waitKey(30)==ord('a'):
        break
cv2.destroyAllWindows()

if len(data) == 100:
    name = input("enter name")
    for i in range(100):
        cv2.imwrite("images/"+name+"_"+str(i+1)+".jpg",data[i])
        
        print("complete")
else:
    print("need more data")
