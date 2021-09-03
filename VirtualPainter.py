import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm


folderPath="Header"
myList=os.listdir(folderPath)
print(myList)
overlayList=[]

for imPath in myList:
    image=cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))



overlayList[0]=cv2.resize(overlayList[0],(640,83),interpolation=cv2.INTER_AREA)
header=overlayList[0]
print("shape,",overlayList[0].shape)
cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,640)
detector=htm.handDetector(detectionCon=0.85)



while True:
    # 1 import image
    success,img=cap.read()
    img=cv2.flip(img,1)
    # print("image shape" ,img.shape)




    # 2 find the hand landmark
    img=detector.findHands(img)
    lmList=detector.findPosition(img,draw=False)

    if len(lmList)!=0:
        # print(lmList)
        # tip of index and middle finger
        x1,y1=lmList[8][1:]
        # middle finger
        x2,y2=lmList[12][1:]

        # 3 check which finger is up
        fingers=detector.fingersUp()
        print(fingers)

        # 4 if selection mode - two fingers are up
        if fingers[1] and fingers[2]:
            # print("selection Mode") 
            pass

        #5 if drawing mode - Index finger is up
        if fingers[1] and fingers[2]==False:
            # print("Drawing Mode") 
            pass

    #setting the header details
    img[0:83, 0:640] = header
    cv2.imshow("Image",img)
    cv2.waitKey(1)