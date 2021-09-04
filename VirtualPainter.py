import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm



# #########
brushThickness=12

folderPath="Header"
myList=os.listdir(folderPath)
print(myList)
overlayList=[]

for imPath in myList:
    image=cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
for i in range(len(overlayList)):
    overlayList[i]=cv2.resize(overlayList[i],(640,83),interpolation=cv2.INTER_AREA)
header=overlayList[0]
drawColor=(255,0,255)
print("shape,",overlayList[0].shape)
cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
detector=htm.handDetector(detectionCon=0.85)

imgCanvas = np.zeros((480, 640, 3), np.uint8)

xp,yp=0,0
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
        # print(fingers)

        # 4 if selection mode - two fingers are up
        if fingers[1] and fingers[2]:
            # print("selection Mode") 
            #checking for the click
            if y1< 83:
                if 84<x1<170:
                    header=overlayList[0]
                    drawColor = (255, 0, 255)
                elif 170<x1<300:
                    header=overlayList[1]
                    drawColor = (255, 0, 0)
                elif 300<x1<435:
                    header=overlayList[2]
                    drawColor = (0, 255, 0)
                elif x1>435:
                    header=overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
        #5 if drawing mode - Index finger is up
        if fingers[1] and fingers[2]==False:
            # print("Drawing Mode") 
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            # pass
            if xp==0 and yp==0:
                xp,yp=x1,y1
            cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp,yp=x1,y1

    #setting the header details
    img[0:83, 0:640] = header
    cv2.imshow("Image",img)
    cv2.imshow("imgCanvas",imgCanvas)
    cv2.waitKey(1)