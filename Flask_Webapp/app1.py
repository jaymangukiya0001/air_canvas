import tkinter as tk
from flask import Flask, render_template, Response
import cv2
import os
import numpy as np
import time
import HandTrackingModule as htm
import datetime
from tkinter import *

app = Flask(__name__)


def gen_frames():  # generate frame by frame from camera
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    root = Tk()
    root.geometry("400x300")
    v1 = DoubleVar()
    v = StringVar(root, "1")
    values = {"with tracing": "1",
              "without tracing": "2"}

    for (text, value) in values.items():
        Radiobutton(root, text=text, variable=v,
                    value=value).pack(side=TOP, ipady=5)

    def show1():
        sel = "brush thickness is : = " + str(v1.get())
        l1.config(text=sel, font=("Courier", 14))
    s1 = Scale(root, variable=v1, from_=1, to=10, orient=HORIZONTAL)

    l3 = Label(root, text="Brush thickness")

    b1 = Button(root, text="brush thickness is:", command=show1, bg="yellow")

    l1 = Label(root)

    s1.pack(anchor=CENTER)
    l3.pack()
    b1.pack(anchor=CENTER)
    l1.pack()
    root.mainloop()

    operation_value = v.get()
    print(operation_value)

    # print("started")
    brushThickness = 5
    eraserThickness = 150
    brushThickness = int(v1.get())
    print(brushThickness)
    folderPath = "Header2"
    folderPath2 = "Header1"
    folderPath3 = "Header3"
    folderPath4 = "Header4"

    myList = os.listdir(folderPath)
    # print(myList)
    overlayList = []
    overlayList3 = []
    overlayList4 = []
    filled = []
    drawFilled = False
    for imPath in myList:
        image = cv2.imread(f'{folderPath}/{imPath}')
        image = cv2.resize(image, (540, 83), interpolation=cv2.INTER_AREA)
        overlayList.append(image)
    myList2 = os.listdir(folderPath2)
    overlayList2 = []
    for imPath in myList2:
        image = cv2.imread(f'{folderPath2}/{imPath}')
        image = cv2.resize(image, (50, 480), interpolation=cv2.INTER_AREA)
        overlayList2.append(image)
    myList3 = os.listdir(folderPath3)
    for imPath in myList3:
        image = cv2.imread(f'{folderPath3}/{imPath}')
        image = cv2.resize(image, (50, 480), interpolation=cv2.INTER_AREA)
        overlayList3.append(image)

    myList4 = os.listdir(folderPath4)
    for imPath in myList4:
        image = cv2.imread(f'{folderPath4}/{imPath}')
        image = cv2.resize(image, (50, 480), interpolation=cv2.INTER_AREA)
        overlayList4.append(image)
    # print(len(overlayList))
    # print(overlayList)

    header = overlayList[0]
    header2 = overlayList2[0]
    header3 = overlayList3[0]
    header4 = overlayList4[0]
    drawColor = (255, 0, 255)
    shape = 'freestyle'

    detector = htm.handDetector(detectionCon=0.85, maxHands=1)
    xp, yp = 0, 0
    imgCanvas = np.zeros((480, 640, 3), np.uint8)

    i = 0

    right_header_change = False
    # for a time management
    cTime = 0
    pTime = datetime.datetime.now()
    watermark = cv2.imread("watermark/w6.png")
    watermark = cv2.resize(watermark, (360, 862),
                           interpolation=cv2.INTER_AREA)
    watermark = cv2.flip(watermark, 1)

    wm_scale = 40
    wm_width = int(watermark.shape[1] * wm_scale/100)
    wm_height = int(watermark.shape[0] * wm_scale/100)
    wm_dim = (wm_width, wm_height)
    resized_wm = cv2.resize(watermark, wm_dim)
    # cv2.imshow("resized",resized_wm)
    overlay = np.zeros((480, 640, 3), dtype="uint8")

    if int(operation_value) == 2:

        while True:

            cTime = datetime.datetime.now()
            # 1. Import image
            success, img = cap.read()
            img = cv2.flip(img, 1)

            # 2. Find Hand Landmarks
            img = detector.findHands(img)
            lmList = detector.findPosition(img, draw=False)

            if len(lmList) != 0:
                # print(lmList)

                # tip of index and middle fingers
                x1, y1 = lmList[8][1:]
                x2, y2 = lmList[12][1:]
                x0, y0 = lmList[4][1:]
                # 3. Check which fingers are up
                fingers = detector.fingersUp()
                # print(fingers)

                # 4. If Selection Mode - Two finger are up
                if fingers[1] and fingers[2]:
                    xp, yp = 0, 0
                    # print("Selection Mode")
                    # # Checking for the click
                    if x1 > 590:
                        if y1 < 83:
                            header3, header4 = header4, header3
                            if right_header_change == False:
                                right_header_change = True
                            else:
                                right_header_change = False
                        elif 83 < y1 < 170:
                            if right_header_change == False:
                                header3 = overlayList3[0]
                                drawColor = (255, 0, 255)
                            else:
                                header3 = overlayList4[0]
                                drawColor = (10, 0, 10)
                        elif 170 < y1 < 270:
                            if right_header_change == False:
                                header3 = overlayList3[1]
                                drawColor = (0, 255, 0)
                            else:
                                header3 = overlayList4[1]
                                drawColor = (0, 0, 255)
                        elif 270 < y1 < 370:
                            if right_header_change == False:
                                header3 = overlayList3[2]
                                drawColor = (255, 0, 0)
                            else:
                                header3 = overlayList4[2]
                                drawColor = (250, 206, 136)
                        # elif 370 < y1 < 480:
                        #     print("more")
                        #     # overlayList3,overlayList4=overlayList4,overlayList3
                        #     header3,header4=header4,header3
                        #     #saving logic
                    if y1 < 83:
                        # cv2.putText(img,str(x1)+","+str(y1),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,3,(244,123,245))

                        if 295 < x1 < 375:
                            if (cTime-pTime).total_seconds() > 0.75:
                                print("inside for unfilled")
                                if drawFilled == True:
                                    header = overlayList[0]
                                    drawFilled = False
                                else:
                                    header = overlayList[1]
                                    drawFilled = True
                                pTime = cTime
                        elif 150 < x1 < 220:
                            drawColor = (0, 0, 0)
                            header = overlayList[2]
                        elif 220 < x1 < 295:
                            # print(i)
                            print((cTime-pTime).total_seconds())
                            if (cTime-pTime).total_seconds() > 1:
                                cv2.imwrite("F:/project/saved/" +
                                            str(i)+".jpg", imgInv)
                                i += 1
                                pTime = cTime

                    if x1 < 50:

                        if y1 < 96:
                            header2 = overlayList2[0]
                            shape = 'freestyle'
                        elif 96 < y1 < 192:
                            header2 = overlayList2[1]
                            shape = 'circle'
                        elif 192 < y1 < 288:
                            header2 = overlayList2[2]
                            shape = 'rectangle'
                        elif 288 < y1 < 384:
                            header2 = overlayList2[3]
                            shape = 'elipse'
                        elif 384 < y1 < 480:
                            header2 = overlayList2[4]
                            shape = 'triangle'
                    cv2.rectangle(img, (x1, y1 - 25),
                                  (x2, y2 + 25), drawColor, cv2.FILLED)
                if fingers[1] and fingers[2] == False:
                    cv2.circle(img, (x1, y1), 15, drawColor)
                    # print("Drawing Mode")
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1

                    cv2.line(img, (xp, yp), (x1, y1),
                             drawColor, brushThickness)

                    if drawColor == (0, 0, 0):
                        eraserThickness = 50
                        z1, z2 = lmList[4][1:]
                        # print(z1,z2)
                        result = int(
                            ((((z1 - x1) ** 2) + ((z2 - y1) ** 2)) ** 0.5))
                        # print(result)
                        if result < 0:
                            result = -1 * result
                        u = result
                        if fingers[1] and fingers[4]:
                            eraserThickness = u
                        if fingers[1] and fingers[4] and fingers[3]:
                            imgCanvas = np.zeros((480, 640, 3), np.uint8)

                        # print(eraserThickness)
                        #cv2.putText(img, str("eraserThickness="), (0, 700), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                        #cv2.putText(img, str(int(eraserThickness)), (450, 700), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 255), 3)
                        cv2.line(img, (xp, yp), (x1, y1),
                                 drawColor, eraserThickness)
                        cv2.line(imgCanvas, (xp, yp), (x1, y1),
                                 drawColor, eraserThickness)

                    else:
                        # brushThickness = 5
                        # z1, z2 = lmList[4][1:]
                        # print(z1,z2)
                        # result = int(((((z1 - x1) ** 2) + ((z2 - y1) ** 2)) ** 0.5))
                        # print(result)
                        # if result < 0:
                        #     result = -1 * result
                        # u = result
                        # brushThickness = int(u)
                        # print(eraserThickness)

                        # draw
                        if shape == 'freestyle':

                            cv2.line(imgCanvas, (xp, yp), (x1, y1),
                                     drawColor, brushThickness)
                            cv2.line(img, (xp, yp), (x1, y1),
                                     drawColor, brushThickness)
                            #cv2.putText(img, str(u), (600, 700), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                            #cv2.putText(img, str("brushThickness="), (0, 700), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                            #cv2.putText(img, str(int(brushThickness)), (450, 700), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255),3)

                        # Rectangle
                        # print(int(cTime-pTime))
                        if shape == 'rectangle' and (cTime-pTime).total_seconds() > 1:
                            # pTime=cTime
                            # cTime=time.time()
                            z1, z2 = lmList[4][1:]
                            # print(z1,z2)
                            result = int(
                                ((((z1 - x1) ** 2) + ((z2 - y1) ** 2)) ** 0.5))
                            # print(result)
                            if result < 0:
                                result = -1 * result
                            u = result
                            cv2.rectangle(img, (x0, y0), (x1, y1), drawColor)
                            # cv2.putText(img, "Length of Diagonal = ", (0, 700), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                            # cv2.putText(img, str(u), (530, 700), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                            if fingers[4]:
                                if drawFilled:
                                    cv2.rectangle(
                                        imgCanvas, (x0, y0), (x1, y1), drawColor, thickness=-1)
                                else:
                                    cv2.rectangle(
                                        imgCanvas, (x0, y0), (x1, y1), drawColor, thickness=brushThickness)
                                pTime = cTime
                                # cv2.circle

                        # Circle
                        if shape == 'circle' and (cTime-pTime).total_seconds() > 1:
                            z1, z2 = lmList[4][1:]
                            # print(z1,z2)
                            result = int(
                                ((((z1 - x1) ** 2) + ((z2 - y1) ** 2)) ** 0.5))
                            # print(result)
                            if result < 0:
                                result = -1 * result
                            u = result
                            # cv2.putText(img, "Radius Of Circe = ", (0, 700), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                            # cv2.putText(img, str(u), (450, 700), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                            cv2.circle(img, (x0, y0), u, drawColor)
                            if fingers[4]:
                                pTime = cTime
                                if drawFilled:
                                    cv2.circle(imgCanvas, (x0, y0),
                                               u, drawColor, thickness=-1)
                                else:
                                    cv2.circle(imgCanvas, (x0, y0), u,
                                               drawColor, thickness=brushThickness)

                        # Ellipse
                        if shape == 'elipse' and (cTime-pTime).total_seconds() > 1:
                            z1, z2 = lmList[4][1:]
                            # cv2.ellipse(img,(x1,y1),(int(z1/2),int(z2/2)),0,0,360,255,0)
                            a = z1 - x1
                            b = (z2 - x2)
                            if x1 > 250:
                                b = int(b / 2)
                            if a < 0:
                                a = -1 * a
                            if b < 0:
                                b = -1 * b
                            cv2.ellipse(img, (x1, y1), (a, b),
                                        0, 0, 360, drawColor, 1)
                            # cv2.putText(img, "Major AL, Minor AL = ", (0, 700), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                            # cv2.putText(img, str(a), (550, 700), cv2.FONT_HERSHEY_PLAIN, 3, (123, 20, 255), 3)
                            # cv2.putText(img, str(b), (700, 700), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                            if fingers[4]:
                                pTime = cTime
                                if drawFilled:
                                    cv2.ellipse(
                                        imgCanvas, (x1, y1), (a, b), 0, 0,  360,  drawColor, thickness=-1)
                                else:
                                    cv2.ellipse(
                                        imgCanvas, (x1, y1), (a, b), 0, 0, 360, drawColor, thickness=brushThickness)

                        if shape == 'triangle' and (cTime-pTime).total_seconds() > 1:
                            z1, z2 = lmList[4][1:]
                            # print(z1,z2)
                            result = int(
                                ((((z1 - x1) ** 2) + ((z2 - y1) ** 2)) ** 0.5))
                            # print(result)
                            if result < 0:
                                result = -1 * result
                            u = result
                            cv2.circle(img, (x0, y0), u, drawColor)
                            f1, f2 = lmList[8][1:]
                            # cv2.putText(img, "Major AL, Minor AL = ", (0, 700), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                            # cv2.putText(img, str(a), (550, 700), cv2.FONT_HERSHEY_PLAIN, 3, (123, 20, 255), 3)
                            # cv2.putText(img, str(b), (700, 700), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                            if fingers[4]:
                                pTime = cTime
                                if drawFilled:
                                    cv2.line(imgCanvas, (f1, f2), (z1-50, z2),
                                             color=drawColor, thickness=brushThickness)
                                    cv2.line(imgCanvas, (f1, f2), (z1+50, z2),
                                             color=drawColor, thickness=brushThickness)
                                    cv2.line(imgCanvas, (z1-50, z2), (z1+50, z2),
                                             color=drawColor, thickness=brushThickness)
                                else:
                                    cv2.line(imgCanvas, (f1, f2), (z1-50, z2),
                                             color=drawColor, thickness=brushThickness)
                                    cv2.line(imgCanvas, (f1, f2), (z1+50, z2),
                                             color=drawColor, thickness=brushThickness)
                                    cv2.line(imgCanvas, (z1-50, z2), (z1+50, z2),
                                             color=drawColor, thickness=brushThickness)

                    xp, yp = x1, y1

                # Clear Canvas when 2 fingers are up
                if fingers[2] and fingers[3] and fingers[0] == 0 and fingers[1] == 0 and fingers[4] == 0:
                    imgCanvas = np.zeros((720, 1280, 3), np.uint8)

            imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
            _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
            imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
            img = cv2.bitwise_and(img, imgInv)
            img = cv2.bitwise_or(img, imgCanvas)
            # pTime=cTime
            # Setting the header image
            img[0:83, 50:590] = header
            img[0:480, 0:50] = header2
            img[0:480, 590:640] = header3
            # img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)

            ret, buffer = cv2.imencode('.jpg', img)

            img = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')  # concat frame one by one and show result

            # cv2.imshow("Image", img)

            # cv2.imshow("Canvas", imgCanvas)
            # cv2.imshow("Inv", imgInv)
            cv2.waitKey(1)
    else:
        while True:
            cTime = datetime.datetime.now()
            # 1. Import image
            success, img = cap.read()
            resized_img = cv2.resize(
                img, (640, 480), interpolation=cv2.INTER_AREA)
            h_img, w_img, _ = resized_img.shape
            center_y = int(h_img/2)
            center_x = int(w_img/2)
            h_wm, w_wm, _ = resized_wm.shape
            top_y = center_y - int(h_wm/2)
            left_x = center_x - int(w_wm/2)
            bottom_y = top_y + h_wm
            right_x = left_x + w_wm
            roi = resized_img[top_y:bottom_y, left_x:right_x]

            result = cv2.addWeighted(roi, 1, resized_wm, 1.0, 0)
            overlay[top_y:bottom_y, left_x:right_x] = resized_wm
            resized_img[top_y:bottom_y, left_x:right_x] = result

            img = resized_img
            img = cv2.flip(img, 1)
            # 2. Find Hand Landmarks
            img = detector.findHands(img)
            lmList = detector.findPosition(img, draw=False)

            if len(lmList) != 0:
                # print(lmList)

                # tip of index and middle fingers
                x1, y1 = lmList[8][1:]
                x2, y2 = lmList[12][1:]
                x0, y0 = lmList[4][1:]
                # 3. Check which fingers are up
                fingers = detector.fingersUp()
                # print(fingers)

                # 4. If Selection Mode - Two finger are up
                if fingers[1] and fingers[2]:
                    xp, yp = 0, 0
                    # print("Selection Mode")
                    # # Checking for the click
                    if x1 > 590:
                        if y1 < 83:
                            header3, header4 = header4, header3
                            if right_header_change == False:
                                right_header_change = True
                            else:
                                right_header_change = False
                        elif 83 < y1 < 170:
                            if right_header_change == False:
                                header3 = overlayList3[0]
                                drawColor = (255, 0, 255)
                            else:
                                header3 = overlayList4[0]
                                drawColor = (10, 0, 10)
                        elif 170 < y1 < 270:
                            if right_header_change == False:
                                header3 = overlayList3[1]
                                drawColor = (0, 255, 0)
                            else:
                                header3 = overlayList4[1]
                                drawColor = (0, 0, 255)
                        elif 270 < y1 < 370:
                            if right_header_change == False:
                                header3 = overlayList3[2]
                                drawColor = (255, 0, 0)
                            else:
                                header3 = overlayList4[2]
                                drawColor = (250, 206, 136)
                        # elif 370 < y1 < 480:
                        #     print("more")
                        #     # overlayList3,overlayList4=overlayList4,overlayList3
                        #     header3,header4=header4,header3
                        #     #saving logic
                    if y1 < 83:
                        # cv2.putText(img,str(x1)+","+str(y1),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,3,(244,123,245))

                        if 295 < x1 < 375:
                            if (cTime-pTime).total_seconds() > 0.75:
                                print("inside for unfilled")
                                if drawFilled == True:
                                    header = overlayList[0]
                                    drawFilled = False
                                else:
                                    header = overlayList[1]
                                    drawFilled = True
                                pTime = cTime
                        elif 150 < x1 < 220:
                            drawColor = (0, 0, 0)
                            header = overlayList[2]
                        elif 220 < x1 < 295:
                            # print(i)
                            print((cTime-pTime).total_seconds())
                            if (cTime-pTime).total_seconds() > 1:
                                cv2.imwrite("F:/project/saved/" +
                                            str(i)+".jpg", imgInv)
                                i += 1
                                pTime = cTime

                    if x1 < 50:

                        if y1 < 96:
                            header2 = overlayList2[0]
                            shape = 'freestyle'
                        elif 96 < y1 < 192:
                            header2 = overlayList2[1]
                            shape = 'circle'
                        elif 192 < y1 < 288:
                            header2 = overlayList2[2]
                            shape = 'rectangle'
                        elif 288 < y1 < 384:
                            header2 = overlayList2[3]
                            shape = 'elipse'
                        elif 384 < y1 < 480:
                            header2 = overlayList2[4]
                            shape = 'triangle'
                    cv2.rectangle(img, (x1, y1 - 25),
                                  (x2, y2 + 25), drawColor, cv2.FILLED)
                if fingers[1] and fingers[2] == False:
                    cv2.circle(img, (x1, y1), 15, drawColor)
                    # print("Drawing Mode")
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1

                    cv2.line(img, (xp, yp), (x1, y1),
                             drawColor, brushThickness)

                    if drawColor == (0, 0, 0):
                        eraserThickness = 50
                        z1, z2 = lmList[4][1:]
                        # print(z1,z2)
                        result = int(
                            ((((z1 - x1) ** 2) + ((z2 - y1) ** 2)) ** 0.5))
                        # print(result)
                        if result < 0:
                            result = -1 * result
                        u = result
                        if fingers[1] and fingers[4]:
                            eraserThickness = u
                        if fingers[1] and fingers[4] and fingers[3]:
                            imgCanvas = np.zeros((480, 640, 3), np.uint8)

                        # print(eraserThickness)
                        #cv2.putText(img, str("eraserThickness="), (0, 700), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                        #cv2.putText(img, str(int(eraserThickness)), (450, 700), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 255), 3)
                        cv2.line(img, (xp, yp), (x1, y1),
                                 drawColor, eraserThickness)
                        cv2.line(imgCanvas, (xp, yp), (x1, y1),
                                 drawColor, eraserThickness)

                    else:
                        # brushThickness = 5
                        # z1, z2 = lmList[4][1:]
                        # print(z1,z2)
                        # result = int(((((z1 - x1) ** 2) + ((z2 - y1) ** 2)) ** 0.5))
                        # print(result)
                        # if result < 0:
                        #     result = -1 * result
                        # u = result
                        # brushThickness = int(u)
                        # print(eraserThickness)

                        # draw
                        if shape == 'freestyle':

                            cv2.line(imgCanvas, (xp, yp), (x1, y1),
                                     drawColor, brushThickness)
                            cv2.line(img, (xp, yp), (x1, y1),
                                     drawColor, brushThickness)
                            #cv2.putText(img, str(u), (600, 700), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                            #cv2.putText(img, str("brushThickness="), (0, 700), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                            #cv2.putText(img, str(int(brushThickness)), (450, 700), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255),3)

                        # Rectangle
                        # print(int(cTime-pTime))
                        if shape == 'rectangle' and (cTime-pTime).total_seconds() > 1:
                            # pTime=cTime
                            # cTime=time.time()
                            z1, z2 = lmList[4][1:]
                            # print(z1,z2)
                            result = int(
                                ((((z1 - x1) ** 2) + ((z2 - y1) ** 2)) ** 0.5))
                            # print(result)
                            if result < 0:
                                result = -1 * result
                            u = result
                            cv2.rectangle(img, (x0, y0), (x1, y1), drawColor)
                            # cv2.putText(img, "Length of Diagonal = ", (0, 700), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                            # cv2.putText(img, str(u), (530, 700), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                            if fingers[4]:
                                if drawFilled:
                                    cv2.rectangle(
                                        imgCanvas, (x0, y0), (x1, y1), drawColor, thickness=-1)
                                else:
                                    cv2.rectangle(
                                        imgCanvas, (x0, y0), (x1, y1), drawColor, thickness=brushThickness)
                                pTime = cTime
                                # cv2.circle

                        # Circle
                        if shape == 'circle' and (cTime-pTime).total_seconds() > 1:
                            z1, z2 = lmList[4][1:]
                            # print(z1,z2)
                            result = int(
                                ((((z1 - x1) ** 2) + ((z2 - y1) ** 2)) ** 0.5))
                            # print(result)
                            if result < 0:
                                result = -1 * result
                            u = result
                            # cv2.putText(img, "Radius Of Circe = ", (0, 700), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                            # cv2.putText(img, str(u), (450, 700), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                            cv2.circle(img, (x0, y0), u, drawColor)
                            if fingers[4]:
                                pTime = cTime
                                if drawFilled:
                                    cv2.circle(imgCanvas, (x0, y0),
                                               u, drawColor, thickness=-1)
                                else:
                                    cv2.circle(imgCanvas, (x0, y0), u,
                                               drawColor, thickness=brushThickness)

                        # Ellipse
                        if shape == 'elipse' and (cTime-pTime).total_seconds() > 1:
                            z1, z2 = lmList[4][1:]
                            # cv2.ellipse(img,(x1,y1),(int(z1/2),int(z2/2)),0,0,360,255,0)
                            a = z1 - x1
                            b = (z2 - x2)
                            if x1 > 250:
                                b = int(b / 2)
                            if a < 0:
                                a = -1 * a
                            if b < 0:
                                b = -1 * b
                            cv2.ellipse(img, (x1, y1), (a, b),
                                        0, 0, 360, drawColor, 1)
                            # cv2.putText(img, "Major AL, Minor AL = ", (0, 700), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                            # cv2.putText(img, str(a), (550, 700), cv2.FONT_HERSHEY_PLAIN, 3, (123, 20, 255), 3)
                            # cv2.putText(img, str(b), (700, 700), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                            if fingers[4]:
                                pTime = cTime
                                if drawFilled:
                                    cv2.ellipse(
                                        imgCanvas, (x1, y1), (a, b), 0, 0,  360,  drawColor, thickness=-1)
                                else:
                                    cv2.ellipse(
                                        imgCanvas, (x1, y1), (a, b), 0, 0, 360, drawColor, thickness=brushThickness)

                        if shape == 'triangle' and (cTime-pTime).total_seconds() > 1:
                            z1, z2 = lmList[4][1:]
                            # print(z1,z2)
                            result = int(
                                ((((z1 - x1) ** 2) + ((z2 - y1) ** 2)) ** 0.5))
                            # print(result)
                            if result < 0:
                                result = -1 * result
                            u = result
                            cv2.circle(img, (x0, y0), u, drawColor)
                            f1, f2 = lmList[8][1:]
                            # cv2.putText(img, "Major AL, Minor AL = ", (0, 700), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                            # cv2.putText(img, str(a), (550, 700), cv2.FONT_HERSHEY_PLAIN, 3, (123, 20, 255), 3)
                            # cv2.putText(img, str(b), (700, 700), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                            if fingers[4]:
                                pTime = cTime
                                if drawFilled:
                                    cv2.line(imgCanvas, (f1, f2), (z1-50, z2),
                                             color=drawColor, thickness=brushThickness)
                                    cv2.line(imgCanvas, (f1, f2), (z1+50, z2),
                                             color=drawColor, thickness=brushThickness)
                                    cv2.line(imgCanvas, (z1-50, z2), (z1+50, z2),
                                             color=drawColor, thickness=brushThickness)
                                else:
                                    cv2.line(imgCanvas, (f1, f2), (z1-50, z2),
                                             color=drawColor, thickness=brushThickness)
                                    cv2.line(imgCanvas, (f1, f2), (z1+50, z2),
                                             color=drawColor, thickness=brushThickness)
                                    cv2.line(imgCanvas, (z1-50, z2), (z1+50, z2),
                                             color=drawColor, thickness=brushThickness)

                    xp, yp = x1, y1

                # Clear Canvas when 2 fingers are up
                if fingers[2] and fingers[3] and fingers[0] == 0 and fingers[1] == 0 and fingers[4] == 0:
                    imgCanvas = np.zeros((720, 1280, 3), np.uint8)

            imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
            _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
            imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
            img = cv2.bitwise_and(img, imgInv)
            img = cv2.bitwise_or(img, imgCanvas)
            # pTime=cTime
            # Setting the header image
            img[0:83, 50:590] = header
            img[0:480, 0:50] = header2
            img[0:480, 590:640] = header3

            ret, buffer = cv2.imencode('.jpg', img)

            img = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')  # concat frame one by one and show result

            # img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)

            # cv2.imshow("Image", img)

            # cv2.imshow("Canvas", imgCanvas)
            # cv2.imshow("Inv", imgInv)
            cv2.waitKey(1)


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
