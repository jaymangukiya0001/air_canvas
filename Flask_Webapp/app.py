from flask import Flask, render_template, Response
import cv2
import os
import numpy as np
import time
import HandTrackingModule as htm
import datetime
app = Flask(__name__)


def gen_frames():  # generate frame by frame from camera
    camera = cv2.VideoCapture(0)
    brushThickness = 5
    eraserThickness = 150

    folderPath = "Header"
    # folderPath2="Header2"
    myList = os.listdir(folderPath)
    myList.sort()
    print(myList)
    overlayList = []
    filled = []
    drawFilled = False
    for imPath in myList:
        image = cv2.imread(f'{folderPath}/{imPath}')
        image = cv2.resize(image, (640, 83), interpolation=cv2.INTER_AREA)
        overlayList.append(image)
    # myList2=os.listdir(folderPath2)
    # overlayList2=[]
    # for imPath in myList2:
    #     image = cv2.imread(f'{folderPath2}/{imPath}')
    #     image=cv2.resize(image,(40,397),interpolation=cv2.INTER_AREA)
    #     overlayList2.append(image)

    # print(len(overlayList))
    # print(overlayList)

    header = overlayList[0]
    # header2=overlayList2[0]
    drawColor = (255, 0, 255)
    # shape = 'freestyle'
    # cap = cv2.VideoCapture(0)
    # cap.set(3, 640)
    # cap.set(4, 480)

    detector = htm.handDetector(detectionCon=0.85, maxHands=1)
    xp, yp = 0, 0

    imgCanvas = np.zeros((480, 640, 3), np.uint8)
    i = 0
    cTime = pTime = 0
    pTime = datetime.datetime.now()
    shape = ""
    while True:
        cTime = datetime.datetime.now()
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        frame = cv2.flip(frame, 1)
        if not success:
            break
        else:
            frame[0:83, 0:640] = header

            frame = detector.findHands(frame)
            lmList = detector.findPosition(frame, draw=False)

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
                    if y1 < 83:
                        if 125 < x1 < 225:
                            header = overlayList[0]
                            drawColor = (255, 0, 255)
                        elif 275 < x1 < 275:
                            header = overlayList[1]
                            drawColor = (255, 0, 0)
                        elif 400 < x1 < 475:
                            header = overlayList[10]
                            drawColor = (0, 255, 0)
                        elif 525 < x1 < 640:
                            header = overlayList[5]
                            drawColor = (0, 0, 0)
                    if x1 < 40:
                        # cv2.putText(img,str(x1)+","+str(y1),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,3,(244,123,245))

                        if y1 > 220:
                            if (cTime-pTime).total_seconds() > 0.75:
                                if drawFilled:
                                    header2 = overlayList[1]
                                    drawFilled = False
                                else:
                                    header2 = overlayList[0]
                                    drawFilled = True
                                pTime = cTime
                        else:
                            # print(i)
                            print((cTime-pTime).total_seconds())
                            if (cTime-pTime).total_seconds() > 1:
                                cv2.imwrite("F:/project/saved/" +
                                            str(i)+".jpg", imgCanvas)
                                i += 1
                                pTime = cTime

                    if y1 > 80 and y1 < 140:
                        if x1 < 125:
                            header = overlayList[9]

                        elif 125 < x1 < 225 and drawColor == (255, 0, 255):
                            header = overlayList[0]
                            shape = 'freestyle'
                        elif 275 < x1 < 375 and drawColor == (255, 0, 255):
                            header = overlayList[6]
                            shape = 'circle'
                        elif 400 < x1 < 475 and drawColor == (255, 0, 255):
                            header = overlayList[7]
                            shape = 'rectangle'
                        elif 525 < x1 < 600 and drawColor == (255, 0, 255):
                            header = overlayList[8]
                            shape = 'elipse'
                        elif 125 < x1 < 225 and drawColor == (255, 0, 0):
                            header = overlayList[10]
                            shape = 'freestyle'
                        elif 275 < x1 < 375 and drawColor == (255, 0, 0):
                            header = overlayList[11]
                            shape = 'circle'
                        elif 400 < x1 < 475 and drawColor == (255, 0, 0):
                            header = overlayList[12]
                            shape = 'rectangle'
                        elif 525 < x1 < 600 and drawColor == (255, 0, 0):
                            header = overlayList[13]
                            shape = 'elipse'
                        if 125 < x1 < 225 and drawColor == (0, 255, 0):
                            header = overlayList[1]
                            shape = 'freestyle'
                        elif 225 < x1 < 375 and drawColor == (0, 255, 0):
                            header = overlayList[2]
                            shape = 'circle'
                        elif 400 < x1 < 475 and drawColor == (0, 255, 0):
                            header = overlayList[3]
                            shape = 'rectangle'
                        elif 525 < x1 < 600 and drawColor == (0, 255, 0):
                            header = overlayList[4]
                            shape = 'elipse'
                    cv2.rectangle(frame, (x1, y1 - 25), (x2, y2 + 25),
                                  drawColor, cv2.FILLED)
                if fingers[1] and fingers[2] == False:
                    cv2.circle(frame, (x1, y1), 15, drawColor)
                    # print("Drawing Mode")
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1

                    cv2.line(frame, (xp, yp), (x1, y1),
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
                        cv2.line(frame, (xp, yp), (x1, y1),
                                 drawColor, eraserThickness)
                        cv2.line(imgCanvas, (xp, yp), (x1, y1),
                                 drawColor, eraserThickness)

                    else:
                        brushThickness = 5
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
                            cv2.line(frame, (xp, yp), (x1, y1),
                                     drawColor, brushThickness)
                            #cv2.putText(img, str(u), (600, 700), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                            #cv2.putText(img, str("brushThickness="), (0, 700), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                            #cv2.putText(img, str(int(brushThickness)), (450, 700), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255),3)

                        # Rectangle
                        # print(int(cTime-pTime))
                        if shape == 'rectangle':
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
                            cv2.rectangle(frame, (x0, y0), (x1, y1), drawColor)
                            # cv2.putText(img, "Length of Diagonal = ", (0, 700), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                            # cv2.putText(img, str(u), (530, 700), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                            if fingers[4]:
                                if drawFilled:
                                    cv2.rectangle(
                                        imgCanvas, (x0, y0), (x1, y1), drawColor, thickness=-1)
                                else:
                                    cv2.rectangle(
                                        imgCanvas, (x0, y0), (x1, y1), drawColor, thickness=1)
                                pTime = cTime
                                # cv2.circle

                        # Circle
                        if shape == 'circle':
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
                            cv2.circle(frame, (x0, y0), u, drawColor)
                            if fingers[4]:
                                pTime = cTime
                                if drawFilled:
                                    cv2.circle(imgCanvas, (x0, y0),
                                               u, drawColor, thickness=-1)
                                else:
                                    cv2.circle(imgCanvas, (x0, y0),
                                               u, drawColor, thickness=1)

                        # Ellipse
                        if shape == 'elipse':
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
                            cv2.ellipse(frame, (x1, y1), (a, b),
                                        0, 0, 360, 255, 0)
                            # cv2.putText(img, "Major AL, Minor AL = ", (0, 700), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                            # cv2.putText(img, str(a), (550, 700), cv2.FONT_HERSHEY_PLAIN, 3, (123, 20, 255), 3)
                            # cv2.putText(img, str(b), (700, 700), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                            if fingers[4]:
                                pTime = cTime
                                if drawFilled:
                                    cv2.ellipse(
                                        imgCanvas, (x1, y1), (a, b), 0, 0, 360, 255, 0, thickness=-1)
                                else:
                                    cv2.ellipse(
                                        imgCanvas, (x1, y1), (a, b), 0, 0, 360, 255, 0, thickness=1)

                    xp, yp = x1, y1

            # # Clear Canvas when 2 fingers are up
            #     if fingers[2] and fingers[3] and fingers[0] == 0 and fingers[1] == 0 and fingers[4] == 0:
            #         imgCanvas = np.zeros((720, 1280, 3), np.uint8)

            imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
            _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
            imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
            frame = cv2.bitwise_and(frame, imgInv)
            frame = cv2.bitwise_or(frame, imgCanvas)
            # pTime=cTime
            # Setting the header image
            frame[0:83, 0:640] = header
            # frame[83:480, 0:40] = header2

            ret, buffer = cv2.imencode('.jpg', frame)

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


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
