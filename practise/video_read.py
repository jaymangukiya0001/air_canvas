import numpy as np
import cv2
import time
 
cap = cv2.VideoCapture(0)
prev_frame_time = 0
new_frame_time = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    gray = frame
    gray = cv2.resize(gray, (500, 300))

    font = cv2.FONT_HERSHEY_SIMPLEX
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)
    cv2.putText(gray, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()