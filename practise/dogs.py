import cv2 as cv
import time
img = cv.imread('cats.jpg')
cv.imshow('Cats', img)

cv.waitKey(0)

# Reading Videos
capture = cv.VideoCapture('dog.mp4')
cTime=0
pTime=0
while True:
    # isTrue, frame = capture.read()
    isTrue,frame=cv.VideoCapture(0)



    # if cv.waitKey(20) & 0xFF==ord('d'):
    # This is the preferred way - if `isTrue` is false (the frame could 
    # not be read, or we're at the end of the video), we immediately
    # break from the loop. 
    # pTime=time.time()
    # frame=int(120/(pTime-cTime))
    # cTime=pTime
    if isTrue:    
        # cv.putText(frame,"frames : "+str(frame),(100,100),cv.FONT_HERSHEY_COMPLEX,1,(255,0,200),2)
        cv.imshow('Video', frame)
        if cv.waitKey(20) & 0xFF==ord('d'):
            break            
    else:
        break

capture.release()
cv.destroyAllWindows()