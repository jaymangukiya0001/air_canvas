import cv2 
import numpy as np
watermark=cv2.imread("watermark/w5.png")
print(watermark)


wm_scale = 40
wm_width = int(watermark.shape[1] * wm_scale/100)
wm_height = int(watermark.shape[0] * wm_scale/100)
wm_dim = (wm_width, wm_height)
resized_wm = cv2.resize(watermark, wm_dim)
# cv2.imshow("resized",resized_wm)
overlay=np.zeros((480,640,3),dtype="uint8")
# cv2.imshow("ov",overlay)
# tmp=cv2.bitwise_and(resized_wm,overlay)
# cv2.imshow("And bitwise",tmp)

cap=cv2.VideoCapture(0)
while True:
    res,frame=cap.read() 
    resized_img = cv2.resize(frame, (640,480), interpolation=cv2.INTER_AREA)
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
    overlay[top_y:bottom_y, left_x:right_x]=resized_wm
    resized_img[top_y:bottom_y, left_x:right_x] = result
    # cv2.imshow("overlay",roi)
    cv2.imshow("img",resized_img)
    cv2.imshow("overlay",overlay)
    cv2.waitKey(1)
# cap.release()
# cv2.destroyAllWindows()
