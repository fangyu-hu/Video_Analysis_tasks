import numpy as np
import cv2

cap = cv2.VideoCapture("mov.avi")

while(True):
    ret,frame = cap.read() #ret是一个布尔值，代表是否到了视频末尾
    if ret == False: #检查是否是最后帧
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#每一帧都转化为灰度
    cv2.imshow("frame",frame)
    cv2.imshow("gray",gray)
    if cv2.waitKey(25) & 0xFF == ord("q"):#linux需要用到后面的0xFF,windows不加也可
        break
cap.release()
cv2.destroyAllWindows()