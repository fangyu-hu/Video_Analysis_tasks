import cv2
import numpy as np

# 初始化背景减除器
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=True)

cap = cv2.VideoCapture("mov.avi")

while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

        # 将图像从BGR转换到灰度，但通常背景减除在BGR上进行
    # 这里直接使用BGR图像
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 应用背景减除
    fgmask = fgbg.apply(frame)

    # 二值化前景掩码
    _, fgmask = cv2.threshold(fgmask, 244, 255, cv2.THRESH_BINARY)

    # 形态学操作以消除噪声
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, (5, 5))
    fgmask = cv2.dilate(fgmask, (3, 3), iterations=2)

    # 查找轮廓
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制轮廓
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # 忽略小区域噪声
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), -1)  # 填充内部

    # 显示结果
    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fgmask)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()