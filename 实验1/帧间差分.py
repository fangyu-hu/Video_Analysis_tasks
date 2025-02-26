import cv2

Fram_last = None  # 定义最后一帧
difImg = None
num = 0  # 计数用变量
cap = cv2.VideoCapture("mov.avi")  # 读入视频
while (cap.isOpened()):
    t = cv2.getTickCount()
    ret, framRGB = cap.read()
    if ret == False:
        break
    fram = cv2.cvtColor(framRGB, cv2.COLOR_BGR2GRAY)  # 图像颜色由彩色转灰度

    if Fram_last is None:
        Fram_last = fram
        continue
    framedelta = cv2.absdiff(Fram_last, fram)  # 两帧之差
    difImg = cv2.threshold(framedelta, 40, 255, cv2.THRESH_BINARY)[1]  # 二值化
    Fram_last = fram.copy()
    num = num + 1
    t = (cv2.getTickCount() - t)
    time = t / cv2.getTickFrequency()  # 计算每两帧所用时间
    print("第", num, "帧所用的时间为", time, "秒")

    cv2.imshow("diff", difImg)
    #     difImg=cv2.morphologyEx(difImg, cv2.MORPH_OPEN, (3,3))
    difImg = cv2.morphologyEx(difImg, cv2.MORPH_OPEN, (5, 5))
    difImg = cv2.dilate(difImg, (5, 5))  # 形态学处理减少噪点
    cv2.imshow("dilated", difImg)
    contours, hierarchy = cv2.findContours(difImg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  # 找出轮廓特征
    img = cv2.drawContours(framRGB, contours, -1, (0, 255, 0), -1)  # 画出轮廓并将轮廓内填充
    cv2.imshow("contours", img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()