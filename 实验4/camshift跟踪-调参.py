# 导入所需的库
import cv2
import numpy as np

# 初始化变量，用于存储选中区域的坐标和大小
xs, ys, ws, hs = 0, 0, 0, 0  # 选中区域的左上角x,y坐标和宽度、高度
xo, yo = 0, 0  # 起始点（鼠标按下时的点）的x,y坐标
selectObject = False  # 标记是否正在选择对象
trackObject = 0  # 标记是否开始跟踪对象


# 鼠标回调函数，用于处理鼠标事件
def onMouse(event, x, y, flags, prams):
    global xs, ys, ws, hs, selectObject, xo, yo, trackObject
    # 如果正在选择对象，则更新选中区域的坐标和大小
    if selectObject == True:
        xs = min(x, xo)
        ys = min(y, yo)
        ws = abs(x - xo)
        hs = abs(y - yo)
        # 如果鼠标左键按下，记录起始点并开始选择对象
    if event == cv2.EVENT_LBUTTONDOWN:
        xo, yo = x, y
        xs, ys, ws, hs = x, y, 0, 0
        selectObject = True
        # 如果鼠标左键释放，结束选择对象并准备开始跟踪
    elif event == cv2.EVENT_LBUTTONUP:
        selectObject = False
        trackObject = -1

    # 打开视频文件


cap = cv2.VideoCapture("shiyan4-3.avi")
# 读取视频的第一帧
ret, frame = cap.read()
# 创建一个窗口用于显示视频
cv2.namedWindow('imshow')
# 设置鼠标回调函数
cv2.setMouseCallback('imshow', onMouse)
# 设置CamShift算法的终止条件
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.1)

# 主循环，持续读取视频帧并处理
while (True):
    ret, frame = cap.read()  # 读取视频帧
    if ret == False:  # 如果视频读取完毕，则退出循环
        break
        # 如果已经开始跟踪对象
    if trackObject != 0:
        # 将视频帧从BGR颜色空间转换为HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 创建一个掩码，只保留特定颜色范围内的像素
        mask = cv2.inRange(hsv, np.array((0., 30., 10.)), np.array((180., 256., 255.)))
        cv2.imshow("mask", mask)  # 显示掩码
        # 如果是刚开始跟踪（trackObject == -1）
        if trackObject == -1:
            track_window = (xs, ys, ws, hs)  # 设置跟踪窗口为选中区域
            maskroi = mask[ys:ys + hs, xs:xs + ws]  # 提取选中区域的掩码
            hsv_roi = hsv[ys:ys + hs, xs:xs + ws]  # 提取选中区域的HSV图像
            # 计算选中区域的HSV直方图
            roi_hist = cv2.calcHist([hsv_roi], [0], maskroi, [180], [0, 180])
            # 归一化直方图
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
            trackObject = 1  # 标记为已经开始跟踪
        # 计算反向投影
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        # 将反向投影与掩码结合，只保留感兴趣区域
        dst &= mask
        # 使用CamShift算法更新跟踪窗口
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        # 获取跟踪窗口的四个顶点
        pts = cv2.boxPoints(ret)
        pts = np.int32(pts)  # 将顶点坐标转换为整数
        # 在视频帧上绘制跟踪窗口
        img2 = cv2.polylines(frame, [pts], True, 255, 2)

        # 如果正在选择对象且选中区域有效（宽度和高度大于0）
    if selectObject == True and ws > 0 and hs > 0:
        # 对选中区域进行反色处理（可选，用于视觉反馈）
        cv2.bitwise_not(frame[ys:ys + hs, xs:xs + ws], frame[ys:ys + hs, xs:xs + ws])
        # 显示视频帧
    cv2.imshow('imshow', frame)
    # 如果按下ESC键，则退出循环
    if cv2.waitKey(50) == 27:
        break
    # 释放视频捕获对象
cap.release()
# 销毁所有窗口
cv2.destroyAllWindows()