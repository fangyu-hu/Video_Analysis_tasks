import numpy as np  # 导入numpy库，用于进行数学运算
import cv2  # 导入OpenCV库，用于图像和视频处理

# 创建一个VideoCapture对象，用于从视频文件'mov.avi'中读取视频
cap = cv2.VideoCapture('mov.avi')

# 设置ShiTomasi角点检测的参数
feature_params = dict(
    maxCorners=100,  # 最大角点数
    qualityLevel=0.3,  # 角点的质量水平参数，值越大，检测到的角点越少
    minDistance=7,  # 角点之间的最小欧氏距离
    blockSize=10  # 用于计算角点检测的邻域大小
)

# 设置Lucas-Kanade光流法的参数
lk_params = dict(
    winSize=(15, 15),  # 搜索窗口的大小
    maxLevel=4,  # 图像金字塔的最大层数，0表示只处理原图
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)  # 迭代搜索算法的终止条件
)

# 创建一些随机颜色，用于绘制轨迹
color = np.random.randint(0, 255, (100, 3))

# 读取视频的第一帧，并转换为灰度图像
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# 使用ShiTomasi角点检测算法在第一帧中找到角点
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# 创建一个与原始帧大小相同的全零掩码图像，用于绘制轨迹
mask = np.zeros_like(old_frame)

# 循环读取视频帧，直到视频结束
while (1):
    ret, frame = cap.read()  # 读取下一帧
    if ret == False:  # 检查是否到达视频末尾
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将当前帧转换为灰度图像

    # 使用Lucas-Kanade光流法计算光流，获取角点的新位置
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # 选择状态为1（即跟踪成功的）的好点
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # 绘制轨迹
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()  # 获取新点的坐标
        c, d = old.ravel()  # 获取旧点的坐标
        a, b, c, d = int(a), int(b), int(c), int(d)  # 转换为整数坐标
        # 在掩码图像上绘制从旧点到新点的线段
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        # 在当前帧上绘制新点
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

        # 将当前帧和掩码图像相加，得到带有轨迹的图像
    img = cv2.add(frame, mask)

    # 显示带有轨迹的图像
    cv2.imshow('frame', img)

    # 按下ESC键（ASCII码为27）时退出循环
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

        # 更新上一帧和上一帧的角点位置
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)  # 重新整理角点的形状以匹配输入要求

# 释放所有窗口并关闭视频捕获对象
cv2.destroyAllWindows()
cap.release()