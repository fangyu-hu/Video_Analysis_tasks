import cv2  # 导入OpenCV库，用于图像和视频处理
import numpy as np  # 导入numpy库，用于数学运算

# 创建一个VideoCapture对象，用于从视频文件"mov.avi"中读取视频
cap = cv2.VideoCapture("mov.avi")

# 读取视频的第一帧
ret, frame1 = cap.read()
# 将第一帧转换为灰度图像，用于后续的光流计算
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# 创建一个与原始帧大小相同的HSV图像，用于将光流信息编码为颜色
hsv = np.zeros_like(frame1)
# 初始化HSV图像的饱和度通道为全白，因为我们将使用色调和亮度通道来表示光流的方向和大小
hsv[..., 1] = 255

# 循环读取视频帧，直到视频结束
while (1):
    # 读取下一帧
    ret, frame2 = cap.read()
    # 检查是否到达视频末尾
    if ret == False:
        break
        # 将当前帧转换为灰度图像
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 使用Farneback方法计算从上一帧到当前帧的稠密光流
    # 参数说明：prvs为前一帧的灰度图，next为当前帧的灰度图，
    # None表示不使用金字塔（即不进行图像尺度变换），
    # 0.5为金字塔层之间的缩放比例（这里未使用），
    # 3为金字塔的层数（这里未使用），
    # 15为计算光流时考虑的像素邻域大小，
    # 3为多项式展开的阶数，
    # 5为用于计算每个像素点光流的迭代次数，
    # 1.2为金字塔层之间的图像平滑程度，
    # 0为计算光流时使用的窗口大小标志（0表示使用整个图像）
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # 将光流的x和y分量转换为极坐标形式，即大小和角度
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # 将光流的角度转换为HSV图像的色调通道值，范围从0到180度
    hsv[..., 0] = ang * 180 / np.pi / 2

    # 将光流的大小（即速度）归一化到0-255范围，并赋值给HSV图像的亮度通道
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # 将HSV图像转换为BGR图像，以便在OpenCV窗口中显示
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 显示带有光流信息的图像
    cv2.imshow('frame2', rgb)

    # 按下ESC键（ASCII码为27）时退出循环
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

        # 注意：这里没有更新prvs为next，这意味着光流是相对于视频的第一帧计算的，


# 释放视频捕获对象
cap.release()
# 关闭所有OpenCV窗口
cv2.destroyAllWindows()