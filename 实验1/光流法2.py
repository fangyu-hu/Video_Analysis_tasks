import numpy as np
import cv2

# 初始化摄像头
cap = cv2.VideoCapture("mov.avi")

# 读取第一帧
ret, old_frame = cap.read()
if not ret:
    print("Cannot read video file")
    exit()

# 转换为灰度图
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Lucas-Kanade光流法参数
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 检测第一帧的特征点
p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# 创建一个用于显示结果的空图像
frame_with_contours = old_frame.copy()

# 初始化一个用于跟踪的mask图像（虽然这里不直接使用，但可以用来绘制轨迹）
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 计算光流
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # 选择有效点
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # 绘制点
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel().astype(int)  # 使用 astype 方法将数组转换为整数类型
        frame_with_contours = cv2.circle(frame_with_contours, (a, b), 5, (0, 255, 0), -1)

        # 使用当前帧的特征点更新为下一帧的p0
    if len(good_new) >= 4:  # 确保有足够的点来检测轮廓
        # 将当前帧的特征点坐标转换为整数并作为轮廓的输入
        # 假设 good_new 是一个形状为 (n, 2) 的浮点数数组
        good_new_int = good_new.astype(int)  # 将浮点数转换为整数
        points = good_new_int.reshape((-1, 1, 2))  # 调整形状为 (n, 1, 2)

        # 现在你可以使用 points 数组了
        # 例如，如果你正在使用 OpenCV 的某些函数，它们可能需要这种形状的点数组
        # 寻找轮廓（这里假设所有点构成了一个封闭的区域，实际上可能需要更复杂的处理）
        # 注意：这里只是一个简化的示例，可能不总是准确
        hull = cv2.convexHull(points)
        # 绘制轮廓和填充内部
        cv2.drawContours(frame_with_contours, [hull], -1, (0, 0, 255), 3)
        cv2.fillPoly(frame_with_contours, [hull], (0, 255, 0))

        # 更新前一帧和特征点
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    # 显示结果
    cv2.imshow('Frame with contours', frame_with_contours)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()