import numpy as np
import cv2

# 初始化摄像头
cap = cv2.VideoCapture("mov.avi")

# 获取视频的第一帧
ret, old_frame = cap.read()
if not ret:
    print("Cannot read video file")
    exit()

# 转换为灰度图
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# 设置Lucas-Kanade光流法的参数
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 检测特征点
p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# 创建一个空白的图像来绘制轨迹
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 计算光流
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # 选择好的点
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # 如果没有检测到特征点，则跳过绘制
    if len(good_new) == 0 or len(good_old) == 0:
        continue

        # 绘制轨迹
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()  # 这里其实不需要ravel()，因为new已经是元组(x, y)
        c, d = old.ravel()  # 同样，old也是元组(x, y)
        # 但为了代码的清晰和避免潜在的错误，我们可以直接使用new和old
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

    img = cv2.add(frame, mask)

    # 更新前一帧和特征点
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    cv2.imshow('frame', img)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()