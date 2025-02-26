# 导入必要的库
import cv2
import numpy as np

# 第一步：使用cv2.VideoCapture读取视频文件
camera = cv2.VideoCapture("1.avi")  # 打开名为"1.avi"的视频文件

# 第二步：使用cv2.getStructuringElement构造形态学操作所需的核（kernel）
# 这里使用了一个3x3的椭圆形结构元素，用于后续的形态学开运算
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# 第三步：构造高斯混合模型（Gaussian Mixture Model, GMM）用于背景减除
# cv2.createBackgroundSubtractorMOG2()函数用于创建背景减除器
model = cv2.createBackgroundSubtractorMOG2()

while (True):
    # 第四步：从视频中读取一帧图像，并检查是否成功读取
    ret, frame = camera.read()  # ret为布尔值，表示是否成功读取；frame为读取到的图像帧
    if ret == False:
        break  # 如果读取失败（例如视频结束），则退出循环

    # 运用高斯混合模型进行背景减除，生成前景掩模（fgmk）
    # 在此掩模中，前景对象被标记为白色（接近255），背景被标记为黑色（接近0）
    fgmk = model.apply(frame)

    # 第五步（实际在第四步之后）：显示原始的前景掩模，以便观察减除效果
    cv2.imshow("noise", fgmk)

    # 使用形态学的开运算去除前景掩模中的噪点
    # 开运算 = 先腐蚀后膨胀，有助于去除小的白色噪点
    fgmk = cv2.morphologyEx(fgmk, cv2.MORPH_OPEN, kernel)

    # 第六步：在去除噪点后的前景掩模上查找轮廓
    contours, hierarchy = cv2.findContours(fgmk.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 遍历找到的轮廓
    for c in contours:
        # 忽略面积小于1500的轮廓，以减少误检
        if cv2.contourArea(c) < 1500:
            continue
            # 计算并绘制轮廓的边界框
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 第七步（实际在第六步之后）：展示处理后的视频帧和前景掩模
    cv2.imshow('fgmk', fgmk)  # 显示处理后的前景掩模
    cv2.imshow('frame', frame)  # 显示带有边界框的视频帧

    # 监听键盘事件，如果按下'Esc'键（ASCII码为27），则退出循环
    if cv2.waitKey(50) & 0xff == 27:
        break

    # 释放视频捕获对象
camera.release()
# 销毁所有OpenCV创建的窗口
cv2.destroyAllWindows()