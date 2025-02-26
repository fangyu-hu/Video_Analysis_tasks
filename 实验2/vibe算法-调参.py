# vibe算法

import numpy as np
import cv2
import random
import time

video = cv2.VideoCapture('1.avi')  # 待处理视频的路径
height = int(video.get(4))
width = int(video.get(3))

# ====================================================================================================
# 超参数设置
# ====================================================================================================

N = 20  # 每个像素的采样数
R = np.ones((height, width, N), dtype=np.uint8) * 20  # 球的半径
Number_min = 2  # 作为背景的一部分的邻近样本的最小数量
Rand_Samples = 20  # 随机采样量

frame_number = 50  # 从哪一帧开始处理
FramesToSkip = 1  # 跳过的帧数（1为正常处理每一帧）
Pad = 5  # 边界框的填充大小

# ====================================================================================================
# 预定义随机向量，以加快计算速度
# ====================================================================================================
Random_Vector_N = [random.randint(0, N - 1) for _ in range(100000)]  # 0到N-1的随机值
Random_Index_N = 0  # 上面的随机向量的索引

Random_Vector_N2 = [random.randint(0, N - 1) for _ in range(100000)]
Random_Index_N2 = 0

Random_Vector_Phai = [random.randint(0, Rand_Samples - 1) for _ in range(100000)]
Random_Index_Phai = 0

Random_Vector_Phai2 = [random.randint(0, Rand_Samples - 1) for _ in range(100000)]
Random_Index_Phai2 = 0

I_vector = np.zeros(100000, dtype=np.int8)  # 用于寻找像素邻近像素的向量
J_vector = np.zeros(100000, dtype=np.int8)

index_neighbour = 0

for index_neighbour in range(0, 100000):
    i, j = 0, 0
    while i == 0 and j == 0:
        i = random.randint(-1, 1)
        j = random.randint(-1, 1)
    I_vector[index_neighbour] = i
    J_vector[index_neighbour] = j
    index_neighbour = index_neighbour + 1

# ====================================================================================================
# 初始化重要矩阵
# ====================================================================================================

print("Height of image: ", height)
print("Width of image: ", width)

Segmentation = np.zeros((height, width), dtype=np.uint8)  # 存储最终图像分割模型
Background_Model = np.zeros((height, width, N), dtype=np.uint8)  # 存储背景模型
frame_3D = np.zeros((height, width, N), dtype=np.uint8)  # 用于后续计算的矩阵
compare_matrix = np.zeros((height, width, N), dtype=np.uint8)  # 用于比较的矩阵


# ====================================================================================================
#  定义一个函数，用于在数字上添加噪声
# ====================================================================================================


def number_plus_noise(number):
    number = number + random.randint(-10, 10)
    if number > 255:
        number = 255
    if number < 0:
        number = 0
    return np.uint8(number)


# ====================================================================================================
#  使用第一帧（带噪声）初始化背景模型
# ====================================================================================================


ret, coloured_frame = video.read()
frame = cv2.cvtColor(coloured_frame, cv2.COLOR_BGR2GRAY)

for x in range(0, height):
    for y in range(0, width):
        for n in range(0, N):
            Background_Model[x, y, n] = number_plus_noise(frame[x, y])
# ====================================================================================================
#  主体部分
# ====================================================================================================

while ret:
    start = time.time()
    video.set(1, frame_number)
    ret, coloured_frame = video.read()
    frame_number = frame_number + FramesToSkip
    frame = cv2.cvtColor(coloured_frame, cv2.COLOR_BGR2GRAY)

    for n in range(0, N):
        frame_3D[:, :, n] = frame[:, :]

    compare_matrix = np.less(np.abs(Background_Model - frame_3D), R)

    for x in range(0, height):
        for y in range(0, width):
            data = 0
            for n in range(0, N):
                if compare_matrix[x, y, n]:
                    data = data + 1
                    if data >= Number_min:
                        Segmentation[x, y] = 0
                        break
                if n == N - 1:
                    Segmentation[x, y] = 255

    for x in range(0, height):
        for y in range(0, width):
            if Segmentation[x, y] == 0:
                rand = Random_Vector_Phai[Random_Index_Phai]
                Random_Index_Phai = Random_Index_Phai + 1
                if Random_Index_Phai == 100000:
                    Random_Index_Phai = 0

                if rand == 0:
                    rand = Random_Vector_N[Random_Index_N]
                    Random_Index_N = Random_Index_N + 1
                    if Random_Index_N == 100000:
                        Random_Index_N = 0
                    Background_Model[x, y, rand] = frame[x, y]

                rand = Random_Vector_Phai2[Random_Index_Phai2]
                Random_Index_Phai2 = Random_Index_Phai2 + 1
                if Random_Index_Phai2 == 100000:
                    Random_Index_Phai2 = 0

                if rand == 0:
                    rand = Random_Vector_N2[Random_Index_N2]
                    Random_Index_N2 = Random_Index_N2 + 1
                    if Random_Index_N2 == 100000:
                        Random_Index_N2 = 0
                    try:
                        Background_Model[x + I_vector[index_neighbour], y + J_vector[index_neighbour], rand] = frame[
                            x, y]
                        index_neighbour = index_neighbour + 1
                        if index_neighbour == 100000:
                            index_neighbour = 0
                    except:
                        pass
    Segmentation = cv2.medianBlur(Segmentation, 9)

    # ====================================================================================================
    #  Bounding Box
    # ====================================================================================================

    contours, hierarchy = cv2.findContours(Segmentation, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # cv2.RETR_LIST表示检索所有轮廓，不建立任何层次结构；cv2.CHAIN_APPROX_NONE表示保存轮廓的每一个点
    # 遍历找到的每一个轮廓
    for c in contours:
        # 使用boundingRect函数计算轮廓的边界矩形，返回矩形左上角的x,y坐标和矩形的宽度、高度
        i, j, h, w = cv2.boundingRect(c)
        # 如果矩形的高度和宽度都大于10，则绘制边界框
        if h > 10 and w > 10:
            # 在原图上绘制绿色（0, 255, 0）的边界框，边框宽度为2
            # 为了使边界框更明显地包围物体，这里对边界框进行了扩展（Pad），但代码中未显示Pad的定义
            cv2.rectangle(coloured_frame, (i - Pad, j - Pad), (i + h + Pad, j + w + Pad), (0, 255, 0), 2)
        else:
            # 如果边界框太小，则遍历该区域内的所有像素，并将其在分割图像（Segmentation）中置为0（背景）
            for x in range(i, i + h):
                for y in range(j, j + w):
                    try:
                        Segmentation[x, y] = 0  # 转换像素为背景
                    except:
                        pass

    # ====================================================================================================
    #  显示带有边界框的彩色帧和分割后的图像
    # ====================================================================================================

    cv2.imshow('Actual Frame!', coloured_frame)
    cv2.imshow('Foreground is white, ', Segmentation)

    # 打印当前帧的编号
    print("Frame number: ", frame_number)
    # 计算并打印处理当前帧所需的时间
    end = time.time()
    print("Time for processing this frame: ", (end - start))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()