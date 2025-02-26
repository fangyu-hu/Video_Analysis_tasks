import cv2
import mediapipe as mp

# 初始化MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, enable_segmentation=False, min_detection_confidence=0.2)  # 调整置信度

# 打开视频文件
cap = cv2.VideoCapture('单人动作素材.avi')

# 设置窗口名称和初始大小（宽度, 高度）
window_name = 'Pose Estimation'
window_width = 800
window_height = 600

# 创建窗口（注意：在某些平台上，后续设置大小可能不起作用）
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 将图像从BGR转换为RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 处理图像并获取姿态估计结果
    results = pose.process(image_rgb)

    # 检查是否检测到人体姿态关键点
    if results.pose_landmarks:
        # 无论 results.pose_landmarks 是列表还是单个对象，都尝试绘制
        try:
            # 尝试作为列表迭代
            for pose_landmarks in results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, pose_landmarks, mp_pose.POSE_CONNECTIONS)
        except TypeError:
            # 如果不是列表，则直接绘制
            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # 显示处理后的图像
    cv2.imshow(window_name, frame)

    # 按'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()