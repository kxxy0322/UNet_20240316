import cv2
import os
import numpy as np
import random

# 设置原始图片路径和增强后的图片保存路径
input_path = "C:/Data_Sentinel2/F/dataset_4/JPEGImages_before"
output_path = "C:/Data_Sentinel2/F/dataset_4/JPEGImages_enhance"
base_img_folder = "C:/Data_Sentinel2/F/data_base_img"

# 如果输出路径不存在，则创建输出路径
if not os.path.exists(output_path):
    os.makedirs(output_path)



#旋转-30度
########################################################################################
# 定义旋转角度
angle = 30

# 读取原始图片文件夹中的所有图片文件
for filename in os.listdir(input_path):
    if filename.endswith(".jpg"):
        # 读取待处理的图片
        img = cv2.imread(os.path.join(input_path, filename))

        # 从base_img_folder文件夹中随机选择一张图片作为base_img
        base_img_filename = random.choice(os.listdir(base_img_folder))
        base_img_path = os.path.join(base_img_folder, base_img_filename)
        base_img = cv2.imread(base_img_path)
        # 确保两张图片具有相同的尺寸
        img = cv2.resize(img, (base_img.shape[1], base_img.shape[0]))
        # 获取图片高度和宽度
        rows, cols = img.shape[:2]
        # 计算旋转中心
        center = (cols // 2, rows // 2)
        # 生成旋转矩阵
        M = cv2.getRotationMatrix2D(center, angle, 1)
        # 进行旋转操作
        rotated_img = cv2.warpAffine(img, M, (cols, rows))
########   图片叠加操作    ##################################################################

        # 创建一个掩码，将旋转后的图像中的黑色区域标记出来
        mask = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)
        mask[mask > 0] = 1
        # 对掩码进行腐蚀操作，减少白色像素值区域的宽度
        kernel = np.ones((3, 3), np.uint8)  # 定义腐蚀核大小
        eroded_mask = cv2.erode(mask, kernel, iterations=1)
        # 将旋转后的图像中的黑色区域填充为基准图像对应区域的像素值
        result = base_img.copy()
        for c in range(base_img.shape[2]):
            result[:, :, c] = result[:, :, c] * (1 - eroded_mask) + rotated_img[:, :, c] * eroded_mask

########   图片叠加操作    ##################################################################

        # # 将旋转后的图像中的黑色区域填充为基准图像对应区域的像素值
        # result = base_img.copy()
        # for c in range(base_img.shape[2]):
        #     result[:,:,c] = result[:,:,c] * (1 - mask) + rotated_img[:,:,c] * mask

        # 构建增强后的文件名
        output_filename = os.path.splitext(filename)[0] + "_rotate30.jpg"

        # 保存增强后的图片到输出路径
        output_filepath = os.path.join(output_path, output_filename)
        cv2.imwrite(output_filepath, result)

print("数据增强--旋转30--处理完成！")