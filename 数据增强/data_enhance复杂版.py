import cv2
import os
import numpy as np


# 设置原始图片路径和增强后的图片保存路径
input_path = "C:/Data_Sentinel2/F/dataset_4/JPEGImages"
output_path = "C:/Data_Sentinel2/F/dataset_4/JPEGImages_enhance"

# 如果输出路径不存在，则创建输出路径
if not os.path.exists(output_path):
    os.makedirs(output_path)

#翻转
########################################################################################
# 遍历原始图片文件夹中的所有图片文件
for filename in os.listdir(input_path):
    if filename.endswith(".jpg"):
        # 读取图片
        img = cv2.imread(os.path.join(input_path, filename))

        # 数据增强操作，这里以水平翻转为例
        flipped_img = cv2.flip(img, 1)

        # 构建增强后的文件名
        output_filename = os.path.splitext(filename)[0] + "_flip.jpg"

        # 保存增强后的图片到输出路径
        output_filepath = os.path.join(output_path, output_filename)
        cv2.imwrite(output_filepath, flipped_img)

print("数据增强--翻转--处理完成！")


#旋转-30度
########################################################################################
# 定义旋转角度
angle = 30

# 遍历原始图片文件夹中的所有图片文件
for filename in os.listdir(input_path):
    if filename.endswith(".jpg"):
        # 读取图片
        img = cv2.imread(os.path.join(input_path, filename))

        # 获取图片高度和宽度
        rows, cols = img.shape[:2]

        # 计算旋转中心
        center = (cols // 2, rows // 2)

        # 生成旋转矩阵
        M = cv2.getRotationMatrix2D(center, angle, 1)
        # 进行旋转操作
        rotated_img = cv2.warpAffine(img, M, (cols, rows))
        # 构建增强后的文件名
        output_filename = os.path.splitext(filename)[0] + "_rotate30.jpg"
        # 保存增强后的图片到输出路径
        output_filepath = os.path.join(output_path, output_filename)
        cv2.imwrite(output_filepath, rotated_img)

print("数据增强--旋转30--处理完成！")

#旋转210度
########################################################################################
# 定义旋转角度
angle = 210

# 遍历原始图片文件夹中的所有图片文件
for filename in os.listdir(input_path):
    if filename.endswith(".jpg"):
        # 读取图片
        img = cv2.imread(os.path.join(input_path, filename))

        # 获取图片高度和宽度
        rows, cols = img.shape[:2]

        # 计算旋转中心
        center = (cols // 2, rows // 2)

        # 生成旋转矩阵
        M = cv2.getRotationMatrix2D(center, angle, 1)
        # 进行旋转操作
        rotated_img = cv2.warpAffine(img, M, (cols, rows))
        # 构建增强后的文件名
        output_filename = os.path.splitext(filename)[0] + "_rotate210.jpg"
        # 保存增强后的图片到输出路径
        output_filepath = os.path.join(output_path, output_filename)
        cv2.imwrite(output_filepath, rotated_img)

print("数据增强--旋转210--处理完成！")


#平移128
########################################################################################
# 定义平移矩阵
tx = 128  # 水平平移像素数
ty = 0  # 垂直平移像素数
M = np.float32([[1, 0, tx], [0, 1, ty]])

# 遍历原始图片文件夹中的所有图片文件
for filename in os.listdir(input_path):
    if filename.endswith(".jpg"):
        # 读取图片
        img = cv2.imread(os.path.join(input_path, filename))

        # 获取图片高度和宽度
        rows, cols = img.shape[:2]

        # 进行平移操作
        translated_img = cv2.warpAffine(img, M, (cols, rows))

        # 构建增强后的文件名
        output_filename = os.path.splitext(filename)[0] + "_translate128.jpg"

        # 保存增强后的图片到输出路径
        output_filepath = os.path.join(output_path, output_filename)
        cv2.imwrite(output_filepath, translated_img)

print("数据增强--平移128--处理完成！")


#平移-128
########################################################################################
# 定义平移矩阵
tx = -128  # 水平平移像素数
ty = 0  # 垂直平移像素数
M = np.float32([[1, 0, tx], [0, 1, ty]])

# 遍历原始图片文件夹中的所有图片文件
for filename in os.listdir(input_path):
    if filename.endswith(".jpg"):
        # 读取图片
        img = cv2.imread(os.path.join(input_path, filename))

        # 获取图片高度和宽度
        rows, cols = img.shape[:2]

        # 进行平移操作
        translated_img = cv2.warpAffine(img, M, (cols, rows))

        # 构建增强后的文件名
        output_filename = os.path.splitext(filename)[0] + "_translate-128.jpg"

        # 保存增强后的图片到输出路径
        output_filepath = os.path.join(output_path, output_filename)
        cv2.imwrite(output_filepath, translated_img)

print("数据增强--平移-128--处理完成！")

#缩放50
########################################################################################
# 定义缩放比例
scale_percent = 0.5  # 缩小50%

# 遍历原始图片文件夹中的所有图片文件
for filename in os.listdir(input_path):
    if filename.endswith(".jpg"):
        # 读取图片
        img = cv2.imread(os.path.join(input_path, filename))

        # 获取原始图像的尺寸
        height, width = img.shape[:2]

        # 计算缩放后的尺寸
        new_width = int(width * scale_percent)
        new_height = int(height * scale_percent)

        # 计算边界填充量
        top = (256 - new_height) // 2
        bottom = 256 - new_height - top
        left = (256 - new_width) // 2
        right = 256 - new_width - left

        # 使用copyMakeBorder函数进行边界填充
        resized_img = cv2.resize(img, (new_width, new_height))
        resized_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # 构建增强后的文件名
        output_filename = os.path.splitext(filename)[0] + "_resize50.jpg"

        # 保存增强后的图片到输出路径
        output_filepath = os.path.join(output_path, output_filename)
        cv2.imwrite(output_filepath, resized_img)

print("数据增强--缩放50--处理完成！")


#缩放75
########################################################################################
# 定义缩放比例
scale_percent = 0.75  # 缩小50%

# 遍历原始图片文件夹中的所有图片文件
for filename in os.listdir(input_path):
    if filename.endswith(".jpg"):
        # 读取图片
        img = cv2.imread(os.path.join(input_path, filename))

        # 获取原始图像的尺寸
        height, width = img.shape[:2]

        # 计算缩放后的尺寸
        new_width = int(width * scale_percent)
        new_height = int(height * scale_percent)

        # 计算边界填充量
        top = (256 - new_height) // 2
        bottom = 256 - new_height - top
        left = (256 - new_width) // 2
        right = 256 - new_width - left

        # 使用copyMakeBorder函数进行边界填充
        resized_img = cv2.resize(img, (new_width, new_height))
        resized_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # 构建增强后的文件名
        output_filename = os.path.splitext(filename)[0] + "_resize75.jpg"

        # 保存增强后的图片到输出路径
        output_filepath = os.path.join(output_path, output_filename)
        cv2.imwrite(output_filepath, resized_img)

print("数据增强--缩放75--处理完成！")




#噪声扰动
########################################################################################

# 定义噪声水平（可根据需要调整）
noise_level = 0.6

# 遍历原始图片文件夹中的所有图片文件
for filename in os.listdir(input_path):
    if filename.endswith(".jpg"):
        # 读取图片
        img = cv2.imread(os.path.join(input_path, filename))

        # 生成与原始图像尺寸相同的高斯噪声
        noise = np.random.normal(0, noise_level, img.shape).astype(np.uint8)

        # 将噪声添加到原始图像
        noisy_img = cv2.add(img, noise)

        # 构建增强后的文件名
        output_filename = os.path.splitext(filename)[0] + "_noise.jpg"

        # 保存增强后的图片到输出路径
        output_filepath = os.path.join(output_path, output_filename)
        cv2.imwrite(output_filepath, noisy_img)

print("数据增强--噪声扰动--处理完成！")

#对比度
########################################################################################

# 定义对比度增强参数
alpha = 1.5  # 对比度增强系数

# 遍历原始图片文件夹中的所有图片文件
for filename in os.listdir(input_path):
    if filename.endswith(".jpg"):
        # 读取图片
        img = cv2.imread(os.path.join(input_path, filename))

        # 对比度增强
        enhanced_img = cv2.convertScaleAbs(img, alpha=alpha)

        # 构建增强后的文件名
        output_filename = os.path.splitext(filename)[0] + "_contrast.jpg"

        # 保存增强后的图片到输出路径
        output_filepath = os.path.join(output_path, output_filename)
        cv2.imwrite(output_filepath, enhanced_img)

print("数据增强--对比度--处理完成！")