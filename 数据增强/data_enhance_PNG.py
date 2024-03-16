import cv2
import os
import numpy as np
import random

# 设置原始图片路径和增强后的图片保存路径
input_path = "C:/Data_Sentinel2/F/dataset_4/SegmentationClass_before"
output_path = "C:/Data_Sentinel2/F/dataset_4/SegmentationClass"
base_img_folder = "C:/Data_Sentinel2/F/data_base_img"

# 如果输出路径不存在，则创建输出路径
if not os.path.exists(output_path):
    os.makedirs(output_path)


# 定义图像增强函数
def augment_image(img, operations, params, output_filename):
    for operation, param in zip(operations, params):
        img = operation(img, *param)
    # 保存增强后的图像
    output_filepath = os.path.join(output_path, output_filename)
    cv2.imwrite(output_filepath, img)


# 翻转图像
def flip(img):
    return cv2.flip(img, 1)


# 旋转图像
def rotate(img, angle):
    rows, cols = img.shape[:2]
    center = (cols // 2, rows // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    return cv2.warpAffine(img, M, (cols, rows))


# 平移图像
def translate(img, tx, ty):
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


# 缩放图像
def resize(img, scale_percent):
    new_width = int(img.shape[1] * scale_percent)
    new_height = int(img.shape[0] * scale_percent)
    resized_img = cv2.resize(img, (new_width, new_height))
    top = (256 - new_height) // 2
    bottom = 256 - new_height - top
    left = (256 - new_width) // 2
    right = 256 - new_width - left
    return cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])


# 添加噪声
def add_noise(img, noise_level):
    noise = np.random.normal(0, noise_level, img.shape).astype(np.uint8)
    return cv2.add(img, noise)


# 对比度增强
def enhance_contrast(img, alpha):
    return cv2.convertScaleAbs(img, alpha=alpha)


# 填充黑色
def fill_black(img):
    # 从base_img_folder文件夹中随机选择一张图片作为base_img
    base_img_filename = random.choice(os.listdir(base_img_folder))
    base_img_path = os.path.join(base_img_folder, base_img_filename)
    base_img = cv2.imread(base_img_path)

    # 创建一个掩码，将旋转后的图像中的黑色区域标记出来
    mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask[mask > 0] = 1
    # 对掩码进行腐蚀操作，减少白色像素值区域的宽度
    kernel = np.ones((3, 3), np.uint8)  # 定义腐蚀核大小
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    # 将旋转后的图像中的黑色区域填充为基准图像对应区域的像素值
    result = base_img.copy()
    for c in range(base_img.shape[2]):
        result[:, :, c] = result[:, :, c] * (1 - eroded_mask) + img[:, :, c] * eroded_mask

    return result


# 组合方式1: 翻转 + 对比度增强
operations_1 = [flip,]
params_1 = [[]]

# 组合方式2: 旋转90度 + 噪声扰动 + 对比度增强
operations_2 = [rotate]
params_2 = [[90]]

# 组合方式3: 旋转180 + 翻转 + 缩放95 + 平移10 + 填充
operations_3 = [rotate, flip, resize, translate]
params_3 = [[180], [], [0.95], [10, 10]]

# 组合方式4: 旋转210度 + 平移64 + 填充+噪声扰动
operations_4 = [rotate, translate]
params_4 = [[210], [-64, -64]]

# 组合方式5: 翻转 + 旋转30度 + 填充 + 对比度增强
operations_5 = [flip, rotate]
params_5 = [[], [30]]

# 组合方式6：旋转270度 + 缩放90 + 填充 + 噪声0.4
operations_6 = [rotate, resize]
params_6 = [[270], [0.9]]

# # 组合方式7：试验
# operations_7 = [rotate, translate]
# params_7 = [[60], [0,32]]

# 遍历原始图片文件夹中的所有图片文件，并进行增强处理
for filename in os.listdir(input_path):
    if filename.endswith(".png"):
        img = cv2.imread(os.path.join(input_path, filename))

        # 组合方式1
        output_filename = os.path.splitext(filename)[0] + "enhance_1.png"
        augment_image(img, operations_1, params_1, output_filename)

        # 组合方式2
        output_filename = os.path.splitext(filename)[0] + "enhance_2.png"
        augment_image(img, operations_2, params_2, output_filename)

        # 组合方式3
        output_filename = os.path.splitext(filename)[0] + "enhance_3.png"
        augment_image(img, operations_3, params_3, output_filename)

        # 组合方式4
        output_filename = os.path.splitext(filename)[0] + "enhance_4.png"
        augment_image(img, operations_4, params_4, output_filename)

        # 组合方式5
        output_filename = os.path.splitext(filename)[0] + "enhance_5.png"
        augment_image(img, operations_5, params_5, output_filename)

        # 组合方式6
        output_filename = os.path.splitext(filename)[0] + "enhance_6.png"
        augment_image(img, operations_6, params_6, output_filename)

        # # 组合方式7
        # output_filename = os.path.splitext(filename)[0] + "enhance_7.png"
        # augment_image(img, operations_7, params_7, output_filename)

print("数据增强处理完成！")
