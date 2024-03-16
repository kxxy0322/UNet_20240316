import cv2
import os
import numpy as np

# 设置原始图片路径和增强后的图片保存路径
input_path = "C:/Data_Sentinel2/F/dataset_4/JPEGImages"
output_path = "C:/Data_Sentinel2/F/dataset_4/JPEGImages_enhance"

# 如果输出路径不存在，则创建输出路径
if not os.path.exists(output_path):
    os.makedirs(output_path)


# 定义图像增强函数
def augment_image(img, operation, params, output_filename):
    # 应用指定的操作和参数增强图像
    augmented_img = operation(img, *params)
    # 保存增强后的图像
    output_filepath = os.path.join(output_path, output_filename)
    cv2.imwrite(output_filepath, augmented_img)


def flip(img):
    return cv2.flip(img, 1)


def rotate(img, angle):
    rows, cols = img.shape[:2]
    center = (cols // 2, rows // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    return cv2.warpAffine(img, M, (cols, rows))


def translate(img, tx, ty):
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


def resize(img, scale_percent):
    new_width = int(img.shape[1] * scale_percent)
    new_height = int(img.shape[0] * scale_percent)
    resized_img = cv2.resize(img, (new_width, new_height))
    top = (256 - new_height) // 2
    bottom = 256 - new_height - top
    left = (256 - new_width) // 2
    right = 256 - new_width - left
    return cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])


def add_noise(img, noise_level):
    noise = np.random.normal(0, noise_level, img.shape).astype(np.uint8)
    return cv2.add(img, noise)


def enhance_contrast(img, alpha):
    return cv2.convertScaleAbs(img, alpha=alpha)


# 遍历原始图片文件夹中的所有图片文件
for filename in os.listdir(input_path):
    if filename.endswith(".jpg"):
        img = cv2.imread(os.path.join(input_path, filename))

        # 翻转
        output_filename = os.path.splitext(filename)[0] + "_flip.jpg"
        augment_image(img, flip, [], output_filename)

        # 旋转30度
        output_filename = os.path.splitext(filename)[0] + "_rotate30.jpg"
        augment_image(img, rotate, [30], output_filename)

        # 旋转210度
        output_filename = os.path.splitext(filename)[0] + "_rotate210.jpg"
        augment_image(img, rotate, [210], output_filename)

        # 平移128
        output_filename = os.path.splitext(filename)[0] + "_translate128.jpg"
        augment_image(img, translate, [128, 0], output_filename)

        # 平移-128
        output_filename = os.path.splitext(filename)[0] + "_translate-128.jpg"
        augment_image(img, translate, [-128, 0], output_filename)

        # 缩放50
        output_filename = os.path.splitext(filename)[0] + "_resize50.jpg"
        augment_image(img, resize, [0.5], output_filename)

        # 缩放75
        output_filename = os.path.splitext(filename)[0] + "_resize75.jpg"
        augment_image(img, resize, [0.75], output_filename)

        # 噪声扰动
        output_filename = os.path.splitext(filename)[0] + "_noise.jpg"
        augment_image(img, add_noise, [0.6], output_filename)

        # 对比度增强
        output_filename = os.path.splitext(filename)[0] + "_contrast.jpg"
        augment_image(img, enhance_contrast, [0.8], output_filename)

print("数据增强处理完成！")
