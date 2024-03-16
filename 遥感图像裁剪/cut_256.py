import cv2

# 图片路径
image_path = r"C:\Data_Sentinel2\data_JPG_all\T51SVA_20211204T024101_TCI_10m.jpg"

# 读取图片
image = cv2.imread(image_path)

# 获取图片尺寸
height, width, _ = image.shape

# 设置每张图片的大小
crop_size = 256

# 计算裁剪后的行列数
rows = height // crop_size
cols = width // crop_size

# 裁剪图片并保存
count = 1
for i in range(rows):
    for j in range(cols):
        start_row = i * crop_size
        end_row = (i + 1) * crop_size
        start_col = j * crop_size
        end_col = (j + 1) * crop_size

        cropped_img = image[start_row:end_row, start_col:end_col]
        cv2.imwrite(r"C:\Data_Sentinel2\F\cut_256_3\{}.jpg".format(count), cropped_img)
        count += 1

print("裁剪完成！")
