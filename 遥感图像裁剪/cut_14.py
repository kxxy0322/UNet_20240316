import cv2

# 图片路径
image_path = r"C:\Data_Sentinel2\F\all_3.jpg"

# 读取图片
image = cv2.imread(image_path)

# 获取图片尺寸
height, width, _ = image.shape

# 计算裁剪后的尺寸
crop_height = height // 2
crop_width = width // 2

# 裁剪图片并保存
top_left = image[:crop_height, :crop_width]
cv2.imwrite(r"C:\Data_Sentinel2\F\top_left1.jpg", top_left)

bottom_left = image[crop_height:, :crop_width]
cv2.imwrite(r"C:\Data_Sentinel2\F\bottom_left1.jpg", bottom_left)

top_right = image[:crop_height, crop_width:]
cv2.imwrite(r"C:\Data_Sentinel2\F\top_right1.jpg", top_right)

bottom_right = image[crop_height:, crop_width:]
cv2.imwrite(r"C:\Data_Sentinel2\F\bottom_right1.jpg", bottom_right)

print("裁剪完成！")
