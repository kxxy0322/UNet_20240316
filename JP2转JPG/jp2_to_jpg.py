import os
import cv2

# 文件夹路径
input_folder = r"C:\Data_Sentinel2\Finish_unzip"
output_folder = r"C:\Data_Sentinel2\data_JPG_all"

# 遍历input_folder中的所有子文件夹并找到以TCI_10m.jp2结尾的文件
for root, dirs, files in os.walk(input_folder):
    for file_name in files:
        if file_name.endswith('TCI_10m.jp2'):
            jp2_path = os.path.join(root, file_name)
            jpg_name = file_name.replace('.jp2', '.jpg')
            jpg_path = os.path.join(output_folder, jpg_name)

            # 使用OpenCV读取.JP2文件，并保存为.JPG文件
            img = cv2.imread(jp2_path)
            cv2.imwrite(jpg_path, img)

print("转换完成！")