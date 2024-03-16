import os

folder_path = r'C:\Data_Sentinel2\F\dataset_4\JPEGImages'
# folder_path = r'C:\Data_Sentinel2\F\dataset_4\SegmentationClass'

# 获取文件夹中所有文件的名称
file_list = os.listdir(folder_path)

# 遍历文件列表，对文件进行重命名
for file_name in file_list:
    old_file_path = os.path.join(folder_path, file_name)
    new_file_name = '1_' + file_name
    new_file_path = os.path.join(folder_path, new_file_name)

    # 重命名文件
    os.rename(old_file_path, new_file_path)
    print(f'Renamed: {file_name} to {new_file_name}')