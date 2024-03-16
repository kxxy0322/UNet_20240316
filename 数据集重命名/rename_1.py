import os

folder_path = r'C:\Data_Sentinel2\F\dataset_4\JPEGImages'
# folder_path = r'C:\Data_Sentinel2\F\dataset_4\SegmentationClass'
start_index = 1  # 起始序号

files = os.listdir(folder_path)

for index, file in enumerate(files):
    file_path = os.path.join(folder_path, file)
    new_file_name = os.path.join(folder_path, f"{start_index+index}{os.path.splitext(file)[1]}")
    os.rename(file_path, new_file_name)
