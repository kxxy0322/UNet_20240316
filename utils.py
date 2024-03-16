# utils 工具类

# 图片等比缩放
from PIL import Image


# 定义一个函数

def keep_image_size_open(path, size=(256, 256)):
    # 读进来图片
    img = Image.open(path)
    # 获取最长边
    temp = max(img.size)
    # 根据最长边做一个mask掩码
    mask = Image.new('RGB', (temp, temp), (0, 0, 0))  # rgb类型，黑色
    mask.paste(img, (0, 0))  # 把原图粘到左上角
    mask = mask.resize(size)  # resize大小是size
    return mask
