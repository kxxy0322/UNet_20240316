# 测试

import os
import cv2
import numpy as np
import torch

from net import *
from utils import *
from data import *
from torchvision.utils import save_image
from PIL import Image

net = UNet().cuda()  # 实例化网络

weights = 'params/unet_raft6.pth'  # 权重地址
if os.path.exists(weights):  # 如果路径中存在权重
    net.load_state_dict(torch.load(weights))  # 则导入权重
    print('successfully')
else:
    print('no loading')

# _input = input('please input JPEGImages path:')  # 输入图片路径
test_path = r'C:\Users\22769\PycharmProjects\UNet_my_20240302\手敲UNet网络\test_data\1.jpg'  # 图片路径
# r的作用是防止\被视为转义字符
# img=keep_image_size_open_rgb(_input)
img = keep_image_size_open(test_path)  # 输出大小为256*256的img
img_data = transform(img).cuda()  # transform操作
print(img_data.shape)
img_data = torch.unsqueeze(img_data, dim=0)  # 升维
out = net(img_data)
save_image(out, 'result/result.jpg')
