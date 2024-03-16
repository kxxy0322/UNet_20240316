# 数据集制作
import os.path
from torch.utils.data import Dataset
from utils import *
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])


class MyDataset(Dataset):  # 定义一个类
    def __init__(self, path):  # 初始化
        self.path = path  # 数据集地址
        # 获取标签所有名字
        self.name = os.listdir(
            os.path.join(path, 'SegmentationClass'))  # 拼接一下path+SegmentationClass，获取下面所有文件名os.listdir

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        # 数据的制作
        segment_name = self.name[index]  # 获取对应下标的数据的名字   名字样式：xxx.png  原图为xxx.jpg 需要转化
        # 标签制作
        segment_path = os.path.join(self.path, 'SegmentationClass', segment_name)  # 拼接地址 原始路径+SegmentationClass
        image_path = os.path.join(self.path, 'JPEGImages', segment_name.replace('png', 'jpg'))  # 原图地址  需要把png替换为jpg
        # 读取图片，网络输入需要固定大小图片，不一样需要缩放（等比缩放）
        # 思想：取每张图的最长边，做一个mask矩形，将原图贴到矩形上面，变成正方形后进行等比的resize就不会变形了
        # 读取图片
        segment_image = keep_image_size_open(segment_path)
        image = keep_image_size_open(image_path)  # 大小默认
        # 图片归一化
        return transform(image), transform(segment_image)


# # 验证以下正确与否
# if __name__ == '__main__':
#     # 定义地址
#     data = MyDataset('C:\Data_Sentinel2\F\cut_256')
#     print(data[0][0].shape)  # 第0张image原图的形状，如果是想象的形状就是正确的
#     print(data[0][1].shape)  # 正常形状应该是3,256,256
