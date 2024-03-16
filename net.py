import torch
from torch import nn
from torch.nn import functional as F  # 插值法


class Conv_Block(nn.Module):  # 第一个板块，卷积板块
    def __init__(self, in_channel, out_channel):  # 输入通道会变，不能定义死（64,128,256...）
        super(Conv_Block, self).__init__()  # 初始化，处理方法
        self.layer = nn.Sequential(  # 序列构造
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            # 卷积，3*3卷积，步长为1，填充padding为1，填充类型为反射加强特征提取能力，偏移量为false，因为要使用batchnorm
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),  # 激活
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),  # 第二个卷积
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )

    def forward(self, x):  # 前项计算
        return self.layer(x)


class DownSample(nn.Module):  # 下采样（最大池化丢特征丢的太大，采用3*3卷积步长为2进行下采样）
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(  # 下采样
            nn.Conv2d(channel, channel, 3, 2, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channel),  # 可用可不用
            nn.LeakyReLU()
        )

    def forward(self, x):  # 前项
        return self.layer(x)


class UpSample(nn.Module):  # 上采样：转置卷积，插值法。这里采用插值法，因为转置卷积会产生空洞。
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(channel, channel // 2, 1, 1)  # 1*1卷积，步长为1，降通道（1024变512）

    def forward(self, x, feature_map):  # feature_map之前的一个特征图，与上采样特征图拼接，实现原理中灰色箭头部分
        up = F.interpolate(x, scale_factor=2, mode='nearest')  # 插值法，（输入，变为原来的2倍，类型：插值法）
        out = self.layer(up)
        return torch.cat((out, feature_map), dim=1)  # 上采样之后还进行了一个concat(拼接)，相当于上采样得到一个特征图，再和之前的特征图拼接到一起，然后再进行卷积卷积
        # 为什么维度是1，因为我们的结构是NCHW，0123，C就对应1


class UNet(nn.Module):  # 定义网络
    # def __init__(self, num_classes):
    def __init__(self):
        super(UNet, self).__init__()
        self.c1 = Conv_Block(3, 64)  # 首先一进来，输入3通道，输出64通道。BLOCK（块）
        self.d1 = DownSample(64)  # 下采样
        self.c2 = Conv_Block(64, 128)
        self.d2 = DownSample(128)
        self.c3 = Conv_Block(128, 256)
        self.d3 = DownSample(256)
        self.c4 = Conv_Block(256, 512)
        self.d4 = DownSample(512)
        self.c5 = Conv_Block(512, 1024)
        self.u1 = UpSample(1024)  # 上采样，降通道
        self.c6 = Conv_Block(1024, 512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512, 256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256, 128)
        self.u4 = UpSample(128)
        self.c9 = Conv_Block(128, 64)
        # self.out = nn.Conv2d(64, num_classes, 3, 1, 1) #最后一个卷积进行输出，因为我们要输出彩色图片，所以我们要输出一个三通道
        self.out = nn.Conv2d(64, 3, 1, 1)  # 最后一个卷积进行输出，因为我们要输出彩色图片，所以我们要输出一个三通道
        self.Th = nn.Sigmoid()  # 激活，使用Sigmoid进行二分类

    def forward(self, x):  # 前项
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))  # 下采样
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))
        O1 = self.c6(self.u1(R5, R4))  # 上采样+拼接
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))

        # return self.out(O4)
        # 最后进行一个Sigmoid
        return self.Th(self.out(O4))


# if __name__ == '__main__':  # 测一下是否正确
#     x = torch.randn(2, 3, 256, 256)  # 给一个随机的两批次3通道256*256的，如果输出相同则没有问题
#     # num_classes = 2  # 假设有两个类别
#     # net = UNet(num_classes)
#     net = UNet()
#     print(net(x).shape)
