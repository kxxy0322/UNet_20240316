import os

import tqdm
from torch import nn, optim
import torch
from torch.utils.data import DataLoader  # 数据加载器
from data import *
from net import *
from torchvision.utils import save_image  # 图片保存的包

#########################################################################################
# 需要注意的参数：1、权重地址pth 2、数据集地址dataset 3、批大小batch_size 4、训练轮次epoch 5、损失值
#########################################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 先把设备定义好，能用则用，不能用使用CPU
weight_path = 'params/unet_raft6.pth'  # 权重地址
data_path = r'C:\Data_Sentinel2\F\dataset_4'  # 数据集地址
save_path = 'train_image'
if __name__ == '__main__':  # 加载数据集
    # num_classes = 2 + 1  # +1是背景也为一类
    data_loader = DataLoader(MyDataset(data_path), batch_size=4, shuffle=True)  # 数据加载器、实例化数据集、batch_size看电脑性能
    # net = UNet(num_classes).to(device)
    net = UNet().to(device)  # 实例化网络
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight！')
    else:
        print('not successful load weight')

    opt = optim.Adam(net.parameters())  # 优化器  20240216:adam优化器可以自动调整学习率
    # loss_fun = nn.CrossEntropyLoss()
    loss_fun = nn.BCELoss()  # 损失函数

    epoch = 1  # 轮次
    while epoch < 21:
        # for i, (image, segment_image) in enumerate(tqdm.tqdm(data_loader)):  # 元组，为了得到下标i
        print('peoch:', epoch)
        for i, (image, segment_image) in enumerate(data_loader):  # 元组，为了得到下标i
            image, segment_image = image.to(device), segment_image.to(device)  # 数据放到设备上面
            out_image = net(image)  # 原图放进网络得到输出图
            train_loss = loss_fun(out_image, segment_image)  # 损失
            opt.zero_grad()  # 清空梯度
            train_loss.backward()  # 方向计算
            opt.step()  # 更新梯度

            # if i % 1 == 0:
            #     print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')

            # 将损失值转换为字符串并写入文件
            # with open('loss_value/loss_value_1000.txt', 'a') as f:  # 使用追加模式打开文件
            #     # f.write(f'{epoch}-{i}: {train_loss.item()}\n')  # 写入epoch, iteration和损失值
            #     f.write(f'{train_loss.item()}\n')  # 写入epoch, iteration和损失值
            if i % 10 == 0:  # 每隔五次打印一下损失值
                print(f'epoch:{epoch}--i:{i}--train_loss===>>{train_loss.item()}')  # 轮次--i--train_loss
                with open('loss_value/loss_value_6.txt', 'a') as f:  # 使用追加模式打开文件
                    # f.write(f'{epoch}-{i}: {train_loss.item()}\n')  # 写入epoch, iteration和损失值
                    f.write(f'{train_loss.item()}\n')  # 写入epoch, iteration和损失值
            # 保存权重
            if i % 50 == 0:  # 每隔50次保存权重
                torch.save(net.state_dict(), weight_path)  # 保存位置weight_path
                # print('save successfully!')
            # 为了方便看训练过程中效果的变化，将每次的图切出来。
            # 取第0张看一下效果
            _image = image[0]
            # _segment_image = torch.unsqueeze(segment_image[0], 0) * 255  # 拿标签的一个图
            _segment_image = segment_image[0]  # 拿标签的一个图
            # _out_image = torch.argmax(out_image[0], dim=0).unsqueeze(0) * 255  # 输出图
            _out_image = out_image[0]  # 输出图
            # 三张图进行拼接
            img = torch.stack([_image, _segment_image, _out_image], dim=0)
            save_image(img, f'{save_path}/{i}.png')
        # if epoch % 20 == 0:
        #     torch.save(net.state_dict(), weight_path)
        #     print('save successfully!')

        epoch += 1
