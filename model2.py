import torch
import torch.nn as nn

# 设置类CNNModel，它继承了torch.nn中的Module模块
class CNNModel2(nn.Module):
    # 定义卷积神经网络
    # 修改初始化函数init的参数列表
    # 需要将训练图片的高height、宽width、
    # 图片中的字符数量digit_num，类别数量class_num传入
    def __init__(self, height, width, digit_num, class_num):
        super(CNNModel2, self).__init__()
        self.digit_num = digit_num # 将digit_num保存在类中

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv6 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2))
        # 完成卷积层的计算后，计算全连接层的输入数据数量input_num
        input_num = 2 * 2 * 2048
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_num, 4096),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(4096, class_num))
        # 后面训练会使用交叉熵损失函数CrossEntropyLoss
        # softmax函数会定义在损失函数中，所以这里就不显示的定义了

    # 前向传播函数
    # 函数输入一个四维张量x
    # 这四个维度分别是样本数量、输入通道、图片的高度和宽度
    def forward(self, x): # [n, 1, 128, 128]
        # 将输入张量x按照顺序，输入至每一层中进行计算
        # 每层都会使张量x的维度发生变化
        out = self.conv1(x) # [n, 64, 64, 64]
        out = self.conv2(out) # [n, 128, 32, 32]
        out = self.conv3(out) # [n, 256, 16, 16]
        out = self.conv4(out) # [n, 512, 8, 8]
        out = self.conv5(out) # [n, 1024, 4, 4]
        out = self.conv6(out) # [n, 2048, 2, 2]
        out = self.fc1(out) # [n, class_num]
        return out

import json
if __name__ == '__main__':
    with open("config.json", "r") as f:
        config = json.load(f)

    height = config["resize_height"]  # 图片的高度
    width = config["resize_width"]  # 图片的宽度
    characters = config["characters"]  # 验证码使用的字符集
    digit_num = config["digit_num"]
    class_num = len(characters) * digit_num

    # 定义一个CNNModelUp1实例
    model = CNNModel2(height, width, digit_num, class_num)
    print(model) #将其打印，观察打印结果可以了解模型的结构
    data = torch.ones(128, 1, 128, 128)
    print(model(data).shape)


