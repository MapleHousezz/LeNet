import torch
from torch import nn
from torchsummary import summary


# 定义类，继承nn.Module
class LeNet(nn.Module):
    # 初始化定义网络层，激活函数。
    #所有网络约定俗称的方法：
    def __init__(self):
        super(LeNet, self).__init__()

        # 卷积层1
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        # 激活函数
        self.sig = nn.Sigmoid()
        # 平均池化
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 卷积层2
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        # 平均池化 和s2相同
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 平展层
        self.flatten = nn.Flatten()
        # 全连接层
        self.f5 = nn.Linear(400, 120)
        self.f6 = nn.Linear(120, 84)
        self.f7 = nn.Linear(84, 10)

    # 前向传播过程：
    def forward(self, x):
        x = self.sig(self.c1(x))    #卷积，激活
        x = self.s2(x)              #平均池化
        x = self.sig(self.c3(x))    #卷积，激活
        x = self.s4(x)              #平均池化
        x = self.flatten(x)         #平展层
        x = self.sig(self.f5(x))
        x = self.sig(self.f6(x))
        x = self.f7(x)
        return x


if __name__ == '__main__':
    # 判断GPU还是CPU
    #这里的("cuda" if torch.cuda.is_available() else "cpu") 是一个三元运算符
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 实例化，放到GPU里面：
    model = LeNet().to(device)
    # 输出摘要信息
    print(summary(model, input_size=(1, 28, 28)))
