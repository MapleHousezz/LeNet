import copy
import time

import pandas as pd
from torchvision.datasets import FashionMNIST
from torchvision.transforms import transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from model import LeNet


##训练集来训练w,b，再用验证集来验证w,b的质量；
# 数据加载处理函数：
def train_val_data_process():
    train_data = FashionMNIST(root='./data',
                              train=True,       #这里的28是什么意思
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=True)
    # 划分数据集,训练集和验证集
    train_data, val_data = Data.random_split(train_data, [round(0.8 * len(train_data)), round(0.2 * len(train_data))])
    #加载处理数据
    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=64,
                                       shuffle=True,
                                       num_workers=2)
    val_dataloader = Data.DataLoader(dataset=val_data,
                                     batch_size=64,
                                     shuffle=True,
                                     num_workers=2)

    return train_dataloader, val_dataloader


#定训练模型
def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []

    since = time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        train_loss = 0.0
        train_correct = 0.0

        val_loss = 0.0
        val_correct = 0.0

        train_num = 0
        val_num = 0

        for step, (b_x, b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.train()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)

            loss = criterion(output, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * b_x.size(0)
            train_correct += torch.sum(pre_lab.cpu() == b_y.data.cpu()).item()
            train_num += b_x.size(0)

        for step, (b_x, b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)

            loss_item = criterion(output, b_y).item()
            val_loss += loss_item * b_x.size(0)
            val_correct += torch.sum(pre_lab.cpu() == b_y.data.cpu()).item()
            val_num += b_x.size(0)

        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_correct / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_correct / val_num)

        print("{} | Train Loss: {:.4f} Train Acc: {:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print("{} | Val Loss: {:.4f} Val Acc: {:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        time_used = time.time() - since
        print("训练耗费的时间{:.0f}m{:.0f}s".format(time_used // 60, time_used % 60))

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'C:/Users/GQL/Desktop/DL/CNN/LeNet/best_model.pth')

    train_process = pd.DataFrame(data={
        "epoch": range(num_epochs),
        "train_loss_all": train_loss_all,
        "val_loss_all": val_loss_all,
        "train_acc_all": train_acc_all,
        "val_acc_all": val_acc_all
    })

    return train_process


def matplot_acc_loss(train_process):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_process["epoch"], train_process.train_loss_all,
             "ro-",label="train loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all,
             "bs-",label="val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")


    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all,
             "ro-",label="train acc")
    plt.plot(train_process["epoch"], train_process.val_acc_all,
             "bs-",label="val acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    #将模型实例化
    LeNet =LeNet()

    train_dataloader,val_dataloader = train_val_data_process()
    train_process = train_model_process(LeNet, train_dataloader,val_dataloader, 50)
    matplot_acc_loss(train_process)

    # print('a')
    # print(train_process)
    #print("推送设置")