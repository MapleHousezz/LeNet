from torchvision.datasets import FashionMNIST #从torchvisiion里面导入数据集
from torchvision.transforms import transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np

# 下载训练数据
train_data = FashionMNIST(root='./data',
                          train=True,
                          transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),
                          download=True)

# 加载数据操作
train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=64,   # 64个数据打捆，为一个批次
                               shuffle=True,    # 打乱
                               num_workers=0)

# 输出展示数据
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break   #   只拿第一批次的数据来测试
batch_x = b_x.squeeze().cpu().numpy()
batch_y = b_y.cpu().numpy()
class_label = train_data.classes  # 数据标签
print(class_label) #

plt.figure(figsize=(12, 5))
for ii in np.arange(len(batch_y)):
    plt.subplot(4, 16, ii + 1)
    plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray)
    plt.title(class_label[batch_y[ii]], size=10)
    plt.axis("off")
    plt.subplots_adjust(wspace=0.5)
plt.show()
