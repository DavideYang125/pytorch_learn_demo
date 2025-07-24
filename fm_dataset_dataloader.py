# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn 
import os

# 下载 FashionMNIST 训练集和测试集
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


# labels_map = {
#     0: "T-Shirt",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle Boot",
# }
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     img, label = training_data[sample_idx]  # 解包
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()


# 用 DataLoader 封装
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
# batch_size=64：每次迭代从训练集中取出 64 个样本。
# shuffle=True：每轮训练（epoch）前会打乱数据顺序，提高训练效果，防止模型记住顺序。
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


# Display image and label.
# 从训练数据集中获取一个批次的图像和标签，用于调试或训练前的验证。展示第一个图片样本
# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")


# 定义神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()  # 将 1x28x28 展平为 784
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)  # 最终10类输出
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# 设置训练设备、实例化模型和损失函数
device = "cuda" if torch.cuda.is_available() else "cpu"
model = NeuralNetwork().to(device) # 创建该网络的一个实例对象并存储到设备

loss_fn = nn.CrossEntropyLoss()  #设置损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)



# 定义训练函数
# 函数逻辑：
# 取一批数据 X, y
# 前向传播 → 得到预测 pred
# 计算损失 loss
# 清空梯度 optimizer.zero_grad()
# 反向传播 loss.backward()
# 参数更新 optimizer.step()
# 继续下一批，直到训练完所有样本

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device) # X 是一批图像（shape 类似于 [64, 1, 28, 28]）  y 是对应的标签（长度为 64）

        # 前向传播
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss_val = loss.item()
            current = batch * len(X)
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")


# 定义测试函数
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# 循环训练若干轮
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), "saved_models/fashion_mnist_model.pth") # 保存训练好的模型参数
print("saved model!")

