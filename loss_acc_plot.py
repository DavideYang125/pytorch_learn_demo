import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from neural_network_model import NeuralNetwork
from torchvision.transforms import ToTensor


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

# 用 DataLoader 封装
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
# batch_size=64：每次迭代从训练集中取出 64 个样本。
# shuffle=True：每轮训练（epoch）前会打乱数据顺序，提高训练效果，防止模型记住顺序。
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


# 设置训练设备、实例化模型和损失函数
device = "cuda" if torch.cuda.is_available() else "cpu"
model = NeuralNetwork().to(device) # 创建该网络的一个实例对象并存储到设备

loss_fn = nn.CrossEntropyLoss()  #设置损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)


# 训练函数
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()

    total_loss = 0
    correct = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    avg_loss = total_loss / num_batches
    accuracy = correct / size
    return avg_loss, accuracy

# 测试函数
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    total_loss = 0
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            total_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    avg_loss = total_loss / num_batches
    accuracy = correct / size
    return avg_loss, accuracy



# 准备画图数据
train_losses, train_accuracies = [], []
test_losses, test_accuracies = [], []


# 训练过程
epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")

    train_loss, train_acc = train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loss, test_acc = test_loop(test_dataloader, model, loss_fn)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    print("-" * 50)


# 设置支持中文显示
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 绘制 loss / accuracy 曲线
epochs_range = range(1, epochs + 1)

plt.figure(figsize=(10, 4))

# Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label="Train Loss 训练损失")
plt.plot(epochs_range, test_losses, label="Test Loss 测试损失")
plt.xlabel("Epoch 训练轮数")
plt.ylabel("Loss 损失")
plt.title("Loss Curve 损失曲线")
plt.legend()
plt.grid(True)

# Accuracy 曲线
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, label="Train Acc 训练准确率")
plt.plot(epochs_range, test_accuracies, label="Test Acc 测试准确率")
plt.xlabel("Epoch 训练轮数")
plt.ylabel("Accuracy 准确率")
plt.title("Accuracy Curve 准确率曲线")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()