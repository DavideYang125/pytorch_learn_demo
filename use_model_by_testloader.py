from neural_network_model import NeuralNetwork
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor



# 设置 device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 下载测试数据
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


# 从 dataloader 中取出一个 batch
data_iter = iter(test_dataloader)
images, labels = next(data_iter)


# 加载模型结构并加载权重
model_path = "saved_models/fashion_mnist_model.pth"
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load(model_path))
model.eval() # 设置模型为测试模式


correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = outputs.argmax(dim=1) # 选出分数最大的类别的索引
        
        # 统计正确数
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        for i in range(len(labels)):
            print(f"真实标签: {labels[i].item()}, 预测结果: {predicted[i].item()}")

accuracy = 100 * correct / total
print(f"\n总数：{total}，正确数：{correct}，测试集准确率: {accuracy:.2f}%")