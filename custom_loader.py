# -*- coding: utf-8 -*-
from custom_dataset import CustomImageDataset
from torch.utils.data   import DataLoader
from torchvision        import transforms
import matplotlib.pyplot as plt
import math


# 通过数据类读取图片数据
img_dir          = 'data/mnist_images/train'                                                 # 图象文件夹
label_file       = 'data/mnist_images/train_labels.csv'                                       # 标签文件
myDataSet        = CustomImageDataset(label_file,img_dir,transform=None,target_transform=None)  # 初始化数据类

# 将图片数据装载到DataLoader中使用
train_dataloader = DataLoader(myDataSet, batch_size=2, shuffle=False)                           # 将数据装载到DataLoader
dataloader_len   = len(train_dataloader)                                                        # dataloader的批数
batch_size       = train_dataloader.batch_size                                                  # dataloader每批的大小

# ---------从DataLoader中获取图片并进行打印-------------
max_images = 10  # 你要显示的最大图片数
count = 0

cols = 5  # 固定列数，比如5列
rows = math.ceil(max_images / cols)  # 根据max_images算行数

figure = plt.figure(figsize=(cols * 3, rows * 3))  # 3英寸宽高的单元格

for i, data in enumerate(train_dataloader):
    imgs, labels = data
    for j in range(len(labels)):
        if count >= max_images:
            break
        ax = figure.add_subplot(rows, cols, count + 1)
        ax.axis('off')  # 关闭坐标轴
        ax.set_title(str(labels[j].item()))
        img = transforms.ToPILImage()(imgs[j])
        ax.imshow(img, cmap='gray')
        count += 1
    if count >= max_images:
        break

plt.tight_layout()
plt.show()