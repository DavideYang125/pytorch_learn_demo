# -*- coding: utf-8 -*-
import torchvision
import os
import gzip
import struct
import numpy as np
from PIL import Image
import pandas as pd


torchvision.datasets.MNIST(
    root='data/mnist_data',                     
    train=True,                                    # 是否是训练集
    transform=torchvision.transforms.ToTensor(),  # 图像转 tensor
    download=True,                                 # 如没有本地数据，则自动下载
    target_transform=None                          # 标签的额外变换（此处没有）
)


# 替换为你的 MNIST 数据路径
RAW_DIR = 'data/mnist_data/MNIST/raw'
OUT_DIR = 'data/mnist_images/train'
LABEL_OUT_DIR= 'data/mnist_images'
IMG_FILE = os.path.join(RAW_DIR, 'train-images-idx3-ubyte.gz')
LABEL_FILE = os.path.join(RAW_DIR, 'train-labels-idx1-ubyte.gz')

if not os.path.exists(IMG_FILE):
    raise FileNotFoundError(f"{IMG_FILE} 不存在")

os.makedirs(OUT_DIR, exist_ok=True)

# 读取图像数据
with gzip.open(IMG_FILE, 'rb') as f:
    magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
    images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)

# 读取标签数据
with gzip.open(LABEL_FILE, 'rb') as f:
    magic, num = struct.unpack('>II', f.read(8))
    labels = np.frombuffer(f.read(), dtype=np.uint8)

# 保存图像和生成 CSV 映射表
records = []

for i, (img, label) in enumerate(zip(images, labels)):
    img_filename = f'{i:05d}.png'
    img_path = os.path.join(OUT_DIR, img_filename)
    
    Image.fromarray(img).save(img_path)  # 保存为 PNG 格式
    
    records.append({'filename': img_filename, 'label': label})

# 生成 CSV 文件
df = pd.DataFrame(records)
df.to_csv(os.path.join(LABEL_OUT_DIR, 'train_labels.csv'), index=False)

print(f'转换完成，共生成 {len(images)} 张图片和 CSV。')
