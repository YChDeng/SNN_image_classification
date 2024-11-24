import numpy as np
import itertools
import cv2
#from google.colab import files
import timeit
from os import listdir
import os
from zipfile import ZipFile
import gdown
import shutil

import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchview import draw_graph
from torchvision.transforms import ToTensor
from torchshape import tensorshape

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from sklearn.metrics import classification_report
import zipfile

# 配置 kaggle.json 文件路径（假设在当前目录下）
kaggle_json_path = "./kaggle.json"
kaggle_destination_path = os.path.expanduser("~/.kaggle")

# 确保 .kaggle 目录存在
os.makedirs(kaggle_destination_path, exist_ok=True)

# 复制 kaggle.json 文件到 .kaggle
kaggle_json_dest = os.path.join(kaggle_destination_path, "kaggle.json")
if not os.path.exists(kaggle_json_dest):
    shutil.copy(kaggle_json_path, kaggle_json_dest)

# 修改权限确保安全
os.chmod(os.path.join(kaggle_destination_path, "kaggle.json"), 0o600)

from kaggle.api.kaggle_api_extended import KaggleApi

# 初始化 Kaggle API
api = KaggleApi()
api.set_config_value(name="proxy", value="http://127.0.0.1:7897") #设置代理
api.authenticate()

# 下载数据集
dataset_name = "ruizgara/socofing"
output_zip_file = "socofing.zip"
if not os.path.exists(output_zip_file):
    print(f"Downloading dataset: {dataset_name}")
    api.dataset_download_files(dataset_name, path=".", quiet=False)

# 下载完成后初始化代理设置
api.unset_config_value(name="proxy")

# 创建解压目录
output_dir = "original_data"
os.makedirs(output_dir, exist_ok=True)

# 解压 ZIP 文件
if os.path.exists(output_zip_file):
    print(f"Extracting {output_zip_file} to {output_dir}")
    with zipfile.ZipFile(output_zip_file, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print("Extraction complete.")
else:
    print(f"ZIP file {output_zip_file} not found.")

# 定义手指类别名称
map_finger_name = [
    'left_thumb', 'left_index', 'left_middle', 'left_ring', 'left_little',
    'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_little'
]

# 定义工作目录
load_dir = 'original_data/SOCOFing/'  # 原始数据文件夹路径
datadir = './datasets/'               # 数据集根目录
finger_dir = os.path.join(datadir, 'fingerprints/')  # 指纹分类目录
finger_train_dir = os.path.join(finger_dir, 'train/')  # 训练集目录
finger_val_dir = os.path.join(finger_dir, 'val/')      # 验证集目录
finger_test_dir = os.path.join(finger_dir, 'test/')    # 测试集目录

# 检查并创建指纹数据集的目录结构
if not os.path.exists(finger_dir):
    print("正在创建数据集目录结构...")
    # 创建主目录
    os.makedirs(finger_dir, exist_ok=True)
    os.makedirs(finger_train_dir, exist_ok=True)
    os.makedirs(finger_val_dir, exist_ok=True)
    os.makedirs(finger_test_dir, exist_ok=True)

    # 为每个手指类别创建对应的子目录（编号 0-9）
    for i in range(10):
        os.makedirs(os.path.join(finger_train_dir, f"{i}/"), exist_ok=True)
        os.makedirs(os.path.join(finger_val_dir, f"{i}/"), exist_ok=True)
        os.makedirs(os.path.join(finger_test_dir, f"{i}/"), exist_ok=True)
    print("目录结构创建完成。")

# 开始处理并按类别分类训练数据
print("正在分类并移动训练数据文件...")
for dir_name in ['Real/', 'Altered/Altered-Easy/']:
    source_dir = os.path.join(load_dir, dir_name)

    # 确保原始数据子目录存在
    if not os.path.exists(source_dir):
        print(f"警告：目录 {source_dir} 不存在，跳过处理。")
        continue

    # 遍历目录中的所有图像文件
    for idx, img_file in enumerate(os.listdir(source_dir), start=1):
        # 根据文件名判断属于哪个类别
        ind = max([i if map_finger_name[i] in img_file.lower() else -1 for i in range(len(map_finger_name))])

        # 确保类别有效
        if ind != -1:
            dest_dir = os.path.join(finger_train_dir, f"{ind}/")
            shutil.copy2(os.path.join(source_dir, img_file), os.path.join(dest_dir, img_file))
            if idx % 100 == 0:
                print(f"已处理 {idx} 个文件，当前文件：{img_file}，目标类别：{ind}")
        else:
            print(f"跳过未识别的文件：{img_file}")

print("训练数据分类完成。")