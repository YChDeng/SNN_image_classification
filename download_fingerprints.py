import numpy as np
import itertools
import cv2
import timeit
from os import listdir
import os
import zipfile
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
# api.set_config_value(name="proxy", value="http://127.0.0.1:7897") #设置代理
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

output_dir = "."

# 解压 ZIP 文件
if os.path.exists(output_zip_file):
    print(f"Extracting {output_zip_file} to {output_dir}")
    with zipfile.ZipFile(output_zip_file, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print("Extraction complete.")
else:
    print(f"ZIP file {output_zip_file} not found.")

# Load fingerprints dataset from kaggle and sort it
# Sort data inside train and test folders on classes folders named from 0 to 9
map_finger_name = ['left_thumb', 'left_index', 'left_middle', 'left_ring', 'left_little',
           'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_little']

# Working directories
load_dir = 'SOCOFing/'

datadir = './datasets/'
finger_dir = datadir + 'fingerprints/'
finger_train_dir = finger_dir + 'train/'
finger_val_dir = finger_dir + 'val/'
finger_test_dir = finger_dir + 'test/'

if not os.path.exists(finger_dir):
    # Create folders
    if not os.path.exists('./datasets'):
        os.mkdir('./datasets')
    if not os.path.exists(finger_dir):
        os.mkdir(finger_dir)
    if not os.path.exists(finger_train_dir):
        os.mkdir(finger_train_dir)
    if not os.path.exists(finger_val_dir):
        os.mkdir(finger_val_dir)
    if not os.path.exists(finger_test_dir):
        os.mkdir(finger_test_dir)

    # Create folders for each class
    for i in range(10):
        if not os.path.exists(finger_train_dir + f"{i}/"):
            os.mkdir(finger_train_dir + f"{i}/")
        if not os.path.exists(finger_val_dir + f"{i}/"):
            os.mkdir(finger_test_dir + f"{i}/")
        if not os.path.exists(finger_test_dir):
            os.mkdir(finger_test_dir + f"{i}/")

    # Move and sort by classes files of train dataset
    # for dir in ['Real/', 'Altered/Altered-Easy/']:
    for dir in ['Real/', 'Altered/Altered-Easy/']:
        for img in os.listdir(load_dir + dir):
            ind = max([i if map_finger_name[i] in img.lower() else -1 for i in range(len(map_finger_name))])
            shutil.copy2(load_dir + dir + img, finger_train_dir + f"{ind}/" + img)