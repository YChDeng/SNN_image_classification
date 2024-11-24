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

# Transforms for fingerprints
class CropInvertPadTransform:
    """
    Crop the frame (2*2*4*4) from the image, reformat to 1 channel, invert colors 
    and put black padding to the squared form. If the size is bigger, than the required, resize it
    """
    def __init__(self, size):
        self.size = size
        

    def __call__(self, x):
        """
        Transform the image by frame cropping, grayscaling and inverting
        Resize to the square by adding black padding

        :param x: tensor (image) to be transformed
        :return: transformed image
        """
        # Crop the frame 2 2 4 4 pixels
        x = TF.crop(x, 2, 2, TF.get_image_size(x)[1] - 6, TF.get_image_size(x)[0] - 6)
        x = TF.rgb_to_grayscale(x)
        x = TF.invert(x)

        # Add black padding to make the image square
        a = max(TF.get_image_size(x)) - TF.get_image_size(x)[0]
        b = max(TF.get_image_size(x)) - TF.get_image_size(x)[1]

        x = TF.pad(x, [a // 2, b // 2, a - a // 2, b - b // 2], fill=0)
        if TF.get_image_size(x)[0] > self.size:
            x = TF.resize(x, [self.size, self.size], antialias=True)
        return x
    
# Set transformations of images
# Transformations for fingerprints
from torchvision import transforms

fingerprints_transform = transforms.Compose([
    transforms.ToTensor(),
    CropInvertPadTransform(97),
    transforms.Normalize((0,), (1,))
])
# Define a transform for emnist and fashion mnist
mnist_transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

# Show example of fingerprint with alterations before and after transform
# Load the image

names = ['107__M_Left_index_finger.BMP', '107__M_Left_index_finger_Obl.BMP', '107__M_Left_index_finger_CR.BMP', '107__M_Left_index_finger_Zcut.BMP']

for i in range(len(names)):
    root='./datasets/fingerprints/'
    dirs = ['train/']
    for d in dirs:
        if names[i] in os.listdir(root+d+'1/'):
            break
    image = cv2.imread(finger_train_dir+'1/'+names[i])
    #Plot the original image
    plt.subplot(2, 4, i+1)
    # plt.title("Original")
    plt.imshow(image)

    #Plot the sharpened image
    plt.subplot(2, 4, i+5)
    
    arr = fingerprints_transform(image)[0].numpy()*255    
    plt.imshow(arr,cmap='gray')

plt.show()

# Transforms-Datasets-Dataloaders

# Directories of datasets
datadir = './datasets/'
finger_dir = datadir + 'fingerprints/'
finger_train_dir = finger_dir + 'train/'
emnist_dir = datadir + 'emnist/'
fashion_dir = datadir + 'fashion_mnist/'

train_dir = 'train/'
val_dir = 'val/'
test_dir = 'test/'

# Split proportions
val_split = 0.1
test_split = 0.1
batch_size = 128

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Fingerprints

# Split on validation and test split
for finger_type in os.listdir(finger_train_dir):
    imgs = os.listdir(finger_train_dir + finger_type)
    # Split with shuffling
    np.random.shuffle(imgs)
    for img in imgs[:int(len(imgs) * test_split) + 1]:
        if not os.path.exists(finger_dir + test_dir + finger_type + "/" + img):
            if not os.path.exists(finger_dir + test_dir + finger_type + "/"):
                os.makedirs(finger_dir + test_dir + finger_type + "/")            
            os.rename(finger_train_dir + finger_type + "/" + img, finger_dir + test_dir + finger_type + "/" + img)
        else:
            os.remove(finger_train_dir + finger_type + "/" + img)

    for img in imgs[int(len(imgs) * test_split) + 1:int(len(imgs) * test_split) + int(len(imgs) * val_split) + 2]:
        if not os.path.exists(finger_dir + val_dir + finger_type + "/" + img):
            if not os.path.exists(finger_dir + val_dir + finger_type + "/"):
                os.makedirs(finger_dir + val_dir + finger_type + "/")
            os.rename(finger_train_dir + finger_type + "/" + img, finger_dir + val_dir + finger_type + "/" + img)
        else:
            os.remove(finger_train_dir + finger_type + "/" + img)


# Set dataset
fingerprints_dataset = {'train': datasets.ImageFolder(root=finger_dir+train_dir, transform=fingerprints_transform),
                        'val': datasets.ImageFolder(root=finger_dir+val_dir, transform=fingerprints_transform),
                        'test': datasets.ImageFolder(root=finger_dir+test_dir, transform=fingerprints_transform)
}

# Set dataloader
fingerprints_dataloader = {'train': DataLoader(fingerprints_dataset['train'], batch_size=batch_size, shuffle=True, drop_last=True),
                        'val': DataLoader(fingerprints_dataset['val'], batch_size=batch_size, shuffle=True, drop_last=True),
                        'test': DataLoader(fingerprints_dataset['test'], batch_size=batch_size, shuffle=True, drop_last=True)
}

# EMNIST
# Set dataset
emnist_dataset = {'train': torchvision.datasets.EMNIST(emnist_dir+train_dir, 'digits', download=True, train=True, transform=mnist_transform),
                  'val': None,
                  'test': torchvision.datasets.EMNIST(emnist_dir+train_dir, 'digits',download=True, train=False, transform=mnist_transform)
}

# Validation split
emnist_dataset['train'], emnist_dataset['val'] = random_split(emnist_dataset['train'], [1-val_split, val_split])

# Set dataloaders
emnist_dataloader = {'train': DataLoader(emnist_dataset['train'], batch_size=batch_size, shuffle=True, drop_last=True),
                     'val': DataLoader(emnist_dataset['val'], batch_size=batch_size, shuffle=True, drop_last=True),
                     'test': DataLoader(emnist_dataset['test'], batch_size=batch_size, shuffle=True, drop_last=True)
}

# Fashion-MNIST
# Set dataset
fashion_dataset = {'train': torchvision.datasets.FashionMNIST(fashion_dir+train_dir, download=True, train=True, transform=mnist_transform),
                  'val': None,
                  'test': torchvision.datasets.FashionMNIST(fashion_dir+train_dir, download=True, train=False, transform=mnist_transform)
}

# Validation split
fashion_dataset['train'], fashion_dataset['val'] = random_split(fashion_dataset['train'], [1-val_split, val_split])

# Set dataloaders
fashion_dataloader = {'train': DataLoader(fashion_dataset['train'], batch_size=batch_size, shuffle=True, drop_last=True),
                     'val': DataLoader(fashion_dataset['val'], batch_size=batch_size, shuffle=True, drop_last=True),
                     'test': DataLoader(fashion_dataset['test'], batch_size=batch_size, shuffle=True, drop_last=True)
}

# Print one picture from each dataset
dataloaders = [fingerprints_dataloader, emnist_dataloader, fashion_dataloader]
for i in range(3):
    pic, _ = next(iter(dataloaders[i]['train']))
    pic = pic[0][0].numpy()*255 

    plt.subplot(1, 3, i+1)
    plt.imshow(pic, cmap='gray')

plt.show()