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

# Batch of images from each dataset
img_finger, label_finger = next(iter(fingerprints_dataloader['train'])) 
img_e, label_e = next(iter(emnist_dataloader['train'])) 
img_fash, label_fash = next(iter(fashion_dataloader['train'])) 

# First image in the batch of images
img_finger, img_e, img_fash = img_finger[0][0], img_e[0][0], img_fash[0][0] 

# Set number of time steps and prbability of spike for most intensive pixel
num_steps = 100
gain = 1

# Encode image with rate coding
encoded_img_finger = spikegen.rate(img_finger, num_steps=num_steps, gain=gain)
encoded_img_e = spikegen.rate(img_e, num_steps=num_steps, gain=gain)
encoded_img_fash = spikegen.rate(img_fash, num_steps=num_steps, gain=gain)

# Base frame of image
fig, ax = plt.subplots(1,3, figsize = (12,4))
plt.tight_layout()
image_plot_finger = ax[0].imshow(img_finger, cmap='gray')
image_plot_e = ax[1].imshow(img_e, cmap='gray')
image_plot_fash = ax[2].imshow(img_fash, cmap='gray')

def init():
    """
    Initialization function
    Plot the background of each frame

    """
    image_plot_finger.set_data(img_finger)
    image_plot_e.set_data(img_e)
    image_plot_fash.set_data(img_fash)
    for x in ax:
      x.set_axis_off()
      
def animate(time):
    """
    Animation function; called sequantially

    :param time: time step
    """
    image_plot_finger.set_array(encoded_img_finger[time])
    image_plot_e.set_array(encoded_img_e[time])
    image_plot_fash.set_array(encoded_img_fash[time])
    ax[0].set_title(f'Fingerprint')
    ax[1].set_title(f'EMNIST')
    ax[2].set_title(f'Fashion-MNIST')

anim = FuncAnimation(fig, animate, init_func=init, frames=len(encoded_img_fash), interval=100)