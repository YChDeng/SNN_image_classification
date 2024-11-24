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

# Animations and visualizations of dataset

#Rate coding

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
#HTML(anim.to_html5_video)

#Raster plot of diagonal pixels for fingerprint#

# Reshape image
raster_img = torch.Tensor([encoded_img_finger[:,i,i].numpy() for i in range(len(encoded_img_finger[0]))]).T

# Raster plot
fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
ax.set_title('Spiking of a diagonal pixels with the time. Fingerprints dataset')
ax.set_xlabel('Time')
ax.set_ylabel('Number of a diagonal pixel')
splt.raster(raster_img, ax, s=1, color='black')

# Membrane's potential

def add_x_ticks(ax):
    """
    Adding ticks

    :param ax: Axes object
    """ 
    ax.set_xticks(list(range(100))[::1])
    ax.tick_params(axis='x', which='major', labelsize=7)
    ax.set_yticks([])

def plot_spikes_with_potential(spikes, in_spikes, potential, 
                               title=None, threshold=1):
    """
    Plot input, output spikes and membrane potential

    :param spikes: output spikes
    :param in_spikes: input spikes
    :param potential: membrane potential
    :param title: title of plot
    :param threshold: value of threshold for membrane potential
    """
    # Unpacking Figure and Axes array
    fig, ax = plt.subplots(3,1, facecolor="w", figsize=(15, 5),
                          gridspec_kw={'height_ratios': [0.5, 4, 0.5]})
    ax[2].autoscale(tight=True)
    ax[1].autoscale(tight=True)
    ax[0].autoscale(tight=True)

    # Input spikes
    ax[0].set_title(f"{int(spikes.sum())} spikes and potential in neuron" if title==None else title)
    ax[0].set_ylabel("Input spikes")
    splt.raster(in_spikes, ax[0], s=200, c="black", marker="|")
    add_x_ticks(ax[0])

    # Output spikes
    splt.raster(spikes, ax[2], s=200, c="black", marker="|")
    ax[2].set_xlabel("Time")
    ax[2].set_ylabel("Output spikes")
    add_x_ticks(ax[2])

    # Membrane potential
    ax[1].plot(potential)
    ax[1].set_ylabel("Membrane potential")
    add_x_ticks(ax[1])

    times = []
    for i,spike in enumerate(spikes):
      if (spike != 0):
        times.append(i)

    ax[1].vlines(x = times, ymin = potential.min(), ymax = potential.max(), 
                 colors = 'gray', ls='dashed', lw=0.5)

    # Threshold line
    ax[1].axhline(y = threshold, color = 'r', linestyle = '--')

    fig.tight_layout()

# Leaky Integrate-and-Fire neuron
delta_t = 1e-3
tau = 5.1*5e-3
b = np.exp(-delta_t/tau)
lif = snn.Leaky(beta = b, threshold = 1.5)

# Initialize input, output spikes and membrane potential
spk_in = spikegen.rate_conv(torch.ones((99))*0.2)
mem = torch.ones(1)*0.1
#print(mem)
spk = torch.zeros(1)
potential = [mem]
#print(potential)
spikes = [spk]

# Neuron simulation
for step in range(99):
  spk, mem = lif(spk_in[step], mem)
  #print(torch.tensor([mem.item()]))
  potential.append(torch.tensor([mem.item()]))
  spikes.append(torch.tensor([spk.item()]))

#print(spikes)
# Plot the spikes and membrane potential
potential = torch.stack(potential)
spikes = torch.stack(spikes)
plot_spikes_with_potential(spikes, spk_in, potential, 
                           title='Membrane potential of an neuron and its spikes', 
                           threshold = 1.5)