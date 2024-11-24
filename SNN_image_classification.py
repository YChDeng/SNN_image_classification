#!/usr/bin/env python
# coding: utf-8

# ## Installing snntorch library

# In[9]:


get_ipython().system('pip install snntorch --quiet')
get_ipython().system('pip install -q kaggle')
get_ipython().system('pip install torchview -q')
get_ipython().system('pip install torchshape -q')


# ## Importing all the necessary tools

# In[10]:


import numpy as np
import itertools
import cv2
# from google.colab import files
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


# ## Loading data from web and arranging in the filesystem
# Before running the following code, make sure that you have loaded the file "kaggle.json" into the working directory. It should contain your API key to Kaggle. Also make sure that you have agreed to the rules for using the fingertips dataset. 
# [Link](https://www.kaggle.com/datasets/ruizgara/socofing) to the SOCOFing dataset.
# 
# Unfortunately, you will not be able to run the code for fingerprints classification without this file. However, you can still use EMNIST and Fashion-MNIST datasets.

# In[12]:


# Load fingerprints dataset from kaggle
get_ipython().run_line_magic('%capture', '')
get_ipython().system('mkdir ~/.kaggle')

get_ipython().system('cp kaggle.json ~/.kaggle/')
get_ipython().system('chmod 600 ~/.kaggle/kaggle.json')
get_ipython().system('kaggle datasets list')
get_ipython().system('kaggle datasets download -d ruizgara/socofing -q')
get_ipython().system('mkdir original_data')
get_ipython().system('unzip socofing.zip -d .')


# In[13]:


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
    os.mkdir('./datasets')
    os.mkdir(finger_dir)
    os.mkdir(finger_train_dir)
    os.mkdir(finger_val_dir)
    os.mkdir(finger_test_dir)

    # Create folders for each class
    for i in range(10):
        os.mkdir(finger_train_dir + f"{i}/")
        os.mkdir(finger_val_dir + f"{i}/")
        os.mkdir(finger_test_dir + f"{i}/")

    # Move and sort by classes files of train dataset
    # for dir in ['Real/', 'Altered/Altered-Easy/']:
    for dir in ['Real/', 'Altered/Altered-Easy/']:
        for img in os.listdir(load_dir + dir):
            ind = max([i if map_finger_name[i] in img.lower() else -1 for i in range(len(map_finger_name))])
            shutil.copy2(load_dir + dir + img, finger_train_dir + f"{ind}/" + img)


# ## Data preprocessing
# 

# In[14]:


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


# In[4]:


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


# In[15]:


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


# ### Transform - Datasets - Dataloaders

# In[16]:


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


# In[17]:


# Fingerprints

# Split on validation and test split
for finger_type in os.listdir(finger_train_dir):
    imgs = os.listdir(finger_train_dir + finger_type)
    # Split with shuffling
    np.random.shuffle(imgs)
    for img in imgs[:int(len(imgs) * test_split) + 1]:
        os.rename(finger_train_dir + finger_type + "/" + img, finger_dir + test_dir + finger_type + "/" + img)

    for img in imgs[int(len(imgs) * test_split) + 1:int(len(imgs) * test_split) + int(len(imgs) * val_split) + 2]:
        os.rename(finger_train_dir + finger_type + "/" + img, finger_dir + val_dir + finger_type + "/" + img)


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


# In[18]:


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


# In[19]:


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


# In[20]:


# Print one picture from each dataset
dataloaders = [fingerprints_dataloader, emnist_dataloader, fashion_dataloader]
for i in range(3):
    pic, _ = next(iter(dataloaders[i]['train']))
    pic = pic[0][0].numpy()*255 

    plt.subplot(1, 3, i+1)
    plt.imshow(pic, cmap='gray')

plt.show()


# ## Animations and visualizations of dataset

# #### Rate coding

# Spiking neural networks are designed to handle time-varying data. In this case, to use spike-based communication in the network, the input data should be encoded from the continuous value form into a time series of spikes.
# 
# In rate coding, each normalized value of input features is used to compute the probability that a spike will occur at any given time step, resulting in a discrete rate-encoded value. One may consider this as a Binomial distribution with number of trials equal to 1 and a probability of success (spike) equal to the normalized value of input feature.

# In[21]:


get_ipython().run_cell_magic('capture', '', '%matplotlib inline\n\n# Batch of images from each dataset\nimg_finger, label_finger = next(iter(fingerprints_dataloader[\'train\'])) \nimg_e, label_e = next(iter(emnist_dataloader[\'train\'])) \nimg_fash, label_fash = next(iter(fashion_dataloader[\'train\'])) \n\n# First image in the batch of images\nimg_finger, img_e, img_fash = img_finger[0][0], img_e[0][0], img_fash[0][0] \n\n# Set number of time steps and prbability of spike for most intensive pixel\nnum_steps = 100\ngain = 1\n\n# Encode image with rate coding\nencoded_img_finger = spikegen.rate(img_finger, num_steps=num_steps, gain=gain)\nencoded_img_e = spikegen.rate(img_e, num_steps=num_steps, gain=gain)\nencoded_img_fash = spikegen.rate(img_fash, num_steps=num_steps, gain=gain)\n\n# Base frame of image\nfig, ax = plt.subplots(1,3, figsize = (12,4))\nplt.tight_layout()\nimage_plot_finger = ax[0].imshow(img_finger, cmap=\'gray\')\nimage_plot_e = ax[1].imshow(img_e, cmap=\'gray\')\nimage_plot_fash = ax[2].imshow(img_fash, cmap=\'gray\')\n\ndef init():\n    """\n    Initialization function\n    Plot the background of each frame\n\n    """\n    image_plot_finger.set_data(img_finger)\n    image_plot_e.set_data(img_e)\n    image_plot_fash.set_data(img_fash)\n    for x in ax:\n      x.set_axis_off()\n      \ndef animate(time):\n    """\n    Animation function; called sequantially\n\n    :param time: time step\n    """\n    image_plot_finger.set_array(encoded_img_finger[time])\n    image_plot_e.set_array(encoded_img_e[time])\n    image_plot_fash.set_array(encoded_img_fash[time])\n    ax[0].set_title(f\'Fingerprint\')\n    ax[1].set_title(f\'EMNIST\')\n    ax[2].set_title(f\'Fashion-MNIST\')\n\nanim = FuncAnimation(fig, animate, init_func=init, frames=len(encoded_img_fash), interval=100)\n')


# In[ ]:


HTML(anim.to_html5_video())


# #### Raster plot of diagonal pixels for fingerprint

# Spike-coded images can be represented in form of raster plot, showing spikes of pixels at each time step

# In[22]:


# Reshape image
raster_img = torch.Tensor([encoded_img_finger[:,i,i].numpy() for i in range(len(encoded_img_finger[0]))]).T

# Raster plot
fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
ax.set_title('Spiking of a diagonal pixels with the time. Fingerprints dataset')
ax.set_xlabel('Time')
ax.set_ylabel('Number of a diagonal pixel')
splt.raster(raster_img, ax, s=1, color='black')


# #### Membrane's potential

# Leaky Integrate-and-Fire (LIF) neuron receives the sum of the weighted input signals. LIF integrates the inputs over time. If the integrated value exceeds a predefined threshold, the LIF neuron produces a spike.
# 
# As a result, the information is not stored in spikes, but rather in its frequency.
# 
# We plotted the behavior of an LIF neuron receiving input spikes over 100 time steps.

# In[23]:


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


# In[24]:


# Leaky Integrate-and-Fire neuron
delta_t = 1e-3
tau = 5.1*5e-3
b = np.exp(-delta_t/tau)
lif = snn.Leaky(beta = b, threshold = 1.5)


# In[44]:


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


# # Experiments

# In[45]:


def training(model, train_loader, val_loader, optimizer, criterion, 
             device, epochs, tolerance = 3, min_delta = 0.01, 
             snn_mode = False, num_steps = None, path_load_model="model.pt"):
    """
    Training of model

    :param model: model to be trained
    :param train_loader: dataloader with train dataset
    :param val_loader: dataloader with validation dataset
    :param optimizer: optimizer
    :param criterion: loss function
    :param device: device on with model run
    :param epochs: number of epochs to be trained
    :param tolerance: number of epochs for stopping criteria
    :param min_delta: min validation loss delta between epochs for stopping criteria
    :param snn_mode: True - SNN training, False - ANN training
    :param num_steps: number of time steps for SNN
    :param path_load_model: path to save trained model
    :return: validation loss and accuracy history, time per epoch
    """
    epochs = epochs
    num_steps = num_steps
    time_delta = 0
    total_val_history = []
    accuracy_history = []
    best_accuracy = 0

    count = 0
    early_stop_val = np.inf
    
    start = timeit.default_timer()

    for epoch in range(epochs):
        # For "epochs" number of epochs
        model.train()

        train_loss, valid_loss = [], []
        total_train = 0
        correct_train = 0
        total_val = 0
        correct_val = 0

        for data, targets in train_loader:
            # For each batch of training set

            # Transfer to device
            data = data.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            
            
            if snn_mode:
                # For SNN models
                # Reshape data
                data = data.view(data.shape[0], -1)

                # Run model on batch and return output spike and membrane potential
                spike, potential = model(data)

                # Training loss of batch
                loss_val = torch.zeros((1), dtype=dtype, device=device)
                for step in range(num_steps):
                    loss_val += criterion(potential[step], targets)

                # Mean training loss
                loss_val /= num_steps

                # Decode output spikes to real-valued label
                _, predicted = spike.sum(dim=0).max(1)

                # Count number targets and correct identified labels
                total_train += targets.size(0)
                correct_train += (predicted == targets).sum().item()
            else:
                # For ANN models
                output = model(data)

                # Training loss of batch
                loss_val = criterion(output, targets)

                # Decode output spikes to real-valued label
                pred = output.argmax(dim=1, keepdim=True) 

                # Count number targets and correct identified labels
                total_train += targets.size(0)
                correct_train += pred.eq(targets.view_as(pred)).sum().item()

            # For next epoch
            loss_val.backward()
            optimizer.step()
            train_loss.append(loss_val.item())

        with torch.no_grad():
            # Validation phase
            model.eval()

            for data, targets in val_loader:
                # For each batch of validation set
                data = data.to(device)
                targets = targets.to(device)
                
                if snn_mode:
                    # For SNN models
                    # Reshape data
                    data = data.view(data.shape[0], -1)

                    # Run model on batch and return output spike and membrane potential
                    spike, potential = model(data)

                    # Validation loss of batch
                    loss_val = torch.zeros((1), dtype=dtype, device=device)
                    for step in range(num_steps):
                        loss_val += criterion(potential[step], targets)

                    # Mean validation loss
                    loss_val /= num_steps

                    # Decode output spikes to real-valued label
                    _, predicted = spike.sum(dim=0).max(1)

                    # Count number targets and correct identified labels
                    total_val += targets.size(0)
                    correct_val += (predicted == targets).sum().item()

                else:
                    # For ANN models
                    output = model(data)

                    # Validation loss of batch
                    loss_val = criterion(output, targets)
                    
                    # Decode output spikes to real-valued label
                    pred = output.argmax(dim=1, keepdim=True) 

                    # Count number targets and correct identified labels
                    total_val += targets.size(0)
                    correct_val += pred.eq(targets.view_as(pred)).sum().item()

                valid_loss.append(loss_val.item())

            # Early stopping
            if abs(early_stop_val - np.mean(valid_loss)) < min_delta:
                count += 1
            else:
                count = 0
            if count == tolerance:
                break

            early_stop_val = np.mean(train_loss)

            # Total validation loss and accuracy
            total_val_history.append(np.mean(valid_loss))
            accuracy_history.append(correct_val/total_val)

            # Save best model
            if best_accuracy <= accuracy_history[-1]:
                torch.save(model.state_dict(), path_load_model)

            print ("Epoch:", epoch, "\n\tTraining Loss:", np.mean(train_loss), 
                f"\n\tTraining Accuracy: {100 * correct_train/total_train:.2f}%", 
                "\n\tValidation Loss:", np.mean(valid_loss),
                f"\n\tValidation Accuracy: {100 * correct_val/total_val:.2f}%")
            
    stop = timeit.default_timer()
    time_delta = stop - start
    return total_val_history, accuracy_history, time_delta/(epoch + 1)


def testing(model, test_loader, device, snn_mode = False):
    """
    Testing of model

    :param model: model to be trained
    :param test_loader: dataloader with test dataset
    :param device: device on with model run
    :param snn_mode: True - SNN training, False - ANN training
    :return: true labels and predictions
    """
    total = 0
    correct = 0
    predictions = np.array([])
    true_labels = np.array([])

    with torch.no_grad():
        model.eval()
        for data, targets in test_loader:
            # For batch in test set
            data = data.to(device)
            targets = targets.to(device)
            
            if snn_mode:
                # SNN model
                # Output spikes
                test_spk, _ = model(data.view(data.size(0), -1))

                # Decode output spikes to real-valued label
                _, predicted = test_spk.sum(dim=0).max(1)
                predictions = np.concatenate((predictions, predicted.cpu()))
            

                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            else:
                # ANN model
                output = model(data)

                predicted = output.argmax(dim=1, keepdim=True) 
                predictions = np.concatenate((predictions, predicted.cpu().numpy().reshape(1,-1)[0]))
            
                
                # Count number targets and correct identified labels
                total += targets.size(0)
                correct += predicted.eq(targets.view_as(predicted)).sum().item()

            true_labels = np.concatenate((true_labels, targets.cpu()))

    print(f"Total correctly classified test set images: {correct}/{total}")
    print(f"Test Set Accuracy: {100 * correct / total:.2f}%\n")
    return true_labels, predictions


# In[46]:


class SNN(nn.Module):
    def __init__(self, num_inputs, num_outputs, 
                 num_hidden = 1024, num_steps = 25, 
                 beta = 0.95):
        """
        SNN with one hidden layer

        :param num_inputs: number of input units
        :param num_outputs: number of output units
        :param num_hidden: number of hidden units
        :param num_steps: number of time steps
        :param beta: beta coefficient value for Leaky model
        """
        super(SNN, self).__init__()

        # Initialize layers
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.linear2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

        # Initialize number of time steps
        self.num_steps = num_steps

    def forward(self, x):
        """
        Forward step

        :param x: input values
        :return: output spikes and membrane potential
        """
        # Initialize hidden states at t=0
        potential1 = self.lif1.init_leaky()
        potential2 = self.lif2.init_leaky()
        
        # Record the output values
        output_spike = []
        output_potential = []

        for step in range(self.num_steps):
            # For each time step run through the SNN
            current1 = self.linear1(x)
            spike1, potential1 = self.lif1(current1, potential1)
            current2 = self.linear2(spike1)
            spike2, potential2 = self.lif2(current2, potential2)
            
            # Record outputs
            output_spike.append(spike2)
            output_potential.append(potential2)
        return torch.stack(output_spike, dim=0), torch.stack(output_potential, dim=0)


# In[47]:


class CNN(nn.Module):
    def __init__(self, inputs_shape=(128, 1,97,97), num_outputs=10, 
                 num_hidden = 1024):
        """
        CNN with one fully-connected hidden layer

        :param inputs_shape: input shape of image
        :param num_outputs: number of output units
        :param num_hidden: number of hidden units
        """
        super(CNN, self).__init__()

        # First convolutional and max-pooling block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,16,3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Second convolutional and max-pooling block
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Output shape
        outshape = [*tensorshape(self.conv2, tensorshape(self.conv1, inputs_shape))][1:]

        # Linear block
        self.linear1 = nn.Sequential(
            nn.Linear(np.prod(outshape), num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_outputs)
        )

    def forward(self, x):
        """
        Forward step

        :param x: input values
        :return: output values
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        return F.log_softmax(x, dim=1)


# ## Create models for different datasets
# 

# In[53]:


# Set parameters common for all models
max_epochs = 20
time_steps = 25
criterion = nn.CrossEntropyLoss

load_path =  "./save_model/"
if not os.path.exists(load_path):
    os.mkdir(load_path)

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(f'Device: {device}')


# ### Fingerprints 
# 
# 

# #### Create models

# In[49]:


# Set parameters for this dataset

# Set input and output shapes
finger_input_shape = next(iter(fingerprints_dataloader['train']))[0].shape
finger_num_inputs = np.prod(finger_input_shape[-2:])
finger_num_outputs = len(np.unique(fingerprints_dataloader['train'].dataset.targets))

# Set path where to load the models
finger_snn_load_path = load_path + "finger_snn_model.pt"
finger_cnn_load_path = load_path + "finger_cnn_model.pt"


# In[50]:


# Create SNN model
finger_snn = SNN(finger_num_inputs, finger_num_outputs).to(device)
finger_snn_optimizer = torch.optim.Adam(finger_snn.parameters(), lr=0.000085, betas=(0.9, 0.999))

print(finger_snn)


# In[52]:


# Train SNN model
finger_snn_loss, finger_snn_accuracy, finger_snn_time = training(finger_snn, fingerprints_dataloader['train'], 
                                  fingerprints_dataloader['val'], finger_snn_optimizer, 
                                  criterion(), device, max_epochs, 
                                  snn_mode=True, num_steps=time_steps, path_load_model=finger_snn_load_path)


# In[ ]:


# Create CNN model
finger_cnn = CNN(finger_input_shape, finger_num_outputs).to(device)
finger_cnn_optimizer = torch.optim.Adam(finger_cnn.parameters(), lr=0.000085, betas=(0.9, 0.999))

print(finger_cnn)


# In[ ]:


# Train CNN model
finger_cnn_loss, finger_cnn_accuracy, finger_cnn_time = training(finger_cnn, fingerprints_dataloader['train'], 
                                  fingerprints_dataloader['val'], finger_cnn_optimizer, 
                                  criterion(), device, max_epochs, snn_mode=False, path_load_model=finger_cnn_load_path)


# #### Load the best obtained models

# In[ ]:


# Load SNN model
finger_snn = SNN(finger_num_inputs, finger_num_outputs)
finger_snn.load_state_dict(torch.load(finger_snn_load_path))
finger_snn.to(device)

print(finger_snn)


# In[ ]:


# Load CNN model
finger_cnn = CNN(finger_input_shape, finger_num_outputs)
finger_cnn.load_state_dict(torch.load(finger_cnn_load_path))
finger_cnn.to(device)

print(finger_cnn)


# #### Analisys of the results obtained by the models
# For each model print the avarage consumed time per epoch, number of correctly classified images and accuracy for test set, as well as classification report for it. 
# Then plot validation loss and accuracy for each epoch during training.

# In[ ]:


print("Average time per epoch for SNN:", finger_snn_time, end=" sec\n")
print(classification_report(*testing(finger_snn, fingerprints_dataloader['test'], device, snn_mode = True)))


# In[ ]:


print("Average time per epoch for CNN:", finger_cnn_time, end=" sec\n")
print(classification_report(*testing(finger_cnn, fingerprints_dataloader['test'], device, snn_mode=False)))


# In[ ]:


# Plot validation loss and accuracy for both models
plt.figure(figsize=(15, 5))

# Plot validation loss
plt.subplot(121)
plt.plot(finger_snn_loss)
plt.plot(finger_cnn_loss)

plt.ylabel('Validation loss')
plt.xlabel('Epochs')
plt.legend(['SNN', 'CNN'])

# Plot validation accuracy
plt.subplot(122)
plt.plot(finger_snn_accuracy)
plt.plot(finger_cnn_accuracy)

plt.ylabel('Validation accuracy')
plt.xlabel('Epochs')
plt.legend(['SNN', 'CNN'])

plt.suptitle('Validation loss and accuracy for fingerprints classification')
plt.show()


# ### EMNIST 
# 
# 

# #### Create models

# In[17]:


# Set parameters for this dataset

# Set input and output shapes
emnist_input_shape = next(iter(emnist_dataloader['train']))[0].shape
emnist_num_inputs = np.prod(emnist_input_shape[-2:])
emnist_num_outputs = len(np.unique(emnist_dataloader['test'].dataset.targets))

# Set path where to load the models
emnist_snn_load_path = load_path + "emnist_snn_model.pt"
emnist_cnn_load_path = load_path + "emnist_cnn_model.pt"


# In[18]:


# Create SNN model
emnist_snn = SNN(emnist_num_inputs, emnist_num_outputs).to(device)
emnist_snn_optimizer = torch.optim.Adam(emnist_snn.parameters(), lr=0.000085, betas=(0.9, 0.999))

print(emnist_snn)


# In[ ]:


# Train SNN model
emnist_snn_loss, emnist_snn_accuracy, emnist_snn_time = training(emnist_snn, emnist_dataloader['train'], 
                                  emnist_dataloader['val'], emnist_snn_optimizer, 
                                  criterion(), device, max_epochs, 
                                  snn_mode=True, num_steps=time_steps, path_load_model=emnist_snn_load_path)


# In[22]:


# Create CNN model
emnist_cnn = CNN(emnist_input_shape, emnist_num_outputs).to(device)
emnist_cnn_optimizer = torch.optim.Adam(emnist_cnn.parameters(), lr=0.000085, betas=(0.9, 0.999))

print(emnist_cnn)


# In[ ]:


# Train CNN model
emnist_cnn_loss, emnist_cnn_accuracy, emnist_cnn_time = training(emnist_cnn, emnist_dataloader['train'], 
                                  emnist_dataloader['val'], emnist_cnn_optimizer, 
                                  criterion(), device, max_epochs, snn_mode=False, path_load_model=emnist_cnn_load_path)


# #### Load best obtained models

# In[ ]:


# Load SNN model
emnist_snn = SNN(emnist_num_inputs, emnist_num_outputs)
emnist_snn.load_state_dict(torch.load(emnist_snn_load_path))
emnist_snn.to(device)

print(emnist_snn)


# In[ ]:


# Load CNN model
emnist_cnn = CNN(emnist_input_shape, emnist_num_outputs)
emnist_cnn.load_state_dict(torch.load(emnist_cnn_load_path))
emnist_cnn.to(device)

print(emnist_cnn)


# #### Analisys of the results obtained by the models
# For each model print the avarage consumed time per epoch, number of correctly classified images and accuracy for test set, as well as classification report for it. 
# Then plot validation loss and accuracy for each epoch during training.

# In[ ]:


print("Average time per epoch for SNN:", emnist_snn_time, end=" sec\n")
print(classification_report(*testing(emnist_snn, emnist_dataloader['test'], device, snn_mode = True)))


# In[ ]:


print("Average time per epoch for CNN:", emnist_cnn_time, end=" sec\n")
print(classification_report(*testing(emnist_cnn, emnist_dataloader['test'], device, snn_mode=False)))


# In[ ]:


# Plot validation loss and accuracy for both models
plt.figure(figsize=(15, 5))

# Plot validation loss
plt.subplot(121)
plt.plot(emnist_snn_loss)
plt.plot(emnist_cnn_loss)

plt.ylabel('Validation loss')
plt.xlabel('Epochs')
plt.legend(['SNN', 'CNN'])

# Plot validation accuracy
plt.subplot(122)
plt.plot(emnist_snn_accuracy)
plt.plot(emnist_cnn_accuracy)

plt.ylabel('Validation accuracy')
plt.xlabel('Epochs')
plt.legend(['SNN', 'CNN'])

plt.suptitle('Validation loss and accuracy for digits classification')
plt.show()


# ### Fashion MNIST
# 
# 

# #### Create models

# In[33]:


# Set parameters for this dataset

# Set input and output shapes
fashion_input_shape = next(iter(fashion_dataloader['train']))[0].shape
fashion_num_inputs = np.prod(fashion_input_shape[-2:])
fashion_num_outputs = len(np.unique(fashion_dataloader['test'].dataset.targets))

# Set path where to load the models
fashion_snn_load_path = load_path + "fashion_snn_model.pt"
fashion_cnn_load_path = load_path + "fashion_cnn_model.pt"


# In[34]:


# Create SNN model
fashion_snn = SNN(fashion_num_inputs, fashion_num_outputs).to(device)
fashion_snn_optimizer = torch.optim.Adam(fashion_snn.parameters(), lr=0.000085, betas=(0.9, 0.999))

print(fashion_snn)


# In[ ]:


# Train SNN model
fashion_snn_loss, fashion_snn_accuracy, fashion_snn_time = training(fashion_snn, fashion_dataloader['train'], 
                                  fashion_dataloader['val'], fashion_snn_optimizer, 
                                  criterion(), device, max_epochs, 
                                  snn_mode=True, num_steps=time_steps, path_load_model=fashion_snn_load_path)


# In[27]:


# Create CNN model
fashion_cnn = CNN(fashion_input_shape, fashion_num_outputs).to(device)
fashion_cnn_optimizer = torch.optim.Adam(fashion_cnn.parameters(), lr=0.000085, betas=(0.9, 0.999))

print(fashion_cnn)


# In[ ]:


# Train CNN model
fashion_cnn_loss, fashion_cnn_accuracy, fashion_cnn_time = training(fashion_cnn, fashion_dataloader['train'], 
                                  fashion_dataloader['val'], fashion_cnn_optimizer, 
                                  criterion(), device, max_epochs, snn_mode=False, path_load_model=fashion_cnn_load_path)


# #### Load best obtained models

# In[ ]:


# Load SNN model
fashion_snn = SNN(fashion_num_inputs, fashion_num_outputs)
fashion_snn.load_state_dict(torch.load(fashion_snn_load_path))
fashion_snn.to(device)

print(fashion_snn)


# In[ ]:


# Load CNN model
fashion_cnn = CNN(fashion_input_shape, fashion_num_outputs)
fashion_cnn.load_state_dict(torch.load(fashion_cnn_load_path))
fashion_cnn.to(device)

print(fashion_cnn)


# #### Analisys of the results obtained by the models
# For each model print the avarage consumed time per epoch, number of correctly classified images and accuracy for test set, as well as classification report for it. 
# Then plot validation loss and accuracy for each epoch during training.

# In[ ]:


print("Average time per epoch for SNN:", fashion_snn_time, end=" sec\n")
print(classification_report(*testing(fashion_snn, fashion_dataloader['test'], device, snn_mode = True)))


# In[ ]:


print("Average time per epoch for CNN:", fashion_cnn_time, end=" sec\n")
print(classification_report(*testing(fashion_cnn, fashion_dataloader['test'], device, snn_mode=False)))


# In[ ]:


# Plot validation loss and accuracy for both models
plt.figure(figsize=(15, 5))

# Plot validation loss
plt.subplot(121)
plt.plot(fashion_snn_loss)
plt.plot(fashion_cnn_loss)

plt.ylabel('Validation loss')
plt.xlabel('Epochs')
plt.legend(['SNN', 'CNN'])

# Plot validation accuracy
plt.subplot(122)
plt.plot(fashion_snn_accuracy)
plt.plot(fashion_cnn_accuracy)

plt.ylabel('Validation accuracy')
plt.xlabel('Epochs')
plt.legend(['SNN', 'CNN'])

plt.suptitle('Validation loss and accuracy for fashion products classification')
plt.show()


# ## Results
# 

# For SOCOFing dataset SNN model outperformed the CNN model and obtained the 98% accuracy, in comparison with 83% for CNN. For both EMNIST, and Fashion-MNIST datasets SNN and CNN showed approximately the same results: 98% accuracy for EMNIST and 86% accuracy for Fashion-MNIST. However, training of one epoch for SNN was, on average, 1.5 times slower than for CNN.
