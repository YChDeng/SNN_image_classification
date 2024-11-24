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

#---------------------下载整理fingerprints数据集---------------------#

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

#---------------------数据预处理---------------------#

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

#---------------------数据集的动画化和可视化---------------------#

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
#---------------------Experiments---------------------#

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
    
# Create models for different datasets

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


# 1.Fingerprints

# create models 

# Set parameters for this dataset

# Set input and output shapes
finger_input_shape = next(iter(fingerprints_dataloader['train']))[0].shape
finger_num_inputs = np.prod(finger_input_shape[-2:])
finger_num_outputs = len(np.unique(fingerprints_dataloader['train'].dataset.targets))

# Set path where to load the models
finger_snn_load_path = load_path + "finger_snn_model.pt"
finger_cnn_load_path = load_path + "finger_cnn_model.pt"

# Create SNN model
finger_snn = SNN(finger_num_inputs, finger_num_outputs).to(device)
finger_snn_optimizer = torch.optim.Adam(finger_snn.parameters(), lr=0.000085, betas=(0.9, 0.999))

print(finger_snn)

# Train SNN model
finger_snn_loss, finger_snn_accuracy, finger_snn_time = training(finger_snn, fingerprints_dataloader['train'], 
                                  fingerprints_dataloader['val'], finger_snn_optimizer, 
                                  criterion(), device, max_epochs, 
                                  snn_mode=True, num_steps=time_steps, path_load_model=finger_snn_load_path)

# Create CNN model
finger_cnn = CNN(finger_input_shape, finger_num_outputs).to(device)
finger_cnn_optimizer = torch.optim.Adam(finger_cnn.parameters(), lr=0.000085, betas=(0.9, 0.999))

print(finger_cnn)

# Train CNN model
finger_cnn_loss, finger_cnn_accuracy, finger_cnn_time = training(finger_cnn, fingerprints_dataloader['train'], 
                                  fingerprints_dataloader['val'], finger_cnn_optimizer, 
                                  criterion(), device, max_epochs, snn_mode=False, path_load_model=finger_cnn_load_path)

# Load SNN model
finger_snn = SNN(finger_num_inputs, finger_num_outputs)
finger_snn.load_state_dict(torch.load(finger_snn_load_path))
finger_snn.to(device)

print(finger_snn)

# Load CNN model
finger_cnn = CNN(finger_input_shape, finger_num_outputs)
finger_cnn.load_state_dict(torch.load(finger_cnn_load_path))
finger_cnn.to(device)

print(finger_cnn)

# Analisys of the results obtained by the models
print("Average time per epoch for SNN:", finger_snn_time, end=" sec\n")
print(classification_report(*testing(finger_snn, fingerprints_dataloader['test'], device, snn_mode = True)))

print("Average time per epoch for CNN:", finger_cnn_time, end=" sec\n")
print(classification_report(*testing(finger_cnn, fingerprints_dataloader['test'], device, snn_mode=False)))

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

# EMNIST

#Create models

# Set parameters for this dataset

# Set input and output shapes
emnist_input_shape = next(iter(emnist_dataloader['train']))[0].shape
emnist_num_inputs = np.prod(emnist_input_shape[-2:])
emnist_num_outputs = len(np.unique(emnist_dataloader['test'].dataset.targets))

# Set path where to load the models
emnist_snn_load_path = load_path + "emnist_snn_model.pt"
emnist_cnn_load_path = load_path + "emnist_cnn_model.pt"

# Create SNN model
emnist_snn = SNN(emnist_num_inputs, emnist_num_outputs).to(device)
emnist_snn_optimizer = torch.optim.Adam(emnist_snn.parameters(), lr=0.000085, betas=(0.9, 0.999))

print(emnist_snn)

# Train SNN model
emnist_snn_loss, emnist_snn_accuracy, emnist_snn_time = training(emnist_snn, emnist_dataloader['train'], 
                                  emnist_dataloader['val'], emnist_snn_optimizer, 
                                  criterion(), device, max_epochs, 
                                  snn_mode=True, num_steps=time_steps, path_load_model=emnist_snn_load_path)

# Create CNN model
emnist_cnn = CNN(emnist_input_shape, emnist_num_outputs).to(device)
emnist_cnn_optimizer = torch.optim.Adam(emnist_cnn.parameters(), lr=0.000085, betas=(0.9, 0.999))

print(emnist_cnn)

# Train CNN model
emnist_cnn_loss, emnist_cnn_accuracy, emnist_cnn_time = training(emnist_cnn, emnist_dataloader['train'], 
                                  emnist_dataloader['val'], emnist_cnn_optimizer, 
                                  criterion(), device, max_epochs, snn_mode=False, path_load_model=emnist_cnn_load_path)

# load best obtained models

# Load SNN model
emnist_snn = SNN(emnist_num_inputs, emnist_num_outputs)
emnist_snn.load_state_dict(torch.load(emnist_snn_load_path))
emnist_snn.to(device)

print(emnist_snn)

# Load CNN model
emnist_cnn = CNN(emnist_input_shape, emnist_num_outputs)
emnist_cnn.load_state_dict(torch.load(emnist_cnn_load_path))
emnist_cnn.to(device)

print(emnist_cnn)

# Analisys of the results obtained by the models

print("Average time per epoch for SNN:", emnist_snn_time, end=" sec\n")
print(classification_report(*testing(emnist_snn, emnist_dataloader['test'], device, snn_mode = True)))

print("Average time per epoch for CNN:", emnist_cnn_time, end=" sec\n")
print(classification_report(*testing(emnist_cnn, emnist_dataloader['test'], device, snn_mode=False)))

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

# Fashion MNIST

# Create models

# Set parameters for this dataset

# Set input and output shapes
fashion_input_shape = next(iter(fashion_dataloader['train']))[0].shape
fashion_num_inputs = np.prod(fashion_input_shape[-2:])
fashion_num_outputs = len(np.unique(fashion_dataloader['test'].dataset.targets))

# Set path where to load the models
fashion_snn_load_path = load_path + "fashion_snn_model.pt"
fashion_cnn_load_path = load_path + "fashion_cnn_model.pt"

# Create SNN model
fashion_snn = SNN(fashion_num_inputs, fashion_num_outputs).to(device)
fashion_snn_optimizer = torch.optim.Adam(fashion_snn.parameters(), lr=0.000085, betas=(0.9, 0.999))

print(fashion_snn)

# Train SNN model
fashion_snn_loss, fashion_snn_accuracy, fashion_snn_time = training(fashion_snn, fashion_dataloader['train'], 
                                  fashion_dataloader['val'], fashion_snn_optimizer, 
                                  criterion(), device, max_epochs, 
                                  snn_mode=True, num_steps=time_steps, path_load_model=fashion_snn_load_path)

# Create CNN model
fashion_cnn = CNN(fashion_input_shape, fashion_num_outputs).to(device)
fashion_cnn_optimizer = torch.optim.Adam(fashion_cnn.parameters(), lr=0.000085, betas=(0.9, 0.999))

print(fashion_cnn)

# Train CNN model
fashion_cnn_loss, fashion_cnn_accuracy, fashion_cnn_time = training(fashion_cnn, fashion_dataloader['train'], 
                                  fashion_dataloader['val'], fashion_cnn_optimizer, 
                                  criterion(), device, max_epochs, snn_mode=False, path_load_model=fashion_cnn_load_path)

# Load best obtained models

# Load SNN model
fashion_snn = SNN(fashion_num_inputs, fashion_num_outputs)
fashion_snn.load_state_dict(torch.load(fashion_snn_load_path))
fashion_snn.to(device)

print(fashion_snn)

# Load CNN model
fashion_cnn = CNN(fashion_input_shape, fashion_num_outputs)
fashion_cnn.load_state_dict(torch.load(fashion_cnn_load_path))
fashion_cnn.to(device)

print(fashion_cnn)

# Analisys of the results obtained by the models

print("Average time per epoch for SNN:", fashion_snn_time, end=" sec\n")
print(classification_report(*testing(fashion_snn, fashion_dataloader['test'], device, snn_mode = True)))

print("Average time per epoch for CNN:", fashion_cnn_time, end=" sec\n")
print(classification_report(*testing(fashion_cnn, fashion_dataloader['test'], device, snn_mode=False)))

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


