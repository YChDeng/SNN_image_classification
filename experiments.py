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