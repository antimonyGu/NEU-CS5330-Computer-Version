# Shi Zhang
# experiment_train.py include the functions for training and evaluating the experiment network

# import statements
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from experiment_network import ExperimentNetwork

# Function to train the network and calculate the average loss
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Function to calculate the accuracy of the model
def evaluate_accuracy(model, device, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return correct / len(test_loader.dataset)

# Function to save the trained model
def save_model(model, save_dir, model_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir, model_name))

