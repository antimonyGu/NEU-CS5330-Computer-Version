# Shi Zhang
# experiment_main.py is the entry point of the experiment part of the project that calls functions from other experiment files.
# This script handles data loading, model initialization, training, and evaluation.
# Additionally, it includes functionality for visualizing data samples and plotting training and testing loss.

# Import statements
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from experiment_network import ExperimentNetwork
from experiment_train import train, evaluate_accuracy, save_model
import csv

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNIST or MNIST Fashion dataset loaders
train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./data_fashion', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
    batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./data_fashion', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
    batch_size=1000, shuffle=True)

# Experiment configurations
num_filters_options = [5, 10, 20, 30]
num_fc_nodes_options = [32, 64, 128, 256]
dropout_rate_options = [0.3, 0.5, 0.7, 0.9]

# Dictionary to store experiment results
experiment_results = []

# Define the number of epochs for training
num_epochs = 10

# Linear search experiment loop
# Iterate over each parameter while holding others constant
for num_fc_nodes in num_fc_nodes_options:
    for dropout_rate in dropout_rate_options:
        for num_filters in num_filters_options:
            model = ExperimentNetwork(num_conv_filters=num_filters, num_fc_nodes=num_fc_nodes, dropout_rate=dropout_rate).to(device)
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
            start_time = time()
            total_loss = 0

            for epoch in range(1, num_epochs + 1):
                epoch_loss = train(model, device, train_loader, optimizer, epoch)
                total_loss += epoch_loss

            training_time = time() - start_time
            average_loss = total_loss / num_epochs
            accuracy = evaluate_accuracy(model, device, test_loader)

            experiment_results.append({
                'num_filters': num_filters,
                'num_fc_nodes': num_fc_nodes,
                'dropout_rate': dropout_rate,
                'accuracy': accuracy,
                'training_time': training_time,
                'loss': average_loss
            })

            model_name = f'model_filters_{num_filters}_nodes_{num_fc_nodes}_dropout_{dropout_rate}.pth'
            save_model(model, './saved_models', model_name)

# Visualization of results

# Define the new keys for plotting
plotting_keys = ['num_filters', 'num_fc_nodes', 'dropout_rate']

# Plotting accuracy vs. each parameter
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, key in enumerate(plotting_keys):
    axes[i].scatter([r[key] for r in experiment_results], [r['accuracy'] for r in experiment_results])
    axes[i].set_xlabel(key)
    axes[i].set_ylabel('Accuracy')
    axes[i].set_title(f'Accuracy vs {key}')

plt.tight_layout()
plt.show()

# Plotting training time vs. each parameter
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, key in enumerate(plotting_keys):
    axes[i].scatter([r[key] for r in experiment_results], [r['training_time'] for r in experiment_results])
    axes[i].set_xlabel(key)
    axes[i].set_ylabel('Training Time (s)')
    axes[i].set_title(f'Training Time vs {key}')

plt.tight_layout()
plt.show()

# Plotting loss vs. each parameter
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, key in enumerate(plotting_keys):
    axes[i].scatter([r[key] for r in experiment_results], [r['loss'] for r in experiment_results])
    axes[i].set_xlabel(key)
    axes[i].set_ylabel('Average Loss')
    axes[i].set_title(f'Average Loss vs {key}')

plt.tight_layout()
plt.show()

# Saving experiment results to a CSV filees
with open('experiment_results.csv', 'w', newline='') as csvfile:
    fieldnames = ['num_filters', 'num_fc_nodes', 'dropout_rate', 'accuracy', 'training_time', 'loss']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for data in experiment_results:
        writer.writerow(data)

print("Experiment results saved to 'experiment_results.csv'")