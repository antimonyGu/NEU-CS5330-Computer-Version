# Shi Zhang
# main.py is the entry point of the project that calls functions from other files.
# This script handles data loading, model initialization, training, and evaluation.
# Additionally, it includes functionality for visualizing data samples and plotting training and testing loss.

# Import statements
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from network import MyNetwork
from train import save_model
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

# Function to visualize the first six digits from the MNIST dataset
def visualize_digits(train_loader):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 2
    for i in range(1, cols * rows + 1):
        batch = next(iter(train_loader))
        image, label = batch
        plt.subplot(rows, cols, i)
        plt.axis('off')
        plt.imshow(image[0].squeeze(), cmap='gray')
        plt.title(f'Label: {label[0]}')
    plt.show()

# Main function that sets up the data, trains the model, and saves it after training
def main():
    # Check if CUDA is available and set device to GPU if it is, else use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # MNIST Dataset transformation: Converts images to tensors and normalizes them
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load the training data
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Load the test data
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)

    # Visualize the first six digits of the MNIST dataset
    visualize_digits(train_loader)

    # Initialize the network and move it to the device (GPU or CPU)
    model = MyNetwork().to(device)
    
    # Define the optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    
    # Initialize lists to keep track of losses and the number of examples seen
    train_losses = []
    test_losses = []
    examples_count = []

    # Evaluate initial test loss before training
    model.eval()  # Set the model to evaluation mode
    initial_test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            initial_test_loss += F.nll_loss(output, target, reduction='sum').item()
    initial_test_loss /= len(test_loader.dataset)
    test_losses.append(initial_test_loss)
    model.train()  # Set the model back to training mode

    # Record the initial test loss
    print(f'Initial test loss: {initial_test_loss}')
    
    # Set the frequency to evaluate the test loss
    evaluation_frequency = 20000

    # Training loop for a specified number of epochs
    for epoch in range(1, 11): 
        model.train()  # Set the model to training mode

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)  # Use negative log likelihood loss
            loss.backward()
            optimizer.step()

            # Record training loss and number of examples seen
            if batch_idx % 10 == 0:
                train_losses.append(loss.item())
                examples_count.append((epoch - 1) * len(train_loader.dataset) + batch_idx * len(data))
            

        # Evaluate the test loss at the end of the epoch
        model.eval()  # Set the model to evaluation mode
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # Sum up batch loss
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        model.train()  # Set the model back to training mode

        # Print the average train loss and current test loss
        print(f'Epoch {epoch}, Train loss: {train_losses[-1]}, Test loss: {test_loss}')

    # After all epochs, plot the training loss and test loss
    plt.figure(figsize=(10, 5))
    plt.plot(examples_count, train_losses, 'b-', label='Train Loss')  # Blue line for training loss

    # Correct the x-values for plotting the test loss, starting with 0 for the initial test loss
    test_loss_x_values = [0] + [(i + 1) * len(train_loader.dataset) for i in range(epoch)]
    plt.plot(test_loss_x_values, test_losses, 'ro', label='Test Loss')  # Red dots for test loss

    plt.title('Negative Log Likelihood Loss over Number of Training Examples Seen')
    plt.xlabel('Number of Training Examples Seen')
    plt.ylabel('Negative Log Likelihood Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save the trained model to a file
    save_model(model, 'mnist_model.pth')

if __name__ == '__main__':
    main()

