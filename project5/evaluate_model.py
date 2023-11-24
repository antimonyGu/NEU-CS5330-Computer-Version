# Shi Zhang
# evaluate_model.py loads the neural network model and evaluates it on a test dataset

# Import statements
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from network import MyNetwork
from PIL import Image

# Load the test dataset
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=10, shuffle=False)

# Initialize the network and load trained model
model = MyNetwork()
model.load_state_dict(torch.load('mnist_model.pth')) 
model.eval()  # Set the model to evaluation mode

# Process the first batch of test images
data, target = next(iter(test_loader))
output = model(data)
_, predicted = torch.max(output, 1)

# Print network output, predicted and actual labels
for i in range(len(data)):
    print(f"Image {i + 1}")
    formatted_output = [f"{value:.2f}" for value in output[i].data.numpy()]
    print("Network Output:", formatted_output)
    print("Predicted Label:", predicted[i].item())
    print("Actual Label:", target[i].item())
    print()

# Plot the first 9 images
fig, axs = plt.subplots(3, 3, figsize=(9, 9))
for i, ax in enumerate(axs.flatten()):
    if i < 9:
        ax.imshow(data[i][0], cmap='gray', interpolation='none')
        ax.set_title(f"Predicted: {predicted[i].item()}")
        ax.set_xticks([])
        ax.set_yticks([])
plt.show()

# Evaluate the model on new handwritten digits
# Define the transform for preprocessing the images
transform = transforms.Compose([
    transforms.Grayscale(), # Convert to grayscale
    # transforms.Resize((28, 28)), # Resize to 28x28
    transforms.ToTensor(), # Convert to a torch tensor
    transforms.Normalize((0.1307,), (0.3081,)) # Normalize like MNIST
])

# Digit image paths assuming they are stored in a 'digits' subdirectory
digit_paths = [f'digits/digit_{i}.png' for i in range(10)]

# Process each digit and run through the network
for path in digit_paths:
    # Load the image
    image = Image.open(path)

    # Apply the transformations
    image = transform(image)

    # Add a batch dimension (B x C x H x W)
    image = image.unsqueeze(0)

    # Forward pass through the network
    output = model(image)

    # Print the output
    formatted_output = [f"{value:.2f}" for value in output[0].data.numpy()]
    print(f"Network Output for {path}: ", formatted_output)
    predicted = torch.argmax(output, 1)
    print("Predicted Label:", predicted.item())
    print()