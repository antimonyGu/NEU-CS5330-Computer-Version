# Author: Zhizhou Gu created at 2023/11/11
# to finish task2 examine the network
import torch
from network import MyNetwork
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from cv2 import filter2D
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import csv

# resume network
continued_network = MyNetwork()
network_state_dict = torch.load('./mnist_model.pth')
continued_network.load_state_dict(network_state_dict)

cols, rows = 4, 3

# fig = plt.figure(layout='constrained', figsize=(10, 4))
fig = plt.figure()

# breakpoint()

for i in range(continued_network.conv1.weight.shape[0]):
    col = (i + 1) // cols
    row = (i + 1) % cols
    
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title("filter {count}".format(count=str(i + 1)))

    # clear x and y ticks
    ax.set_xticks([])
    ax.set_yticks([])

    filter = continued_network.conv1.weight[i, 0]

    ax.imshow(filter.detach().numpy())

# B.
with torch.no_grad():
    # TBD: check weather load data method is right

    # MNIST Dataset transformation: Converts images to tensors and normalizes them
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load the training data
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    batch = next(iter(train_loader))
    image, label = batch

    img_list = []
    weights = continued_network.conv1.weight
    for i in range(10):
        img_list.append(weights[i, 0].numpy())
        filtered_img = filter2D(image[0].numpy()[0], -1, weights[i, 0].numpy())
        img_list.append(filtered_img)

    fig, axs = plt.subplots(5, 4, figsize=(9, 6), subplot_kw={'xticks': [], 'yticks': []})
    for i in range(20):
        row = i // 4
        col = i % 4
        axs[row, col].imshow(img_list[i], cmap='gray')
    plt.show()
