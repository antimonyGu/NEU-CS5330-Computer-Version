# Author: Zhizhou Gu

from network import MyNetwork
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from experiment_train import save_model

# (1) generate the MNIST network (you should import your code from task 1),

model = MyNetwork()

# (2) read an existing model from a file and load the pre-trained weights, 

model.load_state_dict(torch.load('./mnist_model.pth'))

# (3) freeze the network weights, and 
for param in model.parameters():
    param.requires_grad = False

# do not freeze the last layer to train the model
model.fc2 = nn.Linear(50, 3)

print('model', model)

# (4) replace the last layer with a new Linear layer with three nodes. 

# greek data set transform
class GreekTrainTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )
    
class GreekTestTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.Resize(size = (128, 128))(x)
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )

# load training data set
training_set_path = "./greek_train"
greek_train = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder( training_set_path,
                                        transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                                                                                    GreekTrainTransform(),
                                                                                    torchvision.transforms.Normalize(
                                                                                        (0.1307,), (0.3081,) ) ] ) ),
    batch_size = 5,
    shuffle = True )

# TBD: crop and rescale test set
training_set_path = "./greek_test"
greek_test = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder( training_set_path,
                                        transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                                                                                    GreekTestTransform(),
                                                                                    torchvision.transforms.Normalize(
                                                                                        (0.1307,), (0.3081,) ) ] ) ),
    batch_size = 2,
    shuffle = True )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# Function to evaluate the test_accuracy of the model
def evaluate_accuracy(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return correct / len(test_loader.dataset)

# Initialize lists to keep track of losses and the number of examples seen
examples_count = []
accuracys = []
training_errors = []
testing_errors = []
total_training_data_count = 0
correct_training_data_count = 0

# Training loop for a specified number of epochs
for epoch in range(1, 3 + 1):
    model.train()

    for batch_idx, (data, target) in enumerate(greek_train):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)  # Use negative log likelihood loss
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(output.data, 1)

        total_training_data_count += target.size(0)
        correct_training_data_count += (predicted == target).sum().item()

        training_error = correct_training_data_count / total_training_data_count
        training_errors.append(training_error)
        
        test_accuracy = evaluate_accuracy(model, device, greek_test)
        accuracys.append(test_accuracy)
        testing_errors.append(1 - test_accuracy)

# save model
save_model(model, 'saved_models', 'finetuned_model.pth')

# draw Testing Errors/examples plot
plt.plot(range(1, len(training_errors) + 1), training_errors)
plt.title("Training Errors Chart")
plt.xlabel("number of training examples")
plt.ylabel("Training Errors")
plt.show()

# draw Testing Errors/examples plot
plt.plot(range(1, len(testing_errors) + 1), testing_errors)
plt.title("Testing Errors Chart")
plt.xlabel("number of training examples")
plt.ylabel("Testing Errors")
plt.show()

