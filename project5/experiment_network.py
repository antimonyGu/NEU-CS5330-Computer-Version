# Shi Zhang
# experiment_network.py contains the definition of the experiment convolutional neural network

# Import statements
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the network class with customizable parameters
class ExperimentNetwork(nn.Module):
    def __init__(self, num_conv_filters=10, num_fc_nodes=50, dropout_rate=0.5):
        super(ExperimentNetwork, self).__init__()
        # Convolutional layers with reduced kernel size or increased padding
        self.conv1 = nn.Conv2d(1, num_conv_filters, kernel_size=3, padding=1)  # Smaller kernel size and padding
        self.conv2 = nn.Conv2d(num_conv_filters, num_conv_filters * 2, kernel_size=3, padding=1)  # Consistent
        self.dropout = nn.Dropout2d(dropout_rate)

        # Calculate the size of the flattened features after conv layers
        feature_size = self._get_conv_output((1, 28, 28))

        # Fully connected layers
        self.fc1 = nn.Linear(feature_size, num_fc_nodes)
        self.fc2 = nn.Linear(num_fc_nodes, 10)  # Assuming 10 classes for MNIST

    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
