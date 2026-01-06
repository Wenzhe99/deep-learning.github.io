import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # Layer 1: Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,kernel_size=5, stride=1)
        # Layer 2: Pooling Layer
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Layer 3: Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        # Layer 4: Pooling Layer
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(self.relu(x))
        x = self.conv2(x)
        x = self.pool2(self.relu(x))
        x = x.view(-1, 16*5*5)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x