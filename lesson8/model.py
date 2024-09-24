import torch
import torch.nn.functional as F


class LinearRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.l1 = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, 784)
        y_pred = self.l1(x)
        return y_pred


class LinearRegression2(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression2, self).__init__()
        self.l1 = torch.nn.Linear(input_dim, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, output_dim)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        y_pred = self.l5(x)
        return y_pred


# class NET(torch.nn.Module):
#     def __init__(self):
#         super(NET, self).__init__()
#         self.conv1 = torch.nn.Conv2d(1, 10, 3, stride=1, padding=0)
#         self.maxpool1 = torch.nn.MaxPool2d(2)

#         self.conv2 = torch.nn.Conv2d(10, 32, 4, stride=1, padding=0)
#         self.maxpool2 = torch.nn.MaxPool2d(2)

#         self.conv3 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
#         self.flattern = torch.nn.Flatten()  # 64 *5 *5
#         self.l1 = torch.nn.Linear(1600, 128)
#         self.l2 = torch.nn.Linear(128, 10)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.maxpool1(x)
#         x = F.relu(self.conv2(x))
#         x = self.maxpool2(x)
#         x = F.relu(self.conv3(x))
#         x = self.flattern(x)
#         x = self.l1(x)
#         x = self.l2(x)
#         return x

import torch.nn as nn


class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        # First convolutional layer (1 input channel, 10 output channels, 3x3 kernel)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(2)  # 2x2 max pooling

        # Second convolutional layer (10 input channels, 32 output channels, 4x4 kernel)
        self.conv2 = nn.Conv2d(10, 32, kernel_size=4, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(2)  # 2x2 max pooling

        # Third convolutional layer (32 input channels, 64 output channels, 3x3 kernel)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Flatten layer to flatten the 2D matrix into a 1D vector
        self.flatten = nn.Flatten()

        # Fully connected layers (FC layers)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # 1600 inputs to 128 hidden units
        self.fc2 = nn.Linear(
            128, 10
        )  # 128 hidden units to 10 output units (class predictions)

        # Optional: Dropout (to prevent overfitting)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Apply first conv layer, ReLU activation, and max pooling
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)

        # Apply second conv layer, ReLU activation, and max pooling
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)

        # Apply third conv layer and ReLU activation
        x = F.relu(self.conv3(x))

        # Flatten the feature map into a 1D vector
        x = self.flatten(x)

        # Apply first fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))

        # Apply dropout (optional, helps with regularization)
        x = self.dropout(x)

        # Apply second fully connected layer for classification (logits output)
        x = self.fc2(x)

        return x


class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()

        # First convolutional layer (input: 1 channel, output: 16 channels, 3x3 kernel)
        self.conv1 = nn.Conv2d(
            1, 16, kernel_size=3, stride=1, padding=1
        )  # Output: (16, 28, 28)

        # Second convolutional layer (input: 16 channels, output: 32 channels, 3x3 kernel)
        self.conv2 = nn.Conv2d(
            16, 32, kernel_size=3, stride=1, padding=1
        )  # Output: (32, 14, 14)

        # Max pooling layer (2x2 kernel, reduces size by half)
        self.pool = nn.MaxPool2d(2, 2)  # Reduces each dimension by 2

        # Fully connected layers
        self.fc1 = nn.Linear(
            32 * 7 * 7, 128
        )  # 32 channels * 7x7 image size after pooling
        self.fc2 = nn.Linear(128, 10)  # 10 output classes (digits 0-9)

    def forward(self, x):
        # Pass through conv1, ReLU, and then max pooling
        x = self.pool(F.relu(self.conv1(x)))  # Output: (16, 14, 14)

        # Pass through conv2, ReLU, and then max pooling
        x = self.pool(F.relu(self.conv2(x)))  # Output: (32, 7, 7)

        # Flatten the output from (batch_size, 32, 7, 7) to (batch_size, 32*7*7)
        x = x.view(-1, 32 * 7 * 7)  # Flatten the tensor

        # Fully connected layer 1 with ReLU
        x = F.relu(self.fc1(x))

        # Output layer (no softmax, because we'll use CrossEntropyLoss which includes softmax)
        x = self.fc2(x)

        return x
