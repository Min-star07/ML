# -*- coding: utf-8 -*-

"""
@author: Min Li
@e-mail: limin93@ihep.ac.cn
@file: duofenlei.py
@time: 2023/8/23 21:05
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

# 1 Prepare dataset
batch_size = 64
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

train_dataset = datasets.MNIST(
    root="./dataset", train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(
    root="./dataset", train=False, download=True, transform=transform
)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)


# 2 MODEL DESIGN
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooing = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooing(self.conv1(x)))
        x = F.relu(self.pooing(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


model = Net()
# 放到GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
# 3. Construct Loss and Optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# 4 Train and Test
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()
        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print(
                "%d, %5d] loss: %.3f" % (epoch + 1, batch_idx + 1, running_loss / 300)
            )
            running_loss == 0.0


def Test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print("Accuracy on test set : %d %%" % (100 * correct / total))


if __name__ == "__main__":
    for epoch in range(10):
        train(epoch)
        if epoch % 10 == 9:
            Test()
