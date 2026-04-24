# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 10:04:01 2026

@author: oku-hiro
"""

import torch
import torchvision
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import model1

BATCH_SIZE = 16
transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(0.5, 0.5)])

trainsets = torchvision.datasets.MNIST(
    root="./", train=True, download=True, transform=transform)

testsets = torchvision.datasets.MNIST(
    root="./", train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(
    trainsets, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

testloader = torch.utils.data.DataLoader(
    testsets, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

device = torch.device("cpu")
model = model1.ConvNet()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-4,
                      momentum=0.9, weight_decay=5e-3)
EPOCH = 20
print(model)

train_loss = []
test_loss = []
test_acc = []
print(len(trainloader.dataset))

for epoch in range(EPOCH):
    sum_loss = 0.0
    for (inputs, labels) in trainloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
    train_loss.append(sum_loss/len(trainloader.dataset))

    sum_loss = 0.0
    sum_correct = 0.0
    for (inputs, labels) in testloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        # print(predicted)
        loss = criterion(outputs, labels)
        sum_loss += loss.item()
        sum_correct += (predicted == labels).sum().item()
    test_loss.append(sum_loss/len(testloader.dataset))

x = np.linspace(0, EPOCH, EPOCH)
fig, ax = plt.subplots()
ax.plot(x, train_loss, color="b", linewidth=1.0)
ax.plot(x, test_loss, color="k", linewidth=1.0)
ax.grid(True)
ax.plot()
