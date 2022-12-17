import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import matplotlib.pyplot as plt


def training_loop_with_device_switching(model, trainloader, criterion, optimizer, epochs, print_each=200):
    time_begin = time.time()

    # get devices
    if torch.cuda.is_available(): 
        ACCELERATOR = torch.device("cuda")
    elif torch.backends.mps.is_available(): 
        ACCELERATOR = torch.device("mps")
    
    # move model to gpu/mps
    model = model.to(ACCELERATOR)
    time_at_epoch = []

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(ACCELERATOR)
            labels = labels.to(ACCELERATOR)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % print_each == (print_each-1):
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / print_each:.3f}')
                running_loss = 0.0

        current_time = time.time() - time_begin
        time_at_epoch.append(current_time)

    return time_at_epoch


# hyper parameters
BATCH_SIZE = 64
EPOCHS = 10

# data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

# model info
model = models.resnet34()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# start
time_at_epoch = training_loop_with_device_switching(model, trainloader, criterion, optimizer, EPOCHS)
epochs = range(1, len(time_at_epoch)+1)
plt.plot(epochs, time_at_epoch)
plt.xlabel('epoch')
plt.ylabel('time')
plt.title("Training Time vs. Epoch (Normal)")
plt.show()

