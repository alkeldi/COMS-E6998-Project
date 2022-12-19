from __future__ import print_function

import os
import time

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from simple_nn import SimpleNN
from torchvision import datasets, transforms


# Start processes that each train the (shared) model in parallel
def train_model(
        model,
        train_dataloader,
        num_processes,
        epochs_per_process,
        optimizer,
        device='cpu',
        log_interval=10,  # -1 for no logs
        seed=42):

    torch.manual_seed(seed)
    mp.set_start_method('spawn', force=True)
    model = model.to(device)
    processes = []

    for process_num in range(num_processes):
        p = mp.Process(target=train_process, args=(process_num, seed, model, optimizer, device, epochs_per_process,
                                                   train_dataloader, log_interval))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


# Start an individual process to train the (shared) model
def train_process(process_num, seed, model, optimizer, device, epochs_per_process, train_dataloader, log_interval):
    torch.manual_seed(seed + process_num)

    for epoch in range(1, epochs_per_process + 1):
        train_epoch(process_num, epoch, model, device, train_dataloader,
                    optimizer, log_interval)


# Start an individual epoch for a particular process
def train_epoch(process_num, epoch, model, device, train_dataloader, optimizer, log_interval):
    model.train()
    pid = os.getpid()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = criterion(output, target.to(device))
        loss.backward()
        optimizer.step()
        if log_interval != -1 and batch_idx % log_interval == 0:
            print('Process {}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                process_num, epoch, batch_idx *
                len(data), len(train_dataloader.dataset),
                100. * batch_idx / len(train_dataloader), loss.item()))


def test(model, device, test_dataloader, seed=42):
    torch.manual_seed(seed)

    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_dataloader:
            output = model(data.to(device))
            # sum up batch loss
            test_loss += criterion(output, target.to(device)).item()
            pred = output.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.to(device)).sum().item()

    test_loss /= len(test_dataloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)))


if __name__ == '__main__':
    start = time.time()

    model = SimpleNN()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST('../data', train=False,
                                  transform=transform)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=True)

    num_processes = 2

    epochs_per_process = 2

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    device = 'cpu'

    log_interval = 10

    train_model(model, train_dataloader, num_processes, epochs_per_process,
                optimizer, device, log_interval)

    test(model, device, test_dataloader)

    end = time.time()
    print(
        f'Processes: {num_processes}, Epochs: {num_processes} * {epochs_per_process} = {num_processes * epochs_per_process}, Elapsed time: {end - start}')
