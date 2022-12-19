from __future__ import print_function

import os
import time

import matplotlib.pyplot as plt
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
        loss_tracker=None,
        seed=42):

    torch.manual_seed(seed)
    mp.set_start_method('spawn', force=True)
    model = model.to(device)
    processes = []

    for process_num in range(num_processes):
        p = mp.Process(target=train_process, args=(process_num, seed, model, optimizer, device, epochs_per_process,
                                                   train_dataloader, log_interval, loss_tracker))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    if loss_tracker:
        return loss_tracker


# Start an individual process to train the (shared) model
def train_process(process_num, seed, model, optimizer, device, epochs_per_process, train_dataloader, log_interval, loss_tracker=None):
    torch.manual_seed(seed + process_num)

    for epoch in range(1, epochs_per_process + 1):
        train_epoch(process_num, epoch, model, device, train_dataloader,
                    optimizer, log_interval, loss_tracker)


# Start an individual epoch for a particular process
def train_epoch(process_num, epoch, model, device, train_dataloader, optimizer, log_interval, loss_tracker=None):
    model.train()
    pid = os.getpid()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = criterion(output, target.to(device))

        # Optional loss tracking for charting
        if loss_tracker is not None and batch_idx % log_interval == 0:
            loss_tracker.append(loss.item())
        # Optional logging
        if log_interval != -1 and batch_idx % log_interval == 0:
            print('Process {}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                process_num, epoch, batch_idx *
                len(data), len(train_dataloader.dataset),
                100. * batch_idx / len(train_dataloader), loss.item()))

        loss.backward()
        optimizer.step()


# Evaluate the model on test dataset
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


# Log elapsed time
def log_elapsed_time(num_processes, epochs_per_process, start, end):
    print(
        f'Processes: {num_processes}, Epochs: {num_processes} * {epochs_per_process} = {num_processes * epochs_per_process}, Elapsed time: {end - start}')


def chart_losses(losses_dict, title='Comparing Training Losses Across Configurations'):
    for configuration, losses in losses_dict.items():
        batches_range = range(len(losses))
        plt.plot(batches_range, losses, label=configuration)

    plt.xlabel('Batches')
    plt.ylabel('Training Loss')
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Example workflow for comparing the speed/performance of a model built
    # with different configurations of processes and how epochs get split
    # across them. Here, we consider a simple Neural Network trained on MNIST.
    # One version of the model was trained using 1 process and 4 (sequential) epochs,
    # while the other version was trained using 4 processes, with 1 epoch each,
    # executed in parallel.

    # Create dataloaders
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

    # Specify a '4 processes by 1 epoch' model
    start1 = time.time()
    model = SimpleNN()
    num_processes = 4
    epochs_per_process = 1
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    device = 'cpu'
    log_interval = 10
    loss_tracker_4_by_1 = mp.Manager().list()

    train_model(model, train_dataloader, num_processes, epochs_per_process,
                optimizer, device, log_interval, loss_tracker_4_by_1)
    test(model, device, test_dataloader)
    end1 = time.time()

    # Specify a '1 process by 4 epochs' model
    start2 = time.time()
    model = SimpleNN()
    num_processes = 1
    epochs_per_process = 4
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    device = 'cpu'
    log_interval = 10
    loss_tracker_1_by_4 = mp.Manager().list()

    train_model(model, train_dataloader, num_processes, epochs_per_process,
                optimizer, device, log_interval, loss_tracker_1_by_4)
    test(model, device, test_dataloader)
    end2 = time.time()

    # Compare time
    log_elapsed_time(num_processes, epochs_per_process, start1, end1)
    log_elapsed_time(num_processes, epochs_per_process, start2, end2)

    # Compare training loss via chart
    chart_losses({
        '4 processes x 1 epoch NN': loss_tracker_4_by_1,
        '1 process x 4 epochs NN': loss_tracker_1_by_4
    })
