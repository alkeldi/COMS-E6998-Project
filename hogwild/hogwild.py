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


# Chart the training losses over each batch for one or more models. The keys of the
# dict are model labels (ex: '4 process x 1 epoch NN'), while the value are a list
# of loss values.
def chart_losses(losses_dict, title='Comparing Training Losses Across Configurations'):
    for configuration, losses in losses_dict.items():
        batches_range = range(len(losses))
        plt.plot(batches_range, losses, label=configuration)

    plt.xlabel('Batches')
    plt.ylabel('Training Loss')
    plt.title(title)
    plt.legend()
    plt.show()
