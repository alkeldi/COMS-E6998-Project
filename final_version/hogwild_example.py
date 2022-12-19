from __future__ import print_function

import time

import torch
import torch.multiprocessing as mp
import torch.optim as optim
from hogwild import *
from simple_nn import SimpleNN
from torchvision import datasets, transforms

if __name__ == '__main__':
    # Example workflow for comparing the speed/performance of a model built
    # with different configurations of processes and how epochs get split
    # across them. Here, we consider a simple Neural Network trained on MNIST,
    # using different configurations:
    # 1. 1 process with 4 epochs
    # 2. 4 processes with 1 epoch each, trained in parallel
    # 3. 2 processes with 2 epochs each

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

    log_interval = 10
    device = 'cpu'

    # Specify a '4 processes by 1 epoch' model
    start1 = time.perf_counter()
    model = SimpleNN()
    num_processes1 = 4
    epochs_per_process1 = 1
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    loss_tracker_4_by_1 = mp.Manager().list()

    train_model(model, train_dataloader, num_processes1, epochs_per_process1,
                optimizer, device, log_interval, loss_tracker_4_by_1)
    test(model, device, test_dataloader)
    end1 = time.perf_counter()

    # Specify a '1 process by 4 epochs' model
    start2 = time.perf_counter()
    model = SimpleNN()
    num_processes2 = 1
    epochs_per_process2 = 4
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    loss_tracker_1_by_4 = mp.Manager().list()

    train_model(model, train_dataloader, num_processes2, epochs_per_process2,
                optimizer, device, log_interval, loss_tracker_1_by_4)
    test(model, device, test_dataloader)
    end2 = time.perf_counter()

    # Specify a '2 process by 2 epochs' model
    start3 = time.perf_counter()
    model = SimpleNN()
    num_processes3 = 2
    epochs_per_process3 = 2
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    loss_tracker_2_by_2 = mp.Manager().list()

    train_model(model, train_dataloader, num_processes3, epochs_per_process3,
                optimizer, device, log_interval, loss_tracker_2_by_2)
    test(model, device, test_dataloader)
    end3 = time.perf_counter()

    # Compare times
    log_elapsed_time(num_processes1, epochs_per_process1, start1, end1)
    log_elapsed_time(num_processes2, epochs_per_process2, start2, end2)
    log_elapsed_time(num_processes2, epochs_per_process3, start3, end3)

    # Compare training losses via chart
    chart_losses({
        '4 processes x 1 epoch NN': loss_tracker_4_by_1,
        '2 processes x 2 epochs NN': loss_tracker_2_by_2,
        '1 process x 4 epochs NN': loss_tracker_1_by_4,
    })
