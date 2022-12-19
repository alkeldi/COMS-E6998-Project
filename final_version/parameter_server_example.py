import os
import time

import torch
from parameter_server import launch_ps_and_workers
from torchvision import datasets, transforms

if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ["MASTER_PORT"] = '29500'

    num_gpus = 0

    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(
                               (0.1307,), (0.3081,))
                       ])),
        batch_size=32,
        shuffle=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data',
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=32,
        shuffle=True,
    )

    # world size n = 1 parameter server + (n-1) workers
    world_size1 = 2
    epochs_per_worker1 = 4
    start1 = time.perf_counter()
    launch_ps_and_workers(
        world_size1,
        epochs_per_worker1,
        num_gpus,
        train_dataloader,
        test_dataloader)
    end1 = time.perf_counter()

    # world size n = 1 parameter server + (n-1) workers
    world_size2 = 5
    epochs_per_worker2 = 1
    start2 = time.perf_counter()
    launch_ps_and_workers(
        world_size2,
        epochs_per_worker2,
        num_gpus,
        train_dataloader,
        test_dataloader)
    end2 = time.perf_counter()

    print(f'Elapsed time model 1: {end1 - start1}')
    print(f'Elapsed time model 2: {end2 - start2}')
