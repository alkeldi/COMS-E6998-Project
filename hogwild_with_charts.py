from __future__ import print_function

import argparse
# From train.py
import os
import time

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms


def train(rank, args, model, device, train_dataset, val_dataset, dataloader_kwargs):
    torch.manual_seed(args.seed + rank)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, **dataloader_kwargs)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, **dataloader_kwargs)

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)
    for epoch in range(1, args.epochs + 1):
        train_epoch(epoch, args, model, device,
                    train_loader, val_loader, optimizer)


def test(args, model, device, dataset, dataloader_kwargs):
    torch.manual_seed(args.seed)

    test_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    test_epoch(model, device, test_loader)


def train_epoch(epoch, args, model, device, train_data_loader, val_data_loader, optimizer):
    model.train()
    pid = os.getpid()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_data_loader):
        optimizer.zero_grad()
        output = model(data.to(device))
        pred = output.max(1)[1]
        loss = criterion(output, target.to(device))
        loss.backward()
        # output.backward()
        # loss = F.nll_loss(output, target.to(device))
        # loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                pid, epoch, batch_idx *
                len(data), len(train_data_loader.dataset),
                100. * batch_idx / len(train_data_loader), loss.item()))
            if args.dry_run:
                break

    # validation
    # model.eval()
    # valid_loss = 0.0
    # total_correct = 0
    # total = 0

    # # with torch.no_grad():
    # for batch_idx, (data, target) in enumerate(val_data_loader):
    #     output = model(data.to(device))
    #     loss = F.nll_loss(output, target.to(device))
    #     valid_loss += loss.item()

    #     # predictions = torch.argmax(output, dim=1)
    #     pred = output.max(1)[1]
    #     correct = sum(pred == target).item()
    #     total_correct += correct
    #     total += len(data)

    # print('Val Accuracy', total_correct / total)
    # print(
    #     f'Epoch {epoch} \t\t Validation Loss: {valid_loss / len(val_data_loader)}')


def test_epoch(model, device, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))
            # sum up batch loss
            test_loss += F.nll_loss(output, target.to(device),
                                    reduction='sum').item()
            pred = output.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.to(device)).sum().item()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 2)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--mps', action='store_true', default=False,
                    help='enables macOS GPU training')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(256, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        print(x)
        print(x.shape)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)

        # return out


resnet18 = models.resnet18(pretrained=False)

if __name__ == '__main__':
    start = time.time()

    args = parser.parse_args()

    use_cuda = args.cuda and torch.cuda.is_available()
    use_mps = args.mps and torch.backends.mps.is_available()
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST('./data', train=True, download=True,
                                transform=transform)
    train_size = int(len(train_data) * 0.8)
    val_size = int(len(train_data) - train_size)
    train_data, val_data = random_split(train_data, [train_size, val_size])
    test_data = datasets.MNIST('./data', train=False,
                               transform=transform)

    train_data = datasets.CIFAR10(
        root='./CIFAR10', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Grayscale()
        ])
    )
    train_size = int(len(train_data) * 0.8)
    val_size = int(len(train_data) - train_size)
    train_data, val_data = random_split(train_data, [train_size, val_size])
    test_data = datasets.CIFAR10(
        root='./CIFAR10', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Grayscale()
        ])
    )

    kwargs = {'batch_size': args.batch_size,
              'shuffle': True}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       })

    torch.manual_seed(args.seed)
    mp.set_start_method('spawn', force=True)

    # model = Net().to(device)
    # model = LeNet5().to(device)
    model = resnet18.to(device)
    model.share_memory()  # gradients are allocated lazily, so they are not shared here

    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=train, args=(rank, args, model, device,
                                           train_data, val_data, kwargs))
        # We first train the model across `num_processes` processes
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    # Once training is complete, we can test the model
    test(args, model, device, test_data, kwargs)

    end = time.time()
    print(
        f'Processes: {args.num_processes}, Epochs: {args.epochs}, Elapsed time: {end - start}')
