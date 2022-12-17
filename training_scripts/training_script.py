import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models

from torch.utils.data import DataLoader

from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import Accuracy, Loss


# Defining the convolutional neural network
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(256, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


def get_svhn_dataset():
    trainset = datasets.SVHN(
        root='./SVHN', split='train', download=True, 
        transform=transforms.ToTensor()
    )
    testset = datasets.SVHN(
        root='./SVHN',split='test', download=True,
        transform=transforms.ToTensor()
    )
    return trainset, testset


def get_mnist_dataset():
    trainset = datasets.MNIST(
        root='./MNIST', train=True, download=True, 
        transform=transforms.ToTensor()
    )
    testset = datasets.MNIST(
        root='./MNIST', train=False, download=True,
        transform=transforms.ToTensor()
    )
    return trainset, testset


def get_cifar10_dataset():
    trainset = datasets.CIFAR10(
        root='./CIFAR10', train=True, download=True, 
        transform=transforms.ToTensor()
    )
    testset = datasets.CIFAR10(
        root='./CIFAR10', train=False, download=True,
        transform=transforms.ToTensor()
    )
    return trainset, testset


def get_device():
    if torch.cuda.is_available(): 
       return torch.device("cuda")
    elif torch.backends.mps.is_available(): 
        return torch.device("mps")
    else:
        return torch.device("cpu")


class Trainer:
    def __init__(
        self,
        model: str,
        dataset: str,
        batch_size: int=32,
        epochs: int=20,
        lr: float=0.01,
        momentum: float=0,
    ) -> None:
        # model
        if model.lower() == 'lenet5':
            self.model = LeNet5()
        elif model.lower() == 'resnet34':
            self.model = models.resnet34()
        elif model.lower() == 'vgg16':
            self.model = models.vgg16()
        else:
            raise ValueError(f'Unsupported Model {dataset}')

        # dataset
        if dataset.lower() == 'mnist':
            self.trainset, self.testset = get_mnist_dataset()
        elif dataset.lower() == 'cifar10':
            self.trainset, self.testset = get_cifar10_dataset()
        elif dataset.lower() == 'svhn':
            self.trainset, self.testset = get_svhn_dataset()
        else:
            raise ValueError(f'Unsupported Dataset {dataset}')

        # set device
        self.device = get_device()

        # others
        self.batch_size = batch_size
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        # prepare for training
        self.trainloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        self.testloader = DataLoader(self.testset, batch_size=self.batch_size, shuffle=True)        


    def train(self):
        self.model = self.model.to(self.device)

        trainer = create_supervised_trainer(
            self.model, self.optimizer, self.criterion, device=self.device
        )
        # add progress bar to trainer
        progress_bar = ProgressBar()
        progress_bar.attach(trainer, output_transform=lambda x: {'loss': x})

        # start training
        trainer.run(self.trainloader, max_epochs=self.epochs)


    def eval(self, metrics):
        # evaluate the model
        test_evaluator = create_supervised_evaluator(
            self.model, metrics=metrics, device=self.device
        )
        
        time_begin = time.time()
        test_evaluator.run(self.testloader)
        total_time = time.time() - time_begin

        test_metrics = test_evaluator.state.metrics
        test_metrics['Time'] = total_time
        return test_metrics



# train
trainer = Trainer(model='lenet5', dataset='mnist', epochs=1)
trainer.train()
eval_result = trainer.eval({"Accuracy": Accuracy(), "Loss": Loss(trainer.criterion)})
print(eval_result)

