import argparse
import os
import time
from multiprocessing.managers import BaseManager
from threading import Lock

import numpy as np
import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributed.optim import DistributedOptimizer
from torchvision import datasets, transforms

torch.autograd.set_detect_anomaly(True)

# Hardcoded versions according to https://github.com/pytorch/pytorch/issues/68385


def get_device_str(num_gpus):
    if num_gpus == 0:
        print('Using CPU (num_gpus = 0 arg)')
        device_str = 'cpu'
    elif torch.cuda.is_available() and num_gpus > 0:
        print('Using CUDA')
        device_str = 'cuda:0'
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print('Using Apple Silicon')
        device_str = 'mps'
    else:
        print('Using CPU')
        device_str = 'cpu'
    return device_str


# --------- MNIST Network to train, from pytorch/examples -----

class Net(nn.Module):
    def __init__(self, num_gpus=0):
        super(Net, self).__init__()
        print(f"Using {num_gpus} GPUs to train")
        self.num_gpus = num_gpus
        device_str = get_device_str(num_gpus)
        device = torch.device(device_str)
        print(f"Putting first 2 convs on {str(device)}")
        # Put conv layers on the first cuda device, or CPU if no cuda device
        self.conv1 = nn.Conv2d(1, 32, 3, 1).to(device)
        self.conv2 = nn.Conv2d(32, 64, 3, 1).to(device)
        # Put rest of the network on the 2nd cuda device, if there is one
        if "cuda" in str(device) and num_gpus > 1:
            device = torch.device("cuda:1")

        print(f"Putting rest of layers on {str(device)}")
        self.dropout1 = nn.Dropout2d(0.25).to(device)
        self.dropout2 = nn.Dropout2d(0.5).to(device)
        self.fc1 = nn.Linear(9216, 128).to(device)
        self.fc2 = nn.Linear(128, 10).to(device)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)

        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        # Move tensor to next device if necessary
        next_device = next(self.fc1.parameters()).device
        x = x.to(next_device)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def parameters(self):
        # print('looooooool')
        params = super(Net, self).parameters()
        # print(params)
        # print(list(params))
        # print('looooool end')
        return list(params)

# --------- Helper Methods --------------------

# On the local node, call a method with first arg as the value held by the
# RRef. Other args are passed in as arguments to the function called.
# Useful for calling instance methods. method could be any matching function, including
# class methods.


def call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)

# Given an RRef, return the result of calling the passed in method on the value
# held by the RRef. This call is done on the remote node that owns
# the RRef and passes along the given argument.
# Example: If the value held by the RRef is of type Foo, then
# remote_method(Foo.bar, rref, arg1, arg2) is equivalent to calling
# <foo_instance>.bar(arg1, arg2) on the remote node and getting the result
# back.


def remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs)


# --------- Parameter Server --------------------
class ParameterServer(nn.Module):
    def __init__(self, num_gpus=0):
        super().__init__()
        model = Net(num_gpus=num_gpus)
        self.model = model

        # Custom code https://towardsdatascience.com/installing-pytorch-on-apple-m1-chip-with-gpu-acceleration-3351dc44d67c
        print('INITIALIZING PARAMETER SERVER')
        device_str = get_device_str(num_gpus)

        self.input_device = torch.device(device_str)

    def forward(self, inp):
        # inp = inp.to(self.input_device)
        out = self.model(inp)
        # This output is forwarded over RPC, which as of 1.5.0 only accepts CPU tensors.
        # Tensors must be moved in and out of GPU memory due to this.
        # out = out.to("cpu")
        return out

    # Use dist autograd to retrieve gradients accumulated for this model.
    # Primarily used for verification.

    def get_dist_gradients(self, cid):
        grads = dist_autograd.get_gradients(cid)
        # This output is forwarded over RPC, which as of 1.5.0 only accepts CPU tensors.
        # Tensors must be moved in and out of GPU memory due to this.
        cpu_grads = {}
        for k, v in grads.items():
            k_cpu, v_cpu = k.to("cpu"), v.to("cpu")
            # k_cpu, v_cpu = k.to("mps"), v.to("mps")
            cpu_grads[k_cpu] = v_cpu
        return cpu_grads

    # Wrap local parameters in a RRef. Needed for building the
    # DistributedOptimizer which optimizes paramters remotely.

    def get_param_rrefs(self):
        param_rrefs = [rpc.RRef(param) for param in self.model.parameters()]
        return param_rrefs


# The global parameter server instance.
param_server = None
# A lock to ensure we only have one parameter server.
global_lock = Lock()

update_lock = mp.Lock()


def get_parameter_server(num_gpus=0):
    """
    Returns a singleton parameter server to all trainer processes
    """
    global param_server
    # Ensure that we get only one handle to the ParameterServer.
    with global_lock:
        if not param_server:
            # construct it once
            param_server = ParameterServer(num_gpus=num_gpus)
        return param_server


def run_parameter_server(rank, world_size):
    # The parameter server just acts as a host for the model and responds to
    # requests from trainers.
    # rpc.shutdown() will wait for all workers to complete by default, which
    # in this case means that the parameter server will wait for all trainers
    # to complete, and then exit.
    print("PS master initializing RPC")

    # ----------------------------------------------------------------

    # TODO: Remove? copied from https://h-huang.github.io/tutorials/recipes/cuda_rpc.html
    # options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128)
    # options.set_device_map("trainer_1", {'mps:0': 'mps:0'})
    # print('device maps', options.device_maps)
    # rpc.init_rpc(name="parameter_server", rank=rank,
    #              world_size=world_size, rpc_backend_options=options)

    # ----------------------------------------------------------------

    rpc.init_rpc(name="parameter_server", rank=rank, world_size=world_size)
    print("RPC initialized! Running parameter server...")
    rpc.shutdown()
    print("RPC shutdown on parameter server.")


# --------- Trainers --------------------

# nn.Module corresponding to the network trained by this trainer. The
# forward() method simply invokes the network on the given parameter
# server.
class TrainerNet(nn.Module):
    def __init__(self, num_gpus=0):
        super().__init__()
        self.num_gpus = num_gpus
        self.param_server_rref = rpc.remote(
            "parameter_server", get_parameter_server, args=(num_gpus,))

    def get_global_param_rrefs(self):
        remote_params = remote_method(
            ParameterServer.get_param_rrefs,
            self.param_server_rref)
        return remote_params

    def forward(self, x):
        model_output = remote_method(
            ParameterServer.forward, self.param_server_rref, x)
        return model_output


def run_training_loop(rank, world_size, num_gpus, train_loader, test_loader, ulock):
    # Runs the typical nueral network forward + backward + optimizer step, but
    # in a distributed fashion.
    net = TrainerNet(num_gpus=num_gpus)
    # Build DistributedOptimizer.
    param_rrefs = net.get_global_param_rrefs()
    opt = DistributedOptimizer(optim.SGD, param_rrefs, lr=0.03)

    for i, (data, target) in enumerate(train_loader):
        with dist_autograd.context() as cid:
            with ulock:
                model_output = net(data)

                # target = target.to(model_output.device)

                loss = F.nll_loss(model_output, target)

                if i % 5 == 0:
                    print(f"Rank {rank} training batch {i} loss {loss.item()}")
                dist_autograd.backward(cid, [loss])
                # Ensure that dist autograd ran successfully and gradients were
                # returned.
                # assert remote_method(
                #     ParameterServer.get_dist_gradients,
                #     net.param_server_rref,
                #     cid) != {}
                # print(remote_method(
                #     ParameterServer.get_dist_gradients,
                #     net.param_server_rref,
                #     cid))
                opt.step(cid)

    print(f"Worker {rank} Training complete!")

    if rank == world_size - 1:
        print("Getting accuracy....")
        get_accuracy(test_loader, net)


def get_accuracy(test_loader, model):
    model.eval()
    correct_sum = 0
    # Use GPU to evaluate if possible

    device_str = get_device_str(model.num_gpus)
    device = torch.device(device_str)

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            # out = model(data, -1)
            out = model(data)
            pred = out.argmax(dim=1, keepdim=True)
            pred, target = pred.to(device), target.to(device)
            correct = pred.eq(target.view_as(pred)).sum().item()
            correct_sum += correct

    print(f"Accuracy {correct_sum / len(test_loader.dataset)}")


# Main loop for trainers.
def run_worker(rank, world_size, num_gpus, train_loader, test_loader, ulock, epochs=1):
    print(f"Worker rank {rank} initializing RPC")

    # ---------------------------------------------

    # TODO: Remove? copied from https://h-huang.github.io/tutorials/recipes/cuda_rpc.html
    # options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128)
    # options.set_device_map("parameter_server", {'mps:0': 'mps:0'})
    # print('device maps', options.device_maps)
    # rpc.init_rpc(
    #     name=f"trainer_{rank}",
    #     rank=rank,
    #     world_size=world_size, rpc_backend_options=options)

    # ---------------------------------------------

    rpc.init_rpc(
        name=f"trainer_{rank}",
        rank=rank,
        world_size=world_size)

    print(f"Worker {rank} done initializing RPC")

    print(f'worker {rank} acquired lock')
    for e in range(epochs):
        run_training_loop(rank, world_size, num_gpus,
                          train_loader, test_loader, ulock)
    print(f'worker {rank} released lock')
    rpc.shutdown()


def train(args, model, device, train_loader, optimizer, epoch):
    print(dir(model))
    print('train version:', model)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # print('stuff')
        # print(model)
        # print(type(model))
        # print(dir(model))
        with torch.no_grad():
            output = model.forward(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            # if args.dry_run:
            #     break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


class CustomManager(BaseManager):
    # nothing
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Parameter-Server RPC based training")
    parser.add_argument(
        "--world_size",
        type=int,
        default=4,
        help="""Total number of participating processes. Should be the sum of
        master node and all training nodes.""")
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Global rank of this process. Pass in 0 for master.")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=0,
        help="""Number of GPUs to use for training, Currently supports between 0
         and 2 GPUs. Note that this argument will be passed to the parameter servers.""")
    parser.add_argument(
        "--master_addr",
        type=str,
        default="localhost",
        help="""Address of master, will default to localhost if not provided.
        Master must be able to accept network traffic on the address + port.""")
    parser.add_argument(
        "--master_port",
        type=str,
        default="29500",
        help="""Port that master is listening on, will default to 29500 if not
        provided. Master must be able to accept network traffic on the host and port.""")
    parser.add_argument(
        "--epochs_per_worker",
        type=int,
        default=1,
        help="""Number of epochs for each worker.""")

    args = parser.parse_args()
    assert args.rank is not None, "must provide rank argument."
    assert args.num_gpus <= 3, f"Only 0-2 GPUs currently supported (got {args.num_gpus})."
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    processes = []
    world_size = args.world_size
    update_lock = mp.Lock()

    print(f'{world_size=}')

    start = time.perf_counter()


# --------------------------------------------------------

    shared_model = Net()
    print("original:", shared_model)
    # optimizer = optim.SGD(shared_model.parameters(), lr=0.03)

    # from multiprocessing import shared_memory
    # shm_a = shared_memory.SharedMemory(create=True, size=1)

    from multiprocessing import Manager, Process

    # print(shm_a)
    # print(type(shm_a))

    CustomManager.register('net', Net)

    with CustomManager() as manager:

        mm = manager.net()

        # manager_model = manager.Value(Net, shared_model)

        # mm_params = mm.parameters()
        old_m = Net()
        # print('old', old_m)
        # print('old', old_m.parameters())
        # print('mm', mm)
        # print('mm', list(mm.parameters()))

        optimizer = optim.SGD(mm.parameters(), lr=0.03)

        workers = 1

        for w in range(workers):
            mnist_train = datasets.MNIST('../data', train=True, download=True,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize(
                                                 (0.1307,), (0.3081,))
                                         ]))
            train_loader = torch.utils.data.DataLoader(
                mnist_train,
                batch_size=32,
                shuffle=True,
            )
            test_loader = torch.utils.data.DataLoader(
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
            device = 'cpu'
            # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
            # for epoch in range(1, args.epochs + 1):
            for epoch in range(1):
                print('epoch', epoch)
                p = Process(target=train, args=(args, mm,
                            device, train_loader, optimizer, epoch))
                # train(args, manager_model, device,
                #       train_loader, optimizer, epoch)
                # test(manager_model, device, test_loader)
                p.start()
                p.join()
                # scheduler.step()
