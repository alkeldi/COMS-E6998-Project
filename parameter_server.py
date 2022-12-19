import os
import time
from threading import Lock

import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributed.optim import DistributedOptimizer
from torchvision import datasets, transforms

# Refence source: https://pytorch.org/tutorials/intermediate/rpc_param_server_tutorial.html

# Note: For error
# libc++abi: terminating with uncaught exception of type std::runtime_error: In connectFromLoop at tensorpipe/transport/uv/uv.h:297 "rv < 0: too many open files"libc++abi: terminating with uncaught exception of type std::runtime_error: In connectFromLoop at tensorpipe/transport/uv/uv.h:297 "rv < 0: too many open files"
# increase the limit by running `ulimit -n 3000` (or some other large number)

# See UPDATE LOCK NOTE comment below for more info on locking and why this strategy
# ends up seeing performance similar to sequential epochs on a single process.


def get_device_str(num_gpus):
    if num_gpus == 0:
        device_str = 'cpu'
    elif torch.cuda.is_available() and num_gpus > 0:
        device_str = 'cuda:0'
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device_str = 'mps'
    else:
        device_str = 'cpu'
    return device_str


# --------- MNIST Network to train, from pytorch/examples -----

class Net(nn.Module):
    def __init__(self, num_gpus=0):
        super(Net, self).__init__()
        self.num_gpus = num_gpus
        device_str = get_device_str(num_gpus)
        device = torch.device(device_str)
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
            # UPDATE LOCK NOTE: This is the heart of the issue. This lock was added
            # because with it, we see the following error:
            # RuntimeError: Error on Node 0: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [32, 1, 3, 3]] is at version 18; expected version 17 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).
            # Comment out to see the difference.
            with ulock:
                model_output = net(data)

                loss = F.nll_loss(model_output, target)

                if i % 100 == 0:
                    print(f"Rank {rank} training batch {i} loss {loss.item()}")
                dist_autograd.backward(cid, [loss])
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
            out = model(data)
            pred = out.argmax(dim=1, keepdim=True)
            pred, target = pred.to(device), target.to(device)
            correct = pred.eq(target.view_as(pred)).sum().item()
            correct_sum += correct

    print(f"Accuracy {correct_sum / len(test_loader.dataset)}")


# Main loop for trainers.
def run_worker(rank, world_size, num_gpus, train_loader, test_loader, ulock, epochs=1):
    print(f"Worker rank {rank} initializing RPC")
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


def launch_ps_and_workers(
        world_size,
        epochs_per_worker,
        num_gpus,
        train_dataloader,
        test_dataloader):

    print()
    print(
        f'***** Training with {world_size - 1} worker(s) for {epochs_per_worker} epochs each. *****')
    print()

    processes = []
    update_lock = mp.Lock()

    # Launch parameter server and workers
    for r in range(world_size):
        if r == 0:
            # Start parameter server
            p = mp.Process(target=run_parameter_server, args=(0, world_size))
            p.start()
            processes.append(p)
        else:
            # start training worker on this node
            p = mp.Process(
                target=run_worker,
                args=(
                    r,
                    world_size, num_gpus,
                    train_dataloader,
                    test_dataloader,
                    update_lock,
                    epochs_per_worker))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()
