import copy
import numpy as np

import torch
import torch.nn as nn

def init(module, weight_init, bias_init, gain=1):
    """Initialize a module with specified weight and bias initialization."""
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check(input):
    """Check if input is a numpy array or torch tensor."""
    if isinstance(input, np.ndarray):
        return torch.from_numpy(input)
    elif isinstance(input, torch.Tensor):
        return input
    else:
        raise TypeError("Input must be numpy array or torch tensor")

def get_gard_norm(it):
    """Get gradient norm."""
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return torch.sqrt(sum_grad)

def huber_loss(e, d):
    """Compute Huber loss."""
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def mse_loss(e):
    """Compute MSE loss."""
    return e**2/2
