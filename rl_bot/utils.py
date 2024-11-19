import math

from torch import nn


def linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    # linearly reduce espilon from start_e to end_e in "duration" timestep
    # t: current timestep in the training loop
    eps_threshold = end_e + (start_e - end_e) * math.exp(-1. * t / duration)
    return eps_threshold

def beta_annealing(start_b: float, end_b: float, duration: int, t: int) -> float:
    interval = (end_b - start_b) / duration
    return interval * min(duration, t)

def init_uniformly(layer: nn.Linear, init_w: float=3e-3):
    """Initialize the weights and bias uniformly in [-init_w, init_w]."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

def set_init(layers):
    for l in layers:
        nn.init.normal_(l.weight, mean=0, std=0.1)
        nn.init.constant_(l.bias, 0.)