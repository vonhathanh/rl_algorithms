import math

import numpy as np
import torch
import random

from rl_bot.segment_tree import SumSegmentTree


class ReplayMemory:

    def __init__(self, size: int, observation_dim: tuple, device: str = "cpu"):
        self.states = np.zeros((size, *observation_dim))
        self.actions = np.zeros(size)
        self.rewards = np.zeros(size)
        self.next_states = np.zeros((size, *observation_dim))
        self.dones = np.zeros(size)

        self.device = device
        self.ptr = 0
        self.size, self.curr_size = size, 0

    def store(self, memory: tuple):
        states, action, reward, next_state, done = memory

        self.states[self.ptr] = states
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.size
        self.curr_size = min(self.curr_size + 1, self.size)

    def __len__(self) -> int:
        return self.curr_size

    def sample(self, batch_size: int, is_rand=True):
        if is_rand:
            indices = np.random.choice(self.curr_size, batch_size, replace=False)
        else:
            indices = range(0, batch_size)
        return {
            "states": torch.tensor(self.states[indices]).to(self.device, dtype=torch.float32),
            "actions": torch.tensor(self.actions[indices]).to(self.device, dtype=torch.int64),
            "rewards": torch.tensor(self.rewards[indices]).to(self.device, dtype=torch.float32),
            "next_states": torch.tensor(self.next_states[indices]).to(self.device, dtype=torch.float32),
            "dones": torch.tensor(self.dones[indices]).to(self.device, dtype=torch.float32)
        }

    def clear(self):
        # we don't actually clear the underlying data as it's not necessary atm,
        # just reset the ptr and curr_size is enough
        self.ptr = 0
        self.curr_size = 0

class PriporityMemory(ReplayMemory):
    """
    Priority Memory implementation of the paper: PRIORITIZED EXPERIENCE REPLAY
    This version uses sum tree (segment tree) as in the paper reference
    """
    def __init__(self, size: int, observation_dim: tuple, device: str = "cpu", beta: float=0.0):
        super().__init__(size, observation_dim, device)
        # tree capacity must be power of 2
        capacity = 1
        while capacity < size:
            capacity *= 2
        self.p_tree = SumSegmentTree(capacity)
        self.beta = beta
        self.max_priority = 1.0

    def store(self, memory: tuple):
        *m, priority = memory
        self.p_tree[self.ptr] = priority
        super().store(m)

    def sample(self, batch_size: int):
        segment = self.p_tree.sum() / batch_size
        indices = np.zeros(batch_size, dtype=np.int64)
        weights = np.zeros(batch_size)

        for i in range(batch_size):
            upper_bound = random.uniform(segment * i, segment * (i + 1))

            index = self.p_tree.retrieve(upper_bound)
            indices[i] = index

            weights[i] = math.pow(len(self)*self.p_tree[index], self.beta)

        weights /= weights.max()

        return {
            "states": torch.tensor(self.states[indices]).to(self.device, dtype=torch.float32),
            "actions": torch.tensor(self.actions[indices]).to(self.device, dtype=torch.int64),
            "rewards": torch.tensor(self.rewards[indices]).to(self.device, dtype=torch.float32),
            "next_states": torch.tensor(self.next_states[indices]).to(self.device, dtype=torch.float32),
            "dones": torch.tensor(self.dones[indices]).to(self.device, dtype=torch.float32),
            "weights": torch.tensor(weights).to(self.device, dtype=torch.float32),
            "indices": indices
        }