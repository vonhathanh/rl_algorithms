import numpy as np
import torch

from rl_bot.segment_tree import *


class ReplayMemory:

    def __init__(self, size: int, observation_dim: tuple, device):
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

    def sample(self, batch_size: int):
        indices = np.random.choice(self.curr_size, batch_size, replace=False)
        return {
            "states": torch.tensor(self.states[indices]).to(self.device, dtype=torch.float32),
            "actions": torch.tensor(self.actions[indices]).to(self.device, dtype=torch.int64),
            "rewards": torch.tensor(self.rewards[indices]).to(self.device, dtype=torch.float32),
            "next_states": torch.tensor(self.next_states[indices]).to(self.device, dtype=torch.float32),
            "dones": torch.tensor(self.dones[indices]).to(self.device, dtype=torch.float32)
        }

class PriporityMemory(ReplayMemory):
    """
    Priority Memory implementation of the paper: PRIORITIZED EXPERIENCE REPLAY
    This version uses sum tree (segment tree) as in the paper reference
    """
    def __init__(self, size: int, observation_dim: tuple, device):
        super().__init__(size, observation_dim, device)
        self.priorities = np.zeros(size)
        self.sum_tree = np.zeros(size*4)
        init(self.sum_tree, self.priorities, 0, size-1)

    def store(self, memory: tuple):
        *m, priority = memory
        update_tree(self.sum_tree, self.priorities, priority - self.priorities[self.ptr], self.ptr)
        super().store(m)

    def sample(self, batch_size: int):
        pass


if __name__ == '__main__':
    # test PriporityMemory
    obs_shape = (4, 1)
    size = 3
    m = PriporityMemory(size, obs_shape, "cpu")
    print(m.sum_tree)
    print(m.priorities)

    state = np.zeros(obs_shape)

    m.store((state, np.float32(0), np.float32(0), state, np.int64(0), 10))
    print(m.sum_tree)
    print(m.priorities)

    m.store((state, np.float32(0), np.float32(0), state, np.int64(5), -15))
    print(m.sum_tree)
    print(m.priorities)

    m.store((state, np.float32(0), np.float32(0), state, np.int64(5), -3))
    print(m.sum_tree)
    print(m.priorities)

    m.store((state, np.float32(0), np.float32(0), state, np.int64(5), -4))
    print(m.sum_tree)
    print(m.priorities)
