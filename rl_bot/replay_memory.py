import numpy as np
import torch


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