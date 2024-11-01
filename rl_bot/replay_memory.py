import numpy as np


class ReplayMemory:

    def __init__(self, size: int, observation_dim: tuple):
        self.states = np.zeros((size, *observation_dim))
        self.actions = np.zeros(size)
        self.rewards = np.zeros(size)
        self.next_states = np.zeros((size, *observation_dim))
        self.dones = np.zeros(size)

        self.ptr = 0
        self.size, self.curr_size = size, 0

    def store(self, memory: tuple):
        state, action, reward, next_state, done = memory

        self.states[self.ptr] = state
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
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices]
        }