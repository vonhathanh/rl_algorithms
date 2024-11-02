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
        states, actions, rewards, next_states, dones = memory
        # sometimes step_size would make the memory overflow, we have to clip it
        step_size = min(len(states), self.size - self.ptr)

        self.states[self.ptr: self.ptr + step_size] = states[:step_size]
        self.actions[self.ptr: self.ptr + step_size] = actions[:step_size]
        self.rewards[self.ptr: self.ptr + step_size] = rewards[:step_size]
        self.next_states[self.ptr: self.ptr + step_size] = next_states[:step_size]
        self.dones[self.ptr: self.ptr + step_size] = dones[:step_size]

        self.ptr = (self.ptr + step_size) % self.size
        self.curr_size = min(self.curr_size + step_size, self.size)

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