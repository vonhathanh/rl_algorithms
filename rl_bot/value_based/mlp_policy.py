import gymnasium as gym
import numpy as np

from torch import nn

class MLPPolicy(nn.Module):
    def __init__(self, env: gym.vector.VectorEnv):
        super().__init__()

        input_dim = np.array(env.single_observation_space.shape).prod()

        output_dim = env.single_action_space.n

        self.net = nn.Sequential(nn.Linear(input_dim, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, output_dim))

    def forward(self, x):
        return self.net(x)