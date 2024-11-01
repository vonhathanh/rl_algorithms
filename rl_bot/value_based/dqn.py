import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from rl_bot.replay_memory import ReplayMemory
from rl_bot.utils import linear_schedule


class MLPPolicy(nn.Module):
    def __init__(self, env: gym.vector.VectorEnv):
        super().__init__()

        input_dim = np.array(env.single_observation_space.shape).prod()

        output_dim = env.single_action_space.n

        self.net = nn.Sequential(nn.Linear(input_dim, 100), nn.ReLU(), nn.Linear(100, output_dim))

    def forward(self, x):
        return self.net(x)

class DQN:

    def __init__(self, env: gym.vector.VectorEnv, args: dict):
        # if GPU is to avaialble
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        self.env = env
        self.args = args
        self.policy_net = MLPPolicy(env).to(self.device)
        self.target_net = MLPPolicy(env).to(self.device)
        self.replay_memory = ReplayMemory(args["replay_memory_size"], env.single_observation_space.shape)

        # copy weights from policy net to target net
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), self.args["lr"])

    def train(self, n_steps: int):
        for i in range(n_steps):
            # Initialise sequence s1 = {x1} and preprocessed sequenced φ1 = φ(s1), we use mlp policy, no preprocess
            state, info = self.env.reset(seed=self.args["seed"])

            epislon = linear_schedule(self.args["epsilon_start"],
                                      self.args["epsilon_end"],
                                      self.args["explore_duration"],
                                      i)

            for _ in range(self.args["T"]):
                # select action a_t with probability epsilon
                action = self.select_action(torch.tensor(state).to(self.device), epislon)
                # Execute action at in emulator and observe reward r_t and new state s_t+1
                next_state, reward, termination, truncation, _ = self.env.step(action)
                # Store transition in replay memory D
                self.replay_memory.store((state, action, reward, next_state, termination))
                # Set s_t+1 = s_t
                state = next_state
                # Sample random minibatch of transitions (φj, aj, rj, φj+1) from D
                self.train_minibatch()
                # episode is finished
                if termination or truncation:
                    break

    def select_action(self, state, epsilon: float):
        if random.random() <= epsilon:
            return np.random.choice(self.env.action_space.n)
        else:
            logits = self.policy_net(state)
            return torch.argmax(logits)

    def train_minibatch(self):
        if len(self.replay_memory) < self.args["minibatch_size"]:
            return

        minibatch = self.replay_memory.sample(self.args["minibatch_size"])
        # states = minibatch["states"]
        actions = torch.tensor(minibatch["actions"]).to(self.device)
        rewards = torch.tensor(minibatch["rewards"]).to(self.device)
        next_states = torch.tensor(minibatch["next_states"]).to(self.device)
        dones = torch.tensor(minibatch["dones"]).to(self.device)
        # calculate action that has the maximum Q value using target network: Q(s', a', old_phi)
        q_values = torch.max(self.target_net(next_states))
        # y_j = r_j if s_j+1 is terminal state
        # else r_j + gamma*q_values
        target = rewards + self.args["gamma"] * q_values * (1 - dones)

        loss = F.mse_loss(target, actions)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        # soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        TAU = self.args["tau"]
        for k in policy_net_state_dict:
            target_net_state_dict[k] = policy_net_state_dict[k] * TAU + target_net_state_dict[k] * (1 - TAU)
        self.target_net.load_state_dict(target_net_state_dict)

if __name__ == '__main__':

    args = {
        "num_envs": 2,
        "replay_memory_size": 100,
        "minibatch_size": 36,
        "lr": 1e-3,
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "explore_duration": 500,
        "gamma": 0.99,
        "seed": 1993,
        "tau": 0.005,
        "T": 1000,
    }

    envs = gym.make_vec("CartPole-v1", num_envs=args["num_envs"])

    model = DQN(envs, args)

    model.train(n_steps=100)