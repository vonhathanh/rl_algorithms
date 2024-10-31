import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from rl_bot.replay_memory import ReplayMemory


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
            for _ in range(self.args["T"]):
                # select action a_t with probability epsilon
                action = self.select_action(state, self.args["epsilon"])
                # Execute action at in emulator and observe reward r_t and new state s_t+1
                next_state, reward, termination, truncation, _ = self.env.step(action)
                # Store transition in replay memory D
                self.replay_memory.store((state, action, reward, next_state, termination))
                # Set s_t+1 = s_t
                state = next_state
                # Sample random minibatch of transitions (φj, aj, rj, φj+1) from D
                if len(self.replay_memory) >= self.args["minibatch_size"]:
                    minibatch = self.replay_memory.sample(self.args["minibatch_size"])
                    self.train_minibatch(minibatch)
                # episode is finished
                if termination or truncation:
                    break

    def select_action(self, state, epsilon: float):
        if np.random.rand(1)[0] <= epsilon:
            return np.random.choice(self.env.action_space.n)
        else:
            logits = self.policy_net(state)
            return torch.argmax(logits)

    def train_minibatch(self, minibatch):
        # states = minibatch["states"]
        actions = minibatch["actions"]
        rewards = minibatch["rewards"]
        next_states = minibatch["next_states"]
        dones = minibatch["dones"]
        # calculate action that has the maximum Q value using target network: Q(s', a', old_phi)
        q_values = torch.max(self.target_net(next_states))
        # y_j = r_j if s_j+1 is terminal state
        # else r_j + gamma*q_values
        target = rewards + q_values * (1 - dones)

        loss = F.mse_loss(target, actions)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        # copy weights from policy net to target net if update interval % t == 0
        self.target_net.load_state_dict(self.policy_net.state_dict())

        