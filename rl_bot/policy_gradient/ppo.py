import random
from collections import deque
from typing import Deque

import numpy as np
import gymnasium as gym
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from rl_bot.action_normalizer import ActionNormalizer
from rl_bot.utils import init_uniformly


class Actor(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            log_std_min: int = -20,
            log_std_max: int = 0,
    ):
        """Initialize."""
        super(Actor, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.hidden = nn.Linear(in_dim, 32)

        self.mu_layer = nn.Linear(32, out_dim)
        self.mu_layer = init_uniformly(self.mu_layer)

        self.log_std_layer = nn.Linear(32, out_dim)
        self.log_std_layer = init_uniformly(self.log_std_layer)

    def forward(self, state: torch.Tensor):
        """Forward method implementation."""
        x = F.relu(self.hidden(state))

        mu = torch.tanh(self.mu_layer(x))
        log_std = torch.tanh(self.log_std_layer(x))
        log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
        ) * (log_std + 1)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        action = dist.sample()

        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()

        self.hidden = nn.Linear(in_dim, 64)
        self.out = nn.Linear(64, 1)
        self.out = init_uniformly(self.out)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.hidden(state))
        value = self.out(x)

        return value


class PPO:
    def __init__(self, env, args):
        # check if GPU is avaialble
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        self.env = env
        self.args = args
        self.epoch = 64
        self.epsilon = self.args["epsilon"]

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.005)

        # memory
        self.states: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.rewards: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.masks: list[torch.Tensor] = []
        self.log_probs: list[torch.Tensor] = []

        # mode: train/test
        self.is_test = False

        # logging/visualizing
        self.writer = SummaryWriter(args["log_dir"])


    def select_action(self, state):
        action, dist = self.actor(state)
        value = self.critic(state)

        self.states.append(state)
        self.actions.append(action)
        self.values.append(value)
        self.log_probs.append(dist.log_prob(action))

        return action.cpu().detach().numpy()

    def step(self, action):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, (1, -1)).astype(np.float32)
        reward = np.reshape(reward, (1, -1)).astype(np.float32)
        done = np.reshape(done, (1, -1))

        self.rewards.append(torch.FloatTensor(reward).to(self.device))
        self.masks.append(torch.FloatTensor(1 - done).to(self.device))

        return next_state, reward, done


    def train(self, n_steps):
        for i in range(1, n_steps):
            state, _ = self.env.reset()
            state = np.expand_dims(state, axis=0)
            score = 0
            scores = []
            for j in range(self.args["rollout_len"]):
                state = torch.tensor(state).to(self.device)
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                score += reward[0][0]

                if done[0][0]:
                    state, _ = self.env.reset()
                    state = np.expand_dims(state, axis=0)
                    scores.append(score)
                    score = 0
                else:
                    state = next_state
            print(f"step {i=}, score={np.mean(scores)}")
            self.update_model(torch.tensor(next_state).to(self.device))


    def update_model(self, last_state):
        returns = compute_gae(self.critic(last_state),
                              self.rewards, self.masks, self.values, self.args["gamma"], self.args["lambda"])

        self.states = torch.cat(self.states).view(-1, 3)
        self.actions = torch.cat(self.actions)
        returns = torch.cat(returns).detach()
        self.values = torch.cat(self.values).detach()
        self.log_probs = torch.cat(self.log_probs).detach()
        advantages = returns - self.values
        for _ in range(self.epoch):
            for _ in range(self.args["rollout_len"] // self.args["batch_size"]):
                indices = np.random.choice(self.args["rollout_len"], self.args["batch_size"])
                states = self.states[indices, :]
                return_ = returns[indices]
                actions = self.actions[indices]
                log_probs = self.log_probs[indices]
                adv = advantages[indices]

                new_values = self.critic(states)
                value_loss = (new_values - return_).pow(2).mean()

                _, new_dists = self.actor(states)
                new_log_probs = new_dists.log_prob(actions)
                entropy = new_dists.entropy().mean()
                ratio = (new_log_probs - log_probs).exp()

                # entropy
                policy_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - self.epsilon,
                                                                  1 + self.epsilon) * adv).mean() - (
                                          entropy * 0.005)

                self.critic_optimizer.zero_grad()
                value_loss.backward(retain_graph=True)
                self.critic_optimizer.step()

                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                self.actor_optimizer.step()

        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.dones = []
        self.values = []


def compute_gae(
    next_value: list,
    rewards: list,
    masks: list,
    values: list,
    gamma: float,
    tau: float
) -> list:
    """Compute gae."""
    values = values + [next_value]
    gae = 0
    returns: Deque[float] = deque()

    for step in reversed(range(len(rewards))):
        delta = (
            rewards[step]
            + gamma * values[step + 1] * masks[step]
            - values[step]
        )
        gae = delta + gamma * tau * masks[step] * gae
        returns.appendleft(gae + values[step])

    return list(returns)


if __name__ == '__main__':

    args = {
        "env_id": "Pendulum-v1",
        "gamma": 0.9,
        "lambda": 0.8,
        "entropy_weight": 1e-2,
        "seed": 777,
        "n_steps": 10000,
        "rollout_len": 2048,
        "batch_size": 64,
        "epsilon": 0.2,
        "plotting_interval": 100,
        "log_dir": "../../runs",
        "checkpoint": "../../models/pendulum/ppo/"
    }

    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])

    env = gym.make(args["env_id"])
    env = ActionNormalizer(env)

    model = PPO(env, args)

    model.train(n_steps=args["n_steps"])

    # test_env = gym.make(args["env_id"], render_mode="rgb_array")
    # test_env = RecordVideo(test_env, video_folder="../../videos/pendulum-agent", name_prefix="eval",
    #                   episode_trigger=lambda x: True)
    #
    # test_model = PPO(test_env, args)
    # test_model.load()
    # test_model.test()