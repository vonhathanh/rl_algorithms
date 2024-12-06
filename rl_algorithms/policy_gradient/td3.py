import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from rl_algorithms.action_normalizer import ActionNormalizer
from rl_algorithms.ou_noise import OUNoise
from rl_algorithms.replay_memory import ReplayMemory
from rl_algorithms.utils import init_uniformly


class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(Actor, self).__init__()
        self.hidden_1 = nn.Linear(in_dim, 64)
        # self.hidden_2 = nn.Linear(32, 32)
        self.out = nn.Linear(64, out_dim)
        init_uniformly(self.out)

    def forward(self, state):
        x = F.relu(self.hidden_1(state))
        out = F.tanh(self.out(x))

        return out

class Critic(nn.Module):
    def __init__(self, in_dim: int, out_dim: int=1):
        super(Critic, self).__init__()
        self.hidden_1 = nn.Linear(in_dim, 64)
        self.q1 = nn.Linear(64, out_dim)

        self.hidden_2 = nn.Linear(in_dim, 64)
        self.q2 = nn.Linear(64, out_dim)
        init_uniformly(self.q2)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)

        hidden_1 = F.relu(self.hidden_1(x))
        q1 = self.q1(hidden_1)

        hidden_2 = F.relu(self.hidden_2(x))
        q2 = self.q2(hidden_2)

        return q1, q2

    def Q1(self, state, action):
        x = torch.cat((state, action), dim=-1)

        hidden_1 = F.relu(self.hidden_1(x))
        q1 = self.q1(hidden_1)

        return q1


class TD3:
    def __init__(self, env: gym.Env, args):
        self.env = env

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )

        self.n_steps = args["n_steps"]
        self.random_steps = args["random_steps"]
        self.policy_update_interval = args["policy_update_interval"]
        self.batch_size = args["batch_size"]
        self.gamma = args["gamma"]
        self.tau = args["tau"]
        self.noise_scale = 0.1
        self.log_frequency = args["log_frequency"]

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.memory = ReplayMemory(args["replay_memory_size"], env.observation_space.shape, self.device)

        # networks
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim + action_dim).to(self.device)

        self.actor_t = Actor(obs_dim, action_dim).to(self.device)
        self.critic_t = Critic(obs_dim + action_dim).to(self.device)

        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic_t.load_state_dict(self.critic.state_dict())

        # optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.is_test = False
        # logging/visualizing
        self.writer = SummaryWriter(args["log_dir"])

    def select_action(self, state):
        if not self.is_test and len(self.memory) < self.random_steps:
            action = self.env.action_space.sample()
        else:
            action = self.actor(torch.tensor(state, device=self.device)).detach().cpu().numpy()

        if not self.is_test:
            noise = np.random.normal(0, self.noise_scale, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)

        return action

    def train(self):
        state, _ = self.env.reset()
        scores = []
        score = 0

        for i in range(1, self.n_steps):
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.memory.store((state, action, reward, next_state, 1 - done))

            state = next_state
            score += reward
            if done:
                state, _ = self.env.reset()
                scores.append(score)
                score = 0

            self.update_model(i)

            if len(scores) >= 10:
                s = np.mean(scores)
                print(f"{i=}, score={s}")
                self.writer.add_scalar("score", s, i)
                scores.clear()


    def update_model(self, i):
        if len(self.memory) < self.batch_size or len(self.memory) < self.random_steps:
            return 0.0, 0.0

        data = self.memory.sample(self.batch_size)
        states = data["states"]
        next_states = data["next_states"]
        actions = data["actions"]
        with torch.no_grad():
            noise = (torch.rand_like(actions) * self.noise_scale).clamp(-0.5, 0.5)

            action_t = (self.actor_t(next_states) + noise.unsqueeze(-1)).clamp(-1.0, 1.0)
            q_target_1, q_target_2 = self.critic_t(next_states, action_t)
            targets = data["rewards"].unsqueeze(-1) + self.gamma * torch.min(q_target_1, q_target_2) * data["dones"].unsqueeze(-1)

        q1, q2 = self.critic(states, actions.unsqueeze(-1))

        critic_loss = F.mse_loss(q1, targets) + F.mse_loss(q2, targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if i % self.policy_update_interval:
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.targets_soft_update()


    def targets_soft_update(self):
        actor_state_dict = self.actor.state_dict()
        critic_state_dict = self.critic.state_dict()
        actor_tgt_state_dict = self.actor_t.state_dict()
        critic_tgt_state_dict = self.critic_t.state_dict()

        for k in actor_state_dict:
            actor_tgt_state_dict[k] = actor_state_dict[k] * self.tau + actor_tgt_state_dict[k] * (1 - self.tau)
        self.actor_t.load_state_dict(actor_tgt_state_dict)

        for k in critic_state_dict:
            critic_tgt_state_dict[k] = critic_state_dict[k] * self.tau + critic_tgt_state_dict[k] * (1 - self.tau)
        self.critic_t.load_state_dict(critic_tgt_state_dict)


if __name__ == '__main__':
    args = {
        "env_id": "Pendulum-v1",
        "replay_memory_size": 100000,
        "batch_size": 128,
        "gamma": 0.99,
        "seed": 777,
        "tau": 5e-3,
        "n_steps": 100000,
        "random_steps": 10000,
        "policy_update_interval": 2,
        "log_frequency": 10,
        "log_dir": "../../runs"
    }

    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])

    env = gym.make(args["env_id"])
    env = ActionNormalizer(env)

    model = TD3(env, args)
    model.train()
