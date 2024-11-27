import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from rl_bot.action_normalizer import ActionNormalizer
from rl_bot.ou_noise import OUNoise
from rl_bot.replay_memory import ReplayMemory
from rl_bot.utils import init_uniformly


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
        self.out = nn.Linear(64, out_dim)
        init_uniformly(self.out)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.hidden_1(x))
        out = self.out(x)

        return out


class DDPG:
    def __init__(self, env: gym.Env, args):
        self.env = env

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )

        self.n_steps = args["n_steps"]
        self.batch_size = args["batch_size"]
        self.gamma = args["gamma"]
        self.tau = args["tau"]
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
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args["lr"])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args["lr"])

        self.noise = OUNoise(action_dim, theta=args["ou_noise_theta"], sigma=args["ou_noise_sigma"])
        self.is_test = False
        # logging/visualizing
        self.writer = SummaryWriter(args["log_dir"])

    def select_action(self, state):
        action = self.actor(torch.tensor(state, device=self.device)).detach().cpu().numpy()

        if not self.is_test:
            noise = self.noise.sample()
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

            actor_loss, critic_loss = self.update_model()

            if len(scores) >= 10:
                s = np.mean(scores)
                print(f"{i=}, score={s}")
                self.writer.add_scalar("score", s, i)
                self.writer.add_scalar("actor loss", actor_loss, i)
                self.writer.add_scalar("critic loss", critic_loss, i)
                scores.clear()


    def update_model(self):
        if len(self.memory) < self.batch_size:
            return 0.0, 0.0
        data = self.memory.sample(self.batch_size)
        states = data["states"]
        next_states = data["next_states"]
        actions = data["actions"]

        action_t = self.actor_t(next_states)
        q_value_t = self.critic_t(next_states, action_t)
        targets = data["rewards"].unsqueeze(-1) + self.gamma * q_value_t * data["dones"].unsqueeze(-1)

        q_value = self.critic(states, actions.unsqueeze(-1))

        critic_loss = F.mse_loss(q_value, targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.targets_soft_update()

        return actor_loss.item(), critic_loss.item()

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
        "lr": 1e-4,
        "gamma": 0.99,
        "seed": 777,
        "tau": 5e-3,
        "ou_noise_theta": 1.0,
        "ou_noise_sigma": 0.1,
        "n_steps": 100000,
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

    model = DDPG(env, args)
    model.train()
