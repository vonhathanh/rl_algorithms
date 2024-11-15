import os
import datetime
import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn.functional as F
from gymnasium.wrappers import RecordVideo
from torch import nn, optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from rl_bot.utils import init_uniformly


class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(Actor, self).__init__()
        self.hidden = nn.Linear(in_dim, 128)
        self.mu_layer = nn.Linear(128, out_dim)
        self.log_std_layer = nn.Linear(128, out_dim)

        init_uniformly(self.mu_layer)
        init_uniformly(self.log_std_layer)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, Normal]:
        x = F.relu(self.hidden(state))

        mu = torch.tanh(self.mu_layer(x)) * 2
        log_std = F.softplus(self.log_std_layer(x))
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        action = dist.sample()

        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        super(Critic, self).__init__()
        self.hidden = nn.Linear(in_dim, 128)
        self.out = nn.Linear(128, 1)

        init_uniformly(self.out)

    def forward(self, state):
        x = F.relu(self.hidden(state))
        value = self.out(x)

        return value

class A2C:

    def __init__(self, env: gym.Env, args: dict):
        # check if GPU is avaialble
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        self.env = env
        self.args = args

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        # transition (state, log_prob, next_state, reward, done)
        self.transition: list = list()
        # mode: train/test
        self.is_test = False
        # logging/visualizing
        self.writer = SummaryWriter(args["log_dir"])

    def select_action(self, state: np.ndarray):
        state = torch.FloatTensor(state).to(self.device)
        action, dist = self.actor(state)
        selected_action = dist.mean if self.is_test else action

        if not self.is_test:
            log_prob = dist.log_prob(selected_action).sum(dim=-1)
            self.transition = [state, log_prob]

        return selected_action.clamp(-2.0, 2.0).cpu().detach().numpy()

    def step(self, action):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        if not self.is_test:
            self.transition.extend([next_state, reward, done])
        return next_state, reward, done

    def update_model(self):
        """Update the model by gradient descent."""
        state, log_prob, next_state, reward, done = self.transition
        mask = 1 - done
        next_state = torch.FloatTensor(next_state).to(self.device)
        pred_value = self.critic(state)
        target = reward + self.args["gamma"] * self.critic(next_state) * mask
        value_loss = F.smooth_l1_loss(pred_value, target.detach())

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        advantage = (target - pred_value).detach()
        policy_loss = -advantage * log_prob
        policy_loss += self.args["entropy_weight"] * -log_prob

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        return policy_loss.item(), value_loss.item()


    def train(self, n_steps):
        self.is_test = False

        state, _ = self.env.reset()
        score = 0

        for i in range(1, n_steps+1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            actor_loss, critic_loss = self.update_model()

            state = next_state
            score += reward

            if i % self.args["plotting_interval"] == 0:
                print(f"{i=}")
                self.plot(i, score, actor_loss, critic_loss)

            if done:
                state, _ = self.env.reset()
                score = 0

        self.env.close()
        self.save()

    def save(self):
        data = self.args

        data["actor"] = self.actor.state_dict()
        data["critic"] = self.critic.state_dict()

        os.makedirs(self.args["checkpoint"], exist_ok=True)

        curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        torch.save(data, os.path.join(self.args["checkpoint"], f"model_{curr_time}.pt"))

    def load(self):
        if self.args.get("model_id", ""):
            model_id = self.args["model_id"]
        else: # load the last  model
            model_id = sorted(os.listdir(self.args["checkpoint"]))[-1]

        print(f"Loading weights and bias of: {model_id}")
        state_dict = torch.load(os.path.join(self.args["checkpoint"], model_id))

        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])

    def plot(
            self,
            frame_idx: int,
            score,
            actor_loss,
            critic_loss,
    ):
        """Plot the training progresses."""
        self.writer.add_scalar("Score", score, frame_idx)
        self.writer.add_scalar("actor_loss", actor_loss, frame_idx)
        self.writer.add_scalar("critic_loss", critic_loss, frame_idx)


    def test(self):
        self.is_test = True
        state, _ = self.env.reset()
        done = False
        score = 0

        self.actor.eval()
        self.critic.eval()

        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        print(f"{score=}")
        self.env.close()


if __name__ == '__main__':

    args = {
        "env_id": "Pendulum-v1",
        "gamma": 0.9,
        "entropy_weight": 1e-2,
        "seed": 1993,
        "n_steps": 100000,
        "plotting_interval": 100,
        "log_dir": "../../runs",
        "checkpoint": "../../models/pendulum/"
    }

    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])

    env = gym.make(args["env_id"])

    model = A2C(env, args)

    model.train(n_steps=args["n_steps"])

    test_env = gym.make(args["env_id"], render_mode="rgb_array")
    test_env = RecordVideo(test_env, video_folder="../../videos/pendulum-agent", name_prefix="eval",
                      episode_trigger=lambda x: True)

    test_model = A2C(test_env, args)
    test_model.load()
    test_model.test()