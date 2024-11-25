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

class ActionNormalizer(gym.ActionWrapper):
    """Rescale and relocate the actions."""

    def action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (-1, 1) to (low, high)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (low, high) to (-1, 1)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = (action - reloc_factor) / scale_factor
        action = np.clip(action, -1.0, 1.0)

        return action

class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(Actor, self).__init__()
        self.hidden = nn.Linear(in_dim, 64)
        self.mu_layer = nn.Linear(64, out_dim)
        self.log_std_layer = nn.Linear(64, out_dim)

        init_uniformly(self.mu_layer)
        init_uniformly(self.log_std_layer)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, Normal]:
        x = F.relu(self.hidden(state))

        mu = torch.tanh(self.mu_layer(x))
        log_std = F.softplus(self.log_std_layer(x))
        # TODO: normalize log_std?
        # log_std = self.log_std_min + 0.5 * (
        #         self.log_std_max - self.log_std_min
        # ) * (log_std + 1)
        std = torch.exp(log_std)

        dist = Normal(mu.squeeze(-1), std.squeeze(-1))
        action = dist.sample()

        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        super(Critic, self).__init__()
        self.hidden = nn.Linear(in_dim, 64)
        self.out = nn.Linear(64, 1)

        init_uniformly(self.out)

    def forward(self, state):
        x = F.relu(self.hidden(state))
        value = self.out(x)

        return value


class PPO:
    """
    pseudocode for PPO
    for iteration k...n_steps do
        reset the environment
        reset the memory
        get start state s
        for iteration j...roll_out_length do:
            sample action a from policy p
            reward, next_state, done = environment.step(a)
            store the tuple (state, log_prob(a), reward, done) to memory m
            if not done
                state = next_state
            else
                state = environment.reset()
        store the last next_state for compute GAE
        calculate all returns (for value loss):
            initialize R = 0 if done[-1] else V(s[t+1])  # Bootstrap from the value of the last state if not done
            for i in roll_out_length...0:
                R = reward[i] + gamma * R
                returns[i] = R

        calculate GAE from memory:
            init gae = 0
            for t in roll_out_length...0:
                delta[t] = m.reward[t] + gamma*V(s[t+1]) - V(s[t])
                gae[t] = delta[t] + gamma*lamda*gae[t+1]
        for iteration k...roll_out_length/batch_size:
            draw batch_size samples from memory m
            compute value loss L_vf
                v_predict = critic(s)
                v_target = returns(s)
                loss = (v_predict - v_target)**2
            compute surrogate loss L_clip
                action, dist = actor(s)
                new_log_prob = dist.log_prob(action)
                ratio = new_log_prob/log_prob
                loss = min(ratio*gae, clip(ratio, 1-epsilon, 1+epsilon)*gae)
            compute entropy bonus
            use gradient descent to optimize model
    """
    def __init__(self, env, args):
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
        self.actor = Actor(obs_dim, action_dim).to(self.device).to(self.device)
        self.critic = Critic(obs_dim).to(self.device).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.005)

        # memory
        self.states = torch.zeros((self.args["rollout_len"], obs_dim), device=self.device)
        self.actions = torch.zeros(self.args["rollout_len"], device=self.device)
        self.rewards = torch.zeros(self.args["rollout_len"], device=self.device)
        self.log_probs = torch.zeros(self.args["rollout_len"], device=self.device)
        self.dones = torch.zeros(self.args["rollout_len"], device=self.device)
        self.values = torch.zeros(self.args["rollout_len"], device=self.device)

        # mode: train/test
        self.is_test = False

        # logging/visualizing
        self.writer = SummaryWriter(args["log_dir"])

    def store(self, i, state, action, reward, log_prob, done, value):
        self.states[i] = state.detach()
        self.actions[i] = action.detach()
        self.rewards[i] = reward.detach()
        self.log_probs[i] = log_prob.detach()
        self.dones[i] = done
        self.values[i] = value.detach()

    def select_action(self, state):
        with torch.no_grad():
            action, dist = self.actor(state)
            value = self.critic(state)
        return action, dist, value

    def train(self, n_steps):
        scores = []
        for i in range(1, n_steps):
            state, _ = env.reset()
            score = 0

            for j in range(self.args["rollout_len"]):
                state = torch.tensor(state).to(self.device)
                action, dist, value = self.select_action(state)
                log_prob = dist.log_prob(action)

                next_state, reward, terminated, truncated, _ = self.env.step(action.detach().cpu().numpy())
                done = terminated or truncated

                score += reward
                self.store(j, state, action, torch.tensor(reward, device=self.device), log_prob, done, value)

                if done:
                    state, _ = env.reset()
                    # scores.append(score)
                    print(f"step {i=}, {score=}")
                    score = 0
                else:
                    state = next_state

            self.update_model(torch.tensor(next_state).to(self.device))

            # plot_debug_variables()

    def update_model(self, last_state):
        returns = self.compute_gae(last_state)
        returns = torch.tensor(returns, device=self.device).detach()
        avds = returns - self.values
        for _ in range(self.args["rollout_len"] // self.args["batch_size"]):
            indices = np.random.choice(self.args["rollout_len"], self.args["batch_size"])
            states, actions, rewards, log_probs, dones, values = (self.states[indices], self.actions[indices],
                                                                  self.rewards[indices], self.log_probs[indices],
                                                                  self.dones[indices], self.values[indices])

            new_values = self.critic(states)
            value_loss = ((new_values - returns) ** 2).mean()

            self.critic_optimizer.zero_grad()
            # TODO: add retain_graph = True
            value_loss.backward(retain_graph=True)
            self.critic_optimizer.step()

            adv = avds[indices]
            _, new_dists = self.actor(states)
            new_log_probs = new_dists.log_prob(actions)
            ratio = (new_log_probs / log_probs).exp()
            policy_loss = -torch.min(ratio*adv, torch.clamp(ratio, 1 - self.args["epsilon"], 1 + self.args["epsilon"]) * adv).mean()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()



    def compute_gae(self, last_state):
        n = self.args["rollout_len"]
        values = self.values.tolist()
        values.append(self.critic(last_state).detach())
        returns = []
        gae = 0
        gamma = self.args["gamma"]
        lambda_gae = self.args["lambda"]
        for i in reversed(range(n)):
            delta = self.rewards[i] + gamma * values[i+1]*self.dones[i] - values[i]
            gae = delta + gamma * lambda_gae * gae * self.dones[i]
            returns.append(gae + values[i])
        return list(reversed(returns))


if __name__ == '__main__':

    args = {
        "env_id": "Pendulum-v1",
        "gamma": 0.99,
        "lambda": 0.95,
        "entropy_weight": 1e-2,
        "T": 50,
        "seed": 777,
        "n_steps": 10000,
        "rollout_len": 200,
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