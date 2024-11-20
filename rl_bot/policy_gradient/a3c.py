import math
import multiprocessing as mp
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gymnasium as gym
from torch import nn
from torch.distributions import Normal

from rl_bot.shared_adam import SharedAdam
from rl_bot.utils import set_init


class A3C(nn.Module):
    def __init__(self, args: dict, obs_dim, action_dim):
        super(A3C, self).__init__()
        self.args = args

        self.actor_layer = nn.Linear(obs_dim, 200)
        self.mu_layer = nn.Linear(200, action_dim)
        self.sigma_layer = nn.Linear(200, action_dim)

        self.critic_layer = nn.Linear(obs_dim, 128)
        self.value_layer = nn.Linear(128, 1)

        set_init([self.actor_layer, self.mu_layer, self.sigma_layer, self.critic_layer, self.value_layer])

    def forward(self, x):
        h = F.relu6(self.actor_layer(x))
        mu = 2 * F.tanh(self.mu_layer(h))
        sigma = F.softplus(self.sigma_layer(h)) + 0.001

        c = F.relu6(self.critic_layer(x))
        values = self.value_layer(c)
        return mu, sigma, values

    def select_action(self, state):
        with torch.no_grad():
            mu, sigma, _ = self.forward(torch.tensor(state))
            dist = Normal(mu, sigma)
            return dist.sample().numpy()

    def loss(self, state, action, v_targets):
        mu, sigma, values = self.forward(state)
        td = v_targets - values.squeeze(-1)

        critic_loss = td.pow(2)

        dist = Normal(mu.squeeze(-1), sigma.squeeze(-1))
        log_prob = dist.log_prob(action.squeeze(-1))
        entropy = 0.5 + 0.5*math.log(2*math.pi) + torch.log(dist.scale)

        policy_loss = -log_prob * td.detach() + 0.005*entropy

        total_loss = (critic_loss + policy_loss).mean()

        return total_loss


class Worker(mp.Process):
    def __init__(self, gnet, optimizer, global_ep, global_ep_r, res_queue, index):
        super(Worker, self).__init__()
        self.name = f"Worker_{index}"
        self.args = gnet.args
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, optimizer
        self.env = gym.make(self.args["env_id"])
        self.net = A3C(gnet.args, self.env.observation_space.shape[0], self.env.action_space.shape[0])
        self.net.load_state_dict(self.gnet.state_dict())

    def run(self):
        step = 1
        max_t = self.args["max_t"]
        while self.g_ep.value < self.args["n_steps"]:
            state, _ = self.env.reset()
            states, actions, rewards = [], [], []
            episode_reward = 0.0

            for i in range(max_t):
                action = self.net.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action.clip(-2, 2))

                done = terminated or truncated or i == max_t - 1

                episode_reward += reward

                actions.append(action)
                states.append(state)
                rewards.append(max(min(reward, 1), -1))

                if step % self.args["update_global_net_interval"] == 0 or done:
                    push_and_pull(self.opt, self.net, self.gnet, done, next_state, states, actions, rewards, self.args["gamma"])
                    states, actions, rewards = [], [], []
                    if done:
                        record(self.g_ep, self.g_ep_r, episode_reward, self.res_queue, self.name)
                        break

                state = next_state
                step += 1
        self.res_queue.put(None)

def push_and_pull(opt, lnet, gnet, done, next_state, states, actions, rewards, gamma):
    if done:
        value = 0.0
    else:
        value = lnet.forward(torch.tensor(next_state))[-1].detach().numpy()[0]

    v_targers = []
    for r in rewards[::-1]:
        value = r + gamma * value
        v_targers.append(value)
    v_targers.reverse()

    loss = lnet.loss(torch.tensor(np.array(states)),
                     torch.tensor(np.array(actions)),
                     torch.tensor(np.array(v_targers)))

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm(lnet.parameters(), 20)
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    lnet.load_state_dict(gnet.state_dict())

def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(f"{name}, EP: {global_ep.value}, Reward: {global_ep_r.value}")


if __name__ == '__main__':
    args = {
        "env_id": "Pendulum-v1",
        "gamma": 0.9,
        "entropy_weight": 1e-2,
        "seed": 1993,
        "n_steps": 5000,
        "max_t": 200,
        "num_workers": 5,
        "plotting_interval": 100,
        "update_global_net_interval": 10,
        "log_dir": "../../runs",
        "checkpoint": "../../models/pendulum/"
    }

    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])

    g_env = gym.make("Pendulum-v1")

    gnet = A3C(args, g_env.observation_space.shape[0], g_env.action_space.shape[0])
    gnet.share_memory()

    optimizer = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.95, 0.99))

    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    workers = [Worker(gnet, optimizer, global_ep, global_ep_r, res_queue, i) for i in range(args["num_workers"])]
    [w.start() for w in workers]

    res = []
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break

    [w.join() for w in workers]

    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()