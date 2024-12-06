import math
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from rl_algorithms.utils import linear_schedule, beta_annealing
from rl_algorithms.value_based.mlp_policy import MLPPolicy
from rl_algorithms.replay_memory import PriporityMemory


class PER:

    def __init__(self, env: gym.vector.VectorEnv, args: dict):
        # check if GPU is avaialble
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        self.env = env
        self.args = args
        self.args["gamma"] = torch.tensor(args["gamma"]).to(self.device, dtype=torch.float32)
        self.policy_net = MLPPolicy(env).to(self.device)
        self.target_net = MLPPolicy(env).to(self.device)
        self.target_net.eval()
        self.replay_memory = PriporityMemory(args["replay_memory_size"],
                                             env.single_observation_space.shape,
                                             self.device,
                                             args["beta"])

        # copy weights from policy net to target net
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), self.args["lr"], amsgrad=True)

        # logging/visualizing
        self.writer = SummaryWriter(args["log_dir"])

    def train(self, n_steps: int):
        scores = []
        # Initialise sequence s1 = {x1} and preprocessed sequenced φ1 = φ(s1), we use mlp policy, no preprocess
        states, infos = self.env.reset(seed=self.args["seed"])
        # a mask to skip the terminated state to become the parent of a new state
        skipped = np.zeros(envs.num_envs)
        # stores the last reset step t, so we know how good our agents were
        # this is not needed in normal env, but for VecEnv
        last_reset = np.zeros(envs.num_envs)
        for i in range(1, n_steps):
            # select action a_t with probability epsilon
            actions = self.select_action(torch.tensor(states).to(self.device), i)
            # Execute action at in emulator and observe reward r_t and new state s_t+1
            next_states, rewards, terminations, truncations, infos = self.env.step(actions)
            # get sampling priority p_t = max(pi)
            priorities = np.full((envs.num_envs,),
                                 math.pow(self.replay_memory.max_priority, self.args["alpha"]),
                                 dtype=np.float32)
            # autoreset helps us mask and skip terminated or truncated observation
            autoreset = np.logical_or(terminations, truncations)
            # Store transition in replay memory D
            for v in range(self.env.num_envs):
                if not skipped[v]:
                    self.replay_memory.store((states[v],
                                              actions[v],
                                              rewards[v],
                                              next_states[v],
                                              autoreset[v],
                                              priorities[v]))
                if autoreset[v]:
                    scores.append(i - last_reset[v])
                    last_reset[v] = i
            # Set s_t+1 = s_t
            states = next_states
            skipped = autoreset.copy()
            # Sample random minibatch of transitions (φj, aj, rj, φj+1) from D
            loss = self.train_minibatch()
            self.replay_memory.beta = beta_annealing(self.args["beta"], 1.0, self.args["explore_duration"], i)

            if i % self.args["target_net_update_frequency"] == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if len(scores) >= self.args["log_frequency"]:
                self.writer.add_scalar("Avg DDQN Reward", np.mean(scores), i)
                self.writer.add_scalar("DDQN Loss", loss.item(), i)
                print(f"Timestep: {i}, Avg reward: {np.mean(scores)}, Loss: {loss.item()}, ")
                scores.clear()

        self.env.close()
        self.writer.close()

    def select_action(self, state, i: int):
        # epsilon greedy strategy
        epsilon = linear_schedule(self.args["epsilon_start"],
                                  self.args["epsilon_end"],
                                  self.args["explore_duration"],
                                  i)

        if random.random() <= epsilon:
            return self.env.action_space.sample()

        with torch.no_grad():
            logits = self.policy_net(state)
            return torch.argmax(logits, dim=1).cpu().numpy()

    def train_minibatch(self):
        if len(self.replay_memory) < self.args["minibatch_size"]:
            return

        data = self.replay_memory.sample(self.args["minibatch_size"])

        with torch.no_grad():
            actions = torch.argmax(self.policy_net(data["next_states"]), dim=1)
            # calculate q_target
            q_targets = torch.gather(self.target_net(data["next_states"]), 1, actions.unsqueeze(1).long()).squeeze(1)
            # y_j = r_j if s_j+1 is terminal state
            # else r_j + gamma*q_targets
            target = data["rewards"] + self.args["gamma"] * q_targets * (1 - data["dones"])

        q_values = torch.gather(self.policy_net(data["states"]), 1, data["actions"].unsqueeze(1).long()).squeeze(1)

        elementwise_loss = F.mse_loss(q_values, target, reduction="none")
        # typical backward pass
        loss = torch.mean(elementwise_loss * data["weights"])
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()

        # update priorities
        with torch.no_grad():
            elementwise_loss = elementwise_loss.detach().cpu().numpy()
            indices = data["indices"]
            for i in range(len(indices)):
                self.replay_memory.p_tree[indices[i]] = elementwise_loss[i] ** self.args["alpha"] + self.args["epsilon"]
            max_p = elementwise_loss.max()
            self.replay_memory.max_priority = max(self.replay_memory.max_priority, max_p ** self.args["alpha"])

        return loss


if __name__ == '__main__':
    args = {
        "num_envs": 8,
        "replay_memory_size": 10000,
        "minibatch_size": 128,
        "lr": 1e-4,
        "epsilon_start": 0.9,
        "epsilon_end": 0.05,
        "explore_duration": 5000,
        "gamma": 0.99,
        "seed": 1993,
        "target_net_update_frequency": 20,
        "n_steps": 15000,
        "log_frequency": 10,
        "log_dir": "../../runs",
        # PER parameters
        "alpha": 0.5,
        "beta": 0.1,
        "epsilon": 1e-6,
    }

    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])

    envs = gym.make_vec("CartPole-v1", num_envs=args["num_envs"])

    model = PER(envs, args)

    model.train(n_steps=args["n_steps"])
