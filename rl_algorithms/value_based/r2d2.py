import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import random
from collections import namedtuple, deque

"""
Architecture:
- Similar to Ape-X
- Use prioritized exp replay + distributed training
- n-step double Q-learning (n=5)
- 256 actors, 1 learner
- Use dueling network architecture of Wang et al (2016)
- An LSTM layer after the convolutional stack, similar to Gruslys
- Store fixed-length (m=80) sequences of (s, a, r) in replay instead of (s, a, r, s')
- Adjcent sequences overlapping each other by 40 time steps (never crossing episode boundary)
- Have online + target network, unroll both of them in training on the same sequence
- No clipped reward, use invertible value function of the form 
  h(x) = sign(x)(sqrt(abs(x) + 1) - 1) + epsilon*x
- a* = argmax(a, Q_online(s_t+n, a)
- Replay buffer use a mixture of max and mean absolute n-step TD-errors delta_i over the sequence:
  p = eta*max(delta_i) + (1-eta)*mean_delta. eta and priority exponent = 0.9
- This agressive scheme is motivated by observation that averaging over
  long sequences tends to wash out large errors -> compressing the priorities range
  
Training:
- Stored state: Storing the recurrent state in replay and using it to initialize the network at training time
  remedies the weakness of the zero start state strategy but suffers from 
  "representational drift and recurrent state staleness"
- Burn-in: Using a portion of the replay sequence to init the hidden state, and update the net only on
  the remaining part of the sequence
"""

# Prioritized Experience Replay Buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        """
        Prioritized Experience Replay Buffer with recurrent support

        Args:
            capacity (int): Maximum number of experiences to store
            alpha (float): How much prioritization is used (0 - no prioritization, 1 - full prioritization)
            beta (float): To what degree IS weights are used (0 - no corrections, 1 - full correction)
            beta_increment_per_sampling (float): Increment beta by this amount every sampling
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling

        # Store experiences with priorities
        self.memory = []
        self.priorities = []
        self.position = 0

    def add(self, experience, priority):
        """
        Add new experience to the buffer

        Args:
            experience (tuple): Experience to store
            priority (float): Priority of the experience
        """
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
            self.priorities.append(priority)
        else:
            self.memory[self.position] = experience
            self.priorities[self.position] = priority

        # Circular buffer management
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, sequence_length):
        """
        Sample a batch of experiences with prioritized sampling

        Args:
            batch_size (int): Number of sequences to sample
            sequence_length (int): Length of each sequence

        Returns:
            Sampled experiences with importance sampling weights
        """
        # Increase beta
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        # Compute priorities
        if len(self.memory) == 0:
            return None, None, None

        # Compute sampling probabilities
        scaled_priorities = np.array(self.priorities) ** self.alpha
        total_priority = np.sum(scaled_priorities)
        probabilities = scaled_priorities / total_priority

        # Sample sequences
        sampled_indices = np.random.choice(
            len(self.memory),
            size=batch_size,
            p=probabilities,
            replace=False
        )

        # Prepare sequences and corresponding weights
        sequences = []
        is_weights = []
        sampled_priorities = []

        for idx in sampled_indices:
            # Ensure we can sample a full sequence
            start = max(0, idx - sequence_length + 1)
            sequence = self.memory[start:idx + 1]

            # Pad sequence if needed
            if len(sequence) < sequence_length:
                padding = [sequence[0]] * (sequence_length - len(sequence))
                sequence = padding + sequence

            sequences.append(sequence)

            # Compute importance sampling weight
            sample_prob = probabilities[idx]
            weight = (len(self.memory) * sample_prob) ** (-self.beta)
            is_weights.append(weight)

            sampled_priorities.append(self.priorities[idx])

        # Normalize importance sampling weights
        is_weights = np.array(is_weights) / max(is_weights)

        return (
            torch.tensor(sequences, dtype=torch.float32),
            torch.tensor(is_weights, dtype=torch.float32),
            torch.tensor(sampled_priorities, dtype=torch.float32)
        )

    def update_priorities(self, indices, priorities):
        """
        Update priorities of sampled experiences

        Args:
            indices (list): Indices of sampled experiences
            priorities (list): New priorities for those experiences
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority


# Recurrent Neural Network for R2D2
class R2D2Network(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Recurrent Network for R2D2

        Args:
            input_dim (int): Dimension of input state
            hidden_dim (int): Dimension of LSTM hidden state
            output_dim (int): Number of actions
        """
        super(R2D2Network, self).__init__()

        # Feature extraction layers
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim)
        )

        # Recurrent layer
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # Output layers
        self.advantage_stream = nn.Linear(hidden_dim, output_dim)
        self.value_stream = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden_state=None):
        """
        Forward pass through the network

        Args:
            x (torch.Tensor): Input state sequence
            hidden_state (tuple, optional): Previous LSTM hidden state

        Returns:
            Q-values, updated hidden state
        """
        # Feature extraction
        features = self.feature_net(x)

        # Recurrent processing
        lstm_out, new_hidden_state = self.lstm(features, hidden_state)

        # Compute Q-values using Dueling DQN architecture
        advantages = self.advantage_stream(lstm_out)
        values = self.value_stream(lstm_out)

        # Combine advantages and values
        q_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))

        return q_values, new_hidden_state


class R2D2Agent:
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim=64,
                 learning_rate=1e-4,
                 gamma=0.99,
                 tau=0.001,
                 buffer_capacity=10000,
                 sequence_length=8):
        """
        R2D2 Agent implementation

        Args:
            input_dim (int): Dimension of input state
            output_dim (int): Number of actions
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 64.
            learning_rate (float, optional): Learning rate. Defaults to 1e-4.
            gamma (float, optional): Discount factor. Defaults to 0.99.
            tau (float, optional): Soft update coefficient. Defaults to 0.001.
            buffer_capacity (int, optional): Replay buffer capacity. Defaults to 10000.
            sequence_length (int, optional): Length of experience sequences. Defaults to 8.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Network and target networks
        self.online_net = R2D2Network(input_dim, hidden_dim, output_dim).to(self.device)
        self.target_net = R2D2Network(input_dim, hidden_dim, output_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())

        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=learning_rate)

        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.sequence_length = sequence_length

        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(buffer_capacity)

        # Other parameters
        self.input_dim = input_dim
        self.output_dim = output_dim

    def select_action(self, state, hidden_state=None, epsilon=0.1):
        """
        Select action using epsilon-greedy policy

        Args:
            state (torch.Tensor): Current state
            hidden_state (tuple, optional): Previous LSTM hidden state
            epsilon (float, optional): Exploration rate. Defaults to 0.1.

        Returns:
            Action, new hidden state
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.output_dim), hidden_state

        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Get Q-values
        with torch.no_grad():
            q_values, new_hidden_state = self.online_net(state_tensor, hidden_state)

        # Select best action
        action = torch.argmax(q_values.squeeze(), dim=-1).cpu().item()

        return action, new_hidden_state

    def store_experience(self, state, action, reward, next_state, done, priority=1.0):
        """
        Store experience in replay buffer

        Args:
            state (np.array): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.array): Next state
            done (bool): Episode termination flag
            priority (float, optional): Experience priority. Defaults to 1.0.
        """
        experience = (state, action, reward, next_state, done)
        self.replay_buffer.add(experience, priority)

    def train(self, batch_size=32):
        """
        Train the agent using recurrent experience replay

        Args:
            batch_size (int, optional): Number of sequences to sample. Defaults to 32.

        Returns:
            Training loss
        """
        # Sample sequences from replay buffer
        sequences, is_weights, priorities = self.replay_buffer.sample(batch_size, self.sequence_length)

        if sequences is None:
            return None

        # Move to device
        sequences = sequences.to(self.device)
        is_weights = is_weights.to(self.device)

        # Split sequences into components
        states = sequences[:, :, 0]  # First element of each experience
        actions = sequences[:, :, 1].long()
        rewards = sequences[:, :, 2]
        next_states = sequences[:, :, 3]
        dones = sequences[:, :, 4].float()

        # Compute Q-values
        q_values, _ = self.online_net(states)
        next_q_values, _ = self.target_net(next_states)

        # Compute target Q-values
        max_next_q_values = next_q_values.max(dim=-1)[0]
        target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        # Gather Q-values for taken actions
        current_q_values = q_values.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)

        # Compute TD error
        td_error = torch.abs(current_q_values - target_q_values)

        # Compute weighted loss
        loss = F.mse_loss(current_q_values, target_q_values, reduction='none')
        weighted_loss = (loss * is_weights).mean()

        # Optimize
        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()

        # Update priorities
        new_priorities = td_error.detach().cpu().numpy() + 1e-6
        self.replay_buffer.update_priorities(range(batch_size), new_priorities)

        # Soft update target network
        for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )

        return weighted_loss.item()


def train_r2d2(env_name='CartPole-v1',
               episodes=1000,
               max_steps=500,
               batch_size=32):
    """
    Train R2D2 Agent on a given environment

    Args:
        env_name (str): Name of the OpenAI Gym environment
        episodes (int): Number of training episodes
        max_steps (int): Maximum steps per episode
        batch_size (int): Batch size for training
    """
    # Create environment
    env = gym.make(env_name)

    # Get environment dimensions
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    # Initialize R2D2 Agent
    agent = R2D2Agent(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=64,
        learning_rate=1e-4,
        buffer_capacity=10000,
        sequence_length=8
    )

    # Training loop
    episode_rewards = []

    for episode in range(episodes):
        # Reset environment
        state, _ = env.reset()

        # Episode variables
        total_reward = 0
        done = False
        steps = 0

        # Reset hidden state for each episode
        hidden_state = None

        while not done and steps < max_steps:
            # Select action
            action, hidden_state = agent.select_action(
                state,
                hidden_state,
                epsilon=max(0.1, 0.5 - episode / 200)  # Decaying epsilon
            )

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # Store experience
            agent.store_experience(state, action, reward, next_state, done)

            # Update state and reward
            state = next_state
            total_reward += reward
            steps += 1

            # Train agent periodically
            if len(agent.replay_buffer.memory) >= batch_size:
                agent.train(batch_size)

        # Record episode statistics
        episode_rewards.append(total_reward)

        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, Steps: {steps}")

        # Break if solved (for CartPole)
        if len(episode_rewards) >= 100 and np.mean(episode_rewards[-100:]) >= 195:
            print(f"Environment solved in {episode + 1} episodes!")
            break

    # Cleanup
    env.close()

    return agent, episode_rewards


def plot_rewards(rewards):
    """
    Plot training rewards

    Args:
        rewards (list): List of rewards per episode
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('R2D2 Agent Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.smooth_curve('mean')
    plt.grid(True)
    plt.show()


def main():
    """
    Main function to run R2D2 training
    """
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Train on CartPole environment
    agent, rewards = train_r2d2(
        env_name='CartPole-v1',
        episodes=500,
        max_steps=500,
        batch_size=32
    )

    # Plot rewards
    plot_rewards(rewards)

    # Save model
    torch.save(agent.online_net.state_dict(), 'r2d2_cartpole_model.pth')
    print("Model saved successfully!")


if __name__ == '__main__':
    main()