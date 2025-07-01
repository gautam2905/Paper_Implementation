import gymnasium as gym
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# --- 1. Neural Network Definition (DQN) ---
class DQN(nn.Module):
    def __init__(self, hidden_units, state_size, action_size):
        super(DQN, self).__init__()
        # self.q_network = nn.Sequential(
        #     nn.Linear(state_size, hidden_units),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden_units, hidden_units),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden_units, action_size)
        # )
        self.q_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (state_size // 2), hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, action_size)
        )

    def forward(self, x):
        """Forward pass through the network."""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.q_network[0].weight.device)
        return self.q_network(x)

# --- 2. Environment and Action Space Setup ---
env = gym.make('ALE/Breakout-v5')
state_size = env.observation_space.shape[0]

num_joints = env.action_space.shape[0]
base_actions = np.eye(num_joints)
discrete_actions = (
    [[0.0] * num_joints] +
    base_actions.tolist() +
    (-base_actions).tolist()
)
action_size = len(discrete_actions)

# --- 3. Hyperparameters for Long Training ---
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
# Slower decay for very long exploration phase
epsilon_decay = 0.99995
# Stable learning rate
learning_rate = 3e-4
# Larger batch size for GPU training
batch_size = 256
# Very large replay buffer for diverse experiences
memory_size = 1_000_000
# Larger 'warm-up' phase before training starts
learning_starts = 25000
train_frequency = 4
tau = 0.005
# Increased episodes for a long run
episodes = 10000
max_steps_per_episode = 1000
# Checkpointing parameters
save_dir = "dqn_walker_models"
os.makedirs(save_dir, exist_ok=True)
checkpoint_freq = 1000 # Save a checkpoint every 1000 episodes

# --- 4. Initialization ---
memory = deque(maxlen=memory_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"State size: {state_size}")
print(f"Discretized action size: {action_size}")

q_network = DQN(256, state_size=state_size, action_size=action_size).to(device)
target_network = DQN(256, state_size=state_size, action_size=action_size).to(device)
target_network.load_state_dict(q_network.state_dict())
target_network.eval()

optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# --- 5. Core RL Functions ---
def select_action(state, current_epsilon):
    if random.random() < current_epsilon:
        return random.randrange(action_size)
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = q_network(state_tensor)
            return q_values.argmax().item()

def train_model():
    if len(memory) < batch_size:
        return None

    minibatch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*minibatch)

    states = torch.FloatTensor(np.array(states)).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    next_states = torch.FloatTensor(np.array(next_states)).to(device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

    current_q_values = q_network(states).gather(1, actions)

    with torch.no_grad():
        next_q_values = target_network(next_states).max(1)[0].detach().unsqueeze(1)
    
    target_q_values = rewards + (gamma * next_q_values * (1 - dones))

    loss = loss_fn(current_q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_network.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()

def soft_update_target_network():
    for target_param, local_param in zip(target_network.parameters(), q_network.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

# --- 6. Training Loop ---
all_rewards = []
avg_rewards = []
total_timesteps = 0
best_avg_reward = -np.inf

for episode in tqdm(range(episodes)):
    state, _ = env.reset()
    episode_reward = 0
    
    for t in range(max_steps_per_episode):
        total_timesteps += 1
        
        action_index = select_action(state, epsilon)
        continuous_action = discrete_actions[action_index]
        
        next_state, reward, terminated, truncated, _ = env.step(continuous_action)
        done = terminated or truncated
        
        memory.append((state, action_index, reward, next_state, done))
        
        state = next_state
        episode_reward += reward

        if total_timesteps > learning_starts and total_timesteps % train_frequency == 0:
            train_model()
            soft_update_target_network()

        if done:
            break
            
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    all_rewards.append(episode_reward)
    avg_reward = np.mean(all_rewards[-100:])
    avg_rewards.append(avg_reward)

    if episode % 100 == 0:
        print(f"\nEpisode {episode} | Total Reward: {episode_reward:.2f} | Avg Reward (Last 100): {avg_reward:.2f} | Epsilon: {epsilon:.4f} | Timesteps: {total_timesteps}")

    # Checkpointing: Save the best model and periodic checkpoints
    if avg_reward > best_avg_reward:
        best_avg_reward = avg_reward
        torch.save(q_network.state_dict(), os.path.join(save_dir, "best_model.pth"))
        print(f"** New best model saved with avg reward: {best_avg_reward:.2f} **")

    if episode % checkpoint_freq == 0 and episode > 0:
        torch.save(q_network.state_dict(), os.path.join(save_dir, f"checkpoint_episode_{episode}.pth"))
        print(f"++ Checkpoint saved at episode {episode} ++")


# --- 7. Plotting the Results ---
plt.figure(figsize=(12, 6))
plt.plot(all_rewards, label='Episode Reward', alpha=0.6)
plt.plot(avg_rewards, label='Average Reward (100 episodes)', linewidth=2, color='red')
plt.title('DQN Training on Walker2d-v5 (Long Run)')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'training_curve.png'))
plt.show()
