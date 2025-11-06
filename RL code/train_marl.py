import numpy as np
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_tag_v3
from collections import defaultdict
from tqdm import trange
import torch 
from torch import nn
from torch.nn import functional as F
import collections
from tqdm import tqdm
import os

os.makedirs("models", exist_ok=True) 
env = simple_tag_v3.parallel_env(num_good=1, num_adversaries=1, render_mode=None)
env.reset(seed=42)

# OPTIMIZED HYPERPARAMETERS for JAL
alpha_q = 0.005           # Q-network learning rate
alpha_model = 0.01        # Model learning rate (higher to learn opponent faster)
WARMUP_STEPS = 1000        # Very short warmup (just fill buffer minimally)
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.1          # Keep some exploration
epsilon_decay_steps = 50000
decay_rate = (epsilon - epsilon_min) / epsilon_decay_steps
episodes = 100000
BUFFER_CAPACITY = 200000   # Smaller buffer
BATCH_SIZE = 128           # Larger batches
GAMMA = 0.99       
TAU = 0.005                # Target network update rate
UPDATE_EVERY = 4           # Update networks every N steps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

## Deep Q-Network (DQN)
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class ModelNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)
    
    def get_probs(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in batch])
        return (np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)

class ModelBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, opponent_action):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, opponent_action)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        states, actions = zip(*[self.buffer[idx] for idx in batch])
        return (np.array(states), np.array(actions))

    def __len__(self):
        return len(self.buffer)

# Initialize environment dimensions
global_state_dim = env.observation_space('adversary_0').shape[0] + env.observation_space('agent_0').shape[0]
action_dim_0 = env.action_space('agent_0').n
action_dim_1 = env.action_space('adversary_0').n
joint_action_dim = action_dim_0 * action_dim_1

print(f"State dim: {global_state_dim}, Agent actions: {action_dim_0}, Adversary actions: {action_dim_1}")

# Agent 0 (Good Agent)
agent_net = QNetwork(global_state_dim, joint_action_dim).to(device)
agent_target_net = QNetwork(global_state_dim, joint_action_dim).to(device)
agent_target_net.load_state_dict(agent_net.state_dict())
agent_model = ModelNetwork(global_state_dim, action_dim_1).to(device)

# Agent 1 (Adversary)
adversary_net = QNetwork(global_state_dim, joint_action_dim).to(device)
adversary_target_net = QNetwork(global_state_dim, joint_action_dim).to(device)
adversary_target_net.load_state_dict(adversary_net.state_dict())
adversary_model = ModelNetwork(global_state_dim, action_dim_0).to(device)

# Buffers
agent_buffer = ReplayBuffer(BUFFER_CAPACITY)
adversary_buffer = ReplayBuffer(BUFFER_CAPACITY)
agent_model_buffer = ModelBuffer(BUFFER_CAPACITY)
adversary_model_buffer = ModelBuffer(BUFFER_CAPACITY)

# Optimizers
q_optimizer_0 = torch.optim.Adam(agent_net.parameters(), lr=alpha_q)
q_optimizer_1 = torch.optim.Adam(adversary_net.parameters(), lr=alpha_q)
model_optimizer_0 = torch.optim.Adam(agent_model.parameters(), lr=alpha_model)
model_optimizer_1 = torch.optim.Adam(adversary_model.parameters(), lr=alpha_model)

# Loss functions
q_loss_fn = nn.SmoothL1Loss()  # Huber loss (more robust than MSE)
model_loss_fn = nn.CrossEntropyLoss()

# Tracking
recent_rewards_agent_0 = collections.deque(maxlen=100)
recent_rewards_agent_1 = collections.deque(maxlen=100)
recent_model_acc_0 = collections.deque(maxlen=100)
recent_model_acc_1 = collections.deque(maxlen=100)


episode_rewards_0 = []
episode_rewards_1 = []
episode_lengths = []
q_losses_0 = []
q_losses_1 = []
model_losses_0 = []
model_losses_1 = []
model_accuracies_0 = []
model_accuracies_1 = []
epsilon_history = []
avg_rewards_0 = []
avg_rewards_1 = []




progress_bar = tqdm(range(episodes))
global_total_steps = 0

for episode in progress_bar:
    episode_reward_agent_0 = 0
    episode_reward_agent_1 = 0
    cumulative_q_loss_0 = 0
    cumulative_q_loss_1 = 0
    cumulative_model_loss_0 = 0
    cumulative_model_loss_1 = 0
    model_correct_0 = 0
    model_correct_1 = 0
    
    episode_steps = 0
    num_q_updates = 0
    num_model_updates = 0

    done = False
    obs, info = env.reset()
    
    while not done:
        global_total_steps += 1
        episode_steps += 1
        global_state = np.concatenate((obs['adversary_0'], obs['agent_0']))
        state_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(device)

        # Action selection
        with torch.no_grad():
            # Agent 0's action values
            all_joint_q_agent = agent_net(state_tensor)
            probs_opponent_1 = agent_model.get_probs(state_tensor)
            
            q_matrix_agent = all_joint_q_agent.view(action_dim_0, action_dim_1)
            probs_opponent_1_vec = probs_opponent_1.view(action_dim_1, 1)
            action_values_agent = torch.matmul(q_matrix_agent, probs_opponent_1_vec).squeeze()

            # Agent 1's action values
            all_joint_q_adversary = adversary_net(state_tensor)
            probs_opponent_0 = adversary_model.get_probs(state_tensor)
            
            q_matrix_adversary = all_joint_q_adversary.view(action_dim_0, action_dim_1)
            probs_opponent_0_vec = probs_opponent_0.view(action_dim_0, 1)
            action_values_adversary = torch.matmul(q_matrix_adversary.T, probs_opponent_0_vec).squeeze()

        # Epsilon-greedy
        if np.random.rand() < epsilon:
            agent_action = np.random.choice(action_dim_0)
            adversary_action = np.random.choice(action_dim_1)
        else:
            agent_action = torch.argmax(action_values_agent).item()
            adversary_action = torch.argmax(action_values_adversary).item()

        # Environment step
        actions = {'adversary_0': adversary_action, "agent_0": agent_action}
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        done = all(terminations.values()) or all(truncations.values())
        global_next_obs = np.concatenate((next_obs['adversary_0'], next_obs['agent_0']))
        
        episode_reward_agent_0 += rewards['agent_0']
        episode_reward_agent_1 += rewards['adversary_0']

        # Store transitions
        joint_action = agent_action * action_dim_1 + adversary_action
        agent_buffer.push(global_state, joint_action, rewards['agent_0'], global_next_obs, done)
        adversary_buffer.push(global_state, joint_action, rewards['adversary_0'], global_next_obs, done)
        agent_model_buffer.push(global_state, adversary_action)
        adversary_model_buffer.push(global_state, agent_action)

        # Train networks every UPDATE_EVERY steps
        if global_total_steps % UPDATE_EVERY == 0 and len(agent_buffer) >= BATCH_SIZE:
            
            # ===== TRAIN OPPONENT MODELS =====
            if len(agent_model_buffer) >= BATCH_SIZE:
                # Train Agent 0's model of Adversary
                states_m, actions_m = agent_model_buffer.sample(BATCH_SIZE)
                states_m = torch.FloatTensor(states_m).to(device)
                actions_m = torch.LongTensor(actions_m).to(device)
                
                model_optimizer_0.zero_grad()
                logits = agent_model(states_m)
                model_loss_0 = model_loss_fn(logits, actions_m)
                model_loss_0.backward()
                torch.nn.utils.clip_grad_norm_(agent_model.parameters(), 1.0)
                model_optimizer_0.step()
                
                cumulative_model_loss_0 += model_loss_0.item()
                model_correct_0 += (logits.argmax(1) == actions_m).sum().item()
                
                # Train Adversary's model of Agent 0
                states_m, actions_m = adversary_model_buffer.sample(BATCH_SIZE)
                states_m = torch.FloatTensor(states_m).to(device)
                actions_m = torch.LongTensor(actions_m).to(device)
                
                model_optimizer_1.zero_grad()
                logits = adversary_model(states_m)
                model_loss_1 = model_loss_fn(logits, actions_m)
                model_loss_1.backward()
                torch.nn.utils.clip_grad_norm_(adversary_model.parameters(), 1.0)
                model_optimizer_1.step()
                
                cumulative_model_loss_1 += model_loss_1.item()
                model_correct_1 += (logits.argmax(1) == actions_m).sum().item()
                num_model_updates += 1

            # ===== TRAIN Q-NETWORKS =====
            # Train Agent 0's Q-network
            states_b, joint_actions_b, rewards_b, next_states_b, dones_b = agent_buffer.sample(BATCH_SIZE)
            
            states_b = torch.FloatTensor(states_b).to(device)
            joint_actions_b = torch.LongTensor(joint_actions_b).unsqueeze(1).to(device)
            rewards_b = torch.FloatTensor(rewards_b).unsqueeze(1).to(device)
            next_states_b = torch.FloatTensor(next_states_b).to(device)
            dones_b = torch.FloatTensor(dones_b).unsqueeze(1).to(device)

            with torch.no_grad():
                q_next_target_0 = agent_target_net(next_states_b)
                model_next_1 = agent_model.get_probs(next_states_b)
                
                q_matrix_next_0 = q_next_target_0.view(BATCH_SIZE, action_dim_0, action_dim_1)
                model_vec_next_1 = model_next_1.unsqueeze(2)
                
                av_next_0 = torch.bmm(q_matrix_next_0, model_vec_next_1)
                max_av_next_0, _ = torch.max(av_next_0, dim=1)
                target_q_0 = rewards_b + (GAMMA * max_av_next_0 * (1 - dones_b))

            q_current_0 = agent_net(states_b)
            q_current_0_selected = q_current_0.gather(1, joint_actions_b)
            
            q_loss_0 = q_loss_fn(q_current_0_selected, target_q_0)
            q_optimizer_0.zero_grad()
            q_loss_0.backward()
            torch.nn.utils.clip_grad_norm_(agent_net.parameters(), 1.0)
            q_optimizer_0.step()
            cumulative_q_loss_0 += q_loss_0.item()

            # Train Adversary's Q-network
            states_b, joint_actions_b, rewards_b, next_states_b, dones_b = adversary_buffer.sample(BATCH_SIZE)

            states_b = torch.FloatTensor(states_b).to(device)
            joint_actions_b = torch.LongTensor(joint_actions_b).unsqueeze(1).to(device)
            rewards_b = torch.FloatTensor(rewards_b).unsqueeze(1).to(device)
            next_states_b = torch.FloatTensor(next_states_b).to(device)
            dones_b = torch.FloatTensor(dones_b).unsqueeze(1).to(device)

            with torch.no_grad():
                q_next_target_1 = adversary_target_net(next_states_b)
                model_next_0 = adversary_model.get_probs(next_states_b)
                
                q_matrix_next_1 = q_next_target_1.view(BATCH_SIZE, action_dim_0, action_dim_1)
                model_vec_next_0 = model_next_0.unsqueeze(2)
                
                av_next_1 = torch.bmm(q_matrix_next_1.transpose(1, 2), model_vec_next_0)
                max_av_next_1, _ = torch.max(av_next_1, dim=1)
                target_q_1 = rewards_b + (GAMMA * max_av_next_1 * (1 - dones_b))
                
            q_current_1 = adversary_net(states_b)
            q_current_1_selected = q_current_1.gather(1, joint_actions_b)
            
            q_loss_1 = q_loss_fn(q_current_1_selected, target_q_1)
            q_optimizer_1.zero_grad()
            q_loss_1.backward()
            torch.nn.utils.clip_grad_norm_(adversary_net.parameters(), 1.0)
            q_optimizer_1.step()
            cumulative_q_loss_1 += q_loss_1.item()
            
            num_q_updates += 1

            # Soft update target networks
            for target_param, local_param in zip(agent_target_net.parameters(), agent_net.parameters()):
                target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)
            for target_param, local_param in zip(adversary_target_net.parameters(), adversary_net.parameters()):
                target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)

        obs = next_obs
    
    # Episode statistics
    avg_q_loss_0 = cumulative_q_loss_0 / num_q_updates if num_q_updates > 0 else 0
    avg_q_loss_1 = cumulative_q_loss_1 / num_q_updates if num_q_updates > 0 else 0
    avg_model_loss_0 = cumulative_model_loss_0 / num_model_updates if num_model_updates > 0 else 0
    model_acc_0 = model_correct_0 / (num_model_updates * BATCH_SIZE) if num_model_updates > 0 else 0
    model_acc_1 = model_correct_1 / (num_model_updates * BATCH_SIZE) if num_model_updates > 0 else 0

    recent_rewards_agent_0.append(episode_reward_agent_0)
    recent_rewards_agent_1.append(episode_reward_agent_1)
    recent_model_acc_0.append(model_acc_0)
    recent_model_acc_1.append(model_acc_1)
    
    avg_reward_0 = sum(recent_rewards_agent_0) / len(recent_rewards_agent_0)
    avg_reward_1 = sum(recent_rewards_agent_1) / len(recent_rewards_agent_1)
    avg_acc_0 = sum(recent_model_acc_0) / len(recent_model_acc_0)
    avg_acc_1 = sum(recent_model_acc_1) / len(recent_model_acc_1)
    
    progress_bar.set_postfix(
        R0=f"{avg_reward_0:.1f}",
        R1=f"{avg_reward_1:.1f}",
        Eps=f"{epsilon:.2f}",
        QL=f"{avg_q_loss_0:.3f}",
        Acc=f"{avg_acc_0:.2f}"
    )
    
    if episode % 10000 == 0 and episode > 0:
        checkpoint_path = f"models/checkpoint_episode_{episode}.pth"
        torch.save({
            'episode': episode,
            'epsilon': epsilon,
            'agent_net_state_dict': agent_net.state_dict(),
            'adversary_net_state_dict': adversary_net.state_dict(),
            'agent_model_state_dict': agent_model.state_dict(),
            'adversary_model_state_dict': adversary_model.state_dict(),
        }, checkpoint_path)
        print(f"\nEpisode {episode}: Avg Rewards = [{avg_reward_0:.2f}, {avg_reward_1:.2f}], Model Acc = {avg_acc_0:.2%}")
                
    # In your training loop, add these at the end of each episode (before epsilon decay):
    episode_rewards_0.append(episode_reward_agent_0)
    episode_rewards_1.append(episode_reward_agent_1)
    episode_lengths.append(episode_steps)
    q_losses_0.append(avg_q_loss_0)
    q_losses_1.append(avg_q_loss_1)
    model_losses_0.append(avg_model_loss_0)
    model_accuracies_0.append(model_acc_0)
    model_accuracies_1.append(model_acc_1)
    epsilon_history.append(epsilon)
    avg_rewards_0.append(avg_reward_0)
    avg_rewards_1.append(avg_reward_1)

    if episode < epsilon_decay_steps:
        epsilon -= decay_rate
        epsilon = max(epsilon, epsilon_min)

# Save final models
torch.save(adversary_net.state_dict(), "models/adversary_net_final.pth")
torch.save(adversary_model.state_dict(), "models/adversary_model_final.pth")
torch.save(agent_model.state_dict(), "models/agent_model_final.pth")
torch.save(agent_net.state_dict(), "models/agent_net_final.pth")
print("\nTraining complete!")


def plot_training_metrics():
    """Create comprehensive plots for training analysis"""
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Episode Rewards (Raw and Moving Average)
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(episode_rewards_0, alpha=0.3, label='Agent 0 (Raw)', color='blue')
    ax1.plot(episode_rewards_1, alpha=0.3, label='Adversary (Raw)', color='red')
    ax1.plot(avg_rewards_0, label='Agent 0 (MA-100)', color='blue', linewidth=2)
    ax1.plot(avg_rewards_1, label='Adversary (MA-100)', color='red', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Rewards Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Reward Difference (Competition Balance)
    ax2 = plt.subplot(3, 3, 2)
    reward_diff = np.array(episode_rewards_0) - np.array(episode_rewards_1)
    ax2.plot(reward_diff, alpha=0.3, color='purple')
    window = 100
    if len(reward_diff) >= window:
        ma_diff = np.convolve(reward_diff, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(reward_diff)), ma_diff, 
                label=f'MA-{window}', color='purple', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reward Difference')
    ax2.set_title('Agent 0 Reward - Adversary Reward')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Q-Loss Over Time
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(q_losses_0, label='Agent 0 Q-Loss', alpha=0.7, color='blue')
    ax3.plot(q_losses_1, label='Adversary Q-Loss', alpha=0.7, color='red')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Q-Loss')
    ax3.set_title('Q-Network Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. Model Accuracy Over Time
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(model_accuracies_0, label='Agent 0 Model Acc', alpha=0.7, color='blue')
    ax4.plot(model_accuracies_1, label='Adversary Model Acc', alpha=0.7, color='red')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Opponent Model Prediction Accuracy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    # 5. Epsilon Decay
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(epsilon_history, color='green', linewidth=2)
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Epsilon')
    ax5.set_title('Exploration Rate (Epsilon)')
    ax5.grid(True, alpha=0.3)
    
    # 6. Episode Length
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(episode_lengths, alpha=0.3, color='orange')
    if len(episode_lengths) >= 100:
        ma_lengths = np.convolve(episode_lengths, np.ones(100)/100, mode='valid')
        ax6.plot(range(99, len(episode_lengths)), ma_lengths, 
                label='MA-100', color='orange', linewidth=2)
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Steps')
    ax6.set_title('Episode Length')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Reward Distribution (Last 1000 episodes)
    ax7 = plt.subplot(3, 3, 7)
    last_n = min(1000, len(episode_rewards_0))
    ax7.hist(episode_rewards_0[-last_n:], bins=50, alpha=0.5, 
             label='Agent 0', color='blue', density=True)
    ax7.hist(episode_rewards_1[-last_n:], bins=50, alpha=0.5, 
             label='Adversary', color='red', density=True)
    ax7.set_xlabel('Reward')
    ax7.set_ylabel('Density')
    ax7.set_title(f'Reward Distribution (Last {last_n} Episodes)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Convergence Analysis: Rolling Standard Deviation
    ax8 = plt.subplot(3, 3, 8)
    window = 500
    if len(episode_rewards_0) >= window:
        rolling_std_0 = [np.std(episode_rewards_0[i:i+window]) 
                        for i in range(len(episode_rewards_0)-window)]
        rolling_std_1 = [np.std(episode_rewards_1[i:i+window]) 
                        for i in range(len(episode_rewards_1)-window)]
        ax8.plot(rolling_std_0, label='Agent 0', color='blue')
        ax8.plot(rolling_std_1, label='Adversary', color='red')
    ax8.set_xlabel('Episode')
    ax8.set_ylabel('Rolling Std Dev')
    ax8.set_title(f'Reward Stability (Window={window})')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Model Loss Over Time
    ax9 = plt.subplot(3, 3, 9)
    ax9.plot(model_losses_0, label='Agent 0 Model Loss', alpha=0.7, color='blue')
    ax9.set_xlabel('Episode')
    ax9.set_ylabel('Model Loss')
    ax9.set_title('Opponent Model Loss')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    ax9.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('models/training_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved training_analysis.png")
    plt.show()

# Additional diagnostic plots
def plot_convergence_diagnostics():
    """Additional plots for convergence diagnosis"""
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Phase Analysis: Early vs Late Training
    ax1 = plt.subplot(2, 3, 1)
    phase_size = len(episode_rewards_0) // 5
    phases = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    means_0 = [np.mean(episode_rewards_0[i*phase_size:(i+1)*phase_size]) for i in range(5)]
    means_1 = [np.mean(episode_rewards_1[i*phase_size:(i+1)*phase_size]) for i in range(5)]
    
    x = np.arange(len(phases))
    width = 0.35
    ax1.bar(x - width/2, means_0, width, label='Agent 0', color='blue', alpha=0.7)
    ax1.bar(x + width/2, means_1, width, label='Adversary', color='red', alpha=0.7)
    ax1.set_xlabel('Training Phase')
    ax1.set_ylabel('Mean Reward')
    ax1.set_title('Reward by Training Phase')
    ax1.set_xticks(x)
    ax1.set_xticklabels(phases, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Q-Loss vs Reward Correlation
    ax2 = plt.subplot(2, 3, 2)
    if len(q_losses_0) > 100:
        ax2.scatter(q_losses_0[100:], avg_rewards_0[100:], 
                   alpha=0.3, s=10, color='blue', label='Agent 0')
        ax2.scatter(q_losses_1[100:], avg_rewards_1[100:], 
                   alpha=0.3, s=10, color='red', label='Adversary')
    ax2.set_xlabel('Q-Loss')
    ax2.set_ylabel('Average Reward (MA-100)')
    ax2.set_title('Q-Loss vs Reward Correlation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Model Accuracy vs Reward
    ax3 = plt.subplot(2, 3, 3)
    if len(model_accuracies_0) > 100:
        ax3.scatter(model_accuracies_0[100:], avg_rewards_0[100:], 
                   alpha=0.3, s=10, color='blue', label='Agent 0')
        ax3.scatter(model_accuracies_1[100:], avg_rewards_1[100:], 
                   alpha=0.3, s=10, color='red', label='Adversary')
    ax3.set_xlabel('Model Accuracy')
    ax3.set_ylabel('Average Reward (MA-100)')
    ax3.set_title('Model Accuracy vs Reward')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Reward Trend Lines (with polynomial fit)
    ax4 = plt.subplot(2, 3, 4)
    x = np.arange(len(avg_rewards_0))
    if len(x) > 100:
        z0 = np.polyfit(x, avg_rewards_0, 3)
        z1 = np.polyfit(x, avg_rewards_1, 3)
        p0 = np.poly1d(z0)
        p1 = np.poly1d(z1)
        ax4.plot(avg_rewards_0, alpha=0.3, color='blue')
        ax4.plot(avg_rewards_1, alpha=0.3, color='red')
        ax4.plot(x, p0(x), linewidth=3, color='blue', label='Agent 0 Trend')
        ax4.plot(x, p1(x), linewidth=3, color='red', label='Adversary Trend')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Average Reward')
    ax4.set_title('Reward Trends (Polynomial Fit)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Learning Rate Analysis: Recent Performance
    ax5 = plt.subplot(2, 3, 5)
    windows = [100, 500, 1000, 2000]
    recent_means_0 = []
    recent_means_1 = []
    for w in windows:
        if len(episode_rewards_0) >= w:
            recent_means_0.append(np.mean(episode_rewards_0[-w:]))
            recent_means_1.append(np.mean(episode_rewards_1[-w:]))
    
    if recent_means_0:
        x_pos = np.arange(len(recent_means_0))
        ax5.plot(x_pos, recent_means_0, marker='o', linewidth=2, 
                markersize=8, color='blue', label='Agent 0')
        ax5.plot(x_pos, recent_means_1, marker='s', linewidth=2, 
                markersize=8, color='red', label='Adversary')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels([f'Last {w}' for w in windows[:len(recent_means_0)]], rotation=45)
    ax5.set_xlabel('Window Size')
    ax5.set_ylabel('Mean Reward')
    ax5.set_title('Recent Performance Analysis')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Convergence Indicator
    ax6 = plt.subplot(2, 3, 6)
    window = 1000
    if len(episode_rewards_0) >= window * 2:
        improvement_0 = []
        improvement_1 = []
        for i in range(window, len(episode_rewards_0) - window):
            prev_mean_0 = np.mean(episode_rewards_0[i-window:i])
            curr_mean_0 = np.mean(episode_rewards_0[i:i+window])
            improvement_0.append(curr_mean_0 - prev_mean_0)
            
            prev_mean_1 = np.mean(episode_rewards_1[i-window:i])
            curr_mean_1 = np.mean(episode_rewards_1[i:i+window])
            improvement_1.append(curr_mean_1 - prev_mean_1)
        
        ax6.plot(improvement_0, alpha=0.7, color='blue', label='Agent 0')
        ax6.plot(improvement_1, alpha=0.7, color='red', label='Adversary')
        ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Reward Improvement')
    ax6.set_title(f'Learning Progress ({window}-ep windows)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/convergence_diagnostics.png', dpi=300, bbox_inches='tight')
    print("Saved convergence_diagnostics.png")
    plt.show()

# Call plotting functions after training
plot_training_metrics()
plot_convergence_diagnostics()

# Print convergence statistics
print("\n" + "="*60)
print("CONVERGENCE ANALYSIS")
print("="*60)

last_1000 = min(1000, len(episode_rewards_0))
print(f"\nLast {last_1000} episodes statistics:")
print(f"Agent 0: Mean={np.mean(episode_rewards_0[-last_1000:]):.2f}, "
      f"Std={np.std(episode_rewards_0[-last_1000:]):.2f}")
print(f"Adversary: Mean={np.mean(episode_rewards_1[-last_1000:]):.2f}, "
      f"Std={np.std(episode_rewards_1[-last_1000:]):.2f}")

if len(episode_rewards_0) >= 2000:
    first_half = np.mean(episode_rewards_0[:len(episode_rewards_0)//2])
    second_half = np.mean(episode_rewards_0[len(episode_rewards_0)//2:])
    print(f"\nAgent 0 improvement: {second_half - first_half:.2f} "
          f"({(second_half/first_half - 1)*100:.1f}%)")
    
print(f"\nFinal Model Accuracies:")
print(f"Agent 0: {model_accuracies_0[-100:] and np.mean(model_accuracies_0[-100:]):.2%}")
print(f"Adversary: {model_accuracies_1[-100:] and np.mean(model_accuracies_1[-100:]):.2%}")

print(f"\nFinal Q-Loss:")
print(f"Agent 0: {np.mean([x for x in q_losses_0[-100:] if x > 0]):.4f}")
print(f"Adversary: {np.mean([x for x in q_losses_1[-100:] if x > 0]):.4f}")
print("="*60)