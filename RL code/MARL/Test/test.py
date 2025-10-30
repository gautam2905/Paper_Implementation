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

alpha = 0.01
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay_steps = 80000
decay_rate = (epsilon - epsilon_min) / epsilon_decay_steps
episodes = 1000000
BUFFER_CAPACITY = 100000
BATCH_SIZE = 64
GAMMA = 0.99       
TAU = 0.005       

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

## Deep Q-Network (DQN)

class QNetwork(nn.Module):
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
        return F.softmax(self.net(x), dim=-1)

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

# --- Joint Action Learning with Agent Modeling using Deep Q-Networks (DQN) ---

global_state_dim = env.observation_space('adversary_0').shape[0] + env.observation_space('agent_0').shape[0]
action_dim_0 = env.action_space('agent_0').n
action_dim_1 = env.action_space('adversary_0').n
joint_action_dim = action_dim_0 * action_dim_1



# --- Agent 0 (Agent_0) ---
# Q-Net learns Q_0(s, a_0, a_1)
# agent_net = QNetwork(global_state_dim, joint_action_dim)
# agent_target_net = QNetwork(global_state_dim, joint_action_dim)
# agent_target_net.load_state_dict(agent_net.state_dict())

# Model-Net learns pi_1(a_1 | s)
# It models Agent 1 (the opponent)
# agent_model = ModelNetwork(global_state_dim, action_dim_1) 

# --- Agent 0 (Agent_0) ---
agent_net = QNetwork(global_state_dim, joint_action_dim).to(device)
agent_target_net = QNetwork(global_state_dim, joint_action_dim).to(device)
agent_target_net.load_state_dict(agent_net.state_dict())
agent_model = ModelNetwork(global_state_dim, action_dim_1).to(device) 



# --- Agent 1 (Adversary_0) ---
# Q-Net learns Q_1(s, a_0, a_1)
# adversary_net = QNetwork(global_state_dim, joint_action_dim)
# adversary_target_net = QNetwork(global_state_dim, joint_action_dim)
# adversary_target_net.load_state_dict(adversary_net.state_dict())

# Model-Net learns pi_0(a_0 | s)
# It models Agent 0 (the opponent)
# adversary_model = ModelNetwork(global_state_dim, action_dim_0)

adversary_net = QNetwork(global_state_dim, joint_action_dim).to(device)
adversary_target_net = QNetwork(global_state_dim, joint_action_dim).to(device)
adversary_target_net.load_state_dict(adversary_net.state_dict())
adversary_model = ModelNetwork(global_state_dim, action_dim_0).to(device)




agent_buffer = ReplayBuffer(BUFFER_CAPACITY)
adversary_buffer = ReplayBuffer(BUFFER_CAPACITY)

q_optimizer_0 = torch.optim.Adam(agent_net.parameters(), lr=alpha)
q_optimizer_1 = torch.optim.Adam(adversary_net.parameters(), lr=alpha)

model_optimizer_0 = torch.optim.Adam(agent_model.parameters(), lr=alpha)
model_optimizer_1 = torch.optim.Adam(adversary_model.parameters(), lr=alpha)

model_loss_fn = nn.NLLLoss()



recent_rewards_agent_0 = collections.deque(maxlen=100)
recent_rewards_agent_1 = collections.deque(maxlen=100)
progress_bar = tqdm(range(episodes))
for episode in progress_bar:
    episode_reward_agent_0 = 0
    episode_reward_agent_1 = 0
    cumulative_q_loss_0 = 0
    cumulative_q_loss_1 = 0
    cumulative_model_loss_0 = 0
    cumulative_model_loss_1 = 0
    
    episode_steps = 0
    num_train_steps = 0 

    done = False
    obs, info = env.reset()
    while not done:
        global_state = np.concatenate((obs['adversary_0'], obs['agent_0']))
        
        state_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(device)

        with torch.no_grad(): 
            # Get Q_0(s, a_0, a_1) for ALL joint actions
            # Output shape: [1, joint_action_dim]
            all_joint_q_agent = agent_net(state_tensor) 

            # Get pi_1(a_1 | s) -> Agent 0's model of Agent 1
            # Output shape: [1, action_dim_1]
            probs_opponent_1 = agent_model(state_tensor) 

            # Reshape Q-values into a matrix: [action_dim_0, action_dim_1]
            # Each row 'a0' contains Q-values for all opponent actions 'a1'
            q_matrix_agent = all_joint_q_agent.view(action_dim_0, action_dim_1)

            # Reshape opponent probs into a column vector: [action_dim_1, 1]
            probs_opponent_1_vec = probs_opponent_1.view(action_dim_1, 1)

            # This is Equation 6.17: AV_0 = Q_0 * pi_1
            # (dim_0, dim_1) @ (dim_1, 1) -> (dim_0, 1)
            action_values_agent_tensor = torch.matmul(q_matrix_agent, probs_opponent_1_vec)
            
            # Squeeze to get a 1D vector of action values for Agent 0
            # Shape: [action_dim_0]
            action_values_agent = action_values_agent_tensor.squeeze()

        with torch.no_grad():
            # Get Q_1(s, a_0, a_1) for ALL joint actions
            # Output shape: [1, joint_action_dim]
            all_joint_q_adversary = adversary_net(state_tensor)
            
            # Get pi_0(a_0 | s) -> Agent 1's model of Agent 0
            # Output shape: [1, action_dim_0]
            probs_opponent_0 = adversary_model(state_tensor)

            # Reshape Q-values into a matrix: [action_dim_0, action_dim_1]
            q_matrix_adversary = all_joint_q_adversary.view(action_dim_0, action_dim_1)

            # Reshape *this* opponent's probs (Agent 0) into a column vector: [action_dim_0, 1]
            probs_opponent_0_vec = probs_opponent_0.view(action_dim_0, 1)

            # This is Equation 6.17 for Agent 1: AV_1 = (Q_1^T) * pi_0
            # The sum is over a_0, so we must transpose the Q-matrix first.
            # (dim_1, dim_0) @ (dim_0, 1) -> (dim_1, 1)
            action_values_adversary_tensor = torch.matmul(q_matrix_adversary.T, probs_opponent_0_vec)
            
            # Squeeze to get a 1D vector of action values for Agent 1
            # Shape: [action_dim_1]
            action_values_adversary = action_values_adversary_tensor.squeeze()


        if np.random.rand() < epsilon:
            agent_action = np.random.choice(action_dim_0)
            adversary_action = np.random.choice(action_dim_1)
        else:
            agent_action = torch.argmax(action_values_agent).item()
            adversary_action = torch.argmax(action_values_adversary).item()

        actions = {'adversary_0': adversary_action, "agent_0": agent_action}
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        done = all(terminations.values()) or all(truncations.values())
        global_next_obs = np.concatenate((next_obs['adversary_0'], next_obs['agent_0']))
        episode_reward_agent_0 += rewards['agent_0']
        episode_reward_agent_1 += rewards['adversary_0']
        episode_steps += 1

        # === 1. TRAIN AGENT MODELS (Line 11) ===
       
        # We use NLLLoss (Negative Log Likelihood Loss) because your
        # ModelNetwork outputs probabilities (after softmax).


        # --- Train agent_model (Agent 0's model of Agent 1) ---
        model_optimizer_0.zero_grad()
        probs_opponent_1 = agent_model(state_tensor)
        log_probs_1 = torch.log(probs_opponent_1 + 1e-9)
        target_action_1 = torch.LongTensor([adversary_action]).to(device)

        model_loss_0 = model_loss_fn(log_probs_1, target_action_1)
        cumulative_model_loss_0 += model_loss_0.item()
        model_loss_0.backward()
        model_optimizer_0.step()


        model_optimizer_1.zero_grad()
        probs_opponent_0 = adversary_model(state_tensor)
        log_probs_0 = torch.log(probs_opponent_0 + 1e-9)
        target_action_0 = torch.LongTensor([agent_action]).to(device)

        model_loss_1 = model_loss_fn(log_probs_0, target_action_0)
        cumulative_model_loss_1 += model_loss_1.item()
        model_loss_1.backward()
        model_optimizer_1.step()

        # =================================================================
        # === 2. UPDATE BUFFERS (Line 9) ===
        # We must convert the two separate actions into one joint_action index
        # to store in the buffer, as this is what our Q-network learns.
        # Formula: (a0 * num_actions_1) + a1
        joint_action = agent_action * action_dim_1 + adversary_action
        
        agent_buffer.push(global_state, joint_action, rewards['agent_0'], global_next_obs, done)
        
        adversary_buffer.push(global_state, joint_action, rewards['adversary_0'], global_next_obs, done)

        # =================================================================
        # === 3. TRAIN Q-NETWORKS (Line 12) ===
        # This is the Bellman update, using the AV(s') as the target.
        
        q_loss_fn = nn.MSELoss()

        if len(agent_buffer) >= BATCH_SIZE:
            states_b, joint_actions_b, rewards_b, next_states_b, dones_b = agent_buffer.sample(BATCH_SIZE)
            
            states_b = torch.FloatTensor(states_b).to(device)
            joint_actions_b = torch.LongTensor(joint_actions_b).unsqueeze(1).to(device) # Shape: [B, 1]
            rewards_b = torch.FloatTensor(rewards_b).unsqueeze(1).to(device)      # Shape: [B, 1]
            next_states_b = torch.FloatTensor(next_states_b).to(device)
            dones_b = torch.FloatTensor(dones_b).unsqueeze(1).to(device)          # Shape: [B, 1]

            with torch.no_grad():
                q_next_target_0 = agent_target_net(next_states_b) # [B, joint_action_dim]
                # Get pi_1(a_1 | s') from the *opponent model*
                model_next_1 = agent_model(next_states_b) # [B, action_dim_1]
                
                # Reshape for batch matrix multiplication
                q_matrix_next_0 = q_next_target_0.view(BATCH_SIZE, action_dim_0, action_dim_1)
                model_vec_next_1 = model_next_1.unsqueeze(2) # Shape: [B, action_dim_1, 1]
                
                # AV_0(s', a_0) = Q_target_0 * pi_1
                av_next_0 = torch.bmm(q_matrix_next_0, model_vec_next_1) # Shape: [B, action_dim_0, 1]
                
                # max_a0 AV_0(s', a_0)
                max_av_next_0, _ = torch.max(av_next_0, dim=1) # Shape: [B, 1]
                
                # Bellman Target: y_0 = r + gamma * max_AV * (1 - done)
                target_q_0 = rewards_b + (GAMMA * max_av_next_0 * (1 - dones_b))

            # --- Get Current Q-value: Q_0(s, a) ---
            q_current_0 = agent_net(states_b) # [B, joint_action_dim]
            # Get Q_0 for the *specific* joint action taken from the buffer
            q_current_0_selected = q_current_0.gather(1, joint_actions_b) # [B, 1]
            
            # --- Calculate Loss and Optimize ---
            q_loss_0 = q_loss_fn(q_current_0_selected, target_q_0)
            q_optimizer_0.zero_grad()
            cumulative_q_loss_0 += q_loss_0.item()
            num_train_steps += 1 # Increment this *only* when you train
            q_loss_0.backward()
            q_optimizer_0.step()

            # --- Soft update target network ---
            for target_param, local_param in zip(agent_target_net.parameters(), agent_net.parameters()):
                target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)
            
        # --- Train Agent 1's Q-Network (Adversary) ---
        if len(adversary_buffer) >= BATCH_SIZE:
            # Sample a batch
            states_b, joint_actions_b, rewards_b, next_states_b, dones_b = adversary_buffer.sample(BATCH_SIZE)

            # Convert to Tensors
            states_b = torch.FloatTensor(states_b).to(device)
            joint_actions_b = torch.LongTensor(joint_actions_b).unsqueeze(1).to(device)
            rewards_b = torch.FloatTensor(rewards_b).unsqueeze(1).to(device)
            next_states_b = torch.FloatTensor(next_states_b).to(device)
            dones_b = torch.FloatTensor(dones_b).unsqueeze(1).to(device)

            # --- Calculate Target y_1 ---
            # y_1 = r + gamma * max_a1' AV_1(s', a_1')
            with torch.no_grad():
                # Get Q_target_1(s', a_0, a_1) from the *target network*
                q_next_target_1 = adversary_target_net(next_states_b) # [B, joint_action_dim]
                # Get pi_0(a_0 | s') from the *opponent model*
                model_next_0 = adversary_model(next_states_b) # [B, action_dim_0]
                
                # Reshape for batch matrix multiplication
                q_matrix_next_1 = q_next_target_1.view(BATCH_SIZE, action_dim_0, action_dim_1)
                model_vec_next_0 = model_next_0.unsqueeze(2) # [B, action_dim_0, 1]
                
                # AV_1(s', a_1) = (Q_target_1^T) * pi_0
                # We transpose the Q-matrix to sum over a_0
                av_next_1 = torch.bmm(q_matrix_next_1.transpose(1, 2), model_vec_next_0) # [B, action_dim_1, 1]
                
                # max_a1 AV_1(s', a_1)
                max_av_next_1, _ = torch.max(av_next_1, dim=1) # [B, 1]
                
                # Bellman Target: y_1 = r + gamma * max_AV * (1 - done)
                target_q_1 = rewards_b + (GAMMA * max_av_next_1 * (1 - dones_b))
                
            # --- Get Current Q-value: Q_1(s, a) ---
            q_current_1 = adversary_net(states_b)
            q_current_1_selected = q_current_1.gather(1, joint_actions_b)
            
            # --- Calculate Loss and Optimize ---
            q_loss_1 = q_loss_fn(q_current_1_selected, target_q_1)
            q_optimizer_1.zero_grad()
            cumulative_q_loss_1 += q_loss_1.item()
            q_loss_1.backward()
            q_optimizer_1.step()

            # --- Soft update target network ---
            for target_param, local_param in zip(adversary_target_net.parameters(), adversary_net.parameters()):
                target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)

        # =================================================================
        # === 4. PREPARE FOR NEXT STEP ===
        obs = next_obs
        global_state = global_next_obs
    
    # --- (End of `while not done` loop) ---
    avg_q_loss_0 = cumulative_q_loss_0 / num_train_steps if num_train_steps > 0 else 0
    avg_q_loss_1 = cumulative_q_loss_1 / num_train_steps if num_train_steps > 0 else 0
    
    avg_model_loss_0 = cumulative_model_loss_0 / episode_steps if episode_steps > 0 else 0
    avg_model_loss_1 = cumulative_model_loss_1 / episode_steps if episode_steps > 0 else 0

    recent_rewards_agent_0.append(episode_reward_agent_0)
    recent_rewards_agent_1.append(episode_reward_agent_1)
    
    avg_reward_0 = sum(recent_rewards_agent_0) / len(recent_rewards_agent_0)
    avg_reward_1 = sum(recent_rewards_agent_1) / len(recent_rewards_agent_1)
    
    progress_bar.set_postfix(
        Avg_Rwd_A0=f"{avg_reward_0:.2f}",
        Avg_Rwd_A1=f"{avg_reward_1:.2f}",
        Epsilon=f"{epsilon:.3f}",
        Avg_Q_Loss=f"{avg_q_loss_0:.4f}", # Added a bit more precision
        Avg_M_Loss=f"{avg_model_loss_0:.4f}"
    )
    if episode % 50 == 0:
        checkpoint_path = f"models/checkpoint_episode_{episode}.pth"
        print(f"\nSaving checkpoint to {checkpoint_path}...") # Good to add a print
        torch.save({
            'episode': episode,                         # <-- ADD THIS
            'epsilon': epsilon,                         # <-- ADD THIS
            'agent_net_state_dict': agent_net.state_dict(),
            'adversary_net_state_dict': adversary_net.state_dict(),
            'agent_model_state_dict': agent_model.state_dict(),
            'adversary_model_state_dict': adversary_model.state_dict(),
            'q_optimizer_0_state_dict': q_optimizer_0.state_dict(),
            'q_optimizer_1_state_dict': q_optimizer_1.state_dict(),
            'model_optimizer_0_state_dict': model_optimizer_0.state_dict(),
            'model_optimizer_1_state_dict': model_optimizer_1.state_dict(),
        }, checkpoint_path)
                
    if episode < epsilon_decay_steps:
        epsilon -= decay_rate
        epsilon = max(epsilon, epsilon_min)


import os
save_dir = os.curdir
torch.save(adversary_net.state_dict(), os.path.join(save_dir, "models/adversary_net_best_model.pth"))
torch.save(adversary_model.state_dict(), os.path.join(save_dir, "models/adversary_model_best_model.pth"))
torch.save(agent_model.state_dict(), os.path.join(save_dir, "models/agent_model_best_model.pth"))
torch.save(agent_net.state_dict(), os.path.join(save_dir, "models/agent_net_best_model.pth"))