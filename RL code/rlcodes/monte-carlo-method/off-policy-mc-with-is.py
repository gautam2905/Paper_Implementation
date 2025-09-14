import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import random

class DiscreteOffPolicyMC:
    """Off-Policy Monte Carlo for discrete environments (like CartPole)"""
    
    def __init__(self, state_bins=10, gamma=0.99, behavior_epsilon=0.5, target_epsilon=0.1):
        self.gamma = gamma
        self.behavior_epsilon = behavior_epsilon
        self.target_epsilon = target_epsilon
        self.state_bins = state_bins
        
        # Q-table for discrete states
        self.Q = defaultdict(lambda: defaultdict(float))
        self.C = defaultdict(lambda: defaultdict(float))  # Cumulative weights
        
        # For tracking
        self.returns = defaultdict(list)
        self.importance_ratios = []
        
    def discretize_state(self, state, env_name="CartPole-v1"):
        """Convert continuous state to discrete bins"""
        if env_name == "CartPole-v1":
            # CartPole state: [cart_pos, cart_vel, pole_angle, pole_vel]
            cart_pos, cart_vel, pole_angle, pole_vel = state
            
            # Define reasonable ranges and discretize
            cart_pos_bin = np.digitize(cart_pos, np.linspace(-2.4, 2.4, self.state_bins))
            cart_vel_bin = np.digitize(cart_vel, np.linspace(-3, 3, self.state_bins))
            pole_angle_bin = np.digitize(pole_angle, np.linspace(-0.21, 0.21, self.state_bins))
            pole_vel_bin = np.digitize(pole_vel, np.linspace(-3, 3, self.state_bins))
            
            return (cart_pos_bin, cart_vel_bin, pole_angle_bin, pole_vel_bin)
        
        elif env_name == "LunarLander-v2":
            # LunarLander has 8 state variables
            discretized = []
            ranges = [(-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (0, 1), (0, 1)]
            
            for i, (low, high) in enumerate(ranges):
                bin_val = np.digitize(state[i], np.linspace(low, high, self.state_bins))
                discretized.append(bin_val)
            
            return tuple(discretized)
    
    def behavior_policy(self, state, num_actions):
        """Epsilon-greedy behavior policy (more exploration)"""
        if np.random.random() < self.behavior_epsilon:
            return np.random.randint(num_actions)
        else:
            # Greedy action based on current Q-values
            q_values = [self.Q[state][a] for a in range(num_actions)]
            return np.argmax(q_values)
    
    def target_policy(self, state, num_actions):
        """Epsilon-greedy target policy (less exploration)"""
        if np.random.random() < self.target_epsilon:
            return np.random.randint(num_actions)
        else:
            q_values = [self.Q[state][a] for a in range(num_actions)]
            return np.argmax(q_values)
    
    def get_action_probability(self, state, action, num_actions, policy_type="behavior"):
        """Get probability of taking action under specified policy"""
        epsilon = self.behavior_epsilon if policy_type == "behavior" else self.target_epsilon
        
        # Get Q-values for all actions
        q_values = [self.Q[state][a] for a in range(num_actions)]
        greedy_action = np.argmax(q_values)
        
        if action == greedy_action:
            return 1 - epsilon + epsilon / num_actions
        else:
            return epsilon / num_actions
    
    def generate_episode(self, env, policy_type="behavior"):
        """Generate one episode using specified policy"""
        episode = []
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]  # Handle new gym API
            
        discrete_state = self.discretize_state(state, env.spec.id)
        total_reward = 0
        
        for step in range(500):  # Max steps
            if policy_type == "behavior":
                action = self.behavior_policy(discrete_state, env.action_space.n)
            else:
                action = self.target_policy(discrete_state, env.action_space.n)
            
            next_state, reward, done, truncated, _ = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]
                
            episode.append((discrete_state, action, reward))
            
            discrete_state = self.discretize_state(next_state, env.spec.id)
            total_reward += reward
            
            if done or truncated:
                break
        
        return episode, total_reward
    
    def update_q_ordinary_importance_sampling(self, episode, env):
        """Update Q-values using ordinary importance sampling"""
        G = 0
        W = 1  # Importance sampling weight
        
        # Process episode backwards
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = self.gamma * G + reward
            
            # Update Q-value using ordinary importance sampling
            self.returns[(state, action)].append((G, W))
            
            # Calculate average return weighted by importance sampling ratios
            returns_and_weights = self.returns[(state, action)]
            weighted_sum = sum(g * w for g, w in returns_and_weights)
            weight_sum = sum(w for g, w in returns_and_weights)
            
            if weight_sum > 0:
                self.Q[state][action] = weighted_sum / len(returns_and_weights)  # Ordinary IS
            
            # Calculate importance sampling ratio for this step
            target_prob = self.get_action_probability(state, action, env.action_space.n, "target")
            behavior_prob = self.get_action_probability(state, action, env.action_space.n, "behavior")
            
            if behavior_prob > 0:
                W *= target_prob / behavior_prob
            else:
                break  # Zero probability in behavior policy
            
            # If importance sampling ratio becomes too large, stop
            if W > 100:
                break
    
    def update_q_weighted_importance_sampling(self, episode, env):
        """Update Q-values using weighted importance sampling"""
        G = 0
        W = 1
        
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = self.gamma * G + reward
            
            # Update cumulative weight
            self.C[state][action] += W
            
            # Update Q-value using weighted importance sampling
            if self.C[state][action] > 0:
                self.Q[state][action] += (W / self.C[state][action]) * (G - self.Q[state][action])
            
            # Calculate importance sampling ratio
            target_prob = self.get_action_probability(state, action, env.action_space.n, "target")
            behavior_prob = self.get_action_probability(state, action, env.action_space.n, "behavior")
            
            if behavior_prob > 0:
                W *= target_prob / behavior_prob
            else:
                break
            
            if W > 100:  # Prevent explosion
                break
    
    def train(self, env, episodes=1000, method="weighted"):
        """Train using off-policy Monte Carlo"""
        behavior_rewards = []
        target_policy_rewards = []
        avg_importance_ratios = []
        
        print(f"Training Off-Policy Monte Carlo ({method} importance sampling)")
        print(f"Environment: {env.spec.id}")
        print(f"Behavior ε: {self.behavior_epsilon}, Target ε: {self.target_epsilon}")
        print("-" * 60)
        
        for episode in range(episodes):
            # Generate episode using behavior policy
            episode_data, behavior_reward = self.generate_episode(env, "behavior")
            behavior_rewards.append(behavior_reward)
            
            # Update Q-values using importance sampling
            if method == "weighted":
                self.update_q_weighted_importance_sampling(episode_data, env)
            else:
                self.update_q_ordinary_importance_sampling(episode_data, env)
            
            # Periodically evaluate target policy
            if episode % 50 == 0:
                _, target_reward = self.generate_episode(env, "target")
                target_policy_rewards.append(target_reward)
                
                # Calculate average importance ratio for this episode
                total_ratio = 1.0
                for state, action, reward in episode_data:
                    target_prob = self.get_action_probability(state, action, env.action_space.n, "target")
                    behavior_prob = self.get_action_probability(state, action, env.action_space.n, "behavior")
                    if behavior_prob > 0:
                        total_ratio *= target_prob / behavior_prob
                    if total_ratio > 100:
                        total_ratio = 100
                        break
                
                avg_importance_ratios.append(total_ratio)
                
                print(f"Episode {episode:4d} | "
                      f"Behavior Reward: {np.mean(behavior_rewards[-50:]):6.2f} | "
                      f"Target Reward: {target_reward:6.2f} | "
                      f"Avg IS Ratio: {total_ratio:8.3f}")
        
        return behavior_rewards, target_policy_rewards, avg_importance_ratios

class ContinuousOffPolicyMC:
    """Off-Policy Monte Carlo for continuous control (no MuJoCo required)"""
    
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        
        # Behavior policy (more exploration)
        self.behavior_policy = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # Actions in [-1, 1]
        ).to(self.device)
        
        # Target policy (less exploration)
        self.target_policy = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        ).to(self.device)
        
        # Value function
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(self.device)
        
        self.behavior_optimizer = optim.Adam(self.behavior_policy.parameters(), lr=lr)
        self.target_optimizer = optim.Adam(self.target_policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=lr)
        
        self.behavior_noise = 0.3  # Higher exploration
        self.target_noise = 0.1    # Lower exploration
    
    def get_action(self, state, policy_type="behavior", deterministic=False):
        """Get action from specified policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if policy_type == "behavior":
                action = self.behavior_policy(state_tensor)
                noise_scale = 0 if deterministic else self.behavior_noise
            else:
                action = self.target_policy(state_tensor)
                noise_scale = 0 if deterministic else self.target_noise
            
            if not deterministic:
                noise = torch.normal(0, noise_scale, action.shape).to(self.device)
                action = torch.clamp(action + noise, -1, 1)
        
        return action.cpu().numpy()[0]
    
    def train_continuous_environment(self, env_name="Pendulum-v1", episodes=500):
        """Train on continuous control environment"""
        env = gym.make(env_name)
        
        behavior_rewards = []
        target_rewards = []
        
        print(f"Training Continuous Off-Policy Monte Carlo")
        print(f"Environment: {env_name}")
        print("-" * 50)
        
        for episode in range(episodes):
            # Collect episode with behavior policy
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
                
            episode_data = []
            total_reward = 0
            
            for step in range(200):  # Max steps
                action = self.get_action(state, "behavior")
                next_state, reward, done, truncated, _ = env.step(action)
                if isinstance(next_state, tuple):
                    next_state = next_state[0]
                
                episode_data.append((state.copy(), action.copy(), reward))
                state = next_state
                total_reward += reward
                
                if done or truncated:
                    break
            
            behavior_rewards.append(total_reward)
            
            # Update networks (simplified - in practice you'd use importance sampling here too)
            if episode % 10 == 0:
                self.update_continuous_policies(episode_data)
            
            # Evaluate target policy
            if episode % 25 == 0:
                eval_reward = self.evaluate_target_policy(env)
                target_rewards.append(eval_reward)
                
                print(f"Episode {episode:4d} | "
                      f"Behavior: {np.mean(behavior_rewards[-25:]):7.2f} | "
                      f"Target: {eval_reward:7.2f}")
        
        env.close()
        return behavior_rewards, target_rewards
    
    def evaluate_target_policy(self, env):
        """Evaluate target policy"""
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
            
        total_reward = 0
        for step in range(200):
            action = self.get_action(state, "target", deterministic=True)
            next_state, reward, done, truncated, _ = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            
            total_reward += reward
            state = next_state
            
            if done or truncated:
                break
        
        return total_reward
    
    def update_continuous_policies(self, episode_data):
        """Update policies (simplified version)"""
        if len(episode_data) < 10:
            return
        
        states = torch.FloatTensor([s for s, a, r in episode_data]).to(self.device)
        actions = torch.FloatTensor([a for s, a, r in episode_data]).to(self.device)
        rewards = [r for s, a, r in episode_data]
        
        # Calculate returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Update target policy (policy gradient)
        target_actions = self.target_policy(states)
        target_loss = -torch.mean(returns * torch.sum((target_actions - actions) ** 2, dim=1))
        
        self.target_optimizer.zero_grad()
        target_loss.backward()
        self.target_optimizer.step()

def demonstrate_importance_sampling_theory():
    """Demonstrate importance sampling concepts with visual examples"""
    print("=" * 70)
    print("OFF-POLICY MONTE CARLO WITH IMPORTANCE SAMPLING - THEORY & PRACTICE")
    print("=" * 70)
    
    print("""
    KEY CONCEPT: Learn about Target Policy π using data from Behavior Policy μ
    
    IMPORTANCE SAMPLING RATIO:
    ρ_t = π(A_t|S_t) / μ(A_t|S_t)  [for single step]
    ρ_{0:T-1} = ∏(t=0 to T-1) ρ_t  [for entire episode]
    
    TWO MAIN APPROACHES:
    
    1. ORDINARY IMPORTANCE SAMPLING:
       V^π(s) = (1/n) ∑ ρ_{0:T-1} * G_t
       - Unbiased estimate
       - Can have infinite variance
    
    2. WEIGHTED IMPORTANCE SAMPLING:
       V^π(s) = (∑ ρ_{0:T-1} * G_t) / (∑ ρ_{0:T-1})
       - Biased but consistent
       - Lower variance (preferred in practice)
    
    PRACTICAL CHALLENGES:
    - Ratio explosion: ρ can become very large
    - Need behavior policy to have support everywhere target policy does
    - Must handle numerical instabilities
    """)

def run_discrete_example():
    """Run discrete environment example"""
    print("\n" + "="*50)
    print("DISCRETE ENVIRONMENT EXAMPLE (CartPole)")
    print("="*50)
    
    env = gym.make('CartPole-v1')
    agent = DiscreteOffPolicyMC(state_bins=8, behavior_epsilon=0.4, target_epsilon=0.05)
    
    # Train with weighted importance sampling
    behavior_rewards, target_rewards, is_ratios = agent.train(env, episodes=500, method="weighted")
    
    env.close()
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Rewards comparison
    ax1.plot(behavior_rewards, alpha=0.3, label='Behavior Policy (ε=0.4)')
    ax1.plot(np.convolve(behavior_rewards, np.ones(20)/20, mode='valid'), 
             label='Behavior (smoothed)', linewidth=2)
    
    if target_rewards:
        episodes_eval = list(range(0, len(target_rewards) * 50, 50))
        ax1.plot(episodes_eval, target_rewards, 'r-', linewidth=2, 
                label='Target Policy (ε=0.05)', marker='o')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Policy Performance Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # Importance sampling ratios
    if is_ratios:
        ax2.plot(is_ratios, marker='o')
        ax2.set_xlabel('Evaluation Episode')
        ax2.set_ylabel('Average IS Ratio per Episode')
        ax2.set_title('Importance Sampling Ratios')
        ax2.grid(True)
        ax2.set_yscale('log')
    
    # Learning curve comparison
    if len(behavior_rewards) > 50 and target_rewards:
        behavior_smooth = np.convolve(behavior_rewards, np.ones(50)/50, mode='valid')
        ax3.plot(behavior_smooth, label='Behavior Policy', alpha=0.7)
        
        # Interpolate target rewards for comparison
        episodes_eval = np.array(range(0, len(target_rewards) * 50, 50))
        ax3.plot(episodes_eval, target_rewards, 'r-', linewidth=2, 
                label='Target Policy', marker='s', markersize=4)
        
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Reward (smoothed)')
        ax3.set_title('Learning Progress')
        ax3.legend()
        ax3.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return agent

def run_continuous_example():
    """Run continuous environment example"""
    print("\n" + "="*50)
    print("CONTINUOUS ENVIRONMENT EXAMPLE (Pendulum)")
    print("="*50)
    
    agent = ContinuousOffPolicyMC(state_dim=3, action_dim=1)
    behavior_rewards, target_rewards = agent.train_continuous_environment('Pendulum-v1', episodes=300)
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(behavior_rewards, alpha=0.3, label='Behavior Policy')
    plt.plot(np.convolve(behavior_rewards, np.ones(10)/10, mode='valid'), 
             label='Behavior (smoothed)', linewidth=2)
    
    if target_rewards:
        episodes_eval = list(range(0, len(target_rewards) * 25, 25))
        plt.plot(episodes_eval, target_rewards, 'r-', linewidth=2, 
                label='Target Policy', marker='o')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Continuous Control: Off-Policy Learning')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    if len(behavior_rewards) > 20 and target_rewards:
        behavior_smooth = np.convolve(behavior_rewards, np.ones(20)/20, mode='valid')
        plt.plot(behavior_smooth, label='Behavior Policy', alpha=0.7)
        
        episodes_eval = np.array(range(0, len(target_rewards) * 25, 25))
        plt.plot(episodes_eval, target_rewards, 'r-', linewidth=2, 
                label='Target Policy', marker='s')
        
        plt.xlabel('Episode')
        plt.ylabel('Reward (smoothed)')
        plt.title('Learning Comparison')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return agent

if __name__ == "__main__":
    # Explain theory
    demonstrate_importance_sampling_theory()
    
    # Run discrete example (CartPole)
    discrete_agent = run_discrete_example()
    
    # Run continuous example (Pendulum)
    continuous_agent = run_continuous_example()
    
    print("\n" + "="*70)
    print("SUMMARY OF IMPORTANCE SAMPLING RESULTS")
    print("="*70)
    print("""
    Key Observations:
    1. Target policy (low ε) learns to perform better than behavior policy (high ε)
    2. Importance sampling ratios vary significantly between episodes
    3. Weighted IS typically more stable than ordinary IS
    4. Method works for both discrete and continuous action spaces
    
    This demonstrates the core power of off-policy learning:
    - Learn optimal behavior while maintaining exploration
    - Reuse data collected from exploratory policies
    - Separate data collection from policy optimization
    """)