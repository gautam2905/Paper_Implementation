# REINFORCE: Monte-Carlo Policy-Gradient Control (episodic) for optimal policy

import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class PolicyParameter(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyParameter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.fc(x)
    

class REINFORCE:
    def __init__(self, env_name='CartPole-v1', gamma=0.99, learning_rate=0.01):
        self.env = gym.make(env_name)
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.no_of_episodes = 300
        self.policy = PolicyParameter(self.env.observation_space.shape[0], self.env.action_space.n)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.returns = []
        self.episode_lengths = []
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.action_space = self.env.action_space
        self.state = None
        self.running_avg_return = 0.0
        self.alpha_baseline = 0.95 # Smoothing factor for running average

    def generate_episode(self):
        episode = []
        self.state, _ = self.env.reset()
        done = False
        time_steps = 0
        while not done:
            state_tensor = torch.FloatTensor(np.array(self.state)).unsqueeze(0)
            action_probs = self.policy(state_tensor)
            action = np.random.choice(self.action_dim, p=action_probs.detach().numpy()[0])
            step_result = self.env.step(action)

            if (len(step_result) == 4):
                next_state, reward, terminated, truncated, _ = step_result
            else:
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated

            episode.append((self.state, action, reward))
            time_steps += 1
            self.state = next_state
        return episode, time_steps
    
    def compute_returns(self, rewards):
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns

    def train(self):
        for episode in range(self.no_of_episodes):
            generate_episode, time_steps = self.generate_episode()
            states, actions, rewards = zip(*generate_episode)
            returns = self.compute_returns(rewards)

            states_tensor = torch.FloatTensor(np.array(states))
            actions_tensor = torch.LongTensor(np.array(actions))
            returns_tensor = torch.FloatTensor(np.array(returns))
            
            if episode == 0:
                self.running_avg_return = returns_tensor.mean().item()
            else:
                self.running_avg_return = (self.alpha_baseline * self.running_avg_return +
                                          (1 - self.alpha_baseline) * returns_tensor.mean().item())

            adjusted_returns_tensor = returns_tensor - self.running_avg_return
            action_probs = self.policy(states_tensor)
            action_log_probs = torch.log(action_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze())
            policy_loss = -torch.mean(action_log_probs * adjusted_returns_tensor)

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            episode_reward = sum(rewards)
            self.returns.append(episode_reward)
            self.episode_lengths.append(time_steps)

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.returns[-100:])
                print(f'Episode {episode + 1}, Average Reward: {avg_reward:.2f}, Episode Length: {time_steps}')

    def plot_results(self):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.returns)
        plt.title('Episode Returns')
        plt.xlabel('Episode')
        plt.ylabel('Return')

        plt.subplot(1, 2, 2)
        plt.plot(self.episode_lengths)
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Length')

        plt.tight_layout()
        plt.show()

def show_policy(env, policy, num_episodes=5):
    """
    Renders a few episodes using the trained policy to visualize its performance.
    """
    for i in range(num_episodes):
        print(f"Rendering episode {i+1}/{num_episodes}")
        # Reset environment for rendering
        # For Gym versions >= 0.26.0, env.reset() returns (observation, info).
        state, _ = env.reset()
        done = False
        while not done:
            # Convert state to tensor, get action probabilities, and choose action
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = policy(state_tensor)
            action = np.argmax(action_probs.detach().numpy()) # Choose action with highest probability
            
            # Take a step and update state
            # For Gym versions >= 0.26.0, env.step() returns (obs, reward, terminated, truncated, info).
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Render the environment
            env.render()
    env.close() # Close the environment display

if __name__ == "__main__":
    reinforce = REINFORCE(env_name='CartPole-v1', gamma=0.99, learning_rate=0.01)
    reinforce.train()
    reinforce.plot_results()
    env = gym.make('CartPole-v1', render_mode='human')
    show_policy(env, reinforce.policy, num_episodes=5)
    print("Training completed.")