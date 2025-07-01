import gymnasium as gym 
import numpy as np
import random
import matplotlib.pyplot as plt
import torch

class DynaQ:
    def __init__(self, env, action_space, state_space, episode_length, epsilon=0.1, planning_steps=10):
        self.action_space = action_space
        self.state_space = state_space
        self.q_table = np.zeros((state_space, action_space))
        self.model = {}
        self.env = env
        self.epsilon = epsilon
        self.alpha = 0.1
        self.episode_length = episode_length  
        self.planning_steps = planning_steps  
        self.gamma = 0.95  
        self.min_epsilon = 0.01  
        self.max_epsilon = 1.0  
        self.decay_rate = 0.0005  
    
    def get_epsilon_greedy_action(self, state):
        if random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return np.argmax(self.q_table[state])
    
    def train_dyna(self):

        reward_per_ep = []
        
        for episode in range(self.episode_length):
            s, info = self.env.reset()
            done = False
            total_reward = 0

            while not done:

            # Reset the environment for each episode    
            # INTERACT WITH ENVIRONMENT
                a = self.get_epsilon_greedy_action(s)
                next_state, reward, terminated, truncated , info = self.env.step(a)
                done = terminated or truncated
                total_reward += reward
                
                # DIRECT RL
                self.q_table[s, a] += self.alpha * (reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[s, a])

                # MODEL LEARNING
                if s not in self.model:
                    self.model[s] = {}
                self.model[s][a] = (reward, next_state)
                
                for _ in range(self.planning_steps):
                    
                    if done:
                        break
                    
                    if s not in self.model:
                        self.model[s] = {}

                    if not self.model: # Cannot plan if model is empty
                        break
                    s_rand = random.choice(list(self.model.keys()))

                    if not self.model[s_rand]:
                        continue
                    a_rand = random.choice(list(self.model[s_rand].keys()))

                    r_sim, next_s_sim = self.model[s_rand][a_rand]

                    # d. Update Q-table with the simulated experience
                    best_next_q_sim = np.max(self.q_table[next_s_sim])
                    td_target_sim = r_sim + self.gamma * best_next_q_sim
                    td_error_sim = td_target_sim - self.q_table[s_rand, a_rand]
                    self.q_table[s_rand, a_rand] += self.alpha * td_error_sim

                s = next_state

            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)
            
            reward_per_ep.append(total_reward)
            
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{self.episode_length} finished.")

        print("\nTraining completed.")

        plt.figure(figsize=(12, 6))
        plt.plot(reward_per_ep, label='Total Reward')
        moving_avg = np.convolve(reward_per_ep, np.ones(100)/100, mode='valid')
        plt.plot(np.arange(len(moving_avg)) + 99, moving_avg, label='100-episode Moving Average', color='orange', linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Agent's Performance Over Time on FrozenLake")
        plt.legend()
        plt.grid(True)
        plt.show()

def test_policy(q_table, env, num_episodes=2):
    """
    Tests the final learned policy greedily without exploration.
    """
    total_wins = 0
    print("\n--- Testing Final Policy ---")
    env = gym.make(env.spec.id, render_mode="human", is_slippery=False)

    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        print(f"Test Episode {episode + 1}")
        while not done:
            action = np.argmax(q_table[state])
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            if reward > 0: # In FrozenLake, a reward > 0 means a win
                total_wins += 1

    print("\n--- Test Results ---")
    print(f"Win rate over {num_episodes} episodes: {total_wins / num_episodes * 100:.2f}%")
    env.close()


env_name = 'FrozenLake-v1'
train_env = gym.make(env_name, is_slippery=False)

dyna_agent = DynaQ(
    env=train_env,
    action_space=train_env.action_space.n,
    state_space=train_env.observation_space.n,
    planning_steps=5,  # The power of Dyna-Q comes from planning!
    episode_length=50,  # Number of episodes to train
)
dyna_agent.train_dyna()

print("\nFinal Q-Table:")
print(np.round(dyna_agent.q_table, 3))
print("\nModel (State-Action Pairs):")
for state, actions in dyna_agent.model.items():
    for action, (reward, next_state) in actions.items():
        print(f"State {state}, Action {action} -> Reward: {reward}, Next State: {next_state}")

test_policy(dyna_agent.q_table, train_env)

train_env.close()