import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import ale_py
from ale_py import ALEInterface
from torch import nn
ale = ALEInterface()
from PIL import Image
from tqdm import tqdm

def preprocess(rgb_image):
    grayscale_image = np.dot(rgb_image[..., :3], [0.2989, 0.5870, 0.1140])
    grayscale_image = np.clip(grayscale_image, 0, 255)
    grayscale_image = grayscale_image.astype(np.uint8)
    img = Image.fromarray(grayscale_image)
    img = img.resize((110, 84))
    img = img.crop((15, 15, 99, 99))
    # print(np.shape(img))
    return np.array(img)

def render_rgbarray(env):
    plt.imshow(preprocess(env.render()))
    plt.axis('off')
    plt.show()

# def preprocess(img):
#     img = Image.fromarray(img)
#     # Convert to grayscale, resize, and crop
#     img = img.convert('L').resize((84, 110)).crop((0, 26, 84, 110))
#     return np.array(img)


from collections import deque  # Not from pyparsing
import random 

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions, hidden_depth, image_stack):
        super(DQN, self).__init__()

        self.input_shape = input_shape  # input image shape
        self.num_actions = num_actions  # no. of actions
        self.hidden_depth = hidden_depth  # no. of filters in first conv layer
        self.image_stack = image_stack  # no. of previous frames to stack

        # self.replay_memory = torch.zeros((100000, image_stack, *input_shape), dtype=torch.uint8)

        self.dqn = nn.Sequential(
            nn.Conv2d(self.image_stack, hidden_depth, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(hidden_depth, hidden_depth * 2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(hidden_depth * 2, hidden_depth * 2, kernel_size=3, stride=1),
            nn.ReLU(), 
            nn.Flatten(),
            nn.Linear(3136, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        return self.dqn(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    




def epsilon_greedy_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            return policy_net(state).max(1)[1].item()  # return the argmax action
    else:
        return env.action_space.sample()  # return a random action
    

def optimize():
    if len(memory) < BATCH_SIZE:
        return 
    
    transitions = memory.sample(BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*transitions)

    state_batch = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    action_batch = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
    reward_batch = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
    next_state_batch = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    done_batch = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

    q_values = policy_net(state_batch).gather(1, action_batch)    
    with torch.no_grad():
        next_q_values = target_net(next_state_batch).max(1)[0].unsqueeze(1)

    expected_q_values = reward_batch + GAMMA * next_q_values * (1 - done_batch)

    loss = nn.functional.mse_loss(q_values, expected_q_values)
    
    optimizer.zero_grad()
    loss.backward()

    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
            

if __name__ == "__main__":

    gym.register_envs(ale_py)

    env = gym.make("ALE/BeamRider-v5", render_mode="rgb_array", )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    BATCH_SIZE = 64  # Increase from 32 (you have good GPU)
    GAMMA = 0.99  # Keep this
    EPS_START = 1.0 
    EPS_END = 0.01  # Lower from 0.1 for better exploitation later
    EPS_DECAY = 1_000_000  # Increase from 200k for slower decay
    TARGET_UPDATE_FREQUENCY = 10_000  # Update every 10k steps (not episodes!)
    LEARNING_RATE = 0.0001  # Lower from 0.00025 for stability
    MEMORY_SIZE = 200_000  # Increase from 100k (more diverse experiences)
    NUM_EPISODES = 2000  # Increase from 500 for better convergence
    WARMUP_STEPS = 50_000  # Add this - fill buffer before training 

    policy_net = DQN((84, 84), env.action_space.n, 32, 4).to(device)
    target_net = DQN((84, 84), env.action_space.n, 32, 4).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)

    print("training started...")

    for episode in tqdm(range(NUM_EPISODES)):
        obs, info = env.reset()
        frame = preprocess(obs)
        state = np.stack([frame] * 4, axis=0)
        
        done = False
        while not done:
            action = epsilon_greedy_action(state)
            # print(action_tensor)
            # action = action_tensor.item()
            observation, reward, terminated, truncated, info = env.step(action)
            reward = np.clip(reward, -1, 1) 
            done = terminated or truncated

            next_frame = preprocess(observation)
            next_state = np.roll(state, -1, axis=0)
            next_state[-1] = next_frame

            memory.add(state, action, reward, next_state, done)

            state = next_state

            if steps_done % 4 == 0:
                optimize()

        if episode % TARGET_UPDATE_FREQUENCY == 0:
            target_net.load_state_dict(policy_net.state_dict())


    print("Training complete!")
    env.close()


    import os
    save_dir = os.curdir
    torch.save(policy_net.state_dict(), os.path.join(save_dir, "best_model.pth"))
