import time
import torch
import gymnasium as gym
import numpy as np
import cv2
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML  # For Jupyter-compatible animation display

# Assuming DQN is defined in your module; adjust import as needed
# from your_dqn_module import DQN  # Uncomment and adjust
from dqn import DQN

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Preprocessing function for Atari observations (for DQN input)
def preprocess(observation):
    """Convert Atari observation to grayscale, resize to 84x84, and normalize to [0,1]."""
    if len(observation.shape) == 3:
        # RGB to grayscale using standard weights
        gray = np.dot(observation[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
    else:
        gray = observation.astype(np.float32)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0
    return normalized

# Create the test environment with rgb_array rendering (no SDL needed)
test_env = gym.make("ALE/BeamRider-v5", render_mode="rgb_array")

# Initialize the Q-network (adjust DQN params based on your implementation)
test_q_network = DQN((84, 84), test_env.action_space.n, 32, 4).to(device)

# Load the trained model
model_path = '/mnt/ML Summer Learning/Paper_Implementation-master/Paper_Implementation-master/RL code/Deep-Q-Learning/best_model.pth'
test_q_network.load_state_dict(torch.load(model_path, map_location=device))
test_q_network.eval()  # Set the network to evaluation mode

def greedy_action_test(stacked_state):
    """Chooses the best action based on the Q-network's prediction (no exploration).
    Assumes stacked_state is (4, 84, 84)."""
    with torch.no_grad():
        state_tensor = torch.FloatTensor(stacked_state).unsqueeze(0).to(device)  # (1, 4, 84, 84)
        q_values = test_q_network(state_tensor)
        return q_values.argmax().item()

def create_animation(frames):
    """Create and return a matplotlib animation from the list of frames as HTML for Jupyter."""
    if not frames:
        return None
    
    fig, ax = plt.subplots(figsize=(5, 4))  # Adjust size as needed for 160x210 aspect
    ax.set_title("BeamRider Episode Replay")
    ax.axis('off')
    
    im = ax.imshow(frames[0])
    
    def animate(i):
        im.set_array(frames[i])
        return [im]
    
    interval = 50  # ms per frame (20 FPS)
    ani = FuncAnimation(fig, animate, frames=len(frames), interval=interval, blit=True, repeat=True)
    plt.close(fig)  # Close the static fig to avoid display
    return ani.to_jshtml()  # Return HTML string for embedding

num_test_episodes = 10
for episode in range(num_test_episodes):
    # Reset environment and initialize frame stack
    state, info = test_env.reset()
    processed = preprocess(state)
    frame_stack = deque([processed] * 4, maxlen=4)  # Stack 4 identical initial frames
    stacked_state = np.stack(frame_stack, axis=0)  # (4, 84, 84)
    total_reward = 0
    done = False
    frames = []  # List to collect rendered frames for animation

    while not done:
        # Render the current frame (rgb_array: returns numpy array)
        frame = test_env.render()
        frames.append(frame.copy())  # Store a copy to avoid reference issues
        
        # Agent chooses the best action (no epsilon)
        action_index = greedy_action_test(stacked_state)
        
        # Environment takes a step
        next_state, reward, terminated, truncated, _ = test_env.step(action_index)
        done = terminated or truncated
        
        # Preprocess next state and update stack
        next_processed = preprocess(next_state)
        frame_stack.append(next_processed)
        stacked_state = np.stack(frame_stack, axis=0)
        
        total_reward += reward
        
        # Add a small delay to match original timing (optional)
        time.sleep(0.02)

    print(f"Test Episode {episode + 1}, Total Reward: {total_reward:.2f}")
    
    # Create and display the animation as HTML in Jupyter
    html_animation = create_animation(frames)
    display(HTML(html_animation))  # This embeds the interactive animation; it will play automatically

test_env.close()
plt.close('all')  # Clean up figures