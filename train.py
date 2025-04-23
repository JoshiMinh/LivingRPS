import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import numpy as np
from collections import deque
from rps_agent_model import RPSAgentNet  # Custom model defined elsewhere

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_dim = 9  # Input size (state representation)
output_dim = 3  # Number of possible actions
hidden_dim = 32  # Hidden layer size
max_episodes = 1000  # Total training episodes
max_steps = 100  # Max steps per episode
gamma = 0.95  # Discount factor for future rewards
epsilon = 1.0  # Initial exploration rate
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate for epsilon
batch_size = 64  # Batch size for training
lr = 0.001  # Learning rate
target_update_freq = 10  # Frequency to update target network

# Environment parameters
screen_w, screen_h = 750, 750  # Screen dimensions

# Replay buffer
memory = deque(maxlen=10000)  # Stores past experiences for training

# Models
model = RPSAgentNet(input_dim, hidden_dim, output_dim).to(device)  # Main model
target_model = RPSAgentNet(input_dim, hidden_dim, output_dim).to(device)  # Target model
target_model.load_state_dict(model.state_dict())  # Initialize target model weights
target_model.eval()  # Set target model to evaluation mode

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# Utility functions
def random_pos():
    """Generate a random position within the screen."""
    return random.randint(0, screen_w), random.randint(0, screen_h)

def get_state(agent_type, agent_pos, threat_pos, prey_pos):
    """Construct the state representation for the agent."""
    def rel(pos):
        dx = (pos[0] - agent_pos[0]) / screen_w
        dy = (pos[1] - agent_pos[1]) / screen_h
        dist = math.sqrt(dx**2 + dy**2)
        return [dx, dy, dist]
    
    type_vec = [0, 0, 0]
    type_vec[agent_type] = 1  # One-hot encoding for agent type
    return rel(threat_pos) + rel(prey_pos) + type_vec

def get_reward(agent_pos, threat_pos, prey_pos):
    """Calculate the reward based on agent's position relative to threat and prey."""
    d_threat = np.linalg.norm(np.array(agent_pos) - np.array(threat_pos))
    d_prey = np.linalg.norm(np.array(agent_pos) - np.array(prey_pos))
    if d_threat < 30:
        return -1.0  # Negative reward if caught by threat
    if d_prey < 30:
        return 1.0  # Positive reward if prey is caught
    # Encourage moving closer to prey and farther from threat
    return 0.1 * (1.0 - d_prey / screen_w) - 0.1 * (1.0 - d_threat / screen_w)

def step(action, agent_pos, threat_pos, prey_pos):
    """Update agent's position based on the chosen action."""
    dx, dy = 0, 0
    if action == 0:  # Move toward prey
        dx = prey_pos[0] - agent_pos[0]
        dy = prey_pos[1] - agent_pos[1]
    elif action == 1:  # Move away from threat
        dx = agent_pos[0] - threat_pos[0]
        dy = agent_pos[1] - threat_pos[1]
    else:  # Wander randomly
        dx = random.uniform(-1, 1)
        dy = random.uniform(-1, 1)

    # Normalize movement vector
    norm = math.sqrt(dx**2 + dy**2) + 1e-5
    dx /= norm
    dy /= norm

    # Update position and ensure it stays within screen bounds
    new_x = int(agent_pos[0] + dx * 15)
    new_y = int(agent_pos[1] + dy * 15)
    new_x = min(max(0, new_x), screen_w)
    new_y = min(max(0, new_y), screen_h)

    return (new_x, new_y)

# Training loop
for ep in range(max_episodes):
    # Initialize environment
    agent_type = random.randint(0, 2)  # Random agent type
    agent_pos = random_pos()
    threat_type = (agent_type + 2) % 3  # Threat type
    prey_type = (agent_type + 1) % 3  # Prey type
    threat_pos = random_pos()
    prey_pos = random_pos()

    for step_num in range(max_steps):
        # Get current state
        state = get_state(agent_type, agent_pos, threat_pos, prey_pos)
        state_tensor = torch.tensor(state, dtype=torch.float).to(device)

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, 2)  # Explore
        else:
            with torch.no_grad():
                q_values = model(state_tensor)
                action = int(torch.argmax(q_values).item())  # Exploit

        # Take action and observe new state and reward
        new_agent_pos = step(action, agent_pos, threat_pos, prey_pos)
        reward = get_reward(new_agent_pos, threat_pos, prey_pos)
        done = abs(reward) == 1.0  # Episode ends if reward is ±1.0

        next_state = get_state(agent_type, new_agent_pos, threat_pos, prey_pos)
        memory.append((state, action, reward, next_state, done))  # Store experience
        agent_pos = new_agent_pos

        if done:
            break

        # Train the model if enough samples are available
        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            # Convert to tensors
            states = torch.tensor(states, dtype=torch.float).to(device)
            next_states = torch.tensor(next_states, dtype=torch.float).to(device)
            actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float).to(device)
            dones = [bool(d) for d in dones]
            dones = torch.tensor(dones, dtype=torch.bool).to(device)

            # Compute current Q values
            q_pred = model(states).gather(1, actions).squeeze()

            # Compute target Q values using the target model
            q_next = target_model(next_states).max(1)[0]
            q_target = rewards + gamma * q_next * (~dones)

            # Compute loss and update model
            loss = loss_fn(q_pred, q_target.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Update target network periodically
    if (ep + 1) % target_update_freq == 0:
        target_model.load_state_dict(model.state_dict())

    # Print progress every 100 episodes
    if (ep + 1) % 100 == 0:
        print(f"Episode {ep+1}, Epsilon: {epsilon:.3f}")

# Save the trained model
torch.save(model.state_dict(), "rps_agent_model.pth")
print("Training complete. Model saved.")