import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import numpy as np
from collections import deque
from model import RPSAgentNet

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
INPUT_DIM = 9
OUTPUT_DIM = 3
HIDDEN_DIM = 32
MAX_EPISODES = 1000
MAX_STEPS = 100
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
LR = 0.001
TARGET_UPDATE_FREQ = 10
SCREEN_W, SCREEN_H = 750, 750

# Replay buffer and models
memory = deque(maxlen=10000)
model = RPSAgentNet(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
target_model = RPSAgentNet(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

def random_pos():
    """Generate a random position within the screen bounds."""
    return random.randint(0, SCREEN_W), random.randint(0, SCREEN_H)

def get_state(agent_type, agent_pos, threat_pos, prey_pos):
    """
    Construct the state vector for the agent.
    State includes relative positions and one-hot type encoding.
    """
    def rel(pos):
        dx = (pos[0] - agent_pos[0]) / SCREEN_W
        dy = (pos[1] - agent_pos[1]) / SCREEN_H
        dist = math.sqrt(dx**2 + dy**2)
        return [dx, dy, dist]
    type_vec = [0, 0, 0]
    type_vec[agent_type] = 1
    return rel(threat_pos) + rel(prey_pos) + type_vec

def get_reward(agent_pos, threat_pos, prey_pos):
    """
    Reward function:
    -1 if too close to threat, +1 if close to prey, otherwise shaped by distances.
    """
    d_threat = np.linalg.norm(np.array(agent_pos) - np.array(threat_pos))
    d_prey = np.linalg.norm(np.array(agent_pos) - np.array(prey_pos))
    if d_threat < 30:
        return -1.0
    if d_prey < 30:
        return 1.0
    # Shaped reward: encourage getting closer to prey, farther from threat
    return 0.1 * (1.0 - d_prey / SCREEN_W) - 0.1 * (1.0 - d_threat / SCREEN_W)

def step(action, agent_pos, threat_pos, prey_pos):
    """
    Move the agent based on the action:
    0 - move toward prey, 1 - move away from threat, 2 - random move.
    """
    if action == 0:
        dx, dy = prey_pos[0] - agent_pos[0], prey_pos[1] - agent_pos[1]
    elif action == 1:
        dx, dy = agent_pos[0] - threat_pos[0], agent_pos[1] - threat_pos[1]
    else:
        dx, dy = random.uniform(-1, 1), random.uniform(-1, 1)
    norm = math.sqrt(dx**2 + dy**2) + 1e-5
    new_x = int(agent_pos[0] + (dx / norm) * 15)
    new_y = int(agent_pos[1] + (dy / norm) * 15)
    # Clamp to screen bounds
    return min(max(0, new_x), SCREEN_W), min(max(0, new_y), SCREEN_H)

def train_step():
    """Sample a batch from memory and perform a single optimization step."""
    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.tensor(states, dtype=torch.float, device=device)
    next_states = torch.tensor(next_states, dtype=torch.float, device=device)
    actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float, device=device)
    dones = torch.tensor(dones, dtype=torch.bool, device=device)

    # Compute Q values
    q_pred = model(states).gather(1, actions).squeeze()
    with torch.no_grad():
        q_next = target_model(next_states).max(1)[0]
    q_target = rewards + GAMMA * q_next * (~dones)

    # Optimize the model
    loss = loss_fn(q_pred, q_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def main():
    epsilon = EPSILON_START

    for ep in range(1, MAX_EPISODES + 1):
        agent_type = random.randint(0, 2)
        agent_pos = random_pos()
        threat_pos = random_pos()
        prey_pos = random_pos()

        for step_num in range(MAX_STEPS):
            # Get current state
            state = get_state(agent_type, agent_pos, threat_pos, prey_pos)
            state_tensor = torch.tensor(state, dtype=torch.float, device=device)

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, 2)
            else:
                with torch.no_grad():
                    action = int(torch.argmax(model(state_tensor)).item())

            # Take action and observe new state and reward
            new_agent_pos = step(action, agent_pos, threat_pos, prey_pos)
            reward = get_reward(new_agent_pos, threat_pos, prey_pos)
            done = abs(reward) == 1.0
            next_state = get_state(agent_type, new_agent_pos, threat_pos, prey_pos)

            # Store transition in replay buffer
            memory.append((state, action, reward, next_state, done))
            agent_pos = new_agent_pos

            if done:
                break

            # Train the model if enough samples are available
            if len(memory) >= BATCH_SIZE:
                train_step()

        # Decay epsilon
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        # Update target network
        if ep % TARGET_UPDATE_FREQ == 0:
            target_model.load_state_dict(model.state_dict())

        # Print progress
        if ep % 100 == 0:
            print(f"Episode {ep}, Epsilon: {epsilon:.3f}")

    # Save the trained model
    torch.save(model.state_dict(), "../models/rps_agent.pth")
    print("Training complete. Model saved to ../models/rps_agent.pth")

if __name__ == "__main__":
    main()