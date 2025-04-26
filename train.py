import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import numpy as np
from collections import deque
from rps_agent_model import RPSAgentNet

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_dim, output_dim, hidden_dim = 9, 3, 32
max_episodes, max_steps = 1000, 100
gamma, epsilon, epsilon_min, epsilon_decay = 0.95, 1.0, 0.01, 0.995
batch_size, lr, target_update_freq = 64, 0.001, 10
screen_w, screen_h = 750, 750

# Replay buffer and models
memory = deque(maxlen=10000)
model = RPSAgentNet(input_dim, hidden_dim, output_dim).to(device)
target_model = RPSAgentNet(input_dim, hidden_dim, output_dim).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

def random_pos():
    return random.randint(0, screen_w), random.randint(0, screen_h)

def get_state(agent_type, agent_pos, threat_pos, prey_pos):
    def rel(pos):
        dx, dy = (pos[0] - agent_pos[0]) / screen_w, (pos[1] - agent_pos[1]) / screen_h
        return [dx, dy, math.sqrt(dx**2 + dy**2)]
    type_vec = [0, 0, 0]
    type_vec[agent_type] = 1
    return rel(threat_pos) + rel(prey_pos) + type_vec

def get_reward(agent_pos, threat_pos, prey_pos):
    d_threat = np.linalg.norm(np.array(agent_pos) - np.array(threat_pos))
    d_prey = np.linalg.norm(np.array(agent_pos) - np.array(prey_pos))
    if d_threat < 30: return -1.0
    if d_prey < 30: return 1.0
    return 0.1 * (1.0 - d_prey / screen_w) - 0.1 * (1.0 - d_threat / screen_w)

def step(action, agent_pos, threat_pos, prey_pos):
    if action == 0:
        dx, dy = prey_pos[0] - agent_pos[0], prey_pos[1] - agent_pos[1]
    elif action == 1:
        dx, dy = agent_pos[0] - threat_pos[0], agent_pos[1] - threat_pos[1]
    else:
        dx, dy = random.uniform(-1, 1), random.uniform(-1, 1)
    norm = math.sqrt(dx**2 + dy**2) + 1e-5
    new_x = int(agent_pos[0] + (dx / norm) * 15)
    new_y = int(agent_pos[1] + (dy / norm) * 15)
    return min(max(0, new_x), screen_w), min(max(0, new_y), screen_h)

for ep in range(max_episodes):
    agent_type = random.randint(0, 2)
    agent_pos, threat_pos, prey_pos = random_pos(), random_pos(), random_pos()
    for step_num in range(max_steps):
        state = get_state(agent_type, agent_pos, threat_pos, prey_pos)
        state_tensor = torch.tensor(state, dtype=torch.float).to(device)
        action = random.randint(0, 2) if random.random() < epsilon else int(torch.argmax(model(state_tensor)).item())
        new_agent_pos = step(action, agent_pos, threat_pos, prey_pos)
        reward = get_reward(new_agent_pos, threat_pos, prey_pos)
        done = abs(reward) == 1.0
        next_state = get_state(agent_type, new_agent_pos, threat_pos, prey_pos)
        memory.append((state, action, reward, next_state, done))
        agent_pos = new_agent_pos
        if done: break

        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            states = torch.tensor(states, dtype=torch.float).to(device)
            next_states = torch.tensor(next_states, dtype=torch.float).to(device)
            actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float).to(device)
            dones = torch.tensor([bool(d) for d in dones], dtype=torch.bool).to(device)
            q_pred = model(states).gather(1, actions).squeeze()
            q_next = target_model(next_states).max(1)[0]
            q_target = rewards + gamma * q_next * (~dones)
            loss = loss_fn(q_pred, q_target.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    if (ep + 1) % target_update_freq == 0:
        target_model.load_state_dict(model.state_dict())
    if (ep + 1) % 100 == 0:
        print(f"Episode {ep+1}, Epsilon: {epsilon:.3f}")

torch.save(model.state_dict(), "rps_agent_model.pth")
print("Training complete. Model saved.")