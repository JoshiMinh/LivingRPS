import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import numpy as np
from collections import deque
from rps_agent_model import RPSAgentNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_dim = 9
output_dim = 3
max_episodes = 1000
max_steps = 100
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
lr = 0.001

# Environment parameters
screen_w, screen_h = 750, 750

# Replay buffer
memory = deque(maxlen=10000)

# Model
model = RPSAgentNet(input_dim, 32, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

def random_pos():
    return random.randint(0, screen_w), random.randint(0, screen_h)

def get_state(agent_type, agent_pos, threat_pos, prey_pos):
    def rel(pos):
        dx = (pos[0] - agent_pos[0]) / screen_w
        dy = (pos[1] - agent_pos[1]) / screen_h
        dist = math.sqrt(dx**2 + dy**2)
        return [dx, dy, dist]
    
    type_vec = [0, 0, 0]
    type_vec[agent_type] = 1
    return rel(threat_pos) + rel(prey_pos) + type_vec

def get_reward(agent_pos, threat_pos, prey_pos):
    d_threat = np.linalg.norm(np.array(agent_pos) - np.array(threat_pos))
    d_prey = np.linalg.norm(np.array(agent_pos) - np.array(prey_pos))
    if d_threat < 30:
        return -1.0  # caught by threat
    if d_prey < 30:
        return 1.0   # caught prey
    return -0.01    # time penalty

def step(action, agent_pos, threat_pos, prey_pos):
    dx, dy = 0, 0
    if action == 0:  # toward prey
        dx = prey_pos[0] - agent_pos[0]
        dy = prey_pos[1] - agent_pos[1]
    elif action == 1:  # away from threat
        dx = agent_pos[0] - threat_pos[0]
        dy = agent_pos[1] - threat_pos[1]
    else:  # ignore = wander
        dx = random.uniform(-1, 1)
        dy = random.uniform(-1, 1)

    norm = math.sqrt(dx**2 + dy**2) + 1e-5
    dx /= norm
    dy /= norm

    new_x = int(agent_pos[0] + dx * 15)
    new_y = int(agent_pos[1] + dy * 15)

    new_x = min(max(0, new_x), screen_w)
    new_y = min(max(0, new_y), screen_h)

    return (new_x, new_y)

for ep in range(max_episodes):
    agent_type = random.randint(0, 2)
    agent_pos = random_pos()
    threat_type = (agent_type + 2) % 3
    prey_type = (agent_type + 1) % 3
    threat_pos = random_pos()
    prey_pos = random_pos()

    for step_num in range(max_steps):
        state = get_state(agent_type, agent_pos, threat_pos, prey_pos)
        state_tensor = torch.tensor(state, dtype=torch.float).to(device)

        if random.random() < epsilon:
            action = random.randint(0, 2)
        else:
            with torch.no_grad():
                q_values = model(state_tensor)
                action = int(torch.argmax(q_values).item())

        new_agent_pos = step(action, agent_pos, threat_pos, prey_pos)
        reward = get_reward(new_agent_pos, threat_pos, prey_pos)
        done = abs(reward) == 1.0

        next_state = get_state(agent_type, new_agent_pos, threat_pos, prey_pos)
        memory.append((state, action, reward, next_state, done))
        agent_pos = new_agent_pos

        if done:
            break

        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.tensor(states, dtype=torch.float).to(device)
            next_states = torch.tensor(next_states, dtype=torch.float).to(device)
            actions = torch.tensor(actions).unsqueeze(1).to(device)
            rewards = torch.tensor(rewards).to(device)
            dones = torch.tensor(dones).to(device)

            q_pred = model(states).gather(1, actions).squeeze()
            q_next = model(next_states).max(1)[0].detach()
            q_target = rewards + gamma * q_next * (~dones)

            loss = loss_fn(q_pred, q_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if (ep + 1) % 100 == 0:
        print(f"Episode {ep+1}, Epsilon: {epsilon:.3f}")

torch.save(model.state_dict(), "rps_agent_model.pth")
print("Training complete. Model saved.")