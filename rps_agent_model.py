import torch.nn as nn
import torch.nn.functional as F

class RPSAgentNet(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=32, output_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 16)
        self.out = nn.Linear(16, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)