import torch

import torch.nn as nn
import torch.nn.functional as F

class RPSAgentNet(nn.Module):
    """
    Neural network model for a Rock-Paper-Scissors agent.
    Architecture:
        Input layer -> Hidden layer 1 -> Hidden layer 2 -> Output layer
    Args:
        input_dim (int): Number of input features (default: 9)
        hidden_dim (int): Number of units in the first hidden layer (default: 32)
        output_dim (int): Number of output classes (default: 3)
    """
    def __init__(self, input_dim: int = 9, hidden_dim: int = 32, output_dim: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 16)
        self.out = nn.Linear(16, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
        Returns:
            torch.Tensor: Output logits of shape (batch_size, output_dim)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x