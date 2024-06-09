import torch
import torch.nn as nn
import numpy as np

class PPO_NN_actor(nn.Module):
    """
    Neural network model for the actor in Proximal Policy Optimization (PPO).

    This network takes in state observations and outputs a probability distribution over actions.

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Third fully connected layer with small weights initialization.
    """

    def __init__(self, input_dim, output_dim):
        """
        Initializes the PPO_NN_actor network.

        Args:
            input_dim (int): Dimension of the input (state observations).
            output_dim (int): Dimension of the output (action probabilities).
        """
        super(PPO_NN_actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

        # Small weights initialization for the last layer
        nn.init.normal_(self.fc3.weight, mean=0.0, std=0.01)
        # Initialize bias to 0
        nn.init.constant_(self.fc3.bias, 0.0)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor, np.ndarray, or list): Input state observations.

        Returns:
            torch.Tensor: Output action probabilities.
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        if isinstance(x, list):
            x = torch.tensor(x, dtype=torch.float)
        if len(x.shape) == 2:  # Check if the input is in batch form
            x = x.unsqueeze(0)  # Add a batch dimension if not present

        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        x = torch.softmax(x, dim=-1)  # Apply softmax to get probabilities

        return x

class PPO_NN_critic(nn.Module):
    """
    Neural network model for the critic in Proximal Policy Optimization (PPO).

    This network takes in state observations and outputs a value estimate of the given state.

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Third fully connected layer producing a single value output.
    """

    def __init__(self, input_dim):
        """
        Initializes the PPO_NN_critic network.

        Args:
            input_dim (int): Dimension of the input (state observations).
        """
        super(PPO_NN_critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor, np.ndarray, or list): Input state observations.

        Returns:
            torch.Tensor: Output value estimate.
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        if isinstance(x, list):
            x = torch.tensor(x, dtype=torch.float)
        if len(x.shape) == 2:  # Check if the input is in batch form
            x = x.unsqueeze(0)  # Add a batch dimension if not present

        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x