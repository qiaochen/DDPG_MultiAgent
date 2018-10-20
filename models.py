import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary

torch.manual_seed(999)

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class ActorNetwork(nn.Module):
    """
    Actor (Policy) Network.
    """

    def __init__(self, state_dim, action_dim):
        """Initialize parameters and build model.
        :state_dim (int): Dimension of each state
        :action_dim (int): Dimension of each action
        """
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        """
        Maps a state to actions
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))


class CriticNetwork(nn.Module):
    """
    Critic (State-Value) Network.
    """

    def __init__(self, state_dim, action_dim):
        """
        Initialize parameters and build model
        :state_dim (int): Dimension of each state
        :action_dim (int): Dimension of each action
        """
        super(CriticNetwork, self).__init__()
        self.state_fc = nn.Linear(state_dim, 64)
        self.fc1 = nn.Linear(action_dim+64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters
        """
        self.state_fc.weight.data.uniform_(*hidden_init(self.state_fc))
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """
        Maps a state-action pair to Q-values
        """
        state, action = state.squeeze(), action.squeeze()
        x = F.relu(self.state_fc(state))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
if __name__ == "__main__":
    # summarize network structures using torchsummary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim, action_dim = 24, 2
    actor = ActorNetwork(state_dim, action_dim).to(device)
    critic = CriticNetwork(state_dim, action_dim).to(device)
    sum_res = summary(actor, (1, state_dim))
    sum_res = summary(critic, [(1, state_dim), (1, action_dim)])
    
