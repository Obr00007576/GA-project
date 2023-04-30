from typing import Any, Mapping
import torch
from torch import nn
import numpy as np

action_space_len = 4
status_space_len = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {torch.cuda.get_device_name(0)}.")

class GAmodel(nn.Module):
    def __init__(self) -> None:
        super(GAmodel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=status_space_len, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=action_space_len)
        ).to(device)

    def forward(self, x):
        return self.model(x)

    def get_action(self, observation):
        observation = torch.FloatTensor(observation).to(device)
        actions = self(observation)
        action = torch.argmax(actions)
        return action.item()