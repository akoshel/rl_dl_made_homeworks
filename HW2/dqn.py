import random
from collections import deque, namedtuple
from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "mask")
)


class Net(nn.Module):
    def __init__(self, out_channels, kernel_size, actions_space):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels=1, out_channels=out_channels, kernel_size=kernel_size
        )
        self.linear1 = nn.Linear(out_channels, out_channels)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_channels, actions_space)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.conv(state).view(-1, self.out_channels)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class DQN:
    def __init__(
        self, out_channels, kernel_size, actions_space, lr, gamma, game_size
    ) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        self.game_size = game_size
        self.gamma = gamma
        self.model = Net(out_channels, kernel_size, actions_space).to(self.device)
        self.target = Net(out_channels, kernel_size, actions_space).to(self.device)
        self.target.load_state_dict(deepcopy(self.model.state_dict()))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_step(self, transitions):
        batch = Transition(*zip(*transitions))
        # Use batch to update DQN's network.
        state_batch = (
            torch.Tensor(batch.state)
            .view(-1, 1, self.game_size, self.game_size)
            .to(self.device)
        )
        action_batch = torch.Tensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.Tensor(batch.reward).unsqueeze(1).to(self.device)
        next_state_values = (
            torch.Tensor(batch.next_state)
            .view(-1, 1, self.game_size, self.game_size)
            .to(self.device)
        )
        state_action_values = self.model(state_batch).gather(1, action_batch.long())

        expected_state_action_values = (
            self.target(next_state_values).max(1)[0].detach().unsqueeze(1) * self.gamma
        ) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        # Update weights of a target Q-network here. You may use copy.deepcopy to do this or
        # assign a values of network parameters via PyTorch methods.
        self.target.load_state_dict(deepcopy(self.model.state_dict()))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.position = 0

    def store(self, exptuple):
        self.memory.append(Transition(*exptuple))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DuelingNet(nn.Module):
    def __init__(self, out_channels, kernel_size, actions_space):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels=1, out_channels=out_channels, kernel_size=kernel_size
        )
        self.linear1 = nn.Linear(out_channels, out_channels)
        self.relu = nn.ReLU()
        self.linear_a = nn.Linear(out_channels, actions_space)
        self.linear_v = nn.Linear(out_channels, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(state).view(-1, self.out_channels)
        x = self.linear1(x)
        x = self.relu(x)
        advantages = self.linear_a(x)
        values = self.linear_v(x)
        qvals = values + (advantages - advantages.mean())
        return qvals

class DuelingDQN:
    def __init__(
        self, out_channels, kernel_size, actions_space, lr, gamma, game_size
    ) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        self.game_size = game_size
        self.gamma = gamma
        self.model = DuelingNet(out_channels, kernel_size, actions_space).to(self.device)
        self.target = DuelingNet(out_channels, kernel_size, actions_space).to(self.device)
        self.target.load_state_dict(deepcopy(self.model.state_dict()))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_step(self, transitions):
        batch = Transition(*zip(*transitions))
        # Use batch to update DQN's network.
        state_batch = (
            torch.Tensor(batch.state)
            .view(-1, 1, self.game_size, self.game_size)
            .to(self.device)
        )
        action_batch = torch.Tensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.Tensor(batch.reward).unsqueeze(1).to(self.device)
        next_state_values = (
            torch.Tensor(batch.next_state)
            .view(-1, 1, self.game_size, self.game_size)
            .to(self.device)
        )
        state_action_values = self.model(state_batch).gather(1, action_batch.long())

        expected_state_action_values = (
            self.target(next_state_values).max(1)[0].detach().unsqueeze(1) * self.gamma
        ) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        # Update weights of a target Q-network here. You may use copy.deepcopy to do this or
        # assign a values of network parameters via PyTorch methods.
        self.target.load_state_dict(deepcopy(self.model.state_dict()))