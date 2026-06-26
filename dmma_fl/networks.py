from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


class Actor(nn.Module):
    """CNN-style actor over flattened per-device state."""

    def __init__(self, state_dim: int, action_dim: int, num_devices: int, hidden_dim: int):
        super().__init__()
        self.num_devices = num_devices
        self.features = nn.Sequential(
            nn.Conv1d(7, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * num_devices, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = state.view(state.shape[0], self.num_devices, 7).transpose(1, 2)
        h = self.features(x)
        mean = torch.tanh(self.mean(h))
        log_std = self.log_std(h).clamp(-5.0, 1.0)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(state)
        dist = Normal(mean, log_std.exp())
        raw = dist.rsample()
        action = torch.tanh(raw)
        log_prob = dist.log_prob(raw).sum(dim=-1, keepdim=True)
        return action, log_prob


class HybridCritic(nn.Module):
    """LSTM-CNN critic inspired by the paper's Meta-DRL phase."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_devices: int,
        hidden_dim: int,
        history_len: int,
    ):
        super().__init__()
        self.num_devices = num_devices
        self.history_len = history_len
        self.current = nn.Sequential(
            nn.Conv1d(10, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(48, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(48 * num_devices, hidden_dim),
            nn.ReLU(),
        )
        self.history = nn.LSTM(state_dim + action_dim + 1, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        current = torch.cat([state, action], dim=-1)
        current = current.view(current.shape[0], self.num_devices, 10).transpose(1, 2)
        current_feat = self.current(current)

        if history is None:
            hist_feat = torch.zeros_like(current_feat)
        else:
            _, (h_n, _) = self.history(history)
            hist_feat = h_n[-1]
        return self.head(torch.cat([current_feat, hist_feat], dim=-1))

