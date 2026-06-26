from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.data: deque[Transition] = deque(maxlen=capacity)

    def append(self, item: Transition) -> None:
        self.data.append(item)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.data, min(batch_size, len(self.data)))

    def __len__(self) -> int:
        return len(self.data)

