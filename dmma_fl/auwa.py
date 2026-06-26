from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BufferedModel:
    state_dict: dict[str, torch.Tensor]
    utility: float


class AUWAggregator:
    """Asynchronous uploading and weighted aggregation."""

    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.buffer: list[BufferedModel] = []

    def push(self, state_dict: dict[str, torch.Tensor], utility: float) -> dict[str, torch.Tensor] | None:
        self.buffer.append(
            BufferedModel(
                {k: v.detach().cpu().clone() for k, v in state_dict.items()},
                max(float(utility), 1e-8),
            )
        )
        if len(self.buffer) >= self.buffer_size:
            return self.aggregate()
        return None

    def aggregate(self) -> dict[str, torch.Tensor]:
        total = sum(item.utility for item in self.buffer)
        weights = [item.utility / total for item in self.buffer]
        keys = self.buffer[0].state_dict.keys()
        out: dict[str, torch.Tensor] = {}
        for key in keys:
            out[key] = sum(w * item.state_dict[key] for w, item in zip(weights, self.buffer))
        self.buffer.clear()
        return out

