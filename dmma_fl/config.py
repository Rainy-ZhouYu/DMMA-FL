from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class DMMAConfig:
    seed: int = 7
    num_devices: int = 10
    select_k: int = 4
    rounds: int = 8
    q: int = 4
    buffer_size: int = 3

    altitude_m: float = 780_000.0
    service_area_km2: float = 2800.0
    bandwidth_hz: float = 400e6
    carrier_hz: float = 30e9
    noise_density_dbm_hz: float = -174.0
    leo_power_w: float = 40.0
    model_size_bits: float = 1.5e6
    slot_duration_s: float = 0.1
    min_comm_slots: int = 5
    max_comm_slots: int = 15
    local_epochs: int = 1
    cpu_cycles_per_sample: float = 20_000.0
    energy_limit: float = 3.0
    rho: float = 0.5
    gamma: float = 0.95
    actor_lr: float = 3e-4
    critic_lr: float = 5e-4
    meta_lr: float = 0.2
    batch_size: int = 32
    replay_capacity: int = 5000
    action_neighbors: int = 10

    meta_iters: int = 2
    sampled_weights: int = 3
    max_eps: int = 2
    warmup_eps: int = 1
    fine_tune_eps: int = 2

    power_levels: Sequence[float] = (0.05, 0.1, 0.2, 0.4)
    cpu_levels: Sequence[float] = (0.5e9, 0.8e9, 1.1e9, 1.5e9)

    hidden_dim: int = 128
    history_len: int = 4
    synthetic_samples: int = 2000
    num_classes: int = 10
    batch_train: int = 32
    device: str = "cpu"

    @property
    def state_dim(self) -> int:
        # Per-device: b_leo, b_m, h_m, O_m, L_m, D_m, E_t.
        return self.num_devices * 7

    @property
    def action_dim(self) -> int:
        # Per-device: selection logit, CPU level scalar, power level scalar.
        return self.num_devices * 3
