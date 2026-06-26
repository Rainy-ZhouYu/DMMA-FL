from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DMMAConfig


@dataclass
class StepInfo:
    convergence: float
    latency: float
    selected: np.ndarray
    cpu_indices: np.ndarray
    power_indices: np.ndarray
    utility: np.ndarray


class LEOEnvironment:
    """LEO-FL optimization environment from the paper's system model."""

    def __init__(self, cfg: DMMAConfig, data_sizes: np.ndarray, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.data_sizes = data_sizes.astype(np.float32)
        self.power_levels = np.asarray(cfg.power_levels, dtype=np.float32)
        self.cpu_levels = np.asarray(cfg.cpu_levels, dtype=np.float32)
        self.reset()

    def reset(self) -> np.ndarray:
        c = self.cfg
        self.t = 0
        self.age = np.ones(c.num_devices, dtype=np.float32)
        self.loss = self.rng.uniform(0.8, 1.2, c.num_devices).astype(np.float32)
        self.energy = np.zeros(c.num_devices, dtype=np.float32)
        self.distance = self._sample_distances()
        self.channel = self._channel_gain(self.distance)
        return self._state()

    def decode_action(self, raw_action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = self.cfg
        action = raw_action.reshape(c.num_devices, 3)
        select_order = np.argsort(-action[:, 0])
        selected = np.zeros(c.num_devices, dtype=np.float32)
        selected[select_order[: c.select_k]] = 1.0

        cpu_scaled = (action[:, 1] + 1.0) * 0.5
        power_scaled = (action[:, 2] + 1.0) * 0.5
        cpu_idx = np.clip((cpu_scaled * len(self.cpu_levels)).astype(int), 0, len(self.cpu_levels) - 1)
        power_idx = np.clip((power_scaled * len(self.power_levels)).astype(int), 0, len(self.power_levels) - 1)
        return selected, cpu_idx, power_idx

    def step(self, raw_action: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, float, bool, StepInfo]:
        selected, cpu_idx, power_idx = self.decode_action(raw_action)
        return self.step_decoded(selected, cpu_idx, power_idx, weight)

    def step_decoded(
        self,
        selected: np.ndarray,
        cpu_idx: np.ndarray,
        power_idx: np.ndarray,
        weight: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, StepInfo]:
        c = self.cfg
        selected = np.asarray(selected, dtype=np.float32)
        cpu_idx = np.asarray(cpu_idx, dtype=np.int64)
        power_idx = np.asarray(power_idx, dtype=np.int64)
        g = self.cpu_levels[cpu_idx]
        p = self.power_levels[power_idx]

        down = self._downlink_time()
        comp = c.local_epochs * c.cpu_cycles_per_sample * self.data_sizes / g
        uplink = self._uplink_time(p)
        per_device_time = down + comp + uplink
        latency = float(np.max(selected * per_device_time))

        comp_energy = 1e-28 * c.local_epochs * c.cpu_cycles_per_sample * self.data_sizes * (g**2)
        comm_energy = p * uplink
        self.energy = selected * (comp_energy + comm_energy)

        # Local loss improvement proxy. Better channel/compute/power improves expected model utility.
        effort = selected * (0.04 + 0.015 * cpu_idx + 0.01 * power_idx)
        noise = self.rng.normal(0.0, 0.01, c.num_devices).astype(np.float32)
        prev_loss = self.loss.copy()
        self.loss = np.clip(self.loss * (1.0 - effort) + noise, 0.05, 2.5).astype(np.float32)
        self.age = (self.age + 1.0) * (1.0 - selected)

        utility = np.maximum(prev_loss - self.loss, 0.0)
        convergence_pressure = self.data_sizes * self.loss * (self.age + 1.0)
        residual_pressure = np.maximum(
            np.sum(convergence_pressure) - np.sum(selected * convergence_pressure),
            1e-9,
        )
        convergence = float((residual_pressure ** (1.0 - c.rho)) / (1.0 - c.rho))

        objective = np.asarray([convergence, latency], dtype=np.float32)
        reward = -float(np.dot(weight, objective))

        self.t += 1
        self.distance = self._sample_distances()
        self.channel = self._channel_gain(self.distance)
        done = self.t >= c.rounds
        info = StepInfo(convergence=convergence, latency=latency, selected=selected, cpu_indices=cpu_idx, power_indices=power_idx, utility=utility)
        return self._state(), reward, done, info

    def _state(self) -> np.ndarray:
        c = self.cfg
        b_leo = np.full(c.num_devices, c.bandwidth_hz / c.num_devices, dtype=np.float32)
        b_m = np.full(c.num_devices, c.bandwidth_hz / c.num_devices, dtype=np.float32)
        e_t = np.full(c.num_devices, c.energy_limit, dtype=np.float32)
        state = np.stack([b_leo, b_m, self.channel, self.age, self.loss, self.data_sizes, e_t], axis=1)
        denom = np.maximum(np.abs(state).max(axis=0, keepdims=True), 1e-6)
        return (state / denom).astype(np.float32).reshape(-1)

    def _sample_distances(self) -> np.ndarray:
        radius = np.sqrt(self.cfg.service_area_km2 / np.pi) * 1000.0
        ground_radius = self.rng.uniform(0.0, radius, self.cfg.num_devices)
        return np.sqrt(self.cfg.altitude_m**2 + ground_radius**2).astype(np.float32)

    def _channel_gain(self, distance: np.ndarray) -> np.ndarray:
        c0 = 3e8
        path_loss = (c0 / (4.0 * np.pi * self.cfg.carrier_hz * distance)) ** 2
        rician = self.rng.rayleigh(1.0, self.cfg.num_devices).astype(np.float32)
        return np.maximum(path_loss * rician, 1e-16).astype(np.float32)

    def _noise_power(self) -> float:
        noise_dbm = self.cfg.noise_density_dbm_hz + 10.0 * np.log10(self.cfg.bandwidth_hz / self.cfg.num_devices)
        return float(10.0 ** ((noise_dbm - 30.0) / 10.0))

    def _uplink_time(self, power: np.ndarray) -> np.ndarray:
        b = self.cfg.bandwidth_hz / self.cfg.num_devices
        snr = power * self.channel / self._noise_power()
        rate = b * np.log2(1.0 + np.maximum(snr, 1e-9))
        return self._rate_to_slots(rate)

    def _downlink_time(self) -> np.ndarray:
        b = self.cfg.bandwidth_hz / self.cfg.num_devices
        snr = self.cfg.leo_power_w * self.channel / self._noise_power()
        rate = b * np.log2(1.0 + np.maximum(snr, 1e-9))
        return self._rate_to_slots(rate)

    def _rate_to_slots(self, rate: np.ndarray) -> np.ndarray:
        rate = np.asarray(rate, dtype=np.float32)
        lo = float(np.min(rate))
        hi = float(np.max(rate))
        normalized = (rate - lo) / max(hi - lo, 1e-6)
        slots = self.cfg.max_comm_slots - (self.cfg.max_comm_slots - self.cfg.min_comm_slots) * normalized
        slots = np.clip(np.ceil(slots), self.cfg.min_comm_slots, self.cfg.max_comm_slots)
        return (slots * self.cfg.slot_duration_s).astype(np.float32)
