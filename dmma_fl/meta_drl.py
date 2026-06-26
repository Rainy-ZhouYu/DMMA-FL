from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import DMMAConfig
from .decomposition import simplex_weights
from .networks import Actor, HybridCritic
from .replay import ReplayBuffer, Transition
from .satellite import LEOEnvironment, StepInfo


@dataclass
class ParetoSolution:
    weight: np.ndarray
    objectives: tuple[float, float]
    actions: list[np.ndarray]
    infos: list[StepInfo]


class DMMAFL:
    def __init__(self, cfg: DMMAConfig, data_sizes: np.ndarray):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.env = LEOEnvironment(cfg, data_sizes, self.rng)
        self.actor = Actor(cfg.state_dim, cfg.action_dim, cfg.num_devices, cfg.hidden_dim).to(cfg.device)
        self.critic = HybridCritic(cfg.state_dim, cfg.action_dim, cfg.num_devices, cfg.hidden_dim, cfg.history_len).to(cfg.device)

    def train_meta(self) -> None:
        weights = simplex_weights(max(self.cfg.q, self.cfg.sampled_weights), 2)
        for _ in range(self.cfg.meta_iters):
            actor_base = copy.deepcopy(self.actor.state_dict())
            critic_base = copy.deepcopy(self.critic.state_dict())
            actor_states = []
            critic_states = []
            sampled = weights[self.rng.choice(len(weights), size=min(self.cfg.sampled_weights, len(weights)), replace=False)]
            for weight in sampled:
                self.actor.load_state_dict(actor_base)
                self.critic.load_state_dict(critic_base)
                self._train_for_weight(weight, self.cfg.max_eps)
                actor_states.append(copy.deepcopy(self.actor.state_dict()))
                critic_states.append(copy.deepcopy(self.critic.state_dict()))
            self.actor.load_state_dict(self._reptile_update(actor_base, actor_states, self.cfg.meta_lr))
            self.critic.load_state_dict(self._reptile_update(critic_base, critic_states, self.cfg.meta_lr))

    def fine_tune_all(self) -> list[ParetoSolution]:
        meta_actor = copy.deepcopy(self.actor.state_dict())
        meta_critic = copy.deepcopy(self.critic.state_dict())
        solutions = []
        for weight in simplex_weights(self.cfg.q, 2):
            self.actor.load_state_dict(meta_actor)
            self.critic.load_state_dict(meta_critic)
            self._train_for_weight(weight, self.cfg.fine_tune_eps)
            solutions.append(self.rollout(weight))
        return solutions

    def rollout(self, weight: np.ndarray) -> ParetoSolution:
        state = self.env.reset()
        actions: list[np.ndarray] = []
        infos: list[StepInfo] = []
        total_c = 0.0
        total_l = 0.0
        done = False
        while not done:
            action = self._act(state, explore=False)
            state, _, done, info = self.env.step(action, weight)
            total_c += info.convergence
            total_l += info.latency
            actions.append(action)
            infos.append(info)
        return ParetoSolution(weight=weight.copy(), objectives=(total_c, total_l), actions=actions, infos=infos)

    def _train_for_weight(self, weight: np.ndarray, episodes: int) -> None:
        actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)
        replay = ReplayBuffer(self.cfg.replay_capacity)
        for eps in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self._act(state, explore=True)
                next_state, reward, done, _ = self.env.step(action, weight)
                replay.append(Transition(state, action, reward, next_state, done))
                state = next_state
                if eps >= self.cfg.warmup_eps and len(replay) >= 4:
                    self._update_networks(replay, actor_opt, critic_opt)

    def _update_networks(
        self,
        replay: ReplayBuffer,
        actor_opt: torch.optim.Optimizer,
        critic_opt: torch.optim.Optimizer,
    ) -> None:
        batch = replay.sample(self.cfg.batch_size)
        states = torch.as_tensor(np.stack([b.state for b in batch]), dtype=torch.float32, device=self.cfg.device)
        actions = torch.as_tensor(np.stack([b.action for b in batch]), dtype=torch.float32, device=self.cfg.device)
        rewards = torch.as_tensor([[b.reward] for b in batch], dtype=torch.float32, device=self.cfg.device)
        next_states = torch.as_tensor(np.stack([b.next_state for b in batch]), dtype=torch.float32, device=self.cfg.device)
        done = torch.as_tensor([[b.done] for b in batch], dtype=torch.float32, device=self.cfg.device)

        with torch.no_grad():
            next_actions, _ = self.actor.sample(next_states)
            target_q = rewards + self.cfg.gamma * (1.0 - done) * self.critic(next_states, next_actions)
        q = self.critic(states, actions)
        critic_loss = F.mse_loss(q, target_q)
        critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        critic_opt.step()

        pred_actions, _ = self.actor.sample(states)
        actor_loss = -self.critic(states, pred_actions).mean()
        actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        actor_opt.step()

    def _act(self, state: np.ndarray, explore: bool) -> np.ndarray:
        s = torch.as_tensor(state[None, :], dtype=torch.float32, device=self.cfg.device)
        with torch.no_grad():
            if explore:
                action, _ = self.actor.sample(s)
            else:
                action, _ = self.actor(s)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    @staticmethod
    def _reptile_update(base: dict[str, torch.Tensor], trained: list[dict[str, torch.Tensor]], lr: float) -> dict[str, torch.Tensor]:
        out = copy.deepcopy(base)
        if not trained:
            return out
        for key in out:
            delta = sum(state[key] - base[key] for state in trained) / len(trained)
            out[key] = base[key] + lr * delta
        return out

