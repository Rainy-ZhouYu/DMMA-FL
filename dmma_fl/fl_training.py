from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .auwa import AUWAggregator
from .config import DMMAConfig
from .data import loader_for
from .fl_models import CNN1, CNN2, MLP
from .meta_drl import ParetoSolution
from .satellite import LEOEnvironment


@dataclass
class FLRoundMetric:
    round: int
    accuracy: float
    loss: float
    latency: float
    selected: list[int]
    aggregated: bool


@dataclass
class FLRunResult:
    strategy: str
    final_accuracy: float
    final_loss: float
    total_latency: float
    selected_counts: list[int]
    selected_unique: int
    rounds: list[FLRoundMetric]


def build_model(name: str, image_shape: tuple[int, int, int], num_classes: int) -> nn.Module:
    channels, height, width = image_shape
    if name == "mlp":
        return MLP(channels * height * width, num_classes)
    if name == "cnn1":
        return CNN1(channels, num_classes)
    if name == "cnn2":
        return CNN2(channels, num_classes)
    raise ValueError(f"Unsupported model: {name}")


def evaluate(model: nn.Module, dataset: Dataset, batch_size: int, device: str) -> tuple[float, float]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            total_loss += float(criterion(logits, y).item())
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(y.numel())
    return total_loss / max(total, 1), total_correct / max(total, 1)


def client_label_histograms(client_datasets: list[Dataset], num_classes: int) -> np.ndarray:
    hists = np.zeros((len(client_datasets), num_classes), dtype=np.float32)
    for client_idx, dataset in enumerate(client_datasets):
        for _, y in dataset:
            hists[client_idx, int(y)] += 1.0
    return hists


def local_train(
    base_model: nn.Module,
    dataset: Dataset,
    batch_size: int,
    epochs: int,
    lr: float,
    device: str,
    max_batches: int | None = None,
    prox_mu: float = 0.0,
) -> tuple[dict[str, torch.Tensor], float, float]:
    model = copy.deepcopy(base_model).to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    global_params = [param.detach().clone() for param in base_model.parameters()] if prox_mu > 0 else []
    before_loss = _dataset_loss(model, dataset, batch_size, criterion, device, max_batches=2)
    for _ in range(epochs):
        for batch_idx, (x, y) in enumerate(loader_for(dataset, batch_size, shuffle=True)):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            if prox_mu > 0:
                prox = sum(
                    torch.sum((param - global_param.to(param.device)) ** 2)
                    for param, global_param in zip(model.parameters(), global_params)
                )
                loss = loss + 0.5 * prox_mu * prox
            loss.backward()
            optimizer.step()
            if max_batches is not None and batch_idx + 1 >= max_batches:
                break
    after_loss = _dataset_loss(model, dataset, batch_size, criterion, device, max_batches=2)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, before_loss, after_loss


def run_fl_strategy(
    strategy: str,
    cfg: DMMAConfig,
    base_model: nn.Module,
    client_datasets: list[Dataset],
    test_dataset: Dataset,
    solution: ParetoSolution | None,
    data_sizes: np.ndarray,
    seed: int,
    batch_size: int,
    local_lr: float,
    max_local_batches: int | None,
    label_histograms: np.ndarray | None = None,
    fairness_alpha: float = 0.0,
    diversity_alpha: float = 0.0,
    fair_diverse_replace_fraction: float = 0.5,
    prox_mu: float = 0.0,
    eval_interval: int = 1,
) -> FLRunResult:
    rng = np.random.default_rng(seed)
    device = cfg.device
    model = copy.deepcopy(base_model).to(device)
    env = LEOEnvironment(cfg, data_sizes, rng)
    aggregator = AUWAggregator(cfg.buffer_size)
    total_latency = 0.0
    metrics: list[FLRoundMetric] = []
    last_selected = np.full(cfg.num_devices, -1, dtype=np.int32)
    selected_counts = np.zeros(cfg.num_devices, dtype=np.int32)
    label_histograms = label_histograms if label_histograms is not None else client_label_histograms(client_datasets, cfg.num_classes)
    eval_interval = max(int(eval_interval), 1)

    for round_idx in range(cfg.rounds):
        if solution is not None and round_idx < len(solution.actions):
            selected_mask, cpu_idx, power_idx = env.decode_action(solution.actions[round_idx])
            objective_weight = solution.weight
        else:
            action = rng.uniform(-1.0, 1.0, size=cfg.action_dim).astype(np.float32)
            selected_mask, cpu_idx, power_idx = env.decode_action(action)
            objective_weight = np.asarray([0.5, 0.5], dtype=np.float32)

        selected = np.flatnonzero(selected_mask > 0.5).tolist()
        if not selected:
            selected = [int(rng.integers(0, cfg.num_devices))]
        selected = _fair_diverse_selection(
            strategy=strategy,
            selected=selected,
            label_histograms=label_histograms,
            last_selected=last_selected,
            selected_counts=selected_counts,
            round_idx=round_idx,
            select_k=cfg.select_k,
            fairness_alpha=fairness_alpha,
            diversity_alpha=diversity_alpha,
            replace_fraction=fair_diverse_replace_fraction,
            rng=rng,
        )
        selected_mask = np.zeros(cfg.num_devices, dtype=np.float32)
        selected_mask[selected] = 1.0
        _, _, _, info = env.step_decoded(selected_mask, cpu_idx, power_idx, objective_weight)
        total_latency += float(info.latency)

        local_updates = []
        utilities = []
        staleness = _selection_staleness(last_selected, round_idx)
        staleness_norm = staleness / max(float(staleness.max()), 1.0)
        entropy = _normalized_entropy(label_histograms)
        participation = selected_counts / max(float(selected_counts.max()), 1.0)
        underuse = 1.0 - participation
        for client_idx in selected:
            last_selected[client_idx] = round_idx
            selected_counts[client_idx] += 1
        for client_idx in selected:
            state, before_loss, after_loss = local_train(
                model,
                client_datasets[client_idx],
                cfg.batch_train,
                cfg.local_epochs,
                local_lr,
                device,
                max_batches=max_local_batches,
                prox_mu=prox_mu if strategy.endswith("-prox") else 0.0,
            )
            utility = max(before_loss - after_loss, 1e-8) * len(client_datasets[client_idx])
            if strategy in {"dmma-auwa-fair", "dmma-auwa-diverse", "dmma-auwa-diverse-prox", "dmma-auwa-hybrid"}:
                underused = float(underuse[client_idx])
                utility *= 1.0 + fairness_alpha * max(float(staleness_norm[client_idx]), underused)
                if "diverse" in strategy or "hybrid" in strategy:
                    utility *= 1.0 + diversity_alpha * float(entropy[client_idx])
            local_updates.append((client_idx, state))
            utilities.append(utility)

        aggregated = False
        if strategy in {"dmma-auwa", "random-auwa", "dmma-auwa-fair", "dmma-auwa-diverse", "dmma-auwa-diverse-prox", "dmma-auwa-hybrid"}:
            for (_, state), utility in zip(local_updates, utilities):
                new_state = aggregator.push(state, utility)
                if new_state is not None:
                    model.load_state_dict(new_state)
                    aggregated = True
        elif strategy == "fedbuffer":
            for (client_idx, state), _ in zip(local_updates, utilities):
                new_state = aggregator.push(state, len(client_datasets[client_idx]))
                if new_state is not None:
                    model.load_state_dict(new_state)
                    aggregated = True
        elif strategy == "fedasync":
            for (_, state), _ in sorted(zip(local_updates, utilities), key=lambda item: item[1], reverse=True):
                _mix_state_dict(model, state, alpha=0.5)
                aggregated = True
        elif strategy == "fedavg":
            weights = [len(client_datasets[idx]) for idx, _ in local_updates]
            model.load_state_dict(_weighted_average([state for _, state in local_updates], weights))
            aggregated = True
        else:
            raise ValueError(f"Unsupported FL strategy: {strategy}")

        if (round_idx + 1) % eval_interval == 0 or round_idx + 1 == cfg.rounds:
            loss, acc = evaluate(model, test_dataset, batch_size, device)
            metrics.append(
                FLRoundMetric(
                    round=round_idx + 1,
                    accuracy=acc,
                    loss=loss,
                    latency=total_latency,
                    selected=selected,
                    aggregated=aggregated,
                )
            )

    final = metrics[-1]
    return FLRunResult(
        strategy=strategy,
        final_accuracy=final.accuracy,
        final_loss=final.loss,
        total_latency=total_latency,
        selected_counts=selected_counts.astype(int).tolist(),
        selected_unique=int(np.count_nonzero(selected_counts)),
        rounds=metrics,
    )


def _selection_staleness(last_selected: np.ndarray, round_idx: int) -> np.ndarray:
    return np.where(last_selected < 0, round_idx + 1, round_idx - last_selected).astype(np.float32)


def _normalized_entropy(label_histograms: np.ndarray) -> np.ndarray:
    totals = label_histograms.sum(axis=1, keepdims=True)
    probs = np.divide(label_histograms, np.maximum(totals, 1.0), out=np.zeros_like(label_histograms), where=totals > 0)
    entropy = -(probs * np.log(np.maximum(probs, 1e-12))).sum(axis=1)
    return entropy / max(np.log(label_histograms.shape[1]), 1e-12)


def _entropy_of_counts(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    probs = counts / total
    entropy = float(-(probs * np.log(np.maximum(probs, 1e-12))).sum())
    return entropy / max(float(np.log(counts.shape[0])), 1e-12)


def _fair_diverse_selection(
    strategy: str,
    selected: list[int],
    label_histograms: np.ndarray,
    last_selected: np.ndarray,
    selected_counts: np.ndarray,
    round_idx: int,
    select_k: int,
    fairness_alpha: float,
    diversity_alpha: float,
    replace_fraction: float,
    rng: np.random.Generator,
) -> list[int]:
    if strategy not in {"dmma-auwa-fair", "dmma-auwa-diverse", "dmma-auwa-diverse-prox", "dmma-auwa-hybrid"}:
        return selected

    num_clients = label_histograms.shape[0]
    target_k = min(max(select_k, 1), num_clients)
    selected = list(dict.fromkeys(int(idx) for idx in selected if 0 <= int(idx) < num_clients))
    if not selected:
        selected = [int(rng.integers(0, num_clients))]

    staleness = _selection_staleness(last_selected, round_idx)
    staleness_norm = staleness / max(float(staleness.max()), 1.0)
    participation = selected_counts / max(float(selected_counts.max()), 1.0)
    underuse = 1.0 - participation
    client_entropy = _normalized_entropy(label_histograms)
    use_diversity = "diverse" in strategy or "hybrid" in strategy
    if strategy == "dmma-auwa-hybrid":
        replace_fraction = max(replace_fraction, 0.67)
    keep_count = min(len(selected), max(1, int(round(target_k * (1.0 - replace_fraction)))))
    keep_scores = {
        idx: fairness_alpha * float(staleness_norm[idx])
        + (diversity_alpha * float(client_entropy[idx]) if use_diversity else 0.0)
        + (0.5 * float(underuse[idx]) if strategy == "dmma-auwa-hybrid" else 0.0)
        + 0.05
        for idx in selected
    }
    chosen = sorted(selected, key=lambda idx: keep_scores[idx], reverse=True)[:keep_count]
    current_counts = label_histograms[chosen].sum(axis=0) if chosen else np.zeros(label_histograms.shape[1], dtype=np.float32)

    while len(chosen) < target_k:
        best_idx = None
        best_score = -np.inf
        for idx in range(num_clients):
            if idx in chosen:
                continue
            diversity_gain = 0.0
            if use_diversity:
                diversity_gain = max(0.0, _entropy_of_counts(current_counts + label_histograms[idx]) - _entropy_of_counts(current_counts))
                uncovered = (current_counts <= 0) & (label_histograms[idx] > 0)
                diversity_gain += float(uncovered.sum()) / max(label_histograms.shape[1], 1)
            score = fairness_alpha * max(float(staleness_norm[idx]), float(underuse[idx])) + diversity_alpha * diversity_gain
            if strategy == "dmma-auwa-hybrid":
                score += 0.75 * float(underuse[idx])
                if selected_counts[idx] == 0:
                    score += 0.75
            if idx in selected:
                score += 0.05 if strategy == "dmma-auwa-hybrid" else 0.1
            score += float(rng.uniform(0.0, 1e-6))
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx is None:
            break
        chosen.append(best_idx)
        current_counts = current_counts + label_histograms[best_idx]
    return chosen


def _dataset_loss(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int,
    criterion: nn.Module,
    device: str,
    max_batches: int | None,
) -> float:
    model.eval()
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader_for(dataset, batch_size, shuffle=False)):
            x = x.to(device)
            y = y.to(device)
            total_loss += float(criterion(model(x), y).item()) * int(y.numel())
            total += int(y.numel())
            if max_batches is not None and batch_idx + 1 >= max_batches:
                break
    model.train()
    return total_loss / max(total, 1)


def _weighted_average(states: Iterable[dict[str, torch.Tensor]], weights: Iterable[float]) -> dict[str, torch.Tensor]:
    states = list(states)
    weights = [float(w) for w in weights]
    total = sum(weights)
    out: dict[str, torch.Tensor] = {}
    for key in states[0]:
        out[key] = sum((w / total) * state[key] for state, w in zip(states, weights))
    return out


def _mix_state_dict(model: nn.Module, local_state: dict[str, torch.Tensor], alpha: float) -> None:
    current = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    mixed = {key: (1.0 - alpha) * current[key] + alpha * local_state[key] for key in current}
    model.load_state_dict(mixed)
