from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import DMMAConfig
from .decomposition import simplex_weights
from .satellite import LEOEnvironment, StepInfo


@dataclass
class ScheduleSolution:
    method: str
    objectives: tuple[float, float]
    schedule: np.ndarray
    infos: list[StepInfo]
    weight: list[float] | None = None


def evaluate_schedule(
    cfg: DMMAConfig,
    data_sizes: np.ndarray,
    schedule: np.ndarray,
    seed: int,
) -> ScheduleSolution:
    rng = np.random.default_rng(seed)
    env = LEOEnvironment(cfg, data_sizes, rng)
    env.reset()
    infos: list[StepInfo] = []
    total_c = 0.0
    total_l = 0.0
    for step in range(cfg.rounds):
        action = schedule[step]
        _, _, _, info = env.step(action, np.asarray([0.5, 0.5], dtype=np.float32))
        total_c += info.convergence
        total_l += info.latency
        infos.append(info)
    return ScheduleSolution("schedule", (total_c, total_l), schedule.copy(), infos)


def run_nsga_fl(
    cfg: DMMAConfig,
    data_sizes: np.ndarray,
    pop_size: int,
    generations: int,
    seed: int,
) -> list[ScheduleSolution]:
    rng = np.random.default_rng(seed)
    shape = (cfg.rounds, cfg.action_dim)
    population = rng.uniform(-1.0, 1.0, size=(pop_size, *shape)).astype(np.float32)
    scores = _evaluate_population(cfg, data_sizes, population, seed)

    for _ in range(generations):
        ranks, crowd = _rank_and_crowding(scores)
        children = []
        while len(children) < pop_size:
            p1 = population[_tournament(rng, ranks, crowd)]
            p2 = population[_tournament(rng, ranks, crowd)]
            child = _sbx_like_crossover(rng, p1, p2)
            child = _mutate(rng, child, rate=0.08, sigma=0.25)
            children.append(child)
        combined = np.concatenate([population, np.asarray(children, dtype=np.float32)], axis=0)
        combined_scores = _evaluate_population(cfg, data_sizes, combined, seed)
        keep = _select_nsga(combined_scores, pop_size)
        population = combined[keep]
        scores = combined_scores[keep]

    keep = _nondominated_indices(scores)
    return [
        ScheduleSolution("nsga-fl", tuple(map(float, scores[i])), population[i], evaluate_schedule(cfg, data_sizes, population[i], seed).infos)
        for i in keep
    ]


def run_mopso_fl(
    cfg: DMMAConfig,
    data_sizes: np.ndarray,
    swarm_size: int,
    iterations: int,
    archive_size: int,
    seed: int,
) -> list[ScheduleSolution]:
    rng = np.random.default_rng(seed)
    shape = (cfg.rounds, cfg.action_dim)
    pos = rng.uniform(-1.0, 1.0, size=(swarm_size, *shape)).astype(np.float32)
    vel = rng.normal(0.0, 0.1, size=(swarm_size, *shape)).astype(np.float32)
    scores = _evaluate_population(cfg, data_sizes, pos, seed)
    pbest = pos.copy()
    pbest_scores = scores.copy()
    archive_pos, archive_scores = _archive(pos, scores, archive_size)

    for _ in range(iterations):
        for i in range(swarm_size):
            leader = archive_pos[rng.integers(0, len(archive_pos))]
            r1 = rng.random(shape)
            r2 = rng.random(shape)
            vel[i] = 0.5 * vel[i] + 1.4 * r1 * (pbest[i] - pos[i]) + 1.4 * r2 * (leader - pos[i])
            pos[i] = np.clip(pos[i] + vel[i], -1.0, 1.0)
        scores = _evaluate_population(cfg, data_sizes, pos, seed)
        for i, score in enumerate(scores):
            if _dominates(score, pbest_scores[i]) or (not _dominates(pbest_scores[i], score) and rng.random() < 0.5):
                pbest[i] = pos[i]
                pbest_scores[i] = score
        archive_pos, archive_scores = _archive(
            np.concatenate([archive_pos, pos], axis=0),
            np.concatenate([archive_scores, scores], axis=0),
            archive_size,
        )

    return [
        ScheduleSolution("mopso-fl", tuple(map(float, score)), schedule, evaluate_schedule(cfg, data_sizes, schedule, seed).infos)
        for schedule, score in zip(archive_pos, archive_scores)
    ]


def run_moead_fl(
    cfg: DMMAConfig,
    data_sizes: np.ndarray,
    q: int,
    iterations: int,
    neighborhood: int,
    seed: int,
) -> list[ScheduleSolution]:
    rng = np.random.default_rng(seed)
    weights = simplex_weights(q, 2)
    n = len(weights)
    shape = (cfg.rounds, cfg.action_dim)
    population = rng.uniform(-1.0, 1.0, size=(n, *shape)).astype(np.float32)
    scores = _evaluate_population(cfg, data_sizes, population, seed)
    ideal = scores.min(axis=0)
    distances = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=-1)
    neighbors = np.argsort(distances, axis=1)[:, : max(2, neighborhood)]

    for _ in range(iterations):
        for i in range(n):
            pool = neighbors[i]
            a, b = rng.choice(pool, size=2, replace=True)
            child = _mutate(rng, _sbx_like_crossover(rng, population[a], population[b]), rate=0.1, sigma=0.2)
            child_score = _evaluate_population(cfg, data_sizes, child[None, ...], seed)[0]
            ideal = np.minimum(ideal, child_score)
            for j in pool:
                if _tchebycheff(child_score, weights[j], ideal) <= _tchebycheff(scores[j], weights[j], ideal):
                    population[j] = child
                    scores[j] = child_score

    keep = _nondominated_indices(scores)
    return [
        ScheduleSolution("moea-fl", tuple(map(float, scores[i])), population[i], evaluate_schedule(cfg, data_sizes, population[i], seed).infos, weights[i].tolist())
        for i in keep
    ]


def run_single_objective_baseline(
    method: str,
    cfg: DMMAConfig,
    data_sizes: np.ndarray,
    iterations: int,
    seed: int,
) -> list[ScheduleSolution]:
    weights = {
        "fedcs": np.asarray([1.0, 0.0], dtype=np.float32),
        "sdefl": np.asarray([0.0, 1.0], dtype=np.float32),
        "slidingde": np.asarray([1.0, 0.9], dtype=np.float32),
    }
    if method not in weights:
        raise ValueError(f"Unknown single-objective baseline: {method}")
    schedule, score = _differential_evolution(cfg, data_sizes, weights[method], iterations, seed)
    evaluated = evaluate_schedule(cfg, data_sizes, schedule, seed)
    evaluated.method = method
    evaluated.weight = weights[method].tolist()
    # Preserve the scalar-optimized full objective, not the scalar score alone.
    evaluated.objectives = tuple(map(float, score))
    return [evaluated]


def _evaluate_population(cfg: DMMAConfig, data_sizes: np.ndarray, population: np.ndarray, seed: int) -> np.ndarray:
    scores = []
    for idx, schedule in enumerate(population):
        scores.append(evaluate_schedule(cfg, data_sizes, schedule, seed + idx).objectives)
    return np.asarray(scores, dtype=np.float64)


def _differential_evolution(
    cfg: DMMAConfig,
    data_sizes: np.ndarray,
    weight: np.ndarray,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pop_size = 16
    shape = (cfg.rounds, cfg.action_dim)
    population = rng.uniform(-1.0, 1.0, size=(pop_size, *shape)).astype(np.float32)
    scores = _evaluate_population(cfg, data_sizes, population, seed)
    scalar = scores @ weight
    for _ in range(iterations):
        for i in range(pop_size):
            choices = [j for j in range(pop_size) if j != i]
            a, b, c = rng.choice(choices, size=3, replace=False)
            mutant = np.clip(population[a] + 0.5 * (population[b] - population[c]), -1.0, 1.0)
            mask = rng.random(shape) < 0.7
            trial = np.where(mask, mutant, population[i]).astype(np.float32)
            trial_score = _evaluate_population(cfg, data_sizes, trial[None, ...], seed + 10_000 + i)[0]
            trial_scalar = float(trial_score @ weight)
            if trial_scalar <= scalar[i]:
                population[i] = trial
                scores[i] = trial_score
                scalar[i] = trial_scalar
    best = int(np.argmin(scalar))
    return population[best], scores[best]


def _archive(population: np.ndarray, scores: np.ndarray, archive_size: int) -> tuple[np.ndarray, np.ndarray]:
    keep = _nondominated_indices(scores)
    pop = population[keep]
    sc = scores[keep]
    if len(pop) <= archive_size:
        return pop, sc
    _, crowd = _rank_and_crowding(sc)
    order = np.argsort(-crowd)
    selected = order[:archive_size]
    return pop[selected], sc[selected]


def _nondominated_indices(scores: np.ndarray) -> np.ndarray:
    keep = []
    for i, score in enumerate(scores):
        dominated = any(_dominates(other, score) for j, other in enumerate(scores) if i != j)
        if not dominated:
            keep.append(i)
    return np.asarray(keep, dtype=int)


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def _rank_and_crowding(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fronts: list[list[int]] = []
    remaining = set(range(len(scores)))
    ranks = np.zeros(len(scores), dtype=int)
    rank = 0
    while remaining:
        front = []
        for idx in list(remaining):
            if not any(_dominates(scores[j], scores[idx]) for j in remaining if j != idx):
                front.append(idx)
        for idx in front:
            ranks[idx] = rank
            remaining.remove(idx)
        fronts.append(front)
        rank += 1
    crowd = np.zeros(len(scores), dtype=np.float64)
    for front in fronts:
        if len(front) <= 2:
            crowd[front] = np.inf
            continue
        values = scores[front]
        for obj in range(scores.shape[1]):
            order = np.argsort(values[:, obj])
            crowd[np.asarray(front)[order[0]]] = np.inf
            crowd[np.asarray(front)[order[-1]]] = np.inf
            span = values[order[-1], obj] - values[order[0], obj]
            if span <= 1e-12:
                continue
            for k in range(1, len(order) - 1):
                crowd[np.asarray(front)[order[k]]] += (values[order[k + 1], obj] - values[order[k - 1], obj]) / span
    return ranks, crowd


def _tournament(rng: np.random.Generator, ranks: np.ndarray, crowd: np.ndarray) -> int:
    a, b = rng.integers(0, len(ranks), size=2)
    if ranks[a] < ranks[b]:
        return int(a)
    if ranks[b] < ranks[a]:
        return int(b)
    return int(a if crowd[a] >= crowd[b] else b)


def _select_nsga(scores: np.ndarray, n: int) -> np.ndarray:
    ranks, crowd = _rank_and_crowding(scores)
    order = sorted(range(len(scores)), key=lambda i: (ranks[i], -crowd[i]))
    return np.asarray(order[:n], dtype=int)


def _sbx_like_crossover(rng: np.random.Generator, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    alpha = rng.random(a.shape, dtype=np.float32)
    return np.clip(alpha * a + (1.0 - alpha) * b, -1.0, 1.0).astype(np.float32)


def _mutate(rng: np.random.Generator, x: np.ndarray, rate: float, sigma: float) -> np.ndarray:
    mask = rng.random(x.shape) < rate
    noise = rng.normal(0.0, sigma, size=x.shape).astype(np.float32)
    return np.clip(np.where(mask, x + noise, x), -1.0, 1.0).astype(np.float32)


def _tchebycheff(score: np.ndarray, weight: np.ndarray, ideal: np.ndarray) -> float:
    return float(np.max(np.maximum(weight, 1e-3) * np.abs(score - ideal)))

