from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmma_fl.config import DMMAConfig
from dmma_fl.decomposition import hypervolume_2d, simplex_weights
from dmma_fl.satellite import LEOEnvironment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run the DMMA-FL optimization environment without PyTorch.")
    parser.add_argument("--num-devices", type=int, default=5)
    parser.add_argument("--select-k", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--q", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    cfg = DMMAConfig(
        seed=args.seed,
        num_devices=args.num_devices,
        select_k=args.select_k,
        rounds=args.rounds,
        q=args.q,
    )
    data_sizes = rng.integers(80, 180, size=args.num_devices).astype(np.float32)
    env = LEOEnvironment(cfg, data_sizes, rng)

    points = []
    for weight in simplex_weights(cfg.q, 2):
        state = env.reset()
        total_c = 0.0
        total_l = 0.0
        done = False
        while not done:
            action = rng.uniform(-1.0, 1.0, size=cfg.action_dim).astype(np.float32)
            state, reward, done, info = env.step(action, weight)
            total_c += info.convergence
            total_l += info.latency
        points.append([total_c, total_l])

    raw = np.asarray(points, dtype=np.float64)
    normalized = (raw - raw.min(axis=0)) / np.maximum(raw.max(axis=0) - raw.min(axis=0), 1e-9)
    result = {
        "weights": simplex_weights(cfg.q, 2).tolist(),
        "points": raw.tolist(),
        "normalized_hypervolume": hypervolume_2d(normalized, ref=(1.1, 1.1)),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
