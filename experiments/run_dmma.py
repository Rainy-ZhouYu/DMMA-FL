from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmma_fl import DMMAConfig
from dmma_fl.decomposition import hypervolume_2d
from dmma_fl.meta_drl import DMMAFL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a DMMA-FL reproduction experiment.")
    parser.add_argument("--dataset", default="synthetic", choices=["synthetic", "mnist", "cifar10"])
    parser.add_argument("--num-devices", type=int, default=10)
    parser.add_argument("--select-k", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--q", type=int, default=4)
    parser.add_argument("--meta-iters", type=int, default=2)
    parser.add_argument("--max-eps", type=int, default=2)
    parser.add_argument("--fine-tune-eps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="runs/dmma_results.json")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.num_devices = 5
        args.select_k = 2
        args.rounds = 3
        args.q = 2
        args.meta_iters = 1
        args.max_eps = 1
        args.fine_tune_eps = 1

    rng = np.random.default_rng(args.seed)
    data_sizes = rng.integers(80, 180, size=args.num_devices).astype(np.float32)
    cfg = DMMAConfig(
        seed=args.seed,
        num_devices=args.num_devices,
        select_k=args.select_k,
        rounds=args.rounds,
        q=args.q,
        meta_iters=args.meta_iters,
        max_eps=args.max_eps,
        fine_tune_eps=args.fine_tune_eps,
        sampled_weights=min(3, args.q + 1),
        device=args.device,
    )

    algo = DMMAFL(cfg, data_sizes)
    algo.train_meta()
    solutions = algo.fine_tune_all()

    raw_points = np.asarray([s.objectives for s in solutions], dtype=np.float64)
    mins = raw_points.min(axis=0)
    spans = np.maximum(raw_points.max(axis=0) - mins, 1e-9)
    normalized = (raw_points - mins) / spans
    hv = hypervolume_2d(normalized, ref=(1.1, 1.1))

    result = {
        "config": vars(args),
        "points": [
            {
                "weight": s.weight.tolist(),
                "C": float(s.objectives[0]),
                "L": float(s.objectives[1]),
            }
            for s in solutions
        ],
        "normalized_hypervolume": hv,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
