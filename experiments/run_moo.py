from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmma_fl.baselines import (
    run_moead_fl,
    run_mopso_fl,
    run_nsga_fl,
    run_single_objective_baseline,
)
from dmma_fl.config import DMMAConfig
from dmma_fl.decomposition import hypervolume_2d
from dmma_fl.meta_drl import DMMAFL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare DMMA-FL with MOO and single-objective baselines.")
    parser.add_argument("--methods", default="dmma,mopso,nsga,moead,fedcs,slidingde,sdefl")
    parser.add_argument("--num-devices", type=int, default=10)
    parser.add_argument("--select-k", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--q", type=int, default=4)
    parser.add_argument("--meta-iters", type=int, default=2)
    parser.add_argument("--max-eps", type=int, default=2)
    parser.add_argument("--fine-tune-eps", type=int, default=2)
    parser.add_argument("--pop-size", type=int, default=16)
    parser.add_argument("--baseline-iters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="runs/moo_results.json")
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
        args.pop_size = 8
        args.baseline_iters = 2

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
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    grouped: dict[str, list[dict]] = {}

    if "dmma" in methods:
        dmma = DMMAFL(cfg, data_sizes)
        dmma.train_meta()
        grouped["dmma"] = [
            {
                "method": "dmma",
                "weight": sol.weight.tolist(),
                "C": float(sol.objectives[0]),
                "L": float(sol.objectives[1]),
            }
            for sol in dmma.fine_tune_all()
        ]

    if "mopso" in methods:
        grouped["mopso"] = _records(run_mopso_fl(cfg, data_sizes, args.pop_size, args.baseline_iters, args.pop_size, args.seed + 10))
    if "nsga" in methods:
        grouped["nsga"] = _records(run_nsga_fl(cfg, data_sizes, args.pop_size, args.baseline_iters, args.seed + 20))
    if "moead" in methods:
        grouped["moead"] = _records(run_moead_fl(cfg, data_sizes, args.q, args.baseline_iters, neighborhood=3, seed=args.seed + 30))
    for method in ["fedcs", "slidingde", "sdefl"]:
        if method in methods:
            grouped[method] = _records(run_single_objective_baseline(method, cfg, data_sizes, args.baseline_iters, args.seed + 40))

    all_points = np.asarray([[row["C"], row["L"]] for rows in grouped.values() for row in rows], dtype=np.float64)
    mins = all_points.min(axis=0)
    spans = np.maximum(all_points.max(axis=0) - mins, 1e-9)
    hvs = {}
    for method, rows in grouped.items():
        pts = np.asarray([[row["C"], row["L"]] for row in rows], dtype=np.float64)
        hvs[method] = hypervolume_2d((pts - mins) / spans, ref=(1.1, 1.1))

    payload = {
        "config": vars(args),
        "hypervolume": hvs,
        "points": grouped,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def _records(solutions) -> list[dict]:
    rows = []
    for sol in solutions:
        rows.append(
            {
                "method": sol.method,
                "weight": sol.weight,
                "C": float(sol.objectives[0]),
                "L": float(sol.objectives[1]),
            }
        )
    return rows


if __name__ == "__main__":
    main()
