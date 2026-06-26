from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmma_fl.config import DMMAConfig
from dmma_fl.data import load_dataset, split_clients
from dmma_fl.fl_training import build_model, client_label_histograms, run_fl_strategy
from dmma_fl.meta_drl import DMMAFL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DMMA-FL with actual federated model training.")
    parser.add_argument("--dataset", default="synthetic", choices=["synthetic", "mnist", "cifar10"])
    parser.add_argument("--model", default="mlp", choices=["mlp", "cnn1", "cnn2"])
    parser.add_argument("--iid", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--num-devices", type=int, default=10)
    parser.add_argument("--select-k", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--q", type=int, default=4)
    parser.add_argument("--meta-iters", type=int, default=2)
    parser.add_argument("--max-eps", type=int, default=2)
    parser.add_argument("--fine-tune-eps", type=int, default=2)
    parser.add_argument("--weight-index", type=int, default=0)
    parser.add_argument(
        "--weight-policy",
        default="index",
        choices=["index", "balanced", "min-convergence", "min-latency"],
        help="How to choose one Pareto solution for the DMMA-AUWA FL schedule.",
    )
    parser.add_argument("--strategies", default="dmma-auwa,fedavg,fedbuffer,fedasync,random-auwa")
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--local-lr", type=float, default=0.05)
    parser.add_argument("--max-local-batches", type=int, default=2)
    parser.add_argument("--fairness-alpha", type=float, default=0.35)
    parser.add_argument("--diversity-alpha", type=float, default=0.65)
    parser.add_argument("--fair-diverse-replace-fraction", type=float, default=0.5)
    parser.add_argument("--prox-mu", type=float, default=0.01)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="runs/fl_results.json")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.num_devices = 5
        args.select_k = 2
        args.rounds = 2
        args.q = 2
        args.meta_iters = 1
        args.max_eps = 1
        args.fine_tune_eps = 1
        args.samples = 400
        args.max_local_batches = 1
        args.strategies = "dmma-auwa,fedavg"

    train_dataset, test_dataset, image_shape = load_dataset(
        args.dataset,
        args.samples,
        10,
        args.seed,
        download=args.download,
    )
    if args.dataset != "synthetic" and args.samples > 0:
        train_dataset = torch.utils.data.Subset(train_dataset, list(range(min(args.samples, len(train_dataset)))))
        test_dataset = torch.utils.data.Subset(test_dataset, list(range(min(max(args.samples // 5, 200), len(test_dataset)))))

    clients = split_clients(train_dataset, args.num_devices, iid=args.iid, seed=args.seed)
    data_sizes = np.asarray([len(client) for client in clients], dtype=np.float32)
    label_histograms = client_label_histograms(clients, 10)
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
        batch_train=args.batch_size,
        device=args.device,
    )

    dmma = DMMAFL(cfg, data_sizes)
    dmma.train_meta()
    solutions = dmma.fine_tune_all()
    solution = choose_solution(solutions, args.weight_policy, args.weight_index)
    model = build_model(args.model, image_shape, cfg.num_classes)

    results = []
    for strategy in [s.strip() for s in args.strategies.split(",") if s.strip()]:
        print(f"[run_fl] starting strategy={strategy}", file=sys.stderr, flush=True)
        strategy_solution = solution if strategy.startswith("dmma-auwa") else None
        run = run_fl_strategy(
            strategy=strategy,
            cfg=cfg,
            base_model=model,
            client_datasets=clients,
            test_dataset=test_dataset,
            solution=strategy_solution,
            data_sizes=data_sizes,
            seed=args.seed + 100,
            batch_size=args.batch_size,
            local_lr=args.local_lr,
            max_local_batches=args.max_local_batches,
            label_histograms=label_histograms,
            fairness_alpha=args.fairness_alpha,
            diversity_alpha=args.diversity_alpha,
            fair_diverse_replace_fraction=args.fair_diverse_replace_fraction,
            prox_mu=args.prox_mu,
            eval_interval=args.eval_interval,
        )
        print(
            f"[run_fl] finished strategy={strategy} final_accuracy={run.final_accuracy:.4f} latency={run.total_latency:.3f}",
            file=sys.stderr,
            flush=True,
        )
        results.append(asdict(run))

    payload = {
        "config": vars(args),
        "dmma_selected_weight": solution.weight.tolist(),
        "dmma_objectives": {"C": float(solution.objectives[0]), "L": float(solution.objectives[1])},
        "dmma_solution_points": [
            {
                "weight": s.weight.tolist(),
                "C": float(s.objectives[0]),
                "L": float(s.objectives[1]),
            }
            for s in solutions
        ],
        "results": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def choose_solution(solutions, policy: str, index: int):
    if not solutions:
        raise ValueError("No Pareto solutions were produced.")
    points = np.asarray([s.objectives for s in solutions], dtype=np.float64)
    if policy == "index":
        return solutions[min(max(index, 0), len(solutions) - 1)]
    if policy == "min-convergence":
        return solutions[int(np.argmin(points[:, 0]))]
    if policy == "min-latency":
        return solutions[int(np.argmin(points[:, 1]))]
    if policy == "balanced":
        mins = points.min(axis=0)
        spans = np.maximum(points.max(axis=0) - mins, 1e-9)
        normalized = (points - mins) / spans
        # Pick the knee-like point closest to the ideal lower-left objective corner.
        return solutions[int(np.argmin(np.linalg.norm(normalized, axis=1)))]
    raise ValueError(f"Unsupported weight policy: {policy}")


if __name__ == "__main__":
    main()
