# DMMA-FL Reproduction

This repository is an executable reproduction scaffold for:

> Decomposition and Meta-DRL Based Multi-Objective Optimization for
> Asynchronous Federated Learning in 6G-Satellite Systems, IEEE JSAC 2024.

The implementation focuses on the paper's core algorithmic pipeline:

1. Decompose the two-objective LEO-FL MOP into scalar subproblems.
2. Train meta actor/critic networks across sampled weight vectors.
3. Fine-tune each decomposed subproblem.
4. Execute asynchronous uploading and weighted aggregation (AUWA).
5. Report the Pareto points `(C, L)`, hypervolume, and test accuracy.

This version also includes a Non-IID extension used in our experiments:

- `dmma-auwa-fair`: DMMA-AUWA with staleness/fairness-aware client replacement.
- `dmma-auwa-diverse`: DMMA-AUWA with label-diversity-aware replacement.
- `dmma-auwa-hybrid`: a fair-diverse variant that balances DMMA's selected
  clients, low-participation clients, and label coverage.

The default experiment uses synthetic classification data so the code can be
smoke-tested on a fresh server without downloading MNIST/CIFAR-10. Pass
`--dataset mnist` or `--dataset cifar10` to use torchvision datasets when
available.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python experiments/run_dmma.py --smoke
```

For a longer run:

```bash
python experiments/run_dmma.py \
  --dataset synthetic \
  --num-devices 20 \
  --rounds 20 \
  --q 8 \
  --meta-iters 5 \
  --max-eps 4 \
  --fine-tune-eps 4
```

Run actual federated model training with the DMMA schedule and baselines:

```bash
python experiments/run_fl.py --smoke
python experiments/run_fl.py \
  --dataset synthetic \
  --model mlp \
  --num-devices 20 \
  --select-k 6 \
  --rounds 10 \
  --q 8 \
  --strategies dmma-auwa,fedavg,fedbuffer,fedasync,random-auwa
```

For MNIST or CIFAR-10, add `--dataset mnist --download` or
`--dataset cifar10 --model cnn2 --download`.

Run the Non-IID fair-diverse hybrid variant:

```bash
python experiments/run_fl.py \
  --dataset mnist \
  --download \
  --model mlp \
  --num-devices 20 \
  --select-k 6 \
  --rounds 100 \
  --q 8 \
  --meta-iters 18 \
  --max-eps 10 \
  --fine-tune-eps 10 \
  --weight-policy balanced \
  --strategies dmma-auwa,dmma-auwa-diverse,dmma-auwa-hybrid,fedavg \
  --samples 10000 \
  --batch-size 128 \
  --local-lr 0.05 \
  --max-local-batches 8 \
  --fairness-alpha 0.45 \
  --diversity-alpha 0.75 \
  --fair-diverse-replace-fraction 0.67 \
  --eval-interval 5 \
  --out runs/noniid_hybrid_r100.json
```

The BMI9 convenience scripts under `scripts/` reproduce the staged experiments,
including `scripts/run_bmi9_v8_noniid_hybrid_r100.sh`.

Compare optimization baselines on the common LEO-FL objective simulator:

```bash
python experiments/run_moo.py --smoke
python experiments/run_moo.py \
  --methods dmma,mopso,nsga,moead,fedcs,slidingde,sdefl \
  --num-devices 20 --select-k 6 --rounds 20 --q 8 \
  --meta-iters 5 --max-eps 4 --fine-tune-eps 4 \
  --pop-size 24 --baseline-iters 10
```

## Notes

- This is a faithful research-code reproduction from the published equations,
  not an official author release.
- The paper does not publish all engineering details needed for bitwise result
  matching. Defaults are therefore chosen to keep the simulation numerically
  stable and runnable on modest GPUs/CPUs.
- `--smoke` intentionally keeps the model/data tiny and is meant only to verify
  the pipeline.
- The fair/diverse/hybrid strategies are reproduction extensions for Non-IID
  client selection and are not claimed as original paper components.
