#!/usr/bin/env bash
set -euo pipefail

cd /home/yzhou/DMMA-FL

PY=/home/yzhou/anaconda3/envs/dmma-fl/bin/python

"$PY" experiments/run_fl.py \
  --dataset mnist \
  --download \
  --model mlp \
  --num-devices 20 \
  --select-k 6 \
  --rounds 60 \
  --q 8 \
  --meta-iters 14 \
  --max-eps 8 \
  --fine-tune-eps 8 \
  --weight-policy balanced \
  --strategies dmma-auwa,dmma-auwa-diverse,dmma-auwa-hybrid,fedavg \
  --samples 10000 \
  --batch-size 128 \
  --local-lr 0.05 \
  --max-local-batches 8 \
  --fairness-alpha 0.45 \
  --diversity-alpha 0.75 \
  --fair-diverse-replace-fraction 0.67 \
  --prox-mu 0.01 \
  --eval-interval 5 \
  --out runs/bmi9_v7_noniid_mnist_mlp_hybrid_probe_r60.json
