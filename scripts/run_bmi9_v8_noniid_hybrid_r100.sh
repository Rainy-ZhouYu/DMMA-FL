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
  --prox-mu 0.01 \
  --eval-interval 5 \
  --out runs/bmi9_v8_noniid_mnist_mlp_hybrid_r100.json
