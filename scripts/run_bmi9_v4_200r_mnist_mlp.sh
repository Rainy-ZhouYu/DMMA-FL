#!/usr/bin/env bash
set -euo pipefail

cd /home/yzhou/DMMA-FL

PY=/home/yzhou/anaconda3/envs/dmma-fl/bin/python

run_logged() {
  local name=$1
  shift
  local log="runs/${name}.log"
  echo "===== START ${name} $(date '+%F %T') ====="
  "$@" >"${log}" 2>&1
  echo "===== DONE  ${name} $(date '+%F %T') log=${log} ====="
}

COMMON=(
  --dataset mnist
  --model mlp
  --download
  --weight-policy balanced
  --num-devices 20
  --select-k 6
  --rounds 200
  --q 8
  --meta-iters 18
  --max-eps 10
  --fine-tune-eps 10
  --strategies dmma-auwa,fedavg,fedasync
  --samples 10000
  --batch-size 128
  --max-local-batches 8
)

run_logged bmi9_v4_200r_mnist_mlp_iid \
  "$PY" experiments/run_fl.py \
    "${COMMON[@]}" \
    --iid \
    --out runs/bmi9_v4_200r_mnist_mlp_iid.json

run_logged bmi9_v4_200r_mnist_mlp_noniid \
  "$PY" experiments/run_fl.py \
    "${COMMON[@]}" \
    --out runs/bmi9_v4_200r_mnist_mlp_noniid.json

echo "===== ALL V4 200R RUNS DONE $(date '+%F %T') ====="
