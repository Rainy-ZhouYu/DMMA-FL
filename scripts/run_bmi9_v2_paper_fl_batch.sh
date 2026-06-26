#!/usr/bin/env bash
set -euo pipefail

cd /home/yzhou/DMMA-FL

PY=/home/yzhou/anaconda3/envs/dmma-fl/bin/python
COMMON=(
  --download
  --weight-policy balanced
  --num-devices 20
  --select-k 6
  --rounds 10
  --q 8
  --meta-iters 8
  --max-eps 6
  --fine-tune-eps 6
  --strategies dmma-auwa,fedavg,fedbuffer,fedasync,random-auwa
  --samples 5000
  --batch-size 128
  --max-local-batches 3
)

run_one() {
  local dataset=$1
  local model=$2
  local split=$3
  local out=$4
  shift 4
  local split_args=()
  if [[ "$split" == "iid" ]]; then
    split_args=(--iid)
  fi
  echo "===== V2 RUN dataset=${dataset} model=${model} split=${split} out=${out} ====="
  "$PY" experiments/run_fl.py \
    --dataset "$dataset" \
    --model "$model" \
    "${COMMON[@]}" \
    "${split_args[@]}" \
    "$@" \
    --out "$out"
}

run_one mnist mlp iid runs/bmi9_v2_paper_mnist_mlp_iid.json
run_one mnist mlp noniid runs/bmi9_v2_paper_mnist_mlp_noniid.json
run_one mnist cnn1 iid runs/bmi9_v2_paper_mnist_cnn1_iid.json
run_one mnist cnn1 noniid runs/bmi9_v2_paper_mnist_cnn1_noniid.json

run_one cifar10 cnn2 iid runs/bmi9_v2_paper_cifar10_cnn2_iid.json --rounds 8 --samples 5000
run_one cifar10 cnn2 noniid runs/bmi9_v2_paper_cifar10_cnn2_noniid.json --rounds 8 --samples 5000

echo "===== V2 DONE ====="
