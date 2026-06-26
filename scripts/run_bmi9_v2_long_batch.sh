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

run_logged bmi9_v2_long_moo_q8_m30_t30 \
  "$PY" experiments/run_moo.py \
    --methods dmma,mopso,nsga,moead,fedcs,slidingde,sdefl \
    --num-devices 20 \
    --select-k 6 \
    --rounds 30 \
    --q 8 \
    --meta-iters 20 \
    --max-eps 10 \
    --fine-tune-eps 10 \
    --pop-size 32 \
    --baseline-iters 20 \
    --out runs/bmi9_v2_long_moo_q8_m30_t30.json

COMMON=(
  --download
  --weight-policy balanced
  --num-devices 20
  --select-k 6
  --q 8
  --meta-iters 14
  --max-eps 8
  --fine-tune-eps 8
  --strategies dmma-auwa,fedavg,fedbuffer,fedasync,random-auwa
  --samples 10000
  --batch-size 128
  --max-local-batches 5
)

run_fl() {
  local dataset=$1
  local model=$2
  local split=$3
  local rounds=$4
  local out=$5
  local name
  name=$(basename "${out}" .json)
  local split_args=()
  if [[ "${split}" == "iid" ]]; then
    split_args=(--iid)
  fi
  run_logged "${name}" \
    "$PY" experiments/run_fl.py \
      --dataset "${dataset}" \
      --model "${model}" \
      --rounds "${rounds}" \
      "${COMMON[@]}" \
      "${split_args[@]}" \
      --out "${out}"
}

run_fl mnist mlp iid 20 runs/bmi9_v2_long_mnist_mlp_iid.json
run_fl mnist mlp noniid 20 runs/bmi9_v2_long_mnist_mlp_noniid.json
run_fl mnist cnn1 iid 20 runs/bmi9_v2_long_mnist_cnn1_iid.json
run_fl mnist cnn1 noniid 20 runs/bmi9_v2_long_mnist_cnn1_noniid.json

run_fl cifar10 cnn2 iid 12 runs/bmi9_v2_long_cifar10_cnn2_iid.json
run_fl cifar10 cnn2 noniid 12 runs/bmi9_v2_long_cifar10_cnn2_noniid.json

echo "===== ALL LONG RUNS DONE $(date '+%F %T') ====="
