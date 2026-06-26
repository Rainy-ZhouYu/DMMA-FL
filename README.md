# DMMA-FL

This repository contains the source code of Decomposition and Meta-DRL Based Multi-Objective Optimization for Asynchronous Federated Learning in 6G-Satellite Systems, IEEE EEE Journal on Selected Areas in Communications, 2024.

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

## Citation

If you use this code, please cite:

```bibtex
@ARTICLE{10436092,
  author={Zhou, Yu and Lei, Lei and Zhao, Xiaohui and You, Lei and Sun, Yaohua and Chatzinotas, Symeon},
  journal={IEEE Journal on Selected Areas in Communications},
  title={Decomposition and Meta-DRL Based Multi-Objective Optimization for Asynchronous Federated Learning in 6G-Satellite Systems},
  year={2024},
  volume={42},
  number={5},
  pages={1115-1129},
  keywords={Computational modeling;Optimization;Training;Data models;Satellites;Low earth orbit satellites;6G mobile communication;LEO satellite;asynchronous federated learning;multi-objective optimization;meta-reinforcement learning},
  doi={10.1109/JSAC.2024.3365902}
}
```
