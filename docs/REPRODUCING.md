# Reproducing the paper

Every command runs from the repository root with the package installed
(`pip install -e ".[dev]"`). Outputs go wherever `--out` points; the commands
below use `runs/` for training runs and `results/` for evaluation output.

Throughout, `H20` stands for the four main models,

```
--models qtro=runs/h20/qtro,cnto=runs/h20/cnto,seqpred=runs/h20/seqpred,deeponet=runs/h20/deeponet
```

## 1. Environment

```sh
export NS3_DIR=/path/to/ns-allinone-3.42/ns-3.42   # the tree with ns3/qtro-*.cc built in scratch/
export NS3_MODE=wsl NS3_USER=ns3                    # only when ns-3 lives in WSL; see ns3/README.md
```

A GPU is recommended: the reported runs used one 16 GB card, and every
script takes `--device cpu` when there is none.

## 2. Datasets

All datasets come from ns-3.42 (optimised build) through
`scripts/data/generate_dataset.py`; `--workers` sets how many ns-3 processes
run at once. A condition list may override the replication count per
condition, which is how the main dataset's tail settings get more runs.

Main dataset, 300 conditions (120 at 64 replications, the 180 with
`min_th >= 26` at 128, as the condition list says):

```sh
python scripts/data/design_main_conditions.py --out configs/conditions/main.json   # the shipped file, seed 20260826
python scripts/data/generate_dataset.py --conditions configs/conditions/main.json \
    --out data/main --split-name train --replications 64 --samples 600
```

Out-of-envelope dataset, 68 conditions:

```sh
python scripts/data/design_extrapolation_conditions.py --out configs/conditions/extrapolation.json
python scripts/data/generate_dataset.py --conditions configs/conditions/extrapolation.json \
    --out data/extrapolation --split-name ns3_extrap
```

Shift datasets, 120 conditions each, on the main dataset's held-out
coordinates:

```sh
python scripts/data/design_shift_conditions.py --panel data/main --kind burst --out configs/conditions/shift_burst.json
python scripts/data/generate_dataset.py --conditions configs/conditions/shift_burst.json \
    --out data/shift_burst --split-name ns3_burst
python scripts/data/design_shift_conditions.py --panel data/main --kind tandem --out configs/conditions/shift_tandem.json
python scripts/data/generate_dataset.py --conditions configs/conditions/shift_tandem.json \
    --out data/shift_tandem --split-name ns3_tandem --scenario qtro-multibottleneck
```

Validation dataset, 17 conditions (the design had 18; the one that coincides
with a training condition of the main dataset is left out of the shipped
list), and the tail dataset, 12 conditions. Both were generated with 1000
samples per run:

```sh
python scripts/data/generate_dataset.py --conditions configs/conditions/validation.json \
    --out data/validation --split-name ns3_validation --samples 1000
python scripts/data/generate_dataset.py --conditions configs/conditions/tail.json \
    --out data/tail --split-name ns3_tail_eval --samples 1000 --replications 128
```

Load steps (six step conditions, 64 replications each):

```sh
python scripts/data/generate_load_steps.py --out runs/load_steps --replications 64
```

## 3. Training

The four main models, five seeds each, on the recipe in
`configs/ns3_h20.yaml`:

```sh
python scripts/train/train.py --config configs/ns3_h20.yaml --data data/main --out runs/h20/qtro     --model qtro     --variant 4x8    --seeds 0,1,2,3,4
python scripts/train/train.py --config configs/ns3_h20.yaml --data data/main --out runs/h20/cnto     --model cnto     --variant rank27 --seeds 0,1,2,3,4
python scripts/train/train.py --config configs/ns3_h20.yaml --data data/main --out runs/h20/seqpred  --model seqpred  --variant rank27 --seeds 0,1,2,3,4
python scripts/train/train.py --config configs/ns3_h20.yaml --data data/main --out runs/h20/deeponet --model deeponet --variant rank19 --seeds 0,1,2,3,4
```

Rank sweep of the two low-rank baselines (`rank15`, `rank25`, `rank35`,
`rank45`; `rank27` is the main run):

```sh
for m in cnto seqpred; do for r in 15 25 35 45; do
  python scripts/train/train.py --config configs/ns3_h20.yaml --data data/main \
      --out runs/h20_ranks/${m}_rank${r} --model $m --variant rank$r --seeds 0,1,2,3,4
done; done
```

Depth screen (the four variants on the L x d_E = 32 diagonal), the shallow
pair, and the reset ablation:

```sh
python scripts/train/screen_depth.py --data data/main --out runs/depth_screen
python scripts/train/train.py --config configs/ns3_h20.yaml --data data/main --out runs/h20_depth/qtro_4x4 --model qtro --variant 4x4 --seeds 0,1,2,3,4
python scripts/train/train.py --config configs/ns3_h20.yaml --data data/main --out runs/h20_ablation/no_reset   --model qtro --variant 4x8 --seeds 0,1,2,3,4 --set model.reset_max_probability=1e-12
python scripts/train/train.py --config configs/ns3_h20.yaml --data data/main --out runs/h20_ablation/with_reset --model qtro --variant 4x8 --seeds 0,1,2,3,4 --set model.reset_max_probability=1.0
```

Coverage curve (27, 54 and 108 training conditions; the full-data point is
the main run):

```sh
python scripts/train/condition_curve.py --config configs/ns3_h20.yaml --data data/main --out runs/h20_condition_curve --seeds 0,1,2
```

Qubit-native arms (control count, ansätze, further seeds):

```sh
DATA=data/main scripts/qubit/run_control_count.sh
DATA=data/main scripts/qubit/run_ansatz.sh "hea:32 dense:8" 0
DATA=data/main scripts/qubit/run_seeds.sh
```

## 4. Predictive accuracy

Held-out accuracy (table and figure), with the split-half floor:

```sh
python scripts/evaluate/accuracy.py --panel data/main --split ns3_heldout H20 --out results/accuracy_heldout.json
```

Rank sweep, scored the same way:

```sh
python scripts/evaluate/accuracy.py --panel data/main --split ns3_heldout --control qtro \
    --models qtro=runs/h20/qtro,cnto15=runs/h20_ranks/cnto_rank15,cnto25=runs/h20_ranks/cnto_rank25,cnto27=runs/h20/cnto,cnto35=runs/h20_ranks/cnto_rank35,cnto45=runs/h20_ranks/cnto_rank45,sp15=runs/h20_ranks/seqpred_rank15,sp25=runs/h20_ranks/seqpred_rank25,sp27=runs/h20/seqpred,sp35=runs/h20_ranks/seqpred_rank35,sp45=runs/h20_ranks/seqpred_rank45 \
    --out results/accuracy_ranks.json
```

Domain baselines, extrapolation, shifts, envelope, communication observables,
the dataset's own statistics and its Markov gap:

```sh
python scripts/evaluate/domain_baselines.py --panel data/main --split ns3_heldout H20 --out results/domain_baselines.json
python scripts/evaluate/accuracy.py --panel data/extrapolation --split ns3_extrap H20 --out results/accuracy_extrapolation.json
python scripts/evaluate/extrapolation_severity.py --panel data/extrapolation --split ns3_extrap H20 --out results/extrapolation_severity.json
python scripts/evaluate/accuracy.py --panel data/shift_burst  --split ns3_burst  H20 --out results/accuracy_burst.json
python scripts/evaluate/accuracy.py --panel data/shift_tandem --split ns3_tandem H20 --out results/accuracy_tandem.json
python scripts/evaluate/envelope.py --panel data/validation --split ns3_validation H20 --out results/envelope.json
python scripts/evaluate/tail_observables.py --panel data/validation --split ns3_validation --aggregation uniform H20 --out results/tail_observables_validation.json
python scripts/evaluate/tail_observables.py --panel data/shift_burst --split ns3_burst H20 --out results/tail_observables_burst.json
python scripts/data/dataset_statistics.py --panel data/main --split ns3_heldout --out results/dataset_statistics.json
python scripts/evaluate/markov_gap.py --panel data/main --split ns3_heldout --out results/markov_gap.json
```

## 5. Structure, depth and cost

CPTP verification of every main checkpoint, with the empirical lift from the
training half:

```sh
python scripts/evaluate/verify_cptp.py --panel data/validation --panel-split ns3_validation --data data/main \
    $(for f in runs/h20/*/checkpoints/*.pt; do printf -- "--checkpoint %s " "$f"; done) \
    --out results/cptp/results.json
```

Validity under recursion, one report per seed, then the aggregate:

```sh
for s in 0 1 2 3 4; do
  python scripts/evaluate/long_horizon.py --panel data/main --split ns3_heldout H20 --seed $s --out results/long_horizon_seed$s.json
done
python scripts/evaluate/aggregate_long_horizon.py --inputs results/long_horizon_seed*.json --out results/long_horizon.json
```

Depth screen summary, the shallow pair, the reset ablation and the row-sum
bound:

```sh
python scripts/train/screen_depth.py --data data/main --out runs/depth_screen --summarise-only
python scripts/evaluate/accuracy.py --panel data/main --split ns3_heldout --control qtro4 \
    --models qtro4=runs/h20_depth/qtro_4x4,qtro=runs/h20/qtro,cnto=runs/h20/cnto,seqpred=runs/h20/seqpred --out results/accuracy_depth.json
python scripts/evaluate/accuracy.py --panel data/main --split ns3_heldout --horizons 1,5,20 \
    --models qtro=runs/h20_ablation/with_reset,no_reset=runs/h20_ablation/no_reset --out results/accuracy_reset.json
python scripts/evaluate/reset_rowsum.py --run runs/h20/qtro --panel data/main --out results/reset_rowsum
```

Control count of the uniformly controlled ladder, the two ansätze and the
depth pairs, paired against QTRO:

```sh
python scripts/evaluate/accuracy.py --panel data/main --split ns3_heldout --control qtro \
    --models qtro=runs/h20/qtro,k0=runs/ucr_k0,k4=runs/ucr_k4,k8=runs/ucr_k8,k10=runs/ucr_k10,kall=runs/ucr_kall,hea32=runs/qubit_hea_L32,dense8=runs/qubit_dense_L8,k4_L16=runs/ucr_k4_L16,k8_L8=runs/ucr_k8_L8 \
    --out results/accuracy_control_count.json
```

Cost of a decision, learned against the domain solvers and against ns-3:

```sh
python scripts/evaluate/cost.py --panel data/main --split ns3_heldout H20 --skip-ns3 --out results/cost.json
python scripts/evaluate/cost.py --panel data/main --split ns3_heldout H20 --out results/cost_with_ns3.json   # times real ns-3 runs
```

Coverage curve:

```sh
python scripts/evaluate/condition_curve.py --panel data/main --split ns3_heldout --root runs/h20_condition_curve \
    --full qtro=runs/h20/qtro,cnto=runs/h20/cnto,seqpred=runs/h20/seqpred,deeponet=runs/h20/deeponet --out results/condition_curve.json
```

## 6. Closed loop and load steps

The cycling load (every default of `closed_loop.py` is the reported setting),
the best constant setting, and the single surge:

```sh
python scripts/control/closed_loop.py H20 --out results/closed_loop_cycling.json
python scripts/control/closed_loop.py --models qtro=runs/h20/qtro --init-min-th 16 --out results/closed_loop_fixed16.json
python scripts/control/closed_loop.py H20 --flows 4 --surge-flows 4 --surge-start-s 20 --surge-stop-s -1 \
    --init-p-max 0.1 --init-min-th 20 --out results/closed_loop_surge.json
```

Paired intervals and drops over the episodes:

```sh
python scripts/control/closed_loop_bootstrap.py --report results/closed_loop_cycling.json --out results/closed_loop_bootstrap.json
python scripts/control/closed_loop_drops.py --report results/closed_loop_cycling.json --out results/closed_loop_drops.json
```

Load steps and their phases:

```sh
python scripts/evaluate/load_steps.py --transient runs/load_steps H20 --out results/load_steps.json
python scripts/evaluate/load_step_phases.py --comparison results/load_steps.json --metric mean_queue_error --out results/load_step_phases.json
```

The state probes, at the three RED settings and both step sizes:

```sh
for base in 2 4; do for red in "0.05 16" "0.1 20" "0.2 30"; do set -- $red
  python scripts/control/probe_queue_state.py  --base-flows $base --p-max $1 --min-th $2 --out results/probe_state_b${base}_p${1}_t${2}.json
  python scripts/control/probe_delta_rollout.py --base-flows $base --p-max $1 --min-th $2 --out results/probe_delta_b${base}_p${1}_t${2}.json
done; done
```

## 7. Notes

- The published training runs stored the model labels `qkfp` and `softmax`
  for what this repository calls `qtro` and `cnto`. Their checkpoints load
  unchanged (the names are translated on load); when reading their closed-loop
  reports, pass `--control qkfp` to `closed_loop_bootstrap.py` and
  `closed_loop_drops.py`.
- The split halves of a dataset are assigned in the order the ns-3 runs
  finish, so a regenerated dataset is statistically equivalent to the original
  but not bit-identical.
- The seed-0 checkpoints are the ones the closed loop and the cost script use.
