# LILAC Server Validation

Run these commands on the compute server. They are direct commands, not shell launchers.

## Stage 1: Inspect Repository Changes

```bash
git status
git diff --stat
git diff --check
```

Expected new files include `learner/lilac/`, `configs/lilac_arguments.py`, `scripts/check_lilac_environment.py`,
`scripts/inspect_lilac_checkpoint.py`, `tests/lilac/`, `docs/lilac.md`, and this guide. Expected modified
files include `train.py`, README, evaluation scripts, and `utilities/test_arguments.py`.

## Stage 2: Inspect Server Environment

```bash
python scripts/check_lilac_environment.py --env-type cheetah_vel --device cuda:0 --seed 1
```

```bash
python scripts/check_lilac_environment.py --env-type ant_goal --device cuda:0 --seed 1
```

Missing CUDA should fail only when a CUDA device is requested. Missing MuJoCo should produce an actionable
environment creation error.

## Stage 3: Compile And Import

```bash
python -m compileall learner/lilac configs/lilac_arguments.py scripts/check_lilac_environment.py scripts/inspect_lilac_checkpoint.py
```

```bash
python -c "from learner.lilac.agent import LILACAgent; print('LILAC import successful')"
```

## Stage 4: Unit Tests

```bash
python -m pytest tests/lilac -q -m "not mujoco"
```

```bash
python -m pytest tests/lilac -q -m mujoco
```

A true failure is a failed assertion or uncaught exception. An optional skipped MuJoCo test is acceptable only
when it is marked as skipped. A missing dependency means the server environment is incomplete. A numerical
failure usually appears as a non-finite loss or parameter assertion. Pytest logs are printed to the terminal;
redirect them manually if you want a file.

## Stage 5: Minimal Software Smoke Test

```bash
python train.py \
    --env-type cheetah_vel \
    --algo lilac \
    --seed 1 \
    --device cuda:0 \
    --num-processes 1 \
    --num-episodes 4 \
    --latent-dim 2 \
    --actor-hidden-dims 32,32 \
    --critic-hidden-dims 32,32 \
    --value-hidden-dims 32,32 \
    --posterior-hidden-dims 32,32 \
    --decoder-hidden-dims 32,32 \
    --prior-lstm-hidden-dim 32 \
    --transition-embedding-dim 32 \
    --replay-capacity 20 \
    --batch-size 4 \
    --episode-batch-size 2 \
    --subsequence-length 2 \
    --warmup-episodes 2 \
    --warmup-transitions 1 \
    --updates-per-episode 1 \
    --checkpoint-interval 2 \
    --log-interval 1 \
    --output-folder results/lilac_smoke
```

This only checks wiring: environment creation, prior sampling, episode collection, replay insertion, posterior
inference, decoder/KL/critic/actor/value/target updates, and checkpoint saving. It is not meaningful learning.

## Stage 6: Checkpoint Inspection

```bash
python scripts/inspect_lilac_checkpoint.py --checkpoint results/lilac_smoke/<TIMESTAMP>/checkpoints/lilac_final.pt
```

Expected output lists checkpoint version, environment, counters, model components, optimizer components, replay
metadata, RNG state presence, and parameter counts.

## Stage 7: Resume Test

Initial short run:

```bash
python train.py \
    --env-type cheetah_vel \
    --algo lilac \
    --seed 2 \
    --device cuda:0 \
    --num-processes 1 \
    --num-episodes 3 \
    --actor-hidden-dims 32,32 \
    --critic-hidden-dims 32,32 \
    --value-hidden-dims 32,32 \
    --posterior-hidden-dims 32,32 \
    --decoder-hidden-dims 32,32 \
    --prior-lstm-hidden-dim 32 \
    --transition-embedding-dim 32 \
    --replay-capacity 20 \
    --batch-size 4 \
    --episode-batch-size 2 \
    --subsequence-length 2 \
    --warmup-episodes 2 \
    --warmup-transitions 1 \
    --updates-per-episode 1 \
    --checkpoint-interval 1 \
    --output-folder results/lilac_resume_initial
```

Resumed run:

```bash
python train.py \
    --env-type cheetah_vel \
    --algo lilac \
    --seed 2 \
    --device cuda:0 \
    --num-processes 1 \
    --num-episodes 5 \
    --resume-checkpoint results/lilac_resume_initial/<TIMESTAMP>/checkpoints/lilac_ep_000003.pt \
    --output-folder results/lilac_resume_continued
```

Verify counters continue, optimizers restore, target value restores, replay behavior matches whether
`--checkpoint-replay` was enabled, and the output folder is a new timestamped run.

## Stage 8: Evaluation Test

```bash
python cheetah_meta_test.py \
    --algorithms lilac \
    --lilac-checkpoint results/lilac_smoke/<TIMESTAMP>/checkpoints/lilac_final.pt \
    --output-folder results/lilac_eval_cheetah \
    --task-len 1 \
    --num-test-processes 1
```

```bash
python ant_goal_meta_test.py \
    --algorithms lilac \
    --lilac-checkpoint results/lilac_ant/<TIMESTAMP>/checkpoints/lilac_final.pt \
    --output-folder results/lilac_eval_ant \
    --task-len 1 \
    --num-test-processes 1
```

Expected LILAC outputs are `lilac_eval.pkl` and `lilac_eval.csv` in the output folder. Existing baseline pickle
and CSV formats are still produced when those baselines are selected.

## Stage 9: Existing Baseline Parser Checks

```bash
python train.py --env-type cheetah_vel --algo rl2 --help
```

```bash
python train.py --env-type cheetah_vel --algo bayes --help
```

```bash
python train.py --env-type cheetah_vel --algo ts --help
```

These commands should parse without requiring LILAC.

## Stage 10: Short Debugging Run

```bash
python train.py \
    --env-type cheetah_vel \
    --algo lilac \
    --seed 3 \
    --device cuda:0 \
    --num-processes 1 \
    --num-episodes 20 \
    --latent-dim 4 \
    --actor-hidden-dims 64,64 \
    --critic-hidden-dims 64,64 \
    --value-hidden-dims 64,64 \
    --posterior-hidden-dims 64,64 \
    --decoder-hidden-dims 64,64 \
    --prior-lstm-hidden-dim 64 \
    --transition-embedding-dim 64 \
    --replay-capacity 200 \
    --batch-size 32 \
    --episode-batch-size 8 \
    --subsequence-length 4 \
    --warmup-episodes 5 \
    --warmup-transitions 100 \
    --updates-per-episode 2 \
    --checkpoint-interval 5 \
    --log-interval 1 \
    --output-folder results/lilac_debug
```

Inspect finite critic/value/actor losses, finite entropy temperature, finite KL, stable reconstruction losses,
growing replay size, valid prior/posterior standard deviations, bounded actions, target updates, and checkpoints.
Do not expect monotonic returns.

## Stage 11: Production Training

NOT EXECUTED — STARTING CONFIGURATION FOR SERVER EXPERIMENTS:

```bash
python train.py \
    --env-type cheetah_vel \
    --algo lilac \
    --seed 1 \
    --device cuda:0 \
    --num-processes 1 \
    --num-episodes 50000 \
    --latent-dim 8 \
    --actor-hidden-dims 256,256 \
    --critic-hidden-dims 256,256 \
    --value-hidden-dims 256,256 \
    --posterior-hidden-dims 256,256 \
    --decoder-hidden-dims 256,256 \
    --prior-lstm-hidden-dim 128 \
    --transition-embedding-dim 128 \
    --replay-capacity 10000 \
    --batch-size 256 \
    --episode-batch-size 16 \
    --subsequence-length 4 \
    --warmup-episodes 10 \
    --warmup-transitions 1000 \
    --updates-per-episode 1 \
    --checkpoint-interval 500 \
    --output-folder results/lilac_cheetah
```

NOT EXECUTED — STARTING CONFIGURATION FOR SERVER EXPERIMENTS:

```bash
python train.py \
    --env-type ant_goal \
    --algo lilac \
    --seed 1 \
    --device cuda:0 \
    --num-processes 1 \
    --num-episodes 50000 \
    --latent-dim 8 \
    --actor-hidden-dims 256,256 \
    --critic-hidden-dims 256,256 \
    --value-hidden-dims 256,256 \
    --posterior-hidden-dims 256,256 \
    --decoder-hidden-dims 256,256 \
    --prior-lstm-hidden-dim 128 \
    --transition-embedding-dim 128 \
    --replay-capacity 10000 \
    --batch-size 256 \
    --episode-batch-size 16 \
    --subsequence-length 4 \
    --warmup-episodes 10 \
    --warmup-transitions 1000 \
    --updates-per-episode 1 \
    --checkpoint-interval 500 \
    --output-folder results/lilac_ant
```

## Stage 12: Fair Comparison

Use identical task sequences, seeds, sequence lengths, test-process counts, task-transition noise, and evaluation
horizons when comparing LILAC and baselines. LILAC is off-policy with replay and the existing baselines are
primarily PPO-based, so update schedules and replay reuse remain unavoidable algorithmic differences.
