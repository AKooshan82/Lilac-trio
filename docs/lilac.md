# LILAC

LILAC (Lifelong Latent Actor-Critic) is implemented as a first-class off-policy learner under
`learner/lilac`. It follows the information timeline from `Deep Reinforcement Learning amidst
Lifelong Non-Stationarity` (arXiv:2006.10701): a task latent changes between episodes, the policy
acts with a prior prediction for the current episode, and the posterior for that episode is inferred only
after the full trajectory has been collected.

## DP-MDP Timeline

For episode `i`:

1. The previous posterior-derived latent is carried from episode `i - 1`; `z^0 = 0`.
2. `SequentialLatentPrior` predicts `p_psi(z_i | z_{i-1}, h_{i-1})`.
3. The policy uses a fixed prior sample or mean for every step in episode `i`.
4. The complete trajectory is stored in `EpisodicReplayBuffer`.
5. `PosteriorTrajectoryEncoder` infers `q_phi(z_i | tau_i)` from state, action, reward, and next state.
6. The posterior mean or sample is carried into the LSTM prior for episode `i + 1`.

The current episode posterior is never used before acting in that episode.

## Modules

- `actor.py`: squashed Gaussian `pi(a | s, z)` with action-bound scaling and tanh log-prob correction.
- `critic.py`: original-SAC-style `Q(s, a, z)`, `V(s, z)`, and target value network.
- `encoder.py`: feedforward masked trajectory posterior encoder.
- `prior.py`: LSTM inter-episode latent prior.
- `decoder.py`: transition/reward decoder `p(s', r | s, a, z)`.
- `replay_buffer.py`: CPU episodic replay with transition, episode, and subsequence sampling.
- `agent.py`: optimizer ownership and gradient routing.
- `trainer.py`: one-process lifelong episode collection, replay updates, logging, checkpointing, resume.

## Tensor Shapes

- Transitions: `observations [B, obs_dim]`, `actions [B, action_dim]`, `rewards [B, 1]`.
- Episodes: `observations [B, T_max, obs_dim]`, `masks [B, T_max, 1]`.
- Subsequences: `observations [B, L, T_max, obs_dim]`, `episode_masks [B, L, 1]`.
- Latents and Gaussian parameters: `[B, latent_dim]` or `[B, L, latent_dim]`.
- Prior hidden/cell state: `[B, prior_lstm_hidden_dim]`.

## Losses

The decoder uses masked MSE for next-state and reward reconstruction. This is a fixed-variance
Gaussian regression assumption because the paper does not specify decoder likelihood variance.

The KL term is analytic diagonal Gaussian `KL(q_phi(z_i | tau_i) || p_psi(z_i | z_{i-1}, h_{i-1}))`.
SAC uses the paper-indicated value target formulation:

- `J_Q = MSE(Q(s, a, z), r + gamma * V_target(s', z))`
- `J_V = MSE(V(s, z), Q(s, a_pi, z) - alpha log pi(a_pi | s, z))`
- `J_pi = mean(alpha log pi(a_pi | s, z) - Q(s, a_pi, z))`

## Gradient Flow

| Step | Updated parameters | Latent treatment |
| --- | --- | --- |
| Critic | critic only | posterior sample detached |
| Value | value only | posterior sample detached |
| Actor | actor only | posterior sample detached |
| Entropy | `log_alpha` only | actor log-prob detached |
| Encoder | encoder only | reconstruction + KL + critic gradients |
| Decoder | decoder only | posterior sample detached |
| Prior | prior only | posterior parameters and recurrent inputs detached |

The actor loss intentionally does not update the encoder. The encoder receives critic gradients through a
separate critic-loss pass with critic parameters frozen. The code does not use `retain_graph=True`.

## Replay Organization

`EpisodicReplayBuffer` stores complete CPU episodes with observation, action, reward, next observation,
terminated, truncated, episode ID, sequence ID, episode position, timestep, and valid masks. It samples:

- individual transitions with their source padded episodes;
- complete padded episodes;
- contiguous subsequences from the same sequence.

It rejects cross-sequence adjacency and never treats padding as data.

## Sequence And Parallel Semantics

LILAC currently supports `--num-processes 1`. This avoids hidden-state leakage while establishing the
single-sequence implementation. The trainer fails early if more processes are requested. Evaluation resets
the prior hidden state and `z^0` at each independent test sequence.

## Configuration

Important arguments:

- `--latent-dim`
- `--posterior-hidden-dims`
- `--prior-lstm-hidden-dim`
- `--decoder-hidden-dims`
- `--actor-hidden-dims`
- `--critic-hidden-dims`
- `--value-hidden-dims`
- `--replay-capacity`
- `--transition-batch-size`
- `--episode-batch-size`
- `--subsequence-length`
- `--warmup-episodes`
- `--warmup-transitions`
- `--updates-per-episode`
- `--actor-lr`, `--critic-lr`, `--value-lr`
- `--encoder-lr`, `--decoder-lr`, `--prior-lr`, `--entropy-lr`
- `--gamma`, `--polyak-tau`, `--entropy-coef`, `--automatic-entropy-tuning`
- `--kl-coef`, `--transition-reconstruction-coef`, `--reward-reconstruction-coef`
- `--execution-prior sample|mean`
- `--prior-recurrent-input posterior_mean|posterior_sample`
- `--checkpoint-replay`
- `--resume-checkpoint`

Default provenance:

- LILAC paper: episode-level latent timeline, fixed execution latent, posterior-after-episode rule,
  decoder/KL/critic gradient to encoder, LSTM prior, SAC value-target formulation.
- Standard SAC convention: `gamma=0.99`, `polyak_tau=0.005`, `entropy_coef=0.2`, learning rates `3e-4`.
- Repository-specific assumption: random-walk training task stream, one process, masked MSE decoder,
  truncated subsequence prior training, CPU replay.

## Commands

Training:

```bash
python train.py --env-type cheetah_vel --algo lilac --device cuda:0 --num-processes 1
```

Resume:

```bash
python train.py --env-type cheetah_vel --algo lilac --device cuda:0 --resume-checkpoint results/lilac/checkpoints/lilac_ep_000100.pt
```

Evaluation:

```bash
python cheetah_meta_test.py --algorithms lilac --lilac-checkpoint results/lilac/checkpoints/lilac_final.pt --output-folder results/lilac_eval
```

Checkpoint inspection:

```bash
python scripts/inspect_lilac_checkpoint.py --checkpoint results/lilac/checkpoints/lilac_final.pt
```

## Checkpoint Format

LILAC checkpoints are versioned and include actor, critic, value, target value, encoder, decoder, prior,
entropy parameter, all optimizers, counters, config, RNG states, replay metadata and optional replay contents,
and current sequence state.

## Deviations And Limitations

- Full training, benchmark reproduction, GPU validation, and MuJoCo rollouts were not executed locally.
- The prior is trained on contiguous replay subsequences, a truncated-history approximation.
- Parallel LILAC collection is not enabled yet.
- The implementation targets TRIO environments and should not be claimed to reproduce the LILAC paper without
  server-side experiments.
