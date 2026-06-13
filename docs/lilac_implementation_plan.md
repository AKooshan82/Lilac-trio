# LILAC Implementation Plan

This plan was written before implementation, after auditing the repository and the LILAC paper
(`Deep Reinforcement Learning amidst Lifelong Non-Stationarity`, arXiv:2006.10701).

## Repository Architecture Map

- `train.py`: top-level algorithm/environment selector. It currently dispatches `rl2`, `bayes`, and `ts`
  to PPO-oriented learners and saves model objects under timestamped `result/...` folders.
- `configs/`: per-environment/per-algorithm argparse modules. Existing modules expose PPO and inference
  parameters directly as CLI flags.
- `learner/`: algorithm implementations. Existing learners (`recurrent.py`, `bayes.py`,
  `posterior_ts_opt.py`) all wrap `ppo_a2c.PPO` and `RolloutStorage`.
- `inference/`: TRIO inference networks used by Bayes/TS baselines. These infer task posteriors from
  sampled context but are not LILAC's episode-level posterior/prior model.
- `ppo_a2c/`: on-policy rollout storage, policy/value networks, vectorized env helpers, and PPO optimizer.
  These abstractions should not own LILAC updates.
- `task/`: task generators that produce environment kwargs from Gaussian priors or independent task pairs.
- `envs/`: Gym environment registrations and custom tasks (`cheetahvel-v2`, `antgoal-v0`, `golf-v0`,
  `golfsignals-v0`).
- `*_meta_test.py`: environment-specific evaluation scripts with hard-coded meta-test task sequences.
- `utilities/`: folder management, observation augmentation, plotting, and test argument parsing.
- `requirements.txt`: old dependency set (`torch==1.4.0`, `gym==0.15.4`) that constrains API choices.

## Current Training Data Flow

1. `train.py` parses `--env-type` and `--algo`.
2. It selects a config module and environment/task dimensions.
3. Existing learners call `task_generator.sample_pair_tasks(num_processes)` to get independent task kwargs.
4. `ppo_a2c.envs.get_vec_envs_multi_task` creates or retargets vectorized Gym envs.
5. A fixed number of steps is collected into `RolloutStorage`.
6. PPO updates policy/value parameters from that on-policy rollout.
7. Bayes/TS separately train the TRIO inference network from sampled contexts.
8. Models are saved as raw PyTorch objects (`agent_ac`, `agent_vi`, `rl2_actor_critic`).

The current training path does not preserve a lifelong replay stream of complete adjacent episodes.

## Current Evaluation Data Flow

1. `cheetah_meta_test.py`, `ant_goal_meta_test.py`, and MiniGolf scripts define deterministic sequences.
2. They load baseline folders containing one or more saved PyTorch objects.
3. They instantiate the matching learner and call its `meta_test` or `meta_test_sequences`.
4. True task values are used to set environment latents and for CSV/tracking plots.
5. Outputs are pickled and converted to CSV reward/tracking summaries.

## PPO/LILAC Mismatch

TRIO's current abstractions are on-policy and step-window based. LILAC is episodic, off-policy, and needs:

- complete trajectory storage with sequence IDs and episode positions;
- posterior inference from completed episodes;
- a recurrent inter-episode prior state that is independent per lifelong sequence;
- replay batches that keep episode and sequence boundaries intact;
- SAC-style actor, critic, value, entropy, target-network updates;
- decoder and KL losses with explicit optimizer ownership.

Therefore LILAC should use shared environment/task setup where useful, but not `RolloutStorage`, `PPO`, or the
existing policy classes as optimization primitives.

## Paper-To-Code Mapping

| Paper concept | Code target |
| --- | --- |
| DP-MDP sequence of tasks | `LifelongTaskSampler` in `learner/lilac/trainer.py` |
| `q_phi(z_i | tau_i)` | `PosteriorTrajectoryEncoder` |
| `p_psi(z_i | z_{i-1}, h_{i-1})` | `SequentialLatentPrior` |
| `z^0 = 0` | `SequentialLatentPrior.initial_latent` |
| Fixed execution latent per episode | `LILACTrainer.collect_episode` |
| Decoder `p(s', r | s, a, z)` | `TransitionRewardDecoder` |
| SAC policy `pi(a | s, z)` | `SquashedGaussianActor` |
| SAC critic `Q(s, a, z)` | `LatentQNetwork` |
| SAC target value `V(s', z)` | `LatentValueNetwork` and target copy |
| Episodic replay `D[i] <- tau_i` | `EpisodicReplayBuffer` |
| Algorithm 1 optimizer boundaries | `LILACAgent.update` |

## Files To Add

- `configs/lilac_arguments.py`
- `learner/lilac/__init__.py`
- `learner/lilac/agent.py`
- `learner/lilac/actor.py`
- `learner/lilac/checkpoint.py`
- `learner/lilac/critic.py`
- `learner/lilac/decoder.py`
- `learner/lilac/distributions.py`
- `learner/lilac/encoder.py`
- `learner/lilac/losses.py`
- `learner/lilac/prior.py`
- `learner/lilac/replay_buffer.py`
- `learner/lilac/trainer.py`
- `learner/lilac/utils.py`
- `scripts/check_lilac_environment.py`
- `scripts/inspect_lilac_checkpoint.py`
- `tests/lilac/test_*.py`
- `docs/lilac.md`
- `docs/lilac_server_validation.md`

## Existing Files To Modify

- `train.py`: add `--algo lilac` dispatch and construct the LILAC trainer.
- `README.md`: minimal LILAC usage and docs links.
- `utilities/test_arguments.py`: add optional LILAC evaluation arguments.
- `cheetah_meta_test.py`, `ant_goal_meta_test.py`, `golf_meta_test.py`: allow explicit algorithm selection
  and LILAC-only evaluation without requiring baseline checkpoints.

## Tensor Shapes

- Transition observations: `[B, obs_dim]`
- Transition actions: `[B, action_dim]`
- Transition rewards: `[B, 1]`
- Transition terminals/truncations/masks: `[B, 1]`
- Episode observations: `[B, T_max, obs_dim]`
- Episode actions: `[B, T_max, action_dim]`
- Episode rewards: `[B, T_max, 1]`
- Episode next observations: `[B, T_max, obs_dim]`
- Episode masks: `[B, T_max, 1]`
- Episode latent distribution parameters: `[B, latent_dim]`
- Prior LSTM hidden/cell state: `[B, prior_hidden_dim]`
- Subsequence episodes: `[B, L, T_max, ...]`

## Gradient-Flow Decisions

- Critic optimizer updates only critic parameters; posterior samples are detached for the critic step.
- Value optimizer updates only value parameters; posterior samples are detached.
- Actor optimizer updates only actor parameters; posterior samples are detached.
- Entropy optimizer updates only `log_alpha`.
- Encoder optimizer receives gradients from reconstruction, KL, and a separate critic loss pass with critic
  parameters frozen and posterior samples reparameterized.
- Decoder optimizer receives reconstruction gradients with posterior samples detached.
- Prior optimizer receives KL gradients with posterior parameters detached.
- The actor loss does not update the encoder in this implementation.
- No optimizer reuses stale graphs, and `retain_graph=True` is not used as a workaround.

## Replay-Buffer Design

The replay buffer stores CPU numpy arrays per complete episode. Each episode contains observation, action,
reward, next observation, terminated, truncated, valid mask, episode ID, sequence ID, episode position, and
timestep. Sampling returns only the requested batch tensors moved to the selected device by the agent.

Sampling modes:

- transitions with their source episodes for actor/critic updates;
- complete padded trajectories for posterior/decoder losses;
- contiguous episode subsequences from one sequence for prior KL training.

Padding is masked and never counted as data. Adjacent samples are validated to have matching sequence IDs and
consecutive episode positions.

## Sequence-Reset Semantics

- Training initially supports `num_processes == 1` for LILAC and fails clearly otherwise.
- The prior hidden state and previous posterior latent reset at the start of each lifelong sequence.
- A new task is sampled only between episodes, never mid-episode.
- Execution uses the LSTM prior prediction based on previous completed episodes; it never uses the current
  episode posterior before collecting the episode.
- Evaluation resets prior state for every independent evaluation sequence.

## Compatibility Risks

- The repository targets old PyTorch/Gym versions; implementation avoids newer APIs where practical.
- Existing custom Gym registrations may require `import envs` before `gym.make`.
- MuJoCo dependencies are not available locally, so environment runtime checks are server-only.
- Existing evaluation scripts assume all three baselines are present; LILAC integration must make algorithm
  selection explicit.
- Original LILAC experiments use environments that differ from TRIO's exact task generators, so defaults are
  starting points rather than reproduction claims.

## Paper Ambiguities

- The paper states a feedforward inference network but not the exact aggregation architecture over variable
  length trajectories. This implementation uses masked per-transition embeddings followed by masked mean/sum
  features in an MLP, preserving feedforward inference without padding leakage.
- Decoder likelihood variance is not specified. This implementation uses masked mean-squared error, equivalent
  to a fixed-variance Gaussian regression assumption.
- Algorithm 1 lists critic, actor, inference, decoder, and prior updates, while the SAC equations include a
  target value network. This implementation follows the original SAC-style value/target-value formulation
  indicated by the paper.
- Replay-time LSTM training over long histories is expensive. The trainer samples contiguous subsequences and
  unrolls the prior over those subsequences, a truncated-history approximation documented as a limitation.

## Implementation Assumptions

- LILAC training creates a bounded random-walk task stream using the task generator's latent bounds when no
  explicit sequence is provided.
- LILAC supports continuous `Box` observation and action spaces only.
- True task parameters are used only to set environments and for metadata, never as actor/critic/encoder inputs.
- `golf_signals` is treated as compatible only when its observation signals are accepted as environment
  observations; these signals are not separately passed as privileged labels.
- Replay capacity is counted in episodes, with total transition count tracked for diagnostics.

## Server Validation Plan

Local validation is limited to static checks:

- `git status`
- `git diff --check`
- `python -m compileall <files-created-or-modified-for-lilac>`

Server validation should run:

- environment inspection for `cheetah_vel` and `ant_goal`;
- LILAC import and compile checks;
- `pytest tests/lilac -q -m "not mujoco"`;
- optional MuJoCo-marked tests;
- a tiny smoke training run with one process and small networks;
- checkpoint inspection and resume;
- LILAC-only evaluation;
- parser checks for existing baselines.

No local result should be presented as a reproduction or training-success claim.
