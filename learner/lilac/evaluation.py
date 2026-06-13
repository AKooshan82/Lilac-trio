import csv
import os
import pickle
from types import SimpleNamespace
from typing import Dict, List, Optional

import gym
import numpy as np
import torch

import envs  # noqa: F401
from learner.lilac.checkpoint import load_checkpoint
from learner.lilac.trainer import (LILACTrainer, _reset_env, _sanitize_env_kwargs,
                                   _unpack_step, get_lilac_env_spec)


def discover_lilac_checkpoints(checkpoint: Optional[str] = None,
                               checkpoint_folder: Optional[str] = None) -> List[str]:
    if checkpoint is not None:
        return [checkpoint]
    if checkpoint_folder is None:
        raise ValueError("Provide --lilac-checkpoint or --lilac-checkpoint-folder for LILAC evaluation")
    paths = []
    for root, _, files in os.walk(checkpoint_folder):
        for name in files:
            if name.endswith(".pt") and ("lilac" in name):
                paths.append(os.path.join(root, name))
    paths = sorted(paths)
    if not paths:
        raise ValueError("No LILAC .pt checkpoints found under {}".format(checkpoint_folder))
    return paths


def evaluate_lilac_checkpoint(env_type: str,
                              checkpoint_path: str,
                              prior_sequences,
                              task_generator,
                              seed: int,
                              device: torch.device,
                              output_folder: str,
                              task_len: int = 1,
                              deterministic_actions: bool = True,
                              deterministic_prior: bool = True,
                              golf_num_signals: Optional[int] = None,
                              max_episode_steps: Optional[int] = None) -> List[List[float]]:
    """Evaluate a LILAC checkpoint on provided task-prior sequences without optimizer updates."""
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    config = dict(checkpoint.get("config", {}))
    config["device"] = str(device)
    config["resume_checkpoint"] = checkpoint_path
    config["num_processes"] = 1
    config["output_folder"] = output_folder
    config["max_episode_steps"] = max_episode_steps if max_episode_steps is not None else config.get("max_episode_steps")
    args = SimpleNamespace(**config)
    env_spec = get_lilac_env_spec(env_type, golf_num_signals=golf_num_signals)
    trainer = LILACTrainer(args=args, env_spec=env_spec, output_folder=output_folder)
    agent = trainer.agent
    agent.eval()

    all_sequence_rewards = []
    rows = []
    for sequence_index, sequence in enumerate(prior_sequences):
        prior_state = agent.prior.initial_state(1, device)
        prev_latent = agent.prior.initial_latent(1, device)
        sequence_rewards = []
        for episode_position, prior in enumerate(sequence):
            kwargs = _sanitize_env_kwargs(task_generator.sample_task_from_prior(prior))
            prior_dist, next_state = agent.prior.forward_step(prev_latent, prior_state)
            execution_latent = prior_dist.mean if deterministic_prior else prior_dist.sample()
            task_returns = []
            posterior_for_carry = None
            for repeat in range(task_len):
                episode = _rollout_episode(env_spec, kwargs, agent, execution_latent.detach(),
                                           seed + sequence_index * 1000 + episode_position * 17 + repeat,
                                           deterministic_actions, max_episode_steps)
                task_returns.append(float(np.sum(episode["rewards"])))
                posterior_for_carry = agent.posterior_from_numpy_episode(episode["observations"],
                                                                         episode["actions"],
                                                                         episode["rewards"],
                                                                         episode["next_observations"])
            mean_return = float(np.mean(task_returns))
            sequence_rewards.append(mean_return)
            if posterior_for_carry is None:
                raise RuntimeError("LILAC evaluation produced no episode for sequence {}, position {}".format(
                    sequence_index, episode_position))
            prev_latent = posterior_for_carry.mean.detach()
            prior_state = (next_state[0].detach(), next_state[1].detach())
            rows.append({
                "algorithm": "lilac",
                "sequence_index": sequence_index,
                "episode_position": episode_position,
                "mean_return": mean_return,
                "prior_mean_abs": float(prior_dist.mean.detach().abs().mean().cpu().item()),
                "prior_std_mean": float(prior_dist.std.detach().mean().cpu().item()),
                "posterior_mean_abs": float(posterior_for_carry.mean.detach().abs().mean().cpu().item()),
                "posterior_std_mean": float(posterior_for_carry.std.detach().mean().cpu().item()),
            })
        all_sequence_rewards.append(sequence_rewards)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(os.path.join(output_folder, "lilac_eval.pkl"), "wb") as handle:
        pickle.dump(all_sequence_rewards, handle)
    with open(os.path.join(output_folder, "lilac_eval.csv"), "w", newline="") as handle:
        fieldnames = ["algorithm", "sequence_index", "episode_position", "mean_return",
                      "prior_mean_abs", "prior_std_mean", "posterior_mean_abs", "posterior_std_mean"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return all_sequence_rewards


def _rollout_episode(env_spec,
                     kwargs: Dict[str, object],
                     agent,
                     execution_latent: torch.Tensor,
                     seed: int,
                     deterministic_actions: bool,
                     max_episode_steps: Optional[int]) -> Dict[str, np.ndarray]:
    env = gym.make(env_spec.env_name, **kwargs)
    env.seed(seed)
    obs = _reset_env(env)
    observations = []
    actions = []
    rewards = []
    next_observations = []
    terminated = []
    truncated = []
    limit = max_episode_steps
    if limit is None and env.spec is not None and env.spec.max_episode_steps is not None:
        limit = int(env.spec.max_episode_steps)
    if limit is None:
        limit = 200
    for step_idx in range(limit):
        action = agent.act(obs, execution_latent, deterministic=deterministic_actions)
        action = np.clip(action, env_spec.action_low, env_spec.action_high).astype(np.float32)
        next_obs, reward, done, info = _unpack_step(env.step(action))
        reached_limit = step_idx + 1 >= limit
        was_truncated = bool(info.get("TimeLimit.truncated", False) or info.get("bad_transition", False) or reached_limit)
        was_terminated = bool(done and not was_truncated)
        observations.append(np.asarray(obs, dtype=np.float32).reshape(env_spec.state_dim))
        actions.append(np.asarray(action, dtype=np.float32).reshape(env_spec.action_dim))
        rewards.append([float(reward)])
        next_observations.append(np.asarray(next_obs, dtype=np.float32).reshape(env_spec.state_dim))
        terminated.append([1.0 if was_terminated else 0.0])
        truncated.append([1.0 if was_truncated else 0.0])
        obs = next_obs
        if done or reached_limit:
            break
    env.close()
    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "next_observations": np.asarray(next_observations, dtype=np.float32),
        "terminated": np.asarray(terminated, dtype=np.float32),
        "truncated": np.asarray(truncated, dtype=np.float32),
    }
