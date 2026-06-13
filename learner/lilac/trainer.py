import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import gym
import numpy as np
import torch

import envs  # noqa: F401  Ensures custom Gym environments are registered.
from learner.lilac.agent import LILACAgent
from learner.lilac.checkpoint import build_checkpoint, load_checkpoint, restore_agent, save_checkpoint
from learner.lilac.replay_buffer import EpisodicReplayBuffer
from learner.lilac.utils import parse_hidden_dims, set_global_seeds
from task.ant_goal_task_generator import AntGoalTaskGenerator
from task.cheetah_vel_task_generator import CheetahVelTaskGenerator
from task.mini_golf_task_generator import MiniGolfTaskGenerator
from task.mini_golf_with_signals_generator import MiniGolfSignalsTaskGenerator


@dataclass
class LILACEnvSpec:
    env_type: str
    env_name: str
    state_dim: int
    action_dim: int
    default_latent_dim: int
    action_low: np.ndarray
    action_high: np.ndarray
    task_generator: object
    result_folder: str


def get_lilac_env_spec(env_type: str, golf_num_signals: Optional[int] = None) -> LILACEnvSpec:
    if env_type == "cheetah_vel":
        prior_var_min = 0.01
        prior_var_max = 0.3
        action_dim = 6
        return LILACEnvSpec(env_type=env_type,
                            env_name="cheetahvel-v2",
                            state_dim=20,
                            action_dim=action_dim,
                            default_latent_dim=1,
                            action_low=-np.ones(action_dim, dtype=np.float32),
                            action_high=np.ones(action_dim, dtype=np.float32),
                            task_generator=CheetahVelTaskGenerator(prior_var_min, prior_var_max),
                            result_folder="result/cheetahvelv2/lilac/")
    if env_type == "ant_goal":
        prior_var_min = 0.1
        prior_var_max = 0.4
        action_dim = 8
        return LILACEnvSpec(env_type=env_type,
                            env_name="antgoal-v0",
                            state_dim=113,
                            action_dim=action_dim,
                            default_latent_dim=2,
                            action_low=-np.ones(action_dim, dtype=np.float32),
                            action_high=np.ones(action_dim, dtype=np.float32),
                            task_generator=AntGoalTaskGenerator(prior_var_min, prior_var_max),
                            result_folder="result/antgoal/lilac/")
    if env_type == "golf":
        prior_var_min = 0.001
        prior_var_max = 0.2
        return LILACEnvSpec(env_type=env_type,
                            env_name="golf-v0",
                            state_dim=1,
                            action_dim=1,
                            default_latent_dim=1,
                            action_low=np.asarray([1e-5], dtype=np.float32),
                            action_high=np.asarray([10.0], dtype=np.float32),
                            task_generator=MiniGolfTaskGenerator(prior_var_min, prior_var_max),
                            result_folder="result/minigolfv0/lilac/")
    if env_type == "golf_signals":
        if golf_num_signals is None:
            raise ValueError("--golf-num-signals is required for golf_signals")
        prior_var_min = 0.001
        prior_var_max = 0.2
        latent_dim = 1 + int(golf_num_signals)
        return LILACEnvSpec(env_type=env_type,
                            env_name="golfsignals-v0",
                            state_dim=latent_dim,
                            action_dim=1,
                            default_latent_dim=latent_dim,
                            action_low=np.asarray([1e-5], dtype=np.float32),
                            action_high=np.asarray([10.0], dtype=np.float32),
                            task_generator=MiniGolfSignalsTaskGenerator(golf_num_signals, prior_var_min, prior_var_max),
                            result_folder="result/golf_sig_{}/lilac/".format(golf_num_signals))
    raise ValueError("LILAC does not support env-type '{}'".format(env_type))


class LifelongTaskSampler(object):
    """Bounded random-walk task stream for LILAC training."""

    def __init__(self, task_generator: object, latent_dim: int, transition_std: float, seed: int):
        self.task_generator = task_generator
        self.latent_dim = latent_dim
        self.transition_std = transition_std
        self.rng = np.random.RandomState(seed)
        self.low, self.high = self._bounds(task_generator, latent_dim)
        self.current = self.rng.uniform(self.low, self.high).astype(np.float32)
        self.sequence_id = 0
        self.episode_position = 0

    def next_task(self) -> Tuple[Dict[str, object], np.ndarray, int, int]:
        if self.episode_position > 0:
            proposal = self.current + self.rng.normal(0.0, self.transition_std, size=self.latent_dim)
            self.current = np.clip(proposal, self.low, self.high).astype(np.float32)
        prior = torch.tensor([self.current, np.zeros(self.latent_dim, dtype=np.float32)], dtype=torch.float32)
        kwargs = _sanitize_env_kwargs(self.task_generator.sample_task_from_prior(prior))
        seq_id = self.sequence_id
        pos = self.episode_position
        self.episode_position += 1
        return kwargs, self.current.copy(), seq_id, pos

    def reset_sequence(self) -> None:
        self.sequence_id += 1
        self.episode_position = 0
        self.current = self.rng.uniform(self.low, self.high).astype(np.float32)

    def state_dict(self) -> Dict[str, object]:
        return {
            "current": self.current,
            "sequence_id": self.sequence_id,
            "episode_position": self.episode_position,
            "rng_state": self.rng.get_state(),
            "transition_std": self.transition_std,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.current = np.asarray(state["current"], dtype=np.float32)
        self.sequence_id = int(state["sequence_id"])
        self.episode_position = int(state["episode_position"])
        self.transition_std = float(state.get("transition_std", self.transition_std))
        self.rng.set_state(state["rng_state"])

    @staticmethod
    def _bounds(task_generator: object, latent_dim: int) -> Tuple[np.ndarray, np.ndarray]:
        if hasattr(task_generator, "min_vel"):
            return np.asarray([task_generator.min_vel], dtype=np.float32), np.asarray([task_generator.max_vel], dtype=np.float32)
        if hasattr(task_generator, "min") and hasattr(task_generator, "max"):
            return (np.full(latent_dim, task_generator.min, dtype=np.float32),
                    np.full(latent_dim, task_generator.max, dtype=np.float32))
        if hasattr(task_generator, "min_friction"):
            return (np.asarray([task_generator.min_friction], dtype=np.float32),
                    np.asarray([task_generator.max_friction], dtype=np.float32))
        if hasattr(task_generator, "min_value"):
            return (np.asarray(task_generator.min_value.numpy(), dtype=np.float32),
                    np.asarray(task_generator.max_value.numpy(), dtype=np.float32))
        return -np.ones(latent_dim, dtype=np.float32), np.ones(latent_dim, dtype=np.float32)


class LILACTrainer(object):
    def __init__(self, args, env_spec: LILACEnvSpec, output_folder: str):
        if args.num_processes != 1:
            raise ValueError("LILAC currently supports --num-processes 1 only; got {}".format(args.num_processes))
        self.args = args
        self.env_spec = env_spec
        self.output_folder = output_folder
        self.checkpoint_dir = os.path.join(output_folder, "checkpoints")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        set_global_seeds(args.seed, cuda=args.device.startswith("cuda"))
        self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device {} requested but torch.cuda.is_available() is false".format(args.device))

        self.latent_dim = int(args.latent_dim if args.latent_dim is not None else env_spec.default_latent_dim)
        self.agent = LILACAgent(obs_dim=env_spec.state_dim,
                                action_dim=env_spec.action_dim,
                                latent_dim=self.latent_dim,
                                action_low=env_spec.action_low,
                                action_high=env_spec.action_high,
                                device=self.device,
                                actor_hidden_dims=parse_hidden_dims(args.actor_hidden_dims, [256, 256]),
                                critic_hidden_dims=parse_hidden_dims(args.critic_hidden_dims, [256, 256]),
                                value_hidden_dims=parse_hidden_dims(args.value_hidden_dims, [256, 256]),
                                encoder_hidden_dims=parse_hidden_dims(args.posterior_hidden_dims, [256, 256]),
                                decoder_hidden_dims=parse_hidden_dims(args.decoder_hidden_dims, [256, 256]),
                                prior_hidden_dim=args.prior_lstm_hidden_dim,
                                transition_embedding_dim=args.transition_embedding_dim,
                                actor_lr=args.actor_lr,
                                critic_lr=args.critic_lr,
                                value_lr=args.value_lr,
                                encoder_lr=args.encoder_lr,
                                decoder_lr=args.decoder_lr,
                                prior_lr=args.prior_lr,
                                entropy_lr=args.entropy_lr,
                                gamma=args.gamma,
                                polyak_tau=args.polyak_tau,
                                entropy_coef=args.entropy_coef,
                                automatic_entropy_tuning=args.automatic_entropy_tuning,
                                target_entropy=args.target_entropy,
                                kl_coef=args.kl_coef,
                                transition_reconstruction_coef=args.transition_reconstruction_coef,
                                reward_reconstruction_coef=args.reward_reconstruction_coef,
                                critic_encoder_loss_coef=args.critic_encoder_loss_coef,
                                reward_scale=args.reward_scale,
                                max_grad_norm=args.max_grad_norm,
                                activation=args.activation,
                                init=args.network_init,
                                prior_use_posterior_sample=args.prior_recurrent_input == "posterior_sample",
                                deterministic_prior=args.execution_prior == "mean")
        self.replay = EpisodicReplayBuffer(args.replay_capacity,
                                           obs_dim=env_spec.state_dim,
                                           action_dim=env_spec.action_dim,
                                           seed=args.seed)
        self.task_sampler = LifelongTaskSampler(env_spec.task_generator,
                                                latent_dim=self.latent_dim,
                                                transition_std=args.task_transition_std,
                                                seed=args.seed + 17)
        self.counters = {"completed_episodes": 0, "environment_steps": 0, "gradient_updates": 0}
        self.prev_posterior_latent = self.agent.prior.initial_latent(1, self.device)
        self.prior_state = self.agent.prior.initial_state(1, self.device)
        self.env = None
        self.metrics_path = os.path.join(output_folder, "lilac_metrics.csv")
        self.log_path = os.path.join(output_folder, "lilac_log.txt")
        self._csv_fieldnames = None

        if args.resume_checkpoint:
            self._load_resume(args.resume_checkpoint)

    def train(self) -> Dict[str, object]:
        self._log_header()
        while self.counters["completed_episodes"] < self.args.num_episodes:
            metrics = self._train_episode()
            if self.counters["completed_episodes"] % self.args.log_interval == 0:
                self._write_metrics(metrics)
            if self.args.checkpoint_interval > 0 and self.counters["completed_episodes"] % self.args.checkpoint_interval == 0:
                self.save_checkpoint("lilac_ep_{:06d}.pt".format(self.counters["completed_episodes"]))
        self.save_checkpoint("lilac_final.pt")
        if self.env is not None:
            self.env.close()
        return {"output_folder": self.output_folder, "counters": dict(self.counters)}

    def _train_episode(self) -> Dict[str, float]:
        kwargs, task_value, sequence_id, episode_position = self.task_sampler.next_task()
        self._ensure_env(kwargs)
        prior_dist, next_prior_state = self.agent.prior.forward_step(self.prev_posterior_latent, self.prior_state)
        execution_latent = prior_dist.mean if self.args.execution_prior == "mean" else prior_dist.sample()
        execution_latent = execution_latent.detach()

        random_actions = (self.args.warmup_behavior == "random"
                          and (self.counters["completed_episodes"] < self.args.warmup_episodes
                               or self.counters["environment_steps"] < self.args.warmup_transitions))
        episode = self._collect_episode(execution_latent, random_actions)
        episode_id = self.replay.add_episode(sequence_id=sequence_id,
                                             episode_position=episode_position,
                                             **episode)
        self.counters["completed_episodes"] += 1
        self.counters["environment_steps"] += int(episode["observations"].shape[0])

        posterior = self.agent.posterior_from_numpy_episode(episode["observations"],
                                                            episode["actions"],
                                                            episode["rewards"],
                                                            episode["next_observations"])
        if self.args.prior_recurrent_input == "posterior_sample":
            carry_latent = posterior.sample()
        else:
            carry_latent = posterior.mean
        self.prev_posterior_latent = carry_latent.detach()
        self.prior_state = (next_prior_state[0].detach(), next_prior_state[1].detach())

        metrics = {
            "sequence_id": float(sequence_id),
            "episode_position": float(episode_position),
            "episode_id": float(episode_id),
            "episode_return": float(np.sum(episode["rewards"])),
            "episode_length": float(episode["observations"].shape[0]),
            "total_completed_episodes": float(self.counters["completed_episodes"]),
            "total_environment_steps": float(self.counters["environment_steps"]),
            "replay_episode_count": float(self.replay.num_episodes),
            "replay_transition_count": float(self.replay.num_transitions),
            "prior_mean_abs_collect": float(prior_dist.mean.detach().abs().mean().cpu().item()),
            "prior_std_mean_collect": float(prior_dist.std.detach().mean().cpu().item()),
            "posterior_mean_abs_collect": float(posterior.mean.detach().abs().mean().cpu().item()),
            "posterior_std_mean_collect": float(posterior.std.detach().mean().cpu().item()),
            "task_value_mean_metadata": float(np.asarray(task_value).mean()),
        }

        if self._can_update():
            for _ in range(self.args.updates_per_episode):
                try:
                    update_metrics = self.agent.update(self.replay,
                                                       transition_batch_size=self.args.transition_batch_size,
                                                       episode_batch_size=self.args.episode_batch_size,
                                                       subsequence_length=self.args.subsequence_length,
                                                       counters=self.counters)
                except (ValueError, RuntimeError, FloatingPointError) as exc:
                    raise RuntimeError("LILAC update failed at episode {}, env steps {}, updates {}: {}".format(
                        self.counters["completed_episodes"],
                        self.counters["environment_steps"],
                        self.counters["gradient_updates"],
                        exc))
                self.counters["gradient_updates"] += 1
                metrics.update(update_metrics)
        return metrics

    def _can_update(self) -> bool:
        if self.counters["completed_episodes"] < self.args.warmup_episodes:
            return False
        if self.counters["environment_steps"] < self.args.warmup_transitions:
            return False
        if not self.replay.can_sample_transitions(self.args.transition_batch_size):
            return False
        if not self.replay.can_sample_episodes(self.args.episode_batch_size):
            return False
        try:
            self.replay._contiguous_windows(self.args.subsequence_length)
        except ValueError:
            return False
        return len(self.replay._contiguous_windows(self.args.subsequence_length)) > 0

    def _collect_episode(self, execution_latent: torch.Tensor, random_actions: bool) -> Dict[str, np.ndarray]:
        obs = _reset_env(self.env)
        observations = []
        actions = []
        rewards = []
        next_observations = []
        terminated = []
        truncated = []
        max_steps = self.args.max_episode_steps or _env_max_episode_steps(self.env)
        for step_idx in range(max_steps):
            if random_actions:
                action = self.env.action_space.sample()
            else:
                action = self.agent.act(obs, execution_latent, deterministic=False)
            action = np.clip(action, self.env_spec.action_low, self.env_spec.action_high).astype(np.float32)
            step_result = self.env.step(action)
            next_obs, reward, done, info = _unpack_step(step_result)
            reached_limit = step_idx + 1 >= max_steps
            was_truncated = bool(info.get("TimeLimit.truncated", False) or info.get("bad_transition", False))
            if reached_limit:
                was_truncated = True
            was_terminated = bool(done and not was_truncated)
            observations.append(np.asarray(obs, dtype=np.float32).reshape(self.env_spec.state_dim))
            actions.append(np.asarray(action, dtype=np.float32).reshape(self.env_spec.action_dim))
            rewards.append([float(reward)])
            next_observations.append(np.asarray(next_obs, dtype=np.float32).reshape(self.env_spec.state_dim))
            terminated.append([1.0 if was_terminated else 0.0])
            truncated.append([1.0 if was_truncated else 0.0])
            obs = next_obs
            if done or reached_limit:
                break
        return {
            "observations": np.asarray(observations, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "next_observations": np.asarray(next_observations, dtype=np.float32),
            "terminated": np.asarray(terminated, dtype=np.float32),
            "truncated": np.asarray(truncated, dtype=np.float32),
        }

    def _ensure_env(self, kwargs: Dict[str, object]) -> None:
        if self.env is None:
            self.env = gym.make(self.env_spec.env_name, **kwargs)
            self.env.seed(self.args.seed)
            return
        target = self.env
        if hasattr(target, "set_latent"):
            target.set_latent(**kwargs)
        elif hasattr(target, "unwrapped") and hasattr(target.unwrapped, "set_latent"):
            target.unwrapped.set_latent(**kwargs)
        else:
            self.env.close()
            self.env = gym.make(self.env_spec.env_name, **kwargs)
            self.env.seed(self.args.seed)

    def save_checkpoint(self, filename: str) -> str:
        path = os.path.join(self.checkpoint_dir, filename)
        replay_state = self.replay.state_dict(include_episodes=self.args.checkpoint_replay)
        checkpoint = build_checkpoint(agent=self.agent,
                                      config=vars(self.args),
                                      env_type=self.env_spec.env_type,
                                      env_name=self.env_spec.env_name,
                                      obs_dim=self.env_spec.state_dim,
                                      action_dim=self.env_spec.action_dim,
                                      counters=self.counters,
                                      replay_state=replay_state,
                                      normalization_state=None)
        checkpoint["sequence_state"] = {
            "prev_posterior_latent": self.prev_posterior_latent.detach().cpu(),
            "prior_hidden": self.prior_state[0].detach().cpu(),
            "prior_cell": self.prior_state[1].detach().cpu(),
            "task_sampler": self.task_sampler.state_dict(),
        }
        save_checkpoint(path, checkpoint)
        self._log("checkpoint_path={}".format(path))
        return path

    def _load_resume(self, path: str) -> None:
        checkpoint = load_checkpoint(path, map_location=self.device)
        if checkpoint.get("env_type") != self.env_spec.env_type:
            raise ValueError("Checkpoint env_type {} does not match requested {}".format(
                checkpoint.get("env_type"), self.env_spec.env_type))
        if int(checkpoint.get("obs_dim")) != self.env_spec.state_dim:
            raise ValueError("Checkpoint observation dim mismatch")
        if int(checkpoint.get("action_dim")) != self.env_spec.action_dim:
            raise ValueError("Checkpoint action dim mismatch")
        if int(checkpoint.get("latent_dim")) != self.latent_dim:
            raise ValueError("Checkpoint latent dim mismatch")
        restore_agent(self.agent, checkpoint)
        replay_state = checkpoint.get("replay")
        if replay_state is not None:
            self.replay.load_state_dict(replay_state)
        self.counters.update(checkpoint.get("counters", {}))
        seq_state = checkpoint.get("sequence_state", {})
        if seq_state:
            self.prev_posterior_latent = seq_state["prev_posterior_latent"].to(self.device)
            self.prior_state = (seq_state["prior_hidden"].to(self.device), seq_state["prior_cell"].to(self.device))
            if "task_sampler" in seq_state:
                self.task_sampler.load_state_dict(seq_state["task_sampler"])

    def _log_header(self) -> None:
        info = {
            "env_type": self.env_spec.env_type,
            "env_name": self.env_spec.env_name,
            "seed": self.args.seed,
            "device": str(self.device),
            "obs_dim": self.env_spec.state_dim,
            "action_dim": self.env_spec.action_dim,
            "latent_dim": self.latent_dim,
            "action_low": self.env_spec.action_low.tolist(),
            "action_high": self.env_spec.action_high.tolist(),
            "parameter_counts": self.agent.parameter_counts(),
        }
        self._log(json.dumps(info, sort_keys=True))

    def _write_metrics(self, metrics: Dict[str, float]) -> None:
        metrics = dict(metrics)
        metrics["gradient_updates"] = float(self.counters["gradient_updates"])
        if self._csv_fieldnames is None:
            self._csv_fieldnames = sorted(metrics.keys())
            with open(self.metrics_path, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self._csv_fieldnames)
                writer.writeheader()
                writer.writerow({key: metrics.get(key, "") for key in self._csv_fieldnames})
        else:
            with open(self.metrics_path, "a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self._csv_fieldnames)
                writer.writerow({key: metrics.get(key, "") for key in self._csv_fieldnames})
        self._log("episode={} return={} replay_transitions={} updates={}".format(
            int(metrics.get("total_completed_episodes", 0)),
            metrics.get("episode_return", ""),
            int(metrics.get("replay_transition_count", 0)),
            self.counters["gradient_updates"]))

    def _log(self, message: str) -> None:
        print(message)
        with open(self.log_path, "a") as handle:
            handle.write(message + "\n")


def _reset_env(env):
    result = env.reset()
    if isinstance(result, tuple):
        return result[0]
    return result


def _sanitize_env_kwargs(kwargs: Dict[str, object]) -> Dict[str, object]:
    clean = {}
    for key, value in kwargs.items():
        if torch.is_tensor(value):
            if value.numel() == 1:
                clean[key] = float(value.item())
            else:
                clean[key] = value.detach().cpu().numpy()
        else:
            clean[key] = value
    return clean


def _unpack_step(step_result):
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        done = bool(terminated or truncated)
        info = dict(info)
        if truncated:
            info["TimeLimit.truncated"] = True
        return obs, reward, done, info
    obs, reward, done, info = step_result
    return obs, reward, done, info


def _env_max_episode_steps(env) -> int:
    if hasattr(env, "spec") and env.spec is not None and env.spec.max_episode_steps is not None:
        return int(env.spec.max_episode_steps)
    if hasattr(env, "_max_episode_steps"):
        return int(env._max_episode_steps)
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "horizon"):
        return int(env.unwrapped.horizon)
    return 200
