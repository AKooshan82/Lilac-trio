from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class EpisodeRecord:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    episode_id: int
    sequence_id: int
    episode_position: int

    @property
    def length(self) -> int:
        return int(self.observations.shape[0])


class EpisodicReplayBuffer(object):
    """CPU replay buffer organized around complete lifelong episodes.

    Stored episode arrays:
        observations: `[T, obs_dim]`
        actions: `[T, action_dim]`
        rewards: `[T, 1]`
        next_observations: `[T, obs_dim]`
        terminated: `[T, 1]`
        truncated: `[T, 1]`
        valid masks are generated as ones for stored steps and zeros for padding.
    """

    def __init__(self,
                 capacity_episodes: int,
                 obs_dim: int,
                 action_dim: int,
                 seed: int = 1):
        if capacity_episodes <= 0:
            raise ValueError("Replay capacity must be positive")
        self.capacity_episodes = int(capacity_episodes)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self._episodes = []
        self._next_episode_id = 0
        self._rng = np.random.RandomState(seed)

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    @property
    def num_transitions(self) -> int:
        return int(sum(ep.length for ep in self._episodes))

    def add_episode(self,
                    observations: np.ndarray,
                    actions: np.ndarray,
                    rewards: np.ndarray,
                    next_observations: np.ndarray,
                    terminated: np.ndarray,
                    truncated: Optional[np.ndarray],
                    sequence_id: int,
                    episode_position: int) -> int:
        observations = _as_2d_float(observations, self.obs_dim, "observations")
        actions = _as_2d_float(actions, self.action_dim, "actions")
        next_observations = _as_2d_float(next_observations, self.obs_dim, "next_observations")
        rewards = _as_column_float(rewards, "rewards")
        terminated = _as_column_float(terminated, "terminated")
        if truncated is None:
            truncated = np.zeros_like(terminated, dtype=np.float32)
        truncated = _as_column_float(truncated, "truncated")
        length = observations.shape[0]
        if length <= 0:
            raise ValueError("Cannot add empty episode")
        for name, value in [("actions", actions), ("rewards", rewards),
                            ("next_observations", next_observations),
                            ("terminated", terminated), ("truncated", truncated)]:
            if value.shape[0] != length:
                raise ValueError("{} length {} does not match observations length {}".format(
                    name, value.shape[0], length))

        episode_id = self._next_episode_id
        self._next_episode_id += 1
        record = EpisodeRecord(observations=observations,
                               actions=actions,
                               rewards=rewards,
                               next_observations=next_observations,
                               terminated=terminated,
                               truncated=truncated,
                               episode_id=episode_id,
                               sequence_id=int(sequence_id),
                               episode_position=int(episode_position))
        self._episodes.append(record)
        while len(self._episodes) > self.capacity_episodes:
            self._episodes.pop(0)
        return episode_id

    def can_sample_transitions(self, batch_size: int) -> bool:
        return self.num_transitions >= batch_size and batch_size > 0

    def can_sample_episodes(self, batch_size: int) -> bool:
        return self.num_episodes >= batch_size and batch_size > 0

    def sample_transitions(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Sample transitions and their complete source episodes.

        Returns shapes:
            observations/actions/rewards/next_observations/terminated/truncated/masks: `[B, ...]`
            trajectories.*: `[B, T_max, ...]`, one source episode per sampled transition
            timesteps/source_episode_ids/sequence_ids/episode_positions: `[B, 1]`
        """
        if not self.can_sample_transitions(batch_size):
            raise ValueError("Cannot sample {} transitions from {} stored transitions".format(
                batch_size, self.num_transitions))
        flat_index = []
        for ep_idx, episode in enumerate(self._episodes):
            for step in range(episode.length):
                flat_index.append((ep_idx, step))
        choices = self._rng.randint(0, len(flat_index), size=batch_size)
        ep_indices = []
        steps = []
        for choice in choices:
            ep_idx, step = flat_index[int(choice)]
            ep_indices.append(ep_idx)
            steps.append(step)
        episodes = [self._episodes[idx] for idx in ep_indices]

        batch = {
            "observations": np.stack([ep.observations[t] for ep, t in zip(episodes, steps)], axis=0),
            "actions": np.stack([ep.actions[t] for ep, t in zip(episodes, steps)], axis=0),
            "rewards": np.stack([ep.rewards[t] for ep, t in zip(episodes, steps)], axis=0),
            "next_observations": np.stack([ep.next_observations[t] for ep, t in zip(episodes, steps)], axis=0),
            "terminated": np.stack([ep.terminated[t] for ep, t in zip(episodes, steps)], axis=0),
            "truncated": np.stack([ep.truncated[t] for ep, t in zip(episodes, steps)], axis=0),
            "masks": np.ones((batch_size, 1), dtype=np.float32),
            "timesteps": np.asarray(steps, dtype=np.int64).reshape(batch_size, 1),
            "source_episode_ids": np.asarray([ep.episode_id for ep in episodes], dtype=np.int64).reshape(batch_size, 1),
            "sequence_ids": np.asarray([ep.sequence_id for ep in episodes], dtype=np.int64).reshape(batch_size, 1),
            "episode_positions": np.asarray([ep.episode_position for ep in episodes], dtype=np.int64).reshape(batch_size, 1),
        }
        trajectory_batch = self._pad_episodes(episodes)
        for key, value in trajectory_batch.items():
            batch["trajectory_" + key] = value
        return _to_torch(batch, device)

    def sample_episodes(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        if not self.can_sample_episodes(batch_size):
            raise ValueError("Cannot sample {} episodes from {} stored episodes".format(batch_size, self.num_episodes))
        choices = self._rng.randint(0, len(self._episodes), size=batch_size)
        episodes = [self._episodes[int(idx)] for idx in choices]
        return _to_torch(self._pad_episodes(episodes), device)

    def sample_subsequences(self,
                            batch_size: int,
                            subsequence_length: int,
                            device: torch.device) -> Dict[str, torch.Tensor]:
        """Sample contiguous episode subsequences from one sequence.

        Returns shapes:
            observations: `[B, L, T_max, obs_dim]`
            actions: `[B, L, T_max, action_dim]`
            rewards: `[B, L, T_max, 1]`
            next_observations: `[B, L, T_max, obs_dim]`
            terminated/truncated/masks: `[B, L, T_max, 1]`
            sequence_ids/episode_positions/episode_ids/lengths: `[B, L]`
            episode_masks: `[B, L, 1]`
        """
        if subsequence_length <= 0:
            raise ValueError("subsequence_length must be positive")
        candidates = self._contiguous_windows(subsequence_length)
        if not candidates:
            raise ValueError("No contiguous episode subsequence of length {} is available".format(subsequence_length))
        choices = self._rng.randint(0, len(candidates), size=batch_size)
        windows = [candidates[int(idx)] for idx in choices]
        max_len = max(ep.length for window in windows for ep in window)
        batch_np = {}
        for name, width in [("observations", self.obs_dim), ("actions", self.action_dim),
                            ("rewards", 1), ("next_observations", self.obs_dim),
                            ("terminated", 1), ("truncated", 1), ("masks", 1)]:
            batch_np[name] = np.zeros((batch_size, subsequence_length, max_len, width), dtype=np.float32)
        ids = np.zeros((batch_size, subsequence_length), dtype=np.int64)
        sequence_ids = np.zeros((batch_size, subsequence_length), dtype=np.int64)
        positions = np.zeros((batch_size, subsequence_length), dtype=np.int64)
        lengths = np.zeros((batch_size, subsequence_length), dtype=np.int64)
        episode_masks = np.ones((batch_size, subsequence_length, 1), dtype=np.float32)
        for b_idx, window in enumerate(windows):
            self._validate_window(window)
            for l_idx, episode in enumerate(window):
                length = episode.length
                batch_np["observations"][b_idx, l_idx, :length, :] = episode.observations
                batch_np["actions"][b_idx, l_idx, :length, :] = episode.actions
                batch_np["rewards"][b_idx, l_idx, :length, :] = episode.rewards
                batch_np["next_observations"][b_idx, l_idx, :length, :] = episode.next_observations
                batch_np["terminated"][b_idx, l_idx, :length, :] = episode.terminated
                batch_np["truncated"][b_idx, l_idx, :length, :] = episode.truncated
                batch_np["masks"][b_idx, l_idx, :length, :] = 1.0
                ids[b_idx, l_idx] = episode.episode_id
                sequence_ids[b_idx, l_idx] = episode.sequence_id
                positions[b_idx, l_idx] = episode.episode_position
                lengths[b_idx, l_idx] = length
        batch_np["episode_ids"] = ids
        batch_np["sequence_ids"] = sequence_ids
        batch_np["episode_positions"] = positions
        batch_np["lengths"] = lengths
        batch_np["episode_masks"] = episode_masks
        return _to_torch(batch_np, device)

    def state_dict(self, include_episodes: bool = False) -> Dict[str, object]:
        state = {
            "capacity_episodes": self.capacity_episodes,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "next_episode_id": self._next_episode_id,
            "rng_state": self._rng.get_state(),
            "num_episodes": self.num_episodes,
            "num_transitions": self.num_transitions,
        }
        if include_episodes:
            state["episodes"] = self._episodes
        return state

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.capacity_episodes = int(state["capacity_episodes"])
        self.obs_dim = int(state["obs_dim"])
        self.action_dim = int(state["action_dim"])
        self._next_episode_id = int(state["next_episode_id"])
        self._rng.set_state(state["rng_state"])
        episodes = state.get("episodes")
        if episodes is not None:
            self._episodes = list(episodes)

    def _pad_episodes(self, episodes: Sequence[EpisodeRecord]) -> Dict[str, np.ndarray]:
        batch_size = len(episodes)
        max_len = max(ep.length for ep in episodes)
        batch = {
            "observations": np.zeros((batch_size, max_len, self.obs_dim), dtype=np.float32),
            "actions": np.zeros((batch_size, max_len, self.action_dim), dtype=np.float32),
            "rewards": np.zeros((batch_size, max_len, 1), dtype=np.float32),
            "next_observations": np.zeros((batch_size, max_len, self.obs_dim), dtype=np.float32),
            "terminated": np.zeros((batch_size, max_len, 1), dtype=np.float32),
            "truncated": np.zeros((batch_size, max_len, 1), dtype=np.float32),
            "masks": np.zeros((batch_size, max_len, 1), dtype=np.float32),
            "episode_ids": np.zeros((batch_size, 1), dtype=np.int64),
            "sequence_ids": np.zeros((batch_size, 1), dtype=np.int64),
            "episode_positions": np.zeros((batch_size, 1), dtype=np.int64),
            "lengths": np.zeros((batch_size, 1), dtype=np.int64),
        }
        for idx, episode in enumerate(episodes):
            length = episode.length
            batch["observations"][idx, :length, :] = episode.observations
            batch["actions"][idx, :length, :] = episode.actions
            batch["rewards"][idx, :length, :] = episode.rewards
            batch["next_observations"][idx, :length, :] = episode.next_observations
            batch["terminated"][idx, :length, :] = episode.terminated
            batch["truncated"][idx, :length, :] = episode.truncated
            batch["masks"][idx, :length, :] = 1.0
            batch["episode_ids"][idx, 0] = episode.episode_id
            batch["sequence_ids"][idx, 0] = episode.sequence_id
            batch["episode_positions"][idx, 0] = episode.episode_position
            batch["lengths"][idx, 0] = length
        return batch

    def _contiguous_windows(self, length: int) -> List[List[EpisodeRecord]]:
        by_sequence = {}
        for episode in self._episodes:
            by_sequence.setdefault(episode.sequence_id, []).append(episode)
        windows = []
        for episodes in by_sequence.values():
            ordered = sorted(episodes, key=lambda item: item.episode_position)
            for start in range(0, len(ordered) - length + 1):
                window = ordered[start:start + length]
                try:
                    self._validate_window(window)
                except ValueError:
                    continue
                windows.append(window)
        return windows

    @staticmethod
    def _validate_window(window: Sequence[EpisodeRecord]) -> None:
        if not window:
            raise ValueError("Empty replay window")
        sequence_id = window[0].sequence_id
        prev_position = window[0].episode_position - 1
        for episode in window:
            if episode.sequence_id != sequence_id:
                raise ValueError("Replay window crosses sequence boundary")
            if episode.episode_position != prev_position + 1:
                raise ValueError("Replay window is not contiguous")
            prev_position = episode.episode_position


def _as_2d_float(value: np.ndarray, width: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != width:
        raise ValueError("{} must have shape [T, {}], got {}".format(name, width, arr.shape))
    return arr.copy()


def _as_column_float(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.shape[1] != 1:
        raise ValueError("{} must have shape [T, 1], got {}".format(name, arr.shape))
    return arr.copy()


def _to_torch(batch: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    tensors = {}
    for key, value in batch.items():
        if value.dtype.kind in ("i", "u"):
            tensors[key] = torch.from_numpy(value).long().to(device)
        else:
            tensors[key] = torch.from_numpy(value).float().to(device)
    return tensors
