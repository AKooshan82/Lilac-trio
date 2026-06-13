import numpy as np
import pytest
import torch

from learner.lilac.replay_buffer import EpisodicReplayBuffer


def _episode(length, obs_dim=3, action_dim=2, offset=0.0):
    obs = np.ones((length, obs_dim), dtype=np.float32) * offset
    actions = np.ones((length, action_dim), dtype=np.float32) * (offset + 1)
    rewards = np.arange(length, dtype=np.float32).reshape(-1, 1)
    next_obs = obs + 0.5
    terminated = np.zeros((length, 1), dtype=np.float32)
    terminated[-1, 0] = 1.0
    truncated = np.zeros((length, 1), dtype=np.float32)
    return obs, actions, rewards, next_obs, terminated, truncated


def _filled_buffer():
    buffer = EpisodicReplayBuffer(capacity_episodes=10, obs_dim=3, action_dim=2, seed=3)
    for seq in range(2):
        for pos, length in enumerate([2, 4, 3]):
            buffer.add_episode(*_episode(length, offset=seq * 10 + pos),
                               sequence_id=seq, episode_position=pos)
    return buffer


@pytest.mark.unit
def test_replay_buffer_episode_boundaries_and_variable_length_padding():
    buffer = _filled_buffer()
    batch = buffer.sample_episodes(3, torch.device("cpu"))
    assert batch["observations"].dim() == 3
    assert batch["masks"].shape[:2] == batch["rewards"].shape[:2]
    lengths = batch["lengths"].view(-1).tolist()
    for idx, length in enumerate(lengths):
        assert batch["masks"][idx, :length].sum().item() == length
        assert batch["masks"][idx, length:].sum().item() == 0


@pytest.mark.unit
def test_replay_buffer_sequence_boundaries_and_contiguous_sampling():
    buffer = _filled_buffer()
    batch = buffer.sample_subsequences(batch_size=4, subsequence_length=2, device=torch.device("cpu"))
    assert batch["observations"].shape[1] == 2
    for seq_ids, positions in zip(batch["sequence_ids"], batch["episode_positions"]):
        assert torch.all(seq_ids == seq_ids[0])
        assert int(positions[1].item()) == int(positions[0].item()) + 1


@pytest.mark.unit
def test_replay_buffer_transition_sampling_and_valid_masks_are_deterministic():
    b1 = _filled_buffer()
    b2 = _filled_buffer()
    sample1 = b1.sample_transitions(5, torch.device("cpu"))
    sample2 = b2.sample_transitions(5, torch.device("cpu"))
    assert torch.equal(sample1["timesteps"], sample2["timesteps"])
    assert torch.all(sample1["masks"] == 1.0)
    assert sample1["trajectory_observations"].shape[0] == 5


@pytest.mark.unit
def test_replay_buffer_rejects_cross_sequence_windows():
    buffer = EpisodicReplayBuffer(capacity_episodes=10, obs_dim=3, action_dim=2, seed=1)
    buffer.add_episode(*_episode(2), sequence_id=0, episode_position=0)
    buffer.add_episode(*_episode(2), sequence_id=1, episode_position=1)
    with pytest.raises(ValueError):
        buffer.sample_subsequences(1, 2, torch.device("cpu"))
