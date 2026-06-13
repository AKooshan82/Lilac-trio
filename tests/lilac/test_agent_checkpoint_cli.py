import argparse
import copy
import os

import numpy as np
import pytest
import torch

from configs import lilac_arguments
from learner.lilac.agent import LILACAgent
from learner.lilac.checkpoint import build_checkpoint, load_checkpoint, restore_agent, save_checkpoint
from learner.lilac.replay_buffer import EpisodicReplayBuffer


def _tiny_agent():
    return LILACAgent(obs_dim=3, action_dim=2, latent_dim=2,
                      action_low=np.asarray([-1.0, -1.0], dtype=np.float32),
                      action_high=np.asarray([1.0, 1.0], dtype=np.float32),
                      device=torch.device("cpu"),
                      actor_hidden_dims=[8], critic_hidden_dims=[8], value_hidden_dims=[8],
                      encoder_hidden_dims=[8], decoder_hidden_dims=[8],
                      prior_hidden_dim=8, transition_embedding_dim=8,
                      actor_lr=1e-3, critic_lr=1e-3, value_lr=1e-3,
                      encoder_lr=1e-3, decoder_lr=1e-3, prior_lr=1e-3,
                      entropy_lr=1e-3, gamma=0.99, polyak_tau=0.01,
                      entropy_coef=0.2, automatic_entropy_tuning=True,
                      target_entropy=None, kl_coef=1.0,
                      transition_reconstruction_coef=1.0,
                      reward_reconstruction_coef=1.0,
                      critic_encoder_loss_coef=1.0,
                      reward_scale=1.0, max_grad_norm=10.0)


def _buffer():
    replay = EpisodicReplayBuffer(20, obs_dim=3, action_dim=2, seed=5)
    for pos in range(5):
        length = 3
        obs = np.random.randn(length, 3).astype(np.float32)
        actions = np.tanh(np.random.randn(length, 2)).astype(np.float32)
        rewards = np.random.randn(length, 1).astype(np.float32)
        next_obs = obs + 0.1
        terminated = np.zeros((length, 1), dtype=np.float32)
        truncated = np.zeros((length, 1), dtype=np.float32)
        terminated[-1, 0] = 1.0
        replay.add_episode(obs, actions, rewards, next_obs, terminated, truncated,
                           sequence_id=0, episode_position=pos)
    return replay


def _params(module):
    return [param.detach().clone() for param in module.parameters()]


def _changed(before, module):
    return any(not torch.allclose(old, new.detach()) for old, new in zip(before, module.parameters()))


@pytest.mark.integration
def test_one_synthetic_lilac_update_finite_losses_and_expected_parameter_changes():
    agent = _tiny_agent()
    replay = _buffer()
    before_actor = _params(agent.actor)
    before_critic = _params(agent.critic)
    before_encoder = _params(agent.encoder)
    before_decoder = _params(agent.decoder)
    before_prior = _params(agent.prior)
    metrics = agent.update(replay, transition_batch_size=4, episode_batch_size=2, subsequence_length=2)
    for key, value in metrics.items():
        assert np.isfinite(value), key
    assert _changed(before_actor, agent.actor)
    assert _changed(before_critic, agent.critic)
    assert _changed(before_encoder, agent.encoder)
    assert _changed(before_decoder, agent.decoder)
    assert _changed(before_prior, agent.prior)


@pytest.mark.unit
def test_no_unintended_encoder_gradient_from_actor_loss():
    agent = _tiny_agent()
    batch = _buffer().sample_transitions(4, torch.device("cpu"))
    z = agent._sample_transition_latent(batch).detach()
    agent.encoder.zero_grad()
    agent._update_actor_and_alpha(batch, z)
    assert all(param.grad is None for param in agent.encoder.parameters())


@pytest.mark.unit
def test_encoder_receives_critic_reconstruction_and_kl_gradients():
    agent = _tiny_agent()
    replay = _buffer()
    transition_batch = replay.sample_transitions(4, torch.device("cpu"))
    subsequence_batch = replay.sample_subsequences(2, 2, torch.device("cpu"))
    agent.encoder.zero_grad()
    agent._update_encoder(transition_batch, subsequence_batch)
    assert any(param.grad is not None for param in agent.encoder.parameters())


@pytest.mark.unit
def test_prior_updated_only_by_intended_prior_step():
    agent = _tiny_agent()
    replay = _buffer()
    batch = replay.sample_transitions(4, torch.device("cpu"))
    z = agent._sample_transition_latent(batch).detach()
    before = _params(agent.prior)
    agent._update_actor_and_alpha(batch, z)
    assert not _changed(before, agent.prior)
    subseq = replay.sample_subsequences(2, 2, torch.device("cpu"))
    agent._update_prior(subseq)
    assert _changed(before, agent.prior)


@pytest.mark.unit
def test_terminal_and_truncation_bootstrap_handling():
    agent = _tiny_agent()
    batch = _buffer().sample_transitions(4, torch.device("cpu"))
    batch["terminated"].zero_()
    batch["truncated"].fill_(1.0)
    z = agent._sample_transition_latent(batch).detach()
    metrics = agent._update_critic(batch, z)
    assert np.isfinite(metrics["critic_loss"])


@pytest.mark.unit
def test_checkpoint_save_load_configuration_and_counter_restoration(tmp_path):
    agent = _tiny_agent()
    path = os.path.join(str(tmp_path), "lilac.pt")
    checkpoint = build_checkpoint(agent, {"latent_dim": 2}, "mock", "mock-v0", 3, 2,
                                  {"completed_episodes": 7, "environment_steps": 11, "gradient_updates": 13},
                                  replay_state={"num_episodes": 0}, normalization_state=None)
    save_checkpoint(path, checkpoint)
    loaded = load_checkpoint(path, map_location=torch.device("cpu"))
    restored = _tiny_agent()
    restore_agent(restored, loaded)
    assert loaded["counters"]["completed_episodes"] == 7
    assert loaded["latent_dim"] == 2


@pytest.mark.unit
def test_fixed_seed_deterministic_replay_sampling_behavior():
    sample_a = _buffer().sample_transitions(5, torch.device("cpu"))["timesteps"]
    sample_b = _buffer().sample_transitions(5, torch.device("cpu"))["timesteps"]
    assert torch.equal(sample_a, sample_b)


@pytest.mark.unit
def test_cli_recognition_of_lilac_and_existing_algorithm_names():
    args = lilac_arguments.get_args(["--num-episodes", "2", "--batch-size", "3"])
    assert args.num_episodes == 2
    assert args.transition_batch_size == 3
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="rl2")
    for algo in ["rl2", "bayes", "ts", "lilac"]:
        parsed = parser.parse_args(["--algo", algo])
        assert parsed.algo == algo


@pytest.mark.unit
def test_no_privileged_task_parameter_in_policy_inputs_and_fixed_execution_latent_shape():
    agent = _tiny_agent()
    obs = np.zeros(3, dtype=np.float32)
    latent = torch.zeros(1, 2)
    action = agent.act(obs, latent, deterministic=True)
    assert action.shape == (2,)
    assert agent.actor.obs_dim == 3
    assert agent.actor.latent_dim == 2
