import numpy as np
import pytest
import torch

from learner.lilac.actor import SquashedGaussianActor
from learner.lilac.critic import LatentQNetwork, LatentValueNetwork
from learner.lilac.decoder import TransitionRewardDecoder
from learner.lilac.distributions import diagonal_gaussian_kl
from learner.lilac.encoder import PosteriorTrajectoryEncoder
from learner.lilac.losses import reconstruction_losses
from learner.lilac.prior import SequentialLatentPrior
from learner.lilac.utils import hard_update, soft_update


@pytest.mark.unit
def test_posterior_output_shapes_and_reparameterized_samples():
    encoder = PosteriorTrajectoryEncoder(3, 2, 4, [8], transition_embedding_dim=5)
    obs = torch.randn(6, 7, 3)
    actions = torch.randn(6, 7, 2)
    rewards = torch.randn(6, 7, 1)
    next_obs = torch.randn(6, 7, 3)
    masks = torch.ones(6, 7, 1)
    masks[0, 5:] = 0
    dist = encoder(obs, actions, rewards, next_obs, masks)
    assert dist.mean.shape == (6, 4)
    assert dist.rsample().shape == (6, 4)
    assert torch.isfinite(dist.log_std).all()


@pytest.mark.unit
def test_prior_output_shapes_hidden_reset_and_parallel_states():
    prior = SequentialLatentPrior(latent_dim=4, hidden_dim=6)
    z0 = prior.initial_latent(3, torch.device("cpu"))
    state = prior.initial_state(3, torch.device("cpu"))
    dist, next_state = prior.forward_step(z0, state)
    assert dist.mean.shape == (3, 4)
    assert next_state[0].shape == (3, 6)
    reset_state = prior.initial_state(3, torch.device("cpu"))
    assert torch.all(reset_state[0] == 0)
    assert not torch.allclose(next_state[0][0], next_state[0][1]) or torch.allclose(z0[0], z0[1])


@pytest.mark.unit
def test_analytic_diagonal_gaussian_kl_zero_for_equal_distributions():
    mean = torch.zeros(5, 2)
    log_std = torch.zeros(5, 2)
    mask = torch.ones(5, 1)
    assert diagonal_gaussian_kl(mean, log_std, mean, log_std, mask).item() == pytest.approx(0.0)


@pytest.mark.unit
def test_decoder_output_shapes_and_masking_losses():
    decoder = TransitionRewardDecoder(3, 2, 4, [8])
    obs = torch.randn(2, 5, 3)
    actions = torch.randn(2, 5, 2)
    rewards = torch.randn(2, 5, 1)
    next_obs = torch.randn(2, 5, 3)
    z = torch.randn(2, 4)
    masks = torch.ones(2, 5, 1)
    masks[:, 3:] = 0
    pred_next, pred_reward = decoder(obs, actions, z)
    assert pred_next.shape == next_obs.shape
    assert pred_reward.shape == rewards.shape
    transition_loss, reward_loss = reconstruction_losses(decoder, obs, actions, rewards, next_obs, z, masks)
    assert torch.isfinite(transition_loss)
    assert torch.isfinite(reward_loss)


@pytest.mark.unit
def test_actor_action_bounds_logprob_correction_and_deterministic_output():
    actor = SquashedGaussianActor(3, 2, 4, [8],
                                  action_low=np.asarray([-2.0, -1.0], dtype=np.float32),
                                  action_high=np.asarray([2.0, 3.0], dtype=np.float32))
    obs = torch.randn(10, 3)
    z = torch.randn(10, 4)
    action, log_prob, mean_action = actor.sample(obs, z, deterministic=False)
    assert torch.all(action[:, 0] <= 2.0 + 1e-5)
    assert torch.all(action[:, 0] >= -2.0 - 1e-5)
    assert log_prob.shape == (10, 1)
    det = actor.act(obs, z, deterministic=True)
    assert torch.allclose(det, mean_action, atol=1e-6)


@pytest.mark.unit
def test_critic_value_target_initialization_and_polyak_update():
    critic = LatentQNetwork(3, 2, 4, [8])
    value = LatentValueNetwork(3, 4, [8])
    target = LatentValueNetwork(3, 4, [8])
    obs = torch.randn(5, 3)
    action = torch.randn(5, 2)
    z = torch.randn(5, 4)
    assert critic(obs, action, z).shape == (5, 1)
    assert value(obs, z).shape == (5, 1)
    hard_update(target, value)
    for target_param, value_param in zip(target.parameters(), value.parameters()):
        assert torch.allclose(target_param, value_param)
    old = [param.clone() for param in target.parameters()]
    with torch.no_grad():
        for param in value.parameters():
            param.add_(1.0)
    soft_update(target, value, 0.5)
    assert any(not torch.allclose(param, old_param) for param, old_param in zip(target.parameters(), old))
