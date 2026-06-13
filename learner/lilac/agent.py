from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from learner.lilac.actor import SquashedGaussianActor
from learner.lilac.critic import LatentQNetwork, LatentValueNetwork
from learner.lilac.decoder import TransitionRewardDecoder
from learner.lilac.distributions import DiagonalGaussian
from learner.lilac.encoder import PosteriorTrajectoryEncoder
from learner.lilac.losses import kl_loss, q_loss, reconstruction_losses, value_loss
from learner.lilac.prior import SequentialLatentPrior
from learner.lilac.utils import (assert_finite_module, assert_finite_tensor, clip_grad,
                                 count_parameters, frozen, hard_update, soft_update)


class LILACAgent(object):
    """LILAC networks and update rules.

    This implementation follows the original SAC-style value/target-value formulation implied by
    the LILAC paper's critic target `r + V(s', z)`.
    """

    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 latent_dim: int,
                 action_low: np.ndarray,
                 action_high: np.ndarray,
                 device: torch.device,
                 actor_hidden_dims: Sequence[int],
                 critic_hidden_dims: Sequence[int],
                 value_hidden_dims: Sequence[int],
                 encoder_hidden_dims: Sequence[int],
                 decoder_hidden_dims: Sequence[int],
                 prior_hidden_dim: int,
                 transition_embedding_dim: int,
                 actor_lr: float,
                 critic_lr: float,
                 value_lr: float,
                 encoder_lr: float,
                 decoder_lr: float,
                 prior_lr: float,
                 entropy_lr: float,
                 gamma: float,
                 polyak_tau: float,
                 entropy_coef: float,
                 automatic_entropy_tuning: bool,
                 target_entropy: Optional[float],
                 kl_coef: float,
                 transition_reconstruction_coef: float,
                 reward_reconstruction_coef: float,
                 critic_encoder_loss_coef: float,
                 reward_scale: float,
                 max_grad_norm: Optional[float],
                 activation: str = "relu",
                 init: str = "xavier_uniform",
                 prior_use_posterior_sample: bool = False,
                 deterministic_prior: bool = False):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.device = device
        self.gamma = gamma
        self.polyak_tau = polyak_tau
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.target_entropy = float(target_entropy if target_entropy is not None else -action_dim)
        self.kl_coef = kl_coef
        self.transition_reconstruction_coef = transition_reconstruction_coef
        self.reward_reconstruction_coef = reward_reconstruction_coef
        self.critic_encoder_loss_coef = critic_encoder_loss_coef
        self.reward_scale = reward_scale
        self.max_grad_norm = max_grad_norm
        self.prior_use_posterior_sample = prior_use_posterior_sample
        self.deterministic_prior = deterministic_prior
        self.target_update_count = 0

        self.actor = SquashedGaussianActor(obs_dim, action_dim, latent_dim, actor_hidden_dims,
                                           action_low, action_high, activation=activation, init=init).to(device)
        self.critic = LatentQNetwork(obs_dim, action_dim, latent_dim, critic_hidden_dims,
                                     activation=activation, init=init).to(device)
        self.value = LatentValueNetwork(obs_dim, latent_dim, value_hidden_dims,
                                        activation=activation, init=init).to(device)
        self.target_value = LatentValueNetwork(obs_dim, latent_dim, value_hidden_dims,
                                               activation=activation, init=init).to(device)
        self.encoder = PosteriorTrajectoryEncoder(obs_dim, action_dim, latent_dim, encoder_hidden_dims,
                                                  transition_embedding_dim=transition_embedding_dim,
                                                  activation=activation, init=init).to(device)
        self.decoder = TransitionRewardDecoder(obs_dim, action_dim, latent_dim, decoder_hidden_dims,
                                               activation=activation, init=init).to(device)
        self.prior = SequentialLatentPrior(latent_dim, prior_hidden_dim, init=init).to(device)
        hard_update(self.target_value, self.value)

        initial_alpha = max(float(entropy_coef), 1e-8)
        self.log_alpha = torch.tensor(np.log(initial_alpha), dtype=torch.float32,
                                      device=device, requires_grad=True)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=value_lr)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=encoder_lr)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=decoder_lr)
        self.prior_optimizer = torch.optim.Adam(self.prior.parameters(), lr=prior_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=entropy_lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def train(self) -> None:
        for module in self.modules():
            module.train()

    def eval(self) -> None:
        for module in self.modules():
            module.eval()

    def modules(self):
        return [self.actor, self.critic, self.value, self.target_value,
                self.encoder, self.decoder, self.prior]

    def parameter_counts(self) -> Dict[str, int]:
        return {
            "actor": count_parameters(self.actor),
            "critic": count_parameters(self.critic),
            "value": count_parameters(self.value),
            "target_value": count_parameters(self.target_value),
            "encoder": count_parameters(self.encoder),
            "decoder": count_parameters(self.decoder),
            "prior": count_parameters(self.prior),
            "entropy": 1,
        }

    def act(self,
            obs: np.ndarray,
            latent: torch.Tensor,
            deterministic: bool = True) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, self.obs_dim)
        if latent.dim() == 1:
            latent = latent.view(1, self.latent_dim)
        with torch.no_grad():
            action = self.actor.act(obs_tensor, latent.to(self.device), deterministic=deterministic)
        return action.cpu().numpy()[0]

    def infer_posterior(self, episode_batch: Dict[str, torch.Tensor]) -> DiagonalGaussian:
        return self.encoder(episode_batch["observations"],
                            episode_batch["actions"],
                            episode_batch["rewards"],
                            episode_batch["next_observations"],
                            episode_batch["masks"])

    def update(self,
               replay_buffer,
               transition_batch_size: int,
               episode_batch_size: int,
               subsequence_length: int,
               counters: Optional[Dict[str, int]] = None) -> Dict[str, float]:
        """Run one LILAC replay update and return scalar diagnostics."""
        self.train()
        transition_batch = replay_buffer.sample_transitions(transition_batch_size, self.device)
        subsequence_batch = replay_buffer.sample_subsequences(episode_batch_size, subsequence_length, self.device)

        metrics = {}
        sac_latent = self._sample_transition_latent(transition_batch).detach()
        metrics.update(self._update_critic(transition_batch, sac_latent))
        metrics.update(self._update_value(transition_batch, sac_latent))
        metrics.update(self._update_actor_and_alpha(transition_batch, sac_latent))
        metrics.update(self._update_encoder(transition_batch, subsequence_batch))
        metrics.update(self._update_decoder(transition_batch))
        metrics.update(self._update_prior(subsequence_batch))

        soft_update(self.target_value, self.value, self.polyak_tau)
        self.target_update_count += 1
        metrics["target_update_count"] = float(self.target_update_count)
        metrics["alpha"] = float(self.alpha.detach().cpu().item())
        self._check_all_finite()
        return metrics

    def _sample_transition_latent(self, transition_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        episode_batch = _trajectory_from_transition_batch(transition_batch)
        posterior = self.infer_posterior(episode_batch)
        return posterior.rsample()

    def _update_critic(self, batch: Dict[str, torch.Tensor], z: torch.Tensor) -> Dict[str, float]:
        obs = batch["observations"]
        action = batch["actions"]
        reward = batch["rewards"] * self.reward_scale
        next_obs = batch["next_observations"]
        terminated = batch["terminated"]
        nonterminal = 1.0 - terminated
        with torch.no_grad():
            target_v = self.target_value(next_obs, z)
            target_q = reward + self.gamma * nonterminal * target_v
        q_pred = self.critic(obs, action, z)
        loss = q_loss(q_pred, target_q)
        assert_finite_tensor("critic_loss", loss)
        self.critic_optimizer.zero_grad()
        loss.backward()
        grad = clip_grad(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        return {"critic_loss": float(loss.detach().cpu().item()), "critic_grad_norm": float(grad)}

    def _update_value(self, batch: Dict[str, torch.Tensor], z: torch.Tensor) -> Dict[str, float]:
        obs = batch["observations"]
        with torch.no_grad():
            new_action, log_prob, _ = self.actor.sample(obs, z, deterministic=False, with_log_prob=True)
            q_new = self.critic(obs, new_action, z)
            target_v = q_new - self.alpha.detach() * log_prob
        value_pred = self.value(obs, z)
        loss = value_loss(value_pred, target_v)
        assert_finite_tensor("value_loss", loss)
        self.value_optimizer.zero_grad()
        loss.backward()
        grad = clip_grad(self.value.parameters(), self.max_grad_norm)
        self.value_optimizer.step()
        return {"value_loss": float(loss.detach().cpu().item()), "value_grad_norm": float(grad)}

    def _update_actor_and_alpha(self, batch: Dict[str, torch.Tensor], z: torch.Tensor) -> Dict[str, float]:
        obs = batch["observations"]
        self.actor_optimizer.zero_grad()
        with frozen(self.critic):
            new_action, log_prob, _ = self.actor.sample(obs, z, deterministic=False, with_log_prob=True)
            q_new = self.critic(obs, new_action, z)
            actor_loss = (self.alpha.detach() * log_prob - q_new).mean()
        assert_finite_tensor("actor_loss", actor_loss)
        actor_loss.backward()
        actor_grad = clip_grad(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        entropy_loss_value = 0.0
        if self.automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            entropy_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            assert_finite_tensor("entropy_loss", entropy_loss)
            entropy_loss.backward()
            self.alpha_optimizer.step()
            entropy_loss_value = float(entropy_loss.detach().cpu().item())
        return {
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "actor_grad_norm": float(actor_grad),
            "entropy_loss": entropy_loss_value,
        }

    def _update_encoder(self,
                        transition_batch: Dict[str, torch.Tensor],
                        subsequence_batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        episode_batch = _trajectory_from_transition_batch(transition_batch)
        posterior = self.infer_posterior(episode_batch)
        z = posterior.rsample()
        with frozen(self.decoder):
            trans_loss, rew_loss = reconstruction_losses(self.decoder,
                                                         episode_batch["observations"],
                                                         episode_batch["actions"],
                                                         episode_batch["rewards"],
                                                         episode_batch["next_observations"],
                                                         z,
                                                         episode_batch["masks"])
        with frozen(self.critic):
            obs = transition_batch["observations"]
            action = transition_batch["actions"]
            reward = transition_batch["rewards"] * self.reward_scale
            next_obs = transition_batch["next_observations"]
            terminated = transition_batch["terminated"]
            with torch.no_grad():
                target_q = reward + self.gamma * (1.0 - terminated) * self.target_value(next_obs, z.detach())
            critic_encoder_loss = q_loss(self.critic(obs, action, z), target_q)

        q_mean, q_log_std, p_mean, p_log_std, episode_mask = self._prior_kl_tensors(
            subsequence_batch, detach_posterior_inputs=True, detach_posterior_params=False)
        k_loss = kl_loss(q_mean, q_log_std, p_mean.detach(), p_log_std.detach(), episode_mask)
        encoder_loss = (self.transition_reconstruction_coef * trans_loss
                        + self.reward_reconstruction_coef * rew_loss
                        + self.kl_coef * k_loss
                        + self.critic_encoder_loss_coef * critic_encoder_loss)
        assert_finite_tensor("encoder_loss", encoder_loss)
        self.encoder_optimizer.zero_grad()
        encoder_loss.backward()
        grad = clip_grad(self.encoder.parameters(), self.max_grad_norm)
        self.encoder_optimizer.step()
        return {
            "encoder_loss": float(encoder_loss.detach().cpu().item()),
            "encoder_grad_norm": float(grad),
            "transition_reconstruction_loss_encoder": float(trans_loss.detach().cpu().item()),
            "reward_reconstruction_loss_encoder": float(rew_loss.detach().cpu().item()),
            "kl_loss_encoder": float(k_loss.detach().cpu().item()),
            "critic_encoder_loss": float(critic_encoder_loss.detach().cpu().item()),
            "posterior_mean_abs": float(q_mean.detach().abs().mean().cpu().item()),
            "posterior_std_mean": float(torch.exp(q_log_std.detach()).mean().cpu().item()),
            "prior_mean_abs": float(p_mean.detach().abs().mean().cpu().item()),
            "prior_std_mean": float(torch.exp(p_log_std.detach()).mean().cpu().item()),
        }

    def _update_decoder(self, transition_batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        episode_batch = _trajectory_from_transition_batch(transition_batch)
        with torch.no_grad():
            posterior = self.infer_posterior(episode_batch)
            z = posterior.rsample()
        trans_loss, rew_loss = reconstruction_losses(self.decoder,
                                                     episode_batch["observations"],
                                                     episode_batch["actions"],
                                                     episode_batch["rewards"],
                                                     episode_batch["next_observations"],
                                                     z,
                                                     episode_batch["masks"])
        decoder_loss = (self.transition_reconstruction_coef * trans_loss
                        + self.reward_reconstruction_coef * rew_loss)
        assert_finite_tensor("decoder_loss", decoder_loss)
        self.decoder_optimizer.zero_grad()
        decoder_loss.backward()
        grad = clip_grad(self.decoder.parameters(), self.max_grad_norm)
        self.decoder_optimizer.step()
        return {
            "decoder_loss": float(decoder_loss.detach().cpu().item()),
            "decoder_grad_norm": float(grad),
            "transition_reconstruction_loss": float(trans_loss.detach().cpu().item()),
            "reward_reconstruction_loss": float(rew_loss.detach().cpu().item()),
        }

    def _update_prior(self, subsequence_batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        q_mean, q_log_std, p_mean, p_log_std, episode_mask = self._prior_kl_tensors(
            subsequence_batch, detach_posterior_inputs=True, detach_posterior_params=True)
        loss = self.kl_coef * kl_loss(q_mean, q_log_std, p_mean, p_log_std, episode_mask)
        assert_finite_tensor("prior_loss", loss)
        self.prior_optimizer.zero_grad()
        loss.backward()
        grad = clip_grad(self.prior.parameters(), self.max_grad_norm)
        self.prior_optimizer.step()
        return {"prior_loss": float(loss.detach().cpu().item()), "prior_grad_norm": float(grad)}

    def _prior_kl_tensors(self,
                          subsequence_batch: Dict[str, torch.Tensor],
                          detach_posterior_inputs: bool,
                          detach_posterior_params: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, horizon, _ = subsequence_batch["observations"].shape
        flat = {
            "observations": subsequence_batch["observations"].contiguous().view(batch * length, horizon, self.obs_dim),
            "actions": subsequence_batch["actions"].contiguous().view(batch * length, horizon, self.action_dim),
            "rewards": subsequence_batch["rewards"].contiguous().view(batch * length, horizon, 1),
            "next_observations": subsequence_batch["next_observations"].contiguous().view(batch * length, horizon, self.obs_dim),
            "masks": subsequence_batch["masks"].contiguous().view(batch * length, horizon, 1),
        }
        posterior = self.infer_posterior(flat)
        q_mean = posterior.mean.view(batch, length, self.latent_dim)
        q_log_std = posterior.log_std.view(batch, length, self.latent_dim)
        if detach_posterior_params:
            q_mean = q_mean.detach()
            q_log_std = q_log_std.detach()

        if self.prior_use_posterior_sample and not detach_posterior_params:
            prior_inputs = posterior.rsample().view(batch, length, self.latent_dim)
        else:
            prior_inputs = posterior.mean.view(batch, length, self.latent_dim)
        if detach_posterior_inputs:
            prior_inputs = prior_inputs.detach()
        p_mean, p_log_std = self.prior.unroll(prior_inputs)
        episode_mask = subsequence_batch["episode_masks"]
        return q_mean, q_log_std, p_mean, p_log_std, episode_mask

    def posterior_from_numpy_episode(self,
                                     observations: np.ndarray,
                                     actions: np.ndarray,
                                     rewards: np.ndarray,
                                     next_observations: np.ndarray) -> DiagonalGaussian:
        batch = {
            "observations": torch.as_tensor(observations, dtype=torch.float32, device=self.device).unsqueeze(0),
            "actions": torch.as_tensor(actions, dtype=torch.float32, device=self.device).unsqueeze(0),
            "rewards": torch.as_tensor(rewards, dtype=torch.float32, device=self.device).view(1, -1, 1),
            "next_observations": torch.as_tensor(next_observations, dtype=torch.float32, device=self.device).unsqueeze(0),
            "masks": torch.ones(1, observations.shape[0], 1, dtype=torch.float32, device=self.device),
        }
        return self.infer_posterior(batch)

    def _check_all_finite(self) -> None:
        for name, module in [("actor", self.actor), ("critic", self.critic),
                             ("value", self.value), ("encoder", self.encoder),
                             ("decoder", self.decoder), ("prior", self.prior)]:
            assert_finite_module(name, module)


def _trajectory_from_transition_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        "observations": batch["trajectory_observations"],
        "actions": batch["trajectory_actions"],
        "rewards": batch["trajectory_rewards"],
        "next_observations": batch["trajectory_next_observations"],
        "terminated": batch["trajectory_terminated"],
        "truncated": batch["trajectory_truncated"],
        "masks": batch["trajectory_masks"],
    }
