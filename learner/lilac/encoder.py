from typing import Sequence, Tuple

import torch
import torch.nn as nn

from learner.lilac.distributions import DiagonalGaussian
from learner.lilac.utils import build_mlp, initialize_module


class PosteriorTrajectoryEncoder(nn.Module):
    """Feedforward masked trajectory encoder for `q_phi(z_i | tau_i)`.

    Inputs:
        observations: `[B, T, obs_dim]`
        actions: `[B, T, action_dim]`
        rewards: `[B, T, 1]`
        next_observations: `[B, T, obs_dim]`
        masks: `[B, T, 1]`, one for valid steps and zero for padding

    Output:
        `DiagonalGaussian` with mean/log_std `[B, latent_dim]`.
    """

    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 latent_dim: int,
                 hidden_dims: Sequence[int],
                 transition_embedding_dim: int = 128,
                 log_std_min: float = -20.0,
                 log_std_max: float = 2.0,
                 activation: str = "relu",
                 init: str = "xavier_uniform"):
        super(PosteriorTrajectoryEncoder, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        transition_dim = obs_dim + action_dim + 1 + obs_dim
        self.transition_encoder = build_mlp(transition_dim, hidden_dims, transition_embedding_dim,
                                            activation=activation)
        self.head = build_mlp(transition_embedding_dim * 2 + 1, hidden_dims, latent_dim * 2,
                              activation=activation)
        initialize_module(self.transition_encoder, init)
        initialize_module(self.head, init)

    def forward(self,
                observations: torch.Tensor,
                actions: torch.Tensor,
                rewards: torch.Tensor,
                next_observations: torch.Tensor,
                masks: torch.Tensor) -> DiagonalGaussian:
        batch, horizon, _ = observations.shape
        if actions.shape[:2] != (batch, horizon) or rewards.shape[:2] != (batch, horizon):
            raise ValueError("Trajectory encoder batch/time shapes do not match")
        if masks.shape != (batch, horizon, 1):
            raise ValueError("Expected masks shape [{}, {}, 1], got {}".format(batch, horizon, tuple(masks.shape)))
        valid_steps = masks.sum(dim=1)
        if torch.any(valid_steps <= 0):
            raise ValueError("Cannot encode an empty trajectory")

        transition = torch.cat([observations, actions, rewards, next_observations], dim=-1)
        flat = transition.view(batch * horizon, -1)
        emb = self.transition_encoder(flat).view(batch, horizon, -1)
        emb = emb * masks
        denom = valid_steps.clamp(min=1.0)
        mean_emb = emb.sum(dim=1) / denom
        sq_mean_emb = (emb.pow(2).sum(dim=1) / denom)
        length_feature = torch.log(denom)
        pooled = torch.cat([mean_emb, sq_mean_emb, length_feature], dim=-1)
        out = self.head(pooled)
        mean, log_std = torch.chunk(out, 2, dim=-1)
        return DiagonalGaussian(mean, torch.clamp(log_std, self.log_std_min, self.log_std_max))

    def encode_mean(self,
                    observations: torch.Tensor,
                    actions: torch.Tensor,
                    rewards: torch.Tensor,
                    next_observations: torch.Tensor,
                    masks: torch.Tensor) -> torch.Tensor:
        return self.forward(observations, actions, rewards, next_observations, masks).mean
