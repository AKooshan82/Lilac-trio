from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from learner.lilac.utils import build_mlp, initialize_module


class SquashedGaussianActor(nn.Module):
    """Latent-conditioned squashed Gaussian actor.

    Inputs:
        observation: `[B, obs_dim]`
        latent: `[B, latent_dim]`

    Outputs from `sample`:
        action: `[B, action_dim]`, scaled to environment bounds
        log_prob: `[B, 1]`
        mean_action: `[B, action_dim]`
    """

    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 latent_dim: int,
                 hidden_dims: Sequence[int],
                 action_low: np.ndarray,
                 action_high: np.ndarray,
                 log_std_min: float = -20.0,
                 log_std_max: float = 2.0,
                 activation: str = "relu",
                 init: str = "xavier_uniform"):
        super(SquashedGaussianActor, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.net = build_mlp(obs_dim + latent_dim, hidden_dims, 2 * action_dim, activation=activation)
        initialize_module(self.net, init)

        low = torch.tensor(action_low, dtype=torch.float32).view(1, action_dim)
        high = torch.tensor(action_high, dtype=torch.float32).view(1, action_dim)
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

    def forward(self, obs: torch.Tensor, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if obs.dim() != 2 or latent.dim() != 2:
            raise ValueError("Actor expects obs and latent rank 2, got {} and {}".format(obs.dim(), latent.dim()))
        if obs.size(0) != latent.size(0):
            raise ValueError("Actor batch mismatch: obs {}, latent {}".format(obs.size(0), latent.size(0)))
        out = self.net(torch.cat([obs, latent], dim=-1))
        mean, log_std = torch.chunk(out, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self,
               obs: torch.Tensor,
               latent: torch.Tensor,
               deterministic: bool = False,
               with_log_prob: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs, latent)
        std = torch.exp(log_std)
        if deterministic:
            pre_tanh = mean
        else:
            pre_tanh = mean + std * torch.randn_like(std)
        tanh_action = torch.tanh(pre_tanh)
        action = tanh_action * self.action_scale + self.action_bias
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias

        log_prob = torch.zeros(obs.size(0), 1, dtype=obs.dtype, device=obs.device)
        if with_log_prob:
            normal_log_prob = -0.5 * (
                ((pre_tanh - mean) / std).pow(2)
                + 2.0 * log_std
                + torch.log(torch.tensor(2.0 * 3.141592653589793, dtype=obs.dtype, device=obs.device))
            )
            normal_log_prob = normal_log_prob.sum(dim=-1, keepdim=True)
            correction = torch.log(self.action_scale * (1.0 - tanh_action.pow(2)) + 1e-6)
            log_prob = normal_log_prob - correction.sum(dim=-1, keepdim=True)
        return action, log_prob, mean_action

    def act(self, obs: torch.Tensor, latent: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        action, _, _ = self.sample(obs, latent, deterministic=deterministic, with_log_prob=False)
        return action
