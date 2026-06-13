from typing import Sequence

import torch
import torch.nn as nn

from learner.lilac.utils import build_mlp, initialize_module


class LatentQNetwork(nn.Module):
    """Q(s, a, z) network returning `[B, 1]`."""

    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 latent_dim: int,
                 hidden_dims: Sequence[int],
                 activation: str = "relu",
                 init: str = "xavier_uniform"):
        super(LatentQNetwork, self).__init__()
        self.net = build_mlp(obs_dim + action_dim + latent_dim, hidden_dims, 1, activation=activation)
        initialize_module(self.net, init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        if obs.size(0) != action.size(0) or obs.size(0) != latent.size(0):
            raise ValueError("Q batch mismatch: obs {}, action {}, latent {}".format(
                obs.size(0), action.size(0), latent.size(0)))
        return self.net(torch.cat([obs, action, latent], dim=-1))


class LatentValueNetwork(nn.Module):
    """V(s, z) network returning `[B, 1]`.

    The LILAC paper writes the critic target with `V(s', z)`, matching the original SAC formulation.
    """

    def __init__(self,
                 obs_dim: int,
                 latent_dim: int,
                 hidden_dims: Sequence[int],
                 activation: str = "relu",
                 init: str = "xavier_uniform"):
        super(LatentValueNetwork, self).__init__()
        self.net = build_mlp(obs_dim + latent_dim, hidden_dims, 1, activation=activation)
        initialize_module(self.net, init)

    def forward(self, obs: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        if obs.size(0) != latent.size(0):
            raise ValueError("V batch mismatch: obs {}, latent {}".format(obs.size(0), latent.size(0)))
        return self.net(torch.cat([obs, latent], dim=-1))
