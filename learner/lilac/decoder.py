from typing import Sequence, Tuple

import torch
import torch.nn as nn

from learner.lilac.utils import build_mlp, initialize_module


class TransitionRewardDecoder(nn.Module):
    """Predicts next observation and reward from `(s_t, a_t, z_i)`.

    Inputs:
        observations: `[B, T, obs_dim]` or `[B, obs_dim]`
        actions: `[B, T, action_dim]` or `[B, action_dim]`
        latents: `[B, latent_dim]` or `[B, T, latent_dim]`

    Outputs:
        next_observation_pred and reward_pred with matching batch/time axes.
    """

    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 latent_dim: int,
                 hidden_dims: Sequence[int],
                 activation: str = "relu",
                 init: str = "xavier_uniform"):
        super(TransitionRewardDecoder, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.net = build_mlp(obs_dim + action_dim + latent_dim, hidden_dims, obs_dim + 1,
                             activation=activation)
        initialize_module(self.net, init)

    def forward(self,
                observations: torch.Tensor,
                actions: torch.Tensor,
                latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        original_rank = observations.dim()
        if original_rank == 3:
            batch, horizon, _ = observations.shape
            if latents.dim() == 2:
                latents = latents.unsqueeze(1).expand(batch, horizon, latents.size(-1))
            flat_obs = observations.contiguous().view(batch * horizon, -1)
            flat_actions = actions.contiguous().view(batch * horizon, -1)
            flat_latents = latents.contiguous().view(batch * horizon, -1)
            out = self.net(torch.cat([flat_obs, flat_actions, flat_latents], dim=-1))
            out = out.view(batch, horizon, -1)
            return out[:, :, :self.obs_dim], out[:, :, self.obs_dim:self.obs_dim + 1]
        if original_rank == 2:
            out = self.net(torch.cat([observations, actions, latents], dim=-1))
            return out[:, :self.obs_dim], out[:, self.obs_dim:self.obs_dim + 1]
        raise ValueError("Decoder observations must be rank 2 or 3, got {}".format(original_rank))
