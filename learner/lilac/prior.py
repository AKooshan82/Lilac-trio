from typing import Optional, Tuple

import torch
import torch.nn as nn

from learner.lilac.distributions import DiagonalGaussian
from learner.lilac.utils import initialize_module


PriorState = Tuple[torch.Tensor, torch.Tensor]


class SequentialLatentPrior(nn.Module):
    """LSTM prior for `p_psi(z_i | z_{i-1}, h_{i-1})`.

    `forward_step` consumes the previous episode's posterior-derived latent `[B, latent_dim]`
    and previous LSTM state, then returns the prior distribution for the current episode and
    the next LSTM state.
    """

    def __init__(self,
                 latent_dim: int,
                 hidden_dim: int,
                 log_std_min: float = -20.0,
                 log_std_max: float = 2.0,
                 init: str = "xavier_uniform"):
        super(SequentialLatentPrior, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.cell = nn.LSTMCell(latent_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)
        initialize_module(self, init)

    def initial_latent(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.latent_dim, dtype=torch.float32, device=device)

    def initial_state(self, batch_size: int, device: torch.device) -> PriorState:
        h = torch.zeros(batch_size, self.hidden_dim, dtype=torch.float32, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, dtype=torch.float32, device=device)
        return h, c

    def forward_step(self, prev_latent: torch.Tensor, state: Optional[PriorState] = None) -> Tuple[DiagonalGaussian, PriorState]:
        if prev_latent.dim() != 2:
            raise ValueError("Previous latent must have shape [B, latent_dim], got {}".format(tuple(prev_latent.shape)))
        batch_size = prev_latent.size(0)
        if state is None:
            state = self.initial_state(batch_size, prev_latent.device)
        h, c = state
        if h.size(0) != batch_size or c.size(0) != batch_size:
            raise ValueError("Prior hidden-state batch mismatch: latent {}, h {}, c {}".format(
                batch_size, h.size(0), c.size(0)))
        next_h, next_c = self.cell(prev_latent, (h, c))
        mean = self.mean(next_h)
        log_std = torch.clamp(self.log_std(next_h), self.log_std_min, self.log_std_max)
        return DiagonalGaussian(mean, log_std), (next_h, next_c)

    def unroll(self,
               posterior_latents: torch.Tensor,
               initial_latent: Optional[torch.Tensor] = None,
               initial_state: Optional[PriorState] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unroll over a contiguous episode subsequence.

        Args:
            posterior_latents: `[B, L, latent_dim]`, posterior-derived latents for current episodes.
            initial_latent: optional `[B, latent_dim]`; defaults to `z^0 = 0`.
            initial_state: optional prior state.

        Returns:
            prior_means: `[B, L, latent_dim]`
            prior_log_stds: `[B, L, latent_dim]`
        """
        if posterior_latents.dim() != 3:
            raise ValueError("posterior_latents must have shape [B, L, latent_dim]")
        batch, length, _ = posterior_latents.shape
        prev = initial_latent
        if prev is None:
            prev = self.initial_latent(batch, posterior_latents.device)
        state = initial_state
        means = []
        log_stds = []
        for idx in range(length):
            dist, state = self.forward_step(prev, state)
            means.append(dist.mean)
            log_stds.append(dist.log_std)
            prev = posterior_latents[:, idx, :]
        return torch.stack(means, dim=1), torch.stack(log_stds, dim=1)
