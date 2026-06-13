from typing import Optional

import torch


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class DiagonalGaussian(object):
    """Diagonal Gaussian with shape `[batch, latent_dim]`."""

    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor):
        if mean.shape != log_std.shape:
            raise ValueError("mean and log_std must have matching shapes, got {} and {}".format(
                tuple(mean.shape), tuple(log_std.shape)))
        self.mean = mean
        self.log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        self.std = torch.exp(self.log_std)

    def rsample(self) -> torch.Tensor:
        eps = torch.randn_like(self.std)
        return self.mean + eps * self.std

    def sample(self) -> torch.Tensor:
        with torch.no_grad():
            return self.rsample()

    def mode(self) -> torch.Tensor:
        return self.mean

    def log_prob(self, value: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        var = self.std.pow(2)
        log_prob = -0.5 * (((value - self.mean).pow(2) / var) + 2.0 * self.log_std + torch.log(
            torch.tensor(2.0 * 3.141592653589793, dtype=value.dtype, device=value.device)))
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        if keepdim:
            return log_prob
        return log_prob.squeeze(-1)

    def detach(self) -> "DiagonalGaussian":
        return DiagonalGaussian(self.mean.detach(), self.log_std.detach())


def diagonal_gaussian_kl(q_mean: torch.Tensor,
                         q_log_std: torch.Tensor,
                         p_mean: torch.Tensor,
                         p_log_std: torch.Tensor,
                         mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Return masked mean KL(q || p) for diagonal Gaussians.

    Shapes:
        q_mean, q_log_std, p_mean, p_log_std: `[B, latent_dim]` or `[B, L, latent_dim]`
        mask: optional `[B, 1]` or `[B, L, 1]`
    """
    if q_mean.shape != p_mean.shape or q_log_std.shape != p_log_std.shape:
        raise ValueError("KL inputs must have matching shapes")
    q_log_std = torch.clamp(q_log_std, LOG_STD_MIN, LOG_STD_MAX)
    p_log_std = torch.clamp(p_log_std, LOG_STD_MIN, LOG_STD_MAX)
    q_var = torch.exp(2.0 * q_log_std)
    p_var = torch.exp(2.0 * p_log_std)
    kl_per_dim = p_log_std - q_log_std + (q_var + (q_mean - p_mean).pow(2)) / (2.0 * p_var) - 0.5
    kl = kl_per_dim.sum(dim=-1, keepdim=True)
    if mask is None:
        return kl.mean()
    if mask.shape != kl.shape:
        raise ValueError("KL mask shape {} does not match KL shape {}".format(tuple(mask.shape), tuple(kl.shape)))
    denom = mask.sum().clamp(min=1.0)
    return (kl * mask).sum() / denom
