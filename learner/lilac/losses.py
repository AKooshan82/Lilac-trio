from typing import Tuple

import torch
import torch.nn.functional as F

from learner.lilac.distributions import diagonal_gaussian_kl


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked MSE averaged over valid scalar elements."""
    if prediction.shape != target.shape:
        raise ValueError("MSE prediction/target shape mismatch: {} vs {}".format(
            tuple(prediction.shape), tuple(target.shape)))
    if mask.dim() != prediction.dim():
        while mask.dim() < prediction.dim():
            mask = mask.unsqueeze(-1)
    expanded_mask = mask.expand_as(prediction)
    denom = expanded_mask.sum().clamp(min=1.0)
    return ((prediction - target).pow(2) * expanded_mask).sum() / denom


def reconstruction_losses(decoder,
                          observations: torch.Tensor,
                          actions: torch.Tensor,
                          rewards: torch.Tensor,
                          next_observations: torch.Tensor,
                          latents: torch.Tensor,
                          masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    next_pred, reward_pred = decoder(observations, actions, latents)
    transition_loss = masked_mse(next_pred, next_observations, masks)
    reward_loss = masked_mse(reward_pred, rewards, masks)
    return transition_loss, reward_loss


def kl_loss(q_mean: torch.Tensor,
            q_log_std: torch.Tensor,
            p_mean: torch.Tensor,
            p_log_std: torch.Tensor,
            mask: torch.Tensor) -> torch.Tensor:
    return diagonal_gaussian_kl(q_mean, q_log_std, p_mean, p_log_std, mask)


def q_loss(q_value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if q_value.shape != target.shape:
        raise ValueError("Q target shape mismatch: {} vs {}".format(tuple(q_value.shape), tuple(target.shape)))
    return F.mse_loss(q_value, target)


def value_loss(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if value.shape != target.shape:
        raise ValueError("V target shape mismatch: {} vs {}".format(tuple(value.shape), tuple(target.shape)))
    return F.mse_loss(value, target)
