import math
import random
from contextlib import contextmanager
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


def parse_hidden_dims(value: object, default: Sequence[int]) -> List[int]:
    """Parse CLI hidden-size values such as `256,256` or a single integer."""
    if value is None:
        return list(default)
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    text = str(value).strip()
    if not text:
        return list(default)
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def build_mlp(input_dim: int,
              hidden_dims: Sequence[int],
              output_dim: int,
              activation: str = "relu",
              output_activation: Optional[str] = None) -> nn.Sequential:
    dims = [input_dim] + list(hidden_dims)
    layers = []
    for idx in range(len(dims) - 1):
        layers.append(nn.Linear(dims[idx], dims[idx + 1]))
        layers.append(_activation(activation))
    layers.append(nn.Linear(dims[-1], output_dim))
    if output_activation is not None:
        layers.append(_activation(output_activation))
    return nn.Sequential(*layers)


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError("Unsupported activation '{}'".format(name))


def initialize_module(module: nn.Module, mode: str = "xavier_uniform") -> None:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            if mode == "orthogonal":
                nn.init.orthogonal_(child.weight)
            elif mode == "xavier_normal":
                nn.init.xavier_normal_(child.weight)
            elif mode == "xavier_uniform":
                nn.init.xavier_uniform_(child.weight)
            else:
                raise ValueError("Unsupported initialization '{}'".format(mode))
            nn.init.constant_(child.bias, 0.0)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.copy_(target_param * (1.0 - tau) + source_param * tau)


def count_parameters(module: nn.Module) -> int:
    return int(sum(param.numel() for param in module.parameters() if param.requires_grad))


def set_global_seeds(seed: int, cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_from_numpy(array: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.from_numpy(array).to(device=device, dtype=dtype)


def assert_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError("{} contains NaN or infinity with shape {}".format(name, tuple(tensor.shape)))


def assert_finite_module(name: str, module: nn.Module) -> None:
    for param_name, param in module.named_parameters():
        if not torch.isfinite(param).all():
            raise FloatingPointError("{} parameter {} contains NaN or infinity".format(name, param_name))


def grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is not None:
            value = param.grad.detach().norm(2).item()
            total += value * value
    return math.sqrt(total)


@contextmanager
def frozen(module: nn.Module):
    states = [param.requires_grad for param in module.parameters()]
    try:
        for param in module.parameters():
            param.requires_grad = False
        yield
    finally:
        for param, state in zip(module.parameters(), states):
            param.requires_grad = state


def clip_grad(parameters: Iterable[torch.nn.Parameter], max_norm: Optional[float]) -> float:
    params = [param for param in parameters if param.requires_grad]
    if max_norm is None or max_norm <= 0:
        return grad_norm(params)
    return float(torch.nn.utils.clip_grad_norm_(params, max_norm))


def get_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
