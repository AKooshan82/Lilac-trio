import os
from typing import Dict, Optional

import torch

from learner.lilac.utils import count_parameters, get_rng_state, set_rng_state


LILAC_CHECKPOINT_VERSION = 1


def build_checkpoint(agent,
                     config: Dict[str, object],
                     env_type: str,
                     env_name: str,
                     obs_dim: int,
                     action_dim: int,
                     counters: Dict[str, int],
                     replay_state: Optional[Dict[str, object]] = None,
                     normalization_state: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    return {
        "checkpoint_version": LILAC_CHECKPOINT_VERSION,
        "algorithm": "lilac",
        "env_type": env_type,
        "env_name": env_name,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "latent_dim": agent.latent_dim,
        "config": dict(config),
        "counters": dict(counters),
        "models": {
            "actor": agent.actor.state_dict(),
            "critic": agent.critic.state_dict(),
            "value": agent.value.state_dict(),
            "target_value": agent.target_value.state_dict(),
            "encoder": agent.encoder.state_dict(),
            "decoder": agent.decoder.state_dict(),
            "prior": agent.prior.state_dict(),
            "log_alpha": agent.log_alpha.detach().cpu(),
        },
        "optimizers": {
            "actor": agent.actor_optimizer.state_dict(),
            "critic": agent.critic_optimizer.state_dict(),
            "value": agent.value_optimizer.state_dict(),
            "encoder": agent.encoder_optimizer.state_dict(),
            "decoder": agent.decoder_optimizer.state_dict(),
            "prior": agent.prior_optimizer.state_dict(),
            "alpha": agent.alpha_optimizer.state_dict(),
        },
        "rng_state": get_rng_state(),
        "replay": replay_state,
        "normalization": normalization_state,
    }


def save_checkpoint(path: str, checkpoint: Dict[str, object]) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(checkpoint, path)


def load_checkpoint(path: str, map_location: Optional[torch.device] = None) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location=map_location)
    if checkpoint.get("algorithm") != "lilac":
        raise ValueError("Checkpoint is not a LILAC checkpoint: {}".format(path))
    version = checkpoint.get("checkpoint_version")
    if version != LILAC_CHECKPOINT_VERSION:
        raise ValueError("Unsupported LILAC checkpoint version {}; expected {}".format(
            version, LILAC_CHECKPOINT_VERSION))
    return checkpoint


def restore_agent(agent, checkpoint: Dict[str, object]) -> None:
    models = checkpoint.get("models", {})
    required = ["actor", "critic", "value", "target_value", "encoder", "decoder", "prior", "log_alpha"]
    missing = [name for name in required if name not in models]
    if missing:
        raise KeyError("LILAC checkpoint missing model components: {}".format(", ".join(missing)))
    agent.actor.load_state_dict(models["actor"])
    agent.critic.load_state_dict(models["critic"])
    agent.value.load_state_dict(models["value"])
    agent.target_value.load_state_dict(models["target_value"])
    agent.encoder.load_state_dict(models["encoder"])
    agent.decoder.load_state_dict(models["decoder"])
    agent.prior.load_state_dict(models["prior"])
    with torch.no_grad():
        agent.log_alpha.copy_(models["log_alpha"].to(agent.device))

    optimizers = checkpoint.get("optimizers", {})
    for name in ["actor", "critic", "value", "encoder", "decoder", "prior", "alpha"]:
        if name not in optimizers:
            raise KeyError("LILAC checkpoint missing optimizer state '{}'".format(name))
    agent.actor_optimizer.load_state_dict(optimizers["actor"])
    agent.critic_optimizer.load_state_dict(optimizers["critic"])
    agent.value_optimizer.load_state_dict(optimizers["value"])
    agent.encoder_optimizer.load_state_dict(optimizers["encoder"])
    agent.decoder_optimizer.load_state_dict(optimizers["decoder"])
    agent.prior_optimizer.load_state_dict(optimizers["prior"])
    agent.alpha_optimizer.load_state_dict(optimizers["alpha"])
    set_rng_state(checkpoint.get("rng_state", {}))


def checkpoint_summary(checkpoint: Dict[str, object]) -> Dict[str, object]:
    models = checkpoint.get("models", {})
    model_keys = sorted(models.keys())
    optimizers = checkpoint.get("optimizers", {})
    return {
        "checkpoint_version": checkpoint.get("checkpoint_version"),
        "algorithm": checkpoint.get("algorithm"),
        "env_type": checkpoint.get("env_type"),
        "env_name": checkpoint.get("env_name"),
        "obs_dim": checkpoint.get("obs_dim"),
        "action_dim": checkpoint.get("action_dim"),
        "latent_dim": checkpoint.get("latent_dim"),
        "counters": checkpoint.get("counters", {}),
        "model_components": model_keys,
        "optimizer_components": sorted(optimizers.keys()),
        "replay_present": checkpoint.get("replay") is not None,
        "normalization_present": checkpoint.get("normalization") is not None,
        "rng_state_present": checkpoint.get("rng_state") is not None,
    }
