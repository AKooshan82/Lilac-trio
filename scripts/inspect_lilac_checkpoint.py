import argparse
from typing import Dict

import torch

from learner.lilac.checkpoint import LILAC_CHECKPOINT_VERSION, checkpoint_summary, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a LILAC checkpoint without training.")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint, map_location=torch.device("cpu"))
    summary = checkpoint_summary(checkpoint)
    print("LILAC checkpoint summary")
    for key in sorted(summary.keys()):
        print("{}: {}".format(key, summary[key]))
    print("current_checkpoint_version: {}".format(LILAC_CHECKPOINT_VERSION))
    print("parameter_counts:")
    for name, state in sorted(checkpoint.get("models", {}).items()):
        if isinstance(state, dict):
            print("  {}: {}".format(name, _state_dict_parameter_count(state)))
        elif torch.is_tensor(state):
            print("  {}: {}".format(name, int(state.numel())))
        else:
            print("  {}: unknown".format(name))
    print("missing_keys: {}".format(_missing_keys(checkpoint)))
    print("unexpected_top_level_keys: {}".format(sorted(set(checkpoint.keys()) - _expected_top_level_keys())))


def _state_dict_parameter_count(state: Dict[str, torch.Tensor]) -> int:
    total = 0
    for value in state.values():
        if torch.is_tensor(value):
            total += int(value.numel())
    return total


def _expected_top_level_keys():
    return {
        "checkpoint_version", "algorithm", "env_type", "env_name", "obs_dim", "action_dim",
        "latent_dim", "config", "counters", "models", "optimizers", "rng_state",
        "replay", "normalization", "sequence_state",
    }


def _missing_keys(checkpoint):
    missing = []
    for key in sorted(_expected_top_level_keys()):
        if key not in checkpoint:
            missing.append(key)
    for component in ["actor", "critic", "value", "target_value", "encoder", "decoder", "prior", "log_alpha"]:
        if component not in checkpoint.get("models", {}):
            missing.append("models.{}".format(component))
    for component in ["actor", "critic", "value", "encoder", "decoder", "prior", "alpha"]:
        if component not in checkpoint.get("optimizers", {}):
            missing.append("optimizers.{}".format(component))
    return missing


if __name__ == "__main__":
    main()
