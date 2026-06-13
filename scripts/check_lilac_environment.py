import argparse
import platform
import sys
from typing import Any, Dict

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect LILAC environment compatibility without training.")
    parser.add_argument("--env-type", required=True, choices=["cheetah_vel", "ant_goal", "golf", "golf_signals"])
    parser.add_argument("--golf-num-signals", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    report: Dict[str, Any] = {}
    report["python"] = sys.version.replace("\n", " ")
    report["platform"] = platform.platform()
    report["torch_version"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()
    report["cuda_runtime_version"] = getattr(torch.version, "cuda", None)
    report["selected_device"] = args.device
    report["numpy_version"] = np.__version__
    report["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    try:
        import gym
        report["gym_version"] = getattr(gym, "__version__", "unknown")
    except ImportError as exc:
        print_report(report)
        raise RuntimeError("Gym is required for LILAC environment checks: {}".format(exc))

    try:
        import mujoco_py  # noqa: F401
        report["mujoco_py_available"] = True
    except ImportError:
        report["mujoco_py_available"] = False

    try:
        import envs  # noqa: F401
        from learner.lilac.agent import LILACAgent  # noqa: F401
        from learner.lilac.trainer import get_lilac_env_spec
        report["repository_import_status"] = "ok"
        report["lilac_import_status"] = "ok"
    except ImportError as exc:
        print_report(report)
        raise RuntimeError("Repository/LILAC import failed: {}".format(exc))

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device {} was requested, but CUDA is not available".format(args.device))

    env_spec = get_lilac_env_spec(args.env_type, args.golf_num_signals)
    report["env_name"] = env_spec.env_name
    report["expected_observation_dim"] = env_spec.state_dim
    report["expected_action_dim"] = env_spec.action_dim
    report["action_low_config"] = env_spec.action_low.tolist()
    report["action_high_config"] = env_spec.action_high.tolist()

    sampler_kwargs, _, sequence_id, episode_position = _sample_initial_task(env_spec, args.seed)
    report["sampled_task_kwargs_keys"] = sorted(sampler_kwargs.keys())
    report["sequence_id"] = sequence_id
    report["episode_position"] = episode_position

    try:
        env = gym.make(env_spec.env_name, **sampler_kwargs)
        env.seed(args.seed)
        report["environment_registration_status"] = "ok"
    except (gym.error.Error, RuntimeError, ValueError, OSError, ImportError) as exc:
        print_report(report)
        raise RuntimeError("Could not create {}. Check MuJoCo/Gym installation and custom registration. Original error: {}".format(
            env_spec.env_name, exc))

    try:
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        report["reset_return_type"] = type(reset_result).__name__
        report["observation_shape"] = list(np.asarray(obs).shape)
        report["action_shape"] = list(env.action_space.shape)
        report["action_low_env"] = np.asarray(env.action_space.low).tolist()
        report["action_high_env"] = np.asarray(env.action_space.high).tolist()
        action = env.action_space.sample()
        step_result = env.step(action)
        report["step_return_length"] = len(step_result)
        if len(step_result) == 5:
            next_obs, reward, terminated, truncated, info = step_result
            report["step_done_structure"] = "terminated/truncated"
        else:
            next_obs, reward, done, info = step_result
            report["step_done_structure"] = "done"
        report["random_step_succeeds"] = True
        report["next_observation_shape"] = list(np.asarray(next_obs).shape)
        report["reward_type"] = type(reward).__name__
        report["info_keys"] = sorted(list(info.keys()))
    finally:
        env.close()

    print_report(report)


def _sample_initial_task(env_spec, seed: int):
    from learner.lilac.trainer import LifelongTaskSampler
    sampler = LifelongTaskSampler(env_spec.task_generator, env_spec.default_latent_dim, transition_std=0.05, seed=seed)
    return sampler.next_task()


def print_report(report: Dict[str, Any]) -> None:
    for key in sorted(report.keys()):
        print("{}: {}".format(key, report[key]))


if __name__ == "__main__":
    main()
