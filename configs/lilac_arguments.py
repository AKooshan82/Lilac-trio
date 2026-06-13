import argparse


def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "y")


def get_args(rest_args):
    parser = argparse.ArgumentParser(description="LILAC")

    # General settings
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--device", type=str, default="cpu", help="torch device, e.g. cpu or cuda:0")
    parser.add_argument("--output-folder", "--folder", dest="output_folder", type=str, default="",
                        help="folder where LILAC results are written")
    parser.add_argument("--experiment-name", type=str, default="lilac")
    parser.add_argument("--num-processes", type=int, default=1,
                        help="LILAC currently supports exactly one process")
    parser.add_argument("--num-episodes", type=int, default=100000,
                        help="number of complete lifelong episodes to collect")
    parser.add_argument("--max-env-steps", "--max-episode-steps", dest="max_episode_steps",
                        type=int, default=None, help="maximum steps per episode")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--debug", "--debug-logging", dest="debug_logging", action="store_true", default=False)
    parser.add_argument("--resume-checkpoint", type=str, default=None)

    # LILAC latent/model dimensions
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--posterior-hidden-dims", type=str, default="256,256")
    parser.add_argument("--transition-embedding-dim", type=int, default=128)
    parser.add_argument("--prior-lstm-hidden-dim", type=int, default=128)
    parser.add_argument("--decoder-hidden-dims", type=str, default="256,256")
    parser.add_argument("--actor-hidden-dims", type=str, default="256,256")
    parser.add_argument("--critic-hidden-dims", type=str, default="256,256")
    parser.add_argument("--value-hidden-dims", type=str, default="256,256")
    parser.add_argument("--activation", choices=["relu", "elu", "tanh"], default="relu")
    parser.add_argument("--network-init", choices=["xavier_uniform", "xavier_normal", "orthogonal"],
                        default="xavier_uniform")

    # Replay and update schedule
    parser.add_argument("--replay-capacity", type=int, default=10000,
                        help="capacity measured in complete episodes")
    parser.add_argument("--transition-batch-size", "--batch-size", dest="transition_batch_size",
                        type=int, default=256)
    parser.add_argument("--episode-batch-size", type=int, default=16)
    parser.add_argument("--subsequence-length", type=int, default=4)
    parser.add_argument("--warmup-episodes", type=int, default=10)
    parser.add_argument("--warmup-transitions", type=int, default=1000)
    parser.add_argument("--updates-per-episode", type=int, default=1)
    parser.add_argument("--warmup-behavior", choices=["random", "policy"], default="random")
    parser.add_argument("--checkpoint-replay", type=_str_to_bool, default=False,
                        help="whether to include full replay contents in checkpoints")

    # Optimizers
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--value-lr", type=float, default=3e-4)
    parser.add_argument("--encoder-lr", type=float, default=3e-4)
    parser.add_argument("--decoder-lr", type=float, default=3e-4)
    parser.add_argument("--prior-lr", type=float, default=3e-4)
    parser.add_argument("--entropy-lr", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)

    # SAC and LILAC losses
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--polyak-tau", type=float, default=0.005)
    parser.add_argument("--entropy-coef", type=float, default=0.2)
    parser.add_argument("--automatic-entropy-tuning", type=_str_to_bool, default=True)
    parser.add_argument("--target-entropy", type=float, default=None)
    parser.add_argument("--kl-coef", type=float, default=1.0)
    parser.add_argument("--transition-reconstruction-coef", type=float, default=1.0)
    parser.add_argument("--reward-reconstruction-coef", type=float, default=1.0)
    parser.add_argument("--critic-encoder-loss-coef", type=float, default=1.0)
    parser.add_argument("--reward-scale", type=float, default=1.0)

    # Lifelong sequence behavior
    parser.add_argument("--task-transition-std", type=float, default=0.05)
    parser.add_argument("--execution-prior", choices=["sample", "mean"], default="sample",
                        help="use prior sample or prior mean as the fixed execution latent")
    parser.add_argument("--prior-recurrent-input", choices=["posterior_mean", "posterior_sample"],
                        default="posterior_mean",
                        help="posterior-derived latent fed to the LSTM between episodes")
    parser.add_argument("--deterministic-evaluation", type=_str_to_bool, default=True)
    parser.add_argument("--evaluation-interval", type=int, default=None)

    args = parser.parse_args(rest_args)
    return args
