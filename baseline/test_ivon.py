import argparse
import os
import jax
import yaml
import jax.numpy as jnp

from orbax.checkpoint import StandardCheckpointer
from data_loaders.cifar10 import get_cifar10_test_loader, CIFAR10Info
from models import get_cifar10_model, get_supported_models_names
from logger import TestResultsLogger
from trainer.evaluation import evaluate_ivon_with_sampling


def compute_aggregate_metrics(all_metrics):
    metric_names = list(all_metrics[0].keys())
    aggregate_stats = {}

    for metric_name in metric_names:
        values = jnp.array([m[metric_name] for m in all_metrics])
        mean = jnp.mean(values)
        std = jnp.std(values)
        aggregate_stats[metric_name] = {"mean": float(mean), "std": float(std)}

    return aggregate_stats


def reconstruct_opt_state(opt_state_raw):
    """Reconstruct optimizer state from checkpoint."""
    from core.ivon import IVONState

    if not isinstance(opt_state_raw, list):
        return opt_state_raw

    # Get the first element (should be IVONState data)
    ivon_data = opt_state_raw[0]

    # Reconstruct IVONState
    if isinstance(ivon_data, dict):
        ivon_state = IVONState(**ivon_data)
    elif isinstance(ivon_data, (list, tuple)):
        ivon_state = IVONState(*ivon_data)
    elif isinstance(ivon_data, IVONState):
        ivon_state = ivon_data
    else:
        raise ValueError(f"Unexpected IVONState format: {type(ivon_data)}")

    # Reconstruct full tuple
    return (ivon_state, *opt_state_raw[1])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        required=True,
        help="Base directory containing checkpoint folders for each seed",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model to train",
        choices=get_supported_models_names(),
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=64,
        help="Number of parameter samples for uncertainty quantification",
    )

    parser.add_argument(
        "--hessian",
        type=float,
        required=True,
        help="Initial value of the Hessian",
    )

    parser.add_argument(
        "--logdir",
        type=str,
        required=True,
        help="Path to the directory for metrics logger output",
    )

    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), "train_cifar10_config.yaml")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        config = config["cifar10"]["ivon"]

    model = get_cifar10_model(
        model_name=args.model_name, num_classes=CIFAR10Info.num_classes
    )

    init_rng = jax.random.key(0)
    model.init(init_rng, jnp.ones(CIFAR10Info.shape))

    test_ds = get_cifar10_test_loader(
        batch_size=config["test_batch_size"],
    )

    checkpointer = StandardCheckpointer()

    seeds = [0, 1, 2, 3, 4]
    all_metrics = []

    script_dir = os.path.dirname(os.path.abspath(__file__))

    with TestResultsLogger(
        args.logdir,
        f"ivon with {args.n_samples} samples and h0={args.hessian}",
        args.model_name,
    ) as logger:
        for seed in seeds:
            logger.start_seed(seed)
            checkpoint_path = os.path.join(args.checkpoints_dir, str(seed))

            checkpoint_data = checkpointer.restore(directory=checkpoint_path)

            opt_params = checkpoint_data["params"]
            opt_state = reconstruct_opt_state(checkpoint_data["opt_state"])

            # Evaluate with multiple samples
            computed_metrics = evaluate_ivon_with_sampling(
                model,
                opt_params,
                opt_state,
                test_ds,
                seed,
                n_samples=args.n_samples,
                num_classes=CIFAR10Info.num_classes,
            )

            all_metrics.append(computed_metrics)
            logger.log_seed_metrics(seed, computed_metrics)

        aggregate_stats = compute_aggregate_metrics(all_metrics)

        logger.write_aggregate_results(len(seeds), aggregate_stats)
        print(aggregate_stats)
