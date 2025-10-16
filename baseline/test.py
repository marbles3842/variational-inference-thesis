import argparse
import os
import jax
import yaml
import jax.numpy as jnp

from orbax.checkpoint import StandardCheckpointer

from data_loaders.cifar10 import get_cifar10_test_loader
from models import get_cifar10_model, get_supported_models_names
from trainer.metrics import Metrics, cross_entropy_loss
from logger import TestResultsLogger


NUM_CLASSES = 10


def compute_aggregate_metrics(all_metrics):
    metric_names = list(all_metrics[0].keys())
    aggregate_stats = {}

    for metric_name in metric_names:
        values = jnp.array([m[metric_name] for m in all_metrics])
        mean = jnp.mean(values)
        std = jnp.std(values)
        aggregate_stats[metric_name] = {"mean": float(mean), "std": float(std)}

    return aggregate_stats


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model to train",
        choices=get_supported_models_names(),
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        required=True,
        help="Optimizer to evaluate",
        choices=("sgd", "ivon"),
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        required=True,
        help="Base directory containing checkpoint folders for each seed",
    )
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), "train_cifar10_config.yaml")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        config = config["cifar10"][args.optimizer]

    model = get_cifar10_model(model_name=args.model_name, num_classes=NUM_CLASSES)

    init_rng = jax.random.key(0)
    model.init(init_rng, jnp.ones([1, 32, 32, 3]))

    test_ds = get_cifar10_test_loader(
        batch_size=config["test_batch_size"],
    )

    checkpointer = StandardCheckpointer()

    seeds = [0, 1, 2, 3, 4]
    all_metrics = []

    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "..", "out", args.optimizer)

    with TestResultsLogger(log_dir, args.optimizer, args.model_name) as logger:
        for seed in seeds:
            logger.start_seed(seed)

            checkpoint_path = os.path.join(args.checkpoints_dir, str(seed))

            checkpoint_data = checkpointer.restore(directory=checkpoint_path)
            model_params = checkpoint_data["params"]

            metrics = Metrics.empty()

            for test_batch in test_ds:
                test_batch = jax.device_put(test_batch)

                logits = model.apply({"params": model_params}, test_batch["image"])
                loss = cross_entropy_loss(logits=logits, labels=test_batch["label"])

                metric_update = metrics.single_from_model_output(
                    logits=logits, labels=test_batch["label"], loss=loss
                )
                metrics = metrics.merge(metric_update)

            computed_metrics = metrics.compute()
            all_metrics.append(computed_metrics)
            logger.log_seed_metrics(seed, computed_metrics)

        aggregate_stats = compute_aggregate_metrics(all_metrics)

        logger.write_aggregate_results(len(seeds), aggregate_stats)
