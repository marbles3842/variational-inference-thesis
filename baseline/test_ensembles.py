import argparse
import os
import jax
import yaml
import jax.numpy as jnp

from jax.nn import softmax
from jax.tree import map as tree_map
from jax import vmap
from orbax.checkpoint import StandardCheckpointer

from data_loaders.cifar10 import get_cifar10_test_loader, CIFAR10Info
from models import get_cifar10_model, get_supported_models_names
from trainer.metrics import Metrics, cross_entropy_loss
from logger import TestResultsLogger
from typing import Any
from functools import partial


def compute_aggregate_metrics(all_metrics):
    metric_names = list(all_metrics[0].keys())
    aggregate_stats = {}

    for metric_name in metric_names:
        values = jnp.array([m[metric_name] for m in all_metrics])
        mean = jnp.mean(values)
        std = jnp.std(values)
        aggregate_stats[metric_name] = {"mean": float(mean), "std": float(std)}

    return aggregate_stats


def load_ensemble_params(seeds, checkpoints_dir):
    def load_single_model(seed):
        checkpoint_path = os.path.join(checkpoints_dir, str(seed))
        checkpoint_data = checkpointer.restore(directory=checkpoint_path)
        return checkpoint_data["params"]

    ensemble_params = list(map(load_single_model, seeds))
    return tree_map(lambda *args: jnp.stack(args), *ensemble_params)


@partial(jax.jit, static_argnames=["model"])
def evaluate_batch(model, ensemble_params: Any, test_batch):

    batch_metrics = Metrics.empty()

    def evaluate_model(params, images):
        logits = model.apply({"params": params}, images)
        return softmax(logits)

    batch_probs = vmap(lambda params: evaluate_model(params, test_batch["image"]))(
        ensemble_params
    )

    batch_logits = jnp.log(jnp.stack(batch_probs, axis=0).mean(axis=0))
    batch_loss = cross_entropy_loss(logits=batch_logits, labels=test_batch["label"])
    updated_metrics = batch_metrics.single_from_model_output(
        logits=batch_logits, loss=batch_loss, labels=test_batch["label"]
    )
    return batch_metrics.merge(updated_metrics)


def evaluate_ensembles(model, ensemble_params: Any):

    metrics = Metrics.empty()

    for test_batch in test_ds:
        test_batch = jax.device_put(test_batch)

        updated_metrics = evaluate_batch(model, ensemble_params, test_batch)

        metrics = metrics.merge(updated_metrics)

    return metrics.compute()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        default="resnet20",
        help="Model to train",
        choices=get_supported_models_names(),
    )

    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        required=True,
        help="Base directory containing checkpoint folders for each seed",
    )

    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=5,
        help="Number of models in the ensemble",
    )

    parser.add_argument(
        "--ensemble-opt",
        type=str,
        required=True,
        help="Optimizer to evaluate",
        choices=("deep-ensembles-sgd", "multi-ivon"),
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
        config = config["cifar10"]["common"]

    model = get_cifar10_model(
        model_name=args.model_name, num_classes=CIFAR10Info.num_classes
    )

    init_rng = jax.random.key(0)
    model.init(init_rng, jnp.ones(CIFAR10Info.shape))

    test_ds = get_cifar10_test_loader(
        batch_size=config["test_batch_size"], worker_count=0
    )

    checkpointer = StandardCheckpointer()

    seeds = range(25)
    ensembles = list(
        map(
            lambda i: seeds[i : i + args.ensemble_size],
            range(0, len(seeds), args.ensemble_size),
        )
    )
    all_metrics = []

    script_dir = os.path.dirname(os.path.abspath(__file__))

    model = get_cifar10_model(args.model_name, CIFAR10Info.num_classes)

    with TestResultsLogger(args.logdir, args.ensemble_opt, args.model_name) as logger:

        for i, ensemble_seeds in enumerate(ensembles):

            logger.start_seed(i)

            ensemble_params = load_ensemble_params(ensemble_seeds, args.checkpoints_dir)
            ensemble_metrics = evaluate_ensembles(
                model=model, ensemble_params=ensemble_params
            )
            all_metrics.append(ensemble_metrics)
            logger.log_seed_metrics(i, ensemble_metrics)

        aggregate_stats = compute_aggregate_metrics(all_metrics)
        logger.write_aggregate_results(len(ensembles), aggregate_stats)
