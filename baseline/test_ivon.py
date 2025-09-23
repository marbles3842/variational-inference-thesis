import argparse
import os
import jax
import jax.numpy as jnp
import yaml

from orbax.checkpoint import StandardCheckpointer

from data_loaders.cifar10_dataloader import get_cifar10_test_loader
from models.resnet import ResNet20
from logger.concurrent_logger import ConcurrentMetricsLogger
from trainer.train_state import create_eval_state
from trainer.metrics import compute_metrics


NUM_CLASSES = 10


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, required=True, help="Seed for the initialization"
    )
    parser.add_argument(
        "--last-checkpoint", type=str, required=True, help="Path to the last checkpoint"
    )
    parser.add_argument("--job-id", type=int, required=True, help="The job id")
    args = parser.parse_args()

    print(args.last_checkpoint)

    config_path = os.path.join(os.path.dirname(__file__), "train_cifar10_config.yaml")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        config = config["cifar10"]["ivon"]

    model = ResNet20(num_classes=NUM_CLASSES)

    init_rng = jax.random.key(args.seed)

    test_ds = get_cifar10_test_loader(
        batch_size=config["test_batch_size"],
    )

    # init checkpointer
    checkpointer = StandardCheckpointer()
    checkpoint_data = checkpointer.restore(directory=args.last_checkpoint)

    state = create_eval_state(
        model=model,
        rng=init_rng,
        x0=jnp.ones([1, 32, 32, 3]),
        last_checkpoint_data=checkpoint_data,
    )

    logdir = img_dir = os.path.join(os.path.dirname(__file__), "..", "out", "ivon")
    metrics_log_path = os.path.join(logdir, "test-metrics-ivon.csv")

    for test_batch in test_ds:
        test_batch = jax.device_put(test_batch)

        state = compute_metrics(state=state, batch=test_batch)

    with ConcurrentMetricsLogger(metrics_log_path) as logger:
        for metric, value in state.metrics.compute().items():
            logger.update(
                "test", metric=metric, value=value, seed=args.seed, job_id=args.job_id
            )

        logger.write_row()
