import argparse
import os
import jax
import jax.numpy as jnp
import yaml
from orbax.checkpoint import StandardCheckpointer
from jax import random


from core.optimizer import create_cifar_ivon_optimizer
from data_loaders.cifar10 import get_cifar10_train_val_loaders
from models import get_cifar10_model, get_supported_models_names
from logger import MetricsLogger
from trainer.train_state import create_train_state
from trainer.metrics import compute_metrics
from trainer.train_step import train_step_ivon

NUM_CLASSES = 10


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, required=True, help="Seed for the initialization"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model to train",
        choices=get_supported_models_names(),
    )
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), "train_cifar10_config.yaml")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        config = config["cifar10"]["ivon"]

    model = get_cifar10_model(model_name=args.model_name, num_classes=NUM_CLASSES)

    init_rng = random.key(args.seed)

    main_rng, model_rng = random.split(init_rng, num=2)

    train_ds, val_ds = get_cifar10_train_val_loaders(
        train_batch_size=config["train_batch_size"],
        val_batch_size=config["val_batch_size"],
        seed=args.seed,
        num_epochs=config["num_epochs"],
    )

    num_steps_per_epoch = jnp.ceil(
        train_ds._data_source.__len__() / config["train_batch_size"]
    ).astype(jnp.int32)

    optimizer = create_cifar_ivon_optimizer(
        learning_rate=config["learning_rate"],
        warmup_epochs=config["warmup_epochs"],
        total_epochs=config["num_epochs"],
        steps_per_epoch=num_steps_per_epoch,
        momentum=config["momentum"],
        hess_init=config["hess_init"],
        momentum_hess=config["momentum_hess"],
        ess=config["ess"],
        weight_decay=config["weight_decay"],
    )

    state = create_train_state(
        model=model, rng=model_rng, x0=jnp.ones([1, 32, 32, 3]), optimizer=optimizer
    )

    logdir = img_dir = os.path.join(os.path.dirname(__file__), "..", "out", "ivon")
    metrics_log_path = os.path.join(
        logdir, f"train-metrics-ivon-{args.model_name}-{args.seed}.csv"
    )

    # init checkpointer
    checkpointer = StandardCheckpointer()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(
        script_dir, "..", "checkpoints", "ivon", args.model_name, str(args.seed)
    )

    # training loop
    with MetricsLogger(metrics_log_path) as logger:

        for step, batch in enumerate(train_ds):

            batch = jax.device_put(batch)

            main_rng, step_rng = random.split(main_rng)

            state = train_step_ivon(state, batch, step_rng)
            state = compute_metrics(state=state, batch=batch)

            if (step + 1) % num_steps_per_epoch == 0:
                for metric, value in state.metrics.compute().items():
                    logger.update("train", metric, value)

                state = state.replace(metrics=state.metrics.empty())

                val_state = state

                for val_batch in val_ds:
                    val_batch = jax.device_put(val_batch)
                    val_state = compute_metrics(state=val_state, batch=val_batch)

                for metric, value in val_state.metrics.compute().items():
                    logger.update("val", metric, value)

                logger.end_epoch()

    checkpointer.save(checkpoint_dir, state)
    checkpointer.close()
