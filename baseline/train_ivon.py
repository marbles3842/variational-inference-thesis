import argparse
import os
import jax
import jax.numpy as jnp
import yaml
import optax
from orbax.checkpoint import StandardCheckpointer
from jax import lax, random
from jax.tree_util import tree_map


from core.optimizer import create_cifar_ivon_optimizer
from core.ivon import sample_parameters
from data_loaders.cifar10_dataloader import get_cifar10_train_val_loaders
from models.resnet import ResNet20
from logger.metrics_logger import MetricsLogger
from trainer.train_state import create_train_state
from trainer.metrics import compute_metrics, cross_entropy_loss

NUM_CLASSES = 10
CIFAR10_NUM_FILTERS = 16


@jax.jit
def train_step_ivon(state, batch, rng_key, train_mcsamples=1):
    def loss_and_batch_stats(params, batch_stats):
        variables = {"params": params, "batch_stats": batch_stats}
        logits, mutated = state.apply_fn(
            variables,
            batch["image"],
            train=True,
            mutable=["batch_stats"],
        )
        loss = cross_entropy_loss(logits=logits, labels=batch["label"])
        return loss, mutated["batch_stats"]

    keys = random.split(rng_key, train_mcsamples)

    grad_sum = tree_map(jnp.zeros_like, state.params)

    def mc_sampling_step(carry, key):
        params, opt_state, batch_stats, grad_sum = carry
        sample_params, opt_state = sample_parameters(key, params, opt_state)
        (loss, new_batch_stats), grads = jax.value_and_grad(
            loss_and_batch_stats, has_aux=True
        )(sample_params, batch_stats)
        grad_sum = tree_map(lambda a, b: a + b, grad_sum, grads)
        return (params, opt_state, new_batch_stats, grad_sum), loss

    # init run
    (params, opt_state, batch_stats, grad_sum), _ = mc_sampling_step(
        (state.params, state.opt_state, state.batch_stats, grad_sum), keys[0]
    )

    (params, opt_state, final_batch_stats, grad_sum), _ = lax.scan(
        mc_sampling_step, (params, opt_state, batch_stats, grad_sum), keys[1:]
    )

    updates, new_opt_state = state.tx.update(grad_sum, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return state.replace(
        params=new_params,
        opt_state=new_opt_state,
        batch_stats=final_batch_stats,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, required=True, help="Seed for the initialization"
    )
    parser.add_argument("--job-id", type=int, required=True, help="Job id")
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), "train_cifar10_config.yaml")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        config = config["cifar10"]["ivon"]

    model = ResNet20(num_classes=NUM_CLASSES, num_filters=CIFAR10_NUM_FILTERS)

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
    metrics_log_path = os.path.join(logdir, f"metrics-ivon-{args.seed}.csv")

    # init checkpointer
    checkpointer = StandardCheckpointer()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(
        script_dir, "..", "checkpoints", "ivon", str(args.job_id)
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
