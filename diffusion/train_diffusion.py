import jax
import jax.numpy as jnp
import jax.random as jr

from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

from data_loaders.mnist import get_mnist_dataset, MNISTInfo
from trainer.diffusion import (
    diffusion_train_step_with_ivon,
    DiffusionModelSchedule,
)
from models.unet import Unet_MNIST
from trainer.diffusion_train_state import create_train_state
from logger import MetricsLogger

from core.ivon import ivon
from core.serialization import save_params

from .utils import load_diffusion_ivon_config


def get_sharding_for_leaf(leaf):
    if jnp.ndim(leaf) == 0:
        return NamedSharding(mesh, P())
    else:
        return NamedSharding(mesh, P(None))


def get_data_sharding(mesh):
    return NamedSharding(mesh, P("data", None))


def create_mgpu_diffusion_train_step(mesh):
    replicated_sharding = NamedSharding(mesh, P())

    state_sharding = NamedSharding(mesh, P())

    return jax.jit(
        diffusion_train_step_with_ivon,
        in_shardings=(
            state_sharding,
            get_data_sharding(mesh),
            replicated_sharding,
            replicated_sharding,
            replicated_sharding,
        ),
        out_shardings=(state_sharding, replicated_sharding),
        donate_argnums=(0,),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--init-seed",
        type=int,
        default=0,
        help="seed for initializing the model",
    )

    parser.add_argument(
        "--train-mc-samples-seed",
        type=int,
        default=1,
        help="seed for MC samples during the training",
    )

    parser.add_argument(
        "--samples-seed",
        type=int,
        default=42,
        help="seed for samples generating",
    )

    parser.add_argument(
        "--diffusion-step-seed",
        type=int,
        default=68,
        help="seed for samples generating",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=16,
        help="Number of samples to generate",
    )

    parser.add_argument(
        "--logs",
        type=str,
        default="logs.csv",
        metavar="V",
        help="Log file for training",
    )

    args = parser.parse_args()

    ivon_config = load_diffusion_ivon_config()

    devices = jax.devices()

    device_mesh = mesh_utils.create_device_mesh((len(devices),))
    mesh = Mesh(device_mesh, axis_names=("data",))

    total_batch_size = jax.device_count() * ivon_config["batch_size"]

    images = get_mnist_dataset(
        train_batch_size=total_batch_size,
        num_epochs=ivon_config["num_epochs"],
        seed=args.init_seed,
    )

    print(f"Total batch size across devices: {total_batch_size}")

    tx = ivon(
        learning_rate=ivon_config["learning_rate"],
        weight_decay=ivon_config["weight_decay"],
        ess=ivon_config["ess"],
        hess_init=ivon_config["hess_init"],
        clip_radius=ivon_config["clip_radius"],
        rescale_lr=ivon_config["rescale_lr"],
    )

    print(f"Trained with parameters: {ivon_config}")

    model = Unet_MNIST(dtype=jnp.bfloat16)

    state = create_train_state(
        model,
        jr.key(args.init_seed),
        jnp.zeros((1, *MNISTInfo.shape)),
        jnp.zeros((1,)),
        tx,
    )

    state_sharding_tree = jax.tree_util.tree_map(get_sharding_for_leaf, state)

    state = jax.device_put(state, state_sharding_tree)

    diffusion_step_key = jr.key(args.diffusion_step_seed)

    diffusion = DiffusionModelSchedule.create(
        timesteps=ivon_config["diffusion_timesteps"], dtype=jnp.bfloat16
    )

    mgpu_diffusion_train_step = create_mgpu_diffusion_train_step(mesh)

    main_mc_rng = jr.key(args.train_mc_samples_seed)

    num_steps_per_epoch = MNISTInfo.train_length // total_batch_size

    with MetricsLogger(args.logs) as logger:

        for step, batch in enumerate(images):
            batch = batch.astype(jnp.bfloat16)
            batch = jax.device_put(batch, get_data_sharding(mesh))

            main_mc_rng, step_mc_rng = jr.split(main_mc_rng)
            state, diffusion_step_key = mgpu_diffusion_train_step(
                state,
                batch,
                diffusion_step_key,
                diffusion,
                jr.split(step_mc_rng, num=ivon_config["train_mc_samples"]),
            )

            if (step + 1) % num_steps_per_epoch == 0:

                for metric, value in state.metrics.compute().items():
                    logger.update("train", metric, value)

                state = state.replace(metrics=state.metrics.empty())

                logger.end_epoch()

    save_params(f"ddpm-ivon-{args.init_seed}.msgpack", state.params)
    save_params(f"ddpm-ivon-state-{args.init_seed}.msgpack", state.opt_state[0])
