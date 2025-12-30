import os
import jax
import jax.numpy as jnp
import jax.random as jr

from orbax.checkpoint import StandardCheckpointer
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
from flax import serialization

from data_loaders.mnist import get_mnist_dataset, MNIST_LENGTH, MNIST_IMAGE_SHAPE
from trainer.diffusion import (
    diffusion_train_step_with_ivon,
    diffusion_sample,
    DiffusionModelSchedule,
)
from models.unet import Unet_MNIST
from trainer.diffusion_train_state import create_train_state
from logger import MetricsLogger

from core.ivon import sample_parameters, ivon

from .utils import save_samples, load_diffusion_ivon_config


# TODO: refactor and check if this is needed
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
        static_argnames=["train_mcsamples"],
        in_shardings=(
            state_sharding,
            get_data_sharding(mesh),
            replicated_sharding,
            replicated_sharding,
            replicated_sharding,
        ),
        out_shardings=(state_sharding, replicated_sharding),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="model.msgpack",
        help="file to save model to or load model from (default: %(default)s)",
    )
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

    images = get_mnist_dataset(
        train_batch_size=len(devices) * ivon_config["batch_size"],
        num_epochs=ivon_config["num_epochs"],
        seed=args.init_seed,
    )

    print(
        f"Total batch size across devices: {len(devices) * ivon_config['batch_size']}"
    )

    tx = ivon(
        learning_rate=ivon_config["learning_rate"],
        weight_decay=ivon_config["weight_decay"],
        ess=ivon_config["ess"],
        hess_init=ivon_config["hess_init"],
        clip_radius=ivon_config["clip_radius"],
        rescale_lr=ivon_config["rescale_lr"],
    )

    print(f"Trained with parameters: {ivon_config}")

    model = Unet_MNIST()

    state = create_train_state(
        model,
        jr.key(args.init_seed),
        jnp.zeros((1,) + MNIST_IMAGE_SHAPE),
        jnp.zeros((1,)),
        tx,
    )

    state_sharding_tree = jax.tree_util.tree_map(get_sharding_for_leaf, state)

    state = jax.device_put(state, state_sharding_tree)

    diffusion_step_key = jr.key(args.diffusion_step_seed)

    diffusion = DiffusionModelSchedule.create(
        timesteps=ivon_config["diffusion_timesteps"]
    )

    mgpu_diffusion_train_step = create_mgpu_diffusion_train_step(mesh)

    main_mc_rng = jr.key(args.train_mc_samples_seed)

    num_steps_per_epoch = MNIST_LENGTH // ivon_config["batch_size"]

    with MetricsLogger(args.logs) as logger:

        for step, batch in enumerate(images):
            batch = jax.device_put(batch, get_data_sharding(mesh))

            main_mc_rng, step_mc_rng = jr.split(main_mc_rng)
            state, diffusion_step_key = mgpu_diffusion_train_step(
                state,
                batch,
                diffusion_step_key,
                diffusion,
                step_mc_rng,
                ivon_config["train_mc_samples"],
            )

            if (step + 1) % num_steps_per_epoch == 0:

                for metric, value in state.metrics.compute().items():
                    logger.update("train", metric, value)

                state = state.replace(metrics=state.metrics.empty())

                logger.end_epoch()

    # init checkpointer
    bytes_output = serialization.to_bytes(state.params)

    with open(args.model, "wb") as f:
        f.write(bytes_output)

    checkpointer = StandardCheckpointer()
    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints", "diffusion"
    )
    checkpointer.save(checkpoint_dir, state)
    checkpointer.close()

    posterior_params, new_state = sample_parameters(
        rng=jr.key(args.samples_seed), params=state.params, states=state.opt_state
    )

    def generate_and_save_samples(
        key: jr.PRNGKey, params, num_samples: int, filename: str
    ):
        samples = diffusion_sample(
            model=model,
            variables={"params": params},
            shape=(num_samples,) + MNIST_IMAGE_SHAPE,
            key=key,
            schedule=diffusion,
        )
        samples = jnp.clip(samples, -1, 1)
        samples = (samples + 1) / 2
        save_samples(samples=samples, filename=filename)

    generate_and_save_samples(
        key=jr.key(args.samples_seed),
        num_samples=args.num_samples,
        filename="ivon-mean-samples.png",
        params=state.params,
    )

    generate_and_save_samples(
        key=jr.key(args.samples_seed),
        num_samples=args.num_samples,
        filename="ivon-posterior-samples.png",
        params=posterior_params,
    )

    print("Finished!")
