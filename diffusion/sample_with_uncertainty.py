import jax
from typing import Tuple
from jax.lax import map
import jax.random as jr
import jax.numpy as jnp

from trainer.diffusion import DiffusionModelSchedule, diffusion_sample

from models.unet import Unet_MNIST
from data_loaders.mnist import MNISTInfo

from core.ivon import sample_parameters
from core.serialization import load_ivon_state, load_model_params

from .semantic_likelihood import init_semantic_likelihood
from .utils import load_diffusion_ivon_config


if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--init-seed",
        type=int,
        default=0,
        help="seed for initializing the model",
    )

    parser.add_argument(
        "--samples-seed",
        type=int,
        default=42,
        help="seed for samples generating",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        default=700,
        help="seed for samples generating",
    )

    parser.add_argument(
        "--mc-samples",
        type=int,
        default=16,
        help="Number of MC samples for uncertainty estimation",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="A .msgpack file with the model for Unet",
    )

    parser.add_argument(
        "--opt-state",
        type=str,
        required=True,
        help="A .msgpack file with the opt state",
    )

    parser.add_argument(
        "--samples-output-dir",
        required=True,
        type=str,
        help="A dir for the samples output",
    )

    args = parser.parse_args()

    config = load_diffusion_ivon_config()

    semantic_likelihood, sem_params = init_semantic_likelihood(
        rng=jr.key(args.init_seed)
    )

    unet = Unet_MNIST()

    unet_vars = unet.init(
        jr.key(args.init_seed), jnp.zeros((1, *MNISTInfo.shape)), jnp.zeros((1,))
    )

    mean_params = load_model_params(args.model, unet_vars["params"])
    ivon_state = load_ivon_state(args.opt_state, unet_vars["params"])

    diffusion = DiffusionModelSchedule.create(timesteps=config["diffusion_timesteps"])

    def gaussian_entropy(e: jax.Array, sigma_squared: float):
        """
        Calculate the entropy of multivariate Gaussian distributions with covariance
        Eq. 8 from https://arxiv.org/pdf/2502.20946
        """
        D = e.shape[2]

        diag = jnp.mean(e**2, axis=0) - jnp.mean(e, axis=0) ** 2
        diag = jnp.clip(diag, 0.0, None)

        log_det = jnp.sum(jnp.log(diag + sigma_squared), axis=1)

        entropy = 0.5 * log_det + 0.5 * D * (jnp.log(2 * jnp.pi) + 1)

        return entropy

    def sample_with_uncertainty(
        key: jr.PRNGKey, shape: Tuple, mc_samples: int, sem_variance: float
    ):
        """
        Algorithm 1 from Generative Uncertainty in Diffusion Models:
        https://arxiv.org/pdf/2502.20946

        """
        key_diffusion, key_mc = jr.split(key)

        def diffuse_and_embed(params):
            x = diffusion_sample(
                model=unet,
                variables={"params": params},
                shape=shape,
                key=key_diffusion,
                schedule=diffusion,
            )
            e = semantic_likelihood.apply({"params": sem_params}, x, True)
            return x, e

        x0, e0 = diffuse_and_embed(mean_params)

        def mc_sample(key):
            psample, _ = sample_parameters(key, mean_params, (ivon_state,))
            xm, em = diffuse_and_embed(psample)
            return xm, em

        xs, e = map(mc_sample, jr.split(key_mc, num=mc_samples))
        e = jnp.concatenate([e0[None, ...], e])

        return x0, xs, gaussian_entropy(e=e, sigma_squared=sem_variance**2)

    samples, xs, entropy = sample_with_uncertainty(
        key=jr.key(args.samples_seed),
        shape=(args.batch_size, *MNISTInfo.shape),
        mc_samples=args.mc_samples,
        sem_variance=config["semantic_likelihood_noise"],
    )

    samples_output_dir = os.path.join(
        args.samples_output_dir, f"samples-seed-{args.samples_seed}.npy"
    )

    samples_xs_output_dir = os.path.join(
        args.samples_output_dir, f"samples-xs-seed-{args.samples_seed}.npy"
    )
    entropy_output_dir = os.path.join(
        args.samples_output_dir, f"entropy-seed-{args.samples_seed}.npy"
    )

    jnp.save(samples_output_dir, samples)

    jnp.save(entropy_output_dir, entropy)

    jnp.save(samples_xs_output_dir, xs)
