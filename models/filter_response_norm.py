from jax import lax
import jax.numpy as jnp
import flax.linen as nn


class FilterResponseNorm(nn.Module):
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        par_shape = (1, 1, 1, x.shape[-1])
        tau = self.param("tau", nn.initializers.zeros, par_shape)
        beta = self.param("beta", nn.initializers.zeros, par_shape)
        gamma = self.param("gamma", nn.initializers.ones, par_shape)

        nu2 = jnp.mean(jnp.square(x), axis=[1, 2], keepdims=True)

        x = x * lax.rsqrt(nu2 + self.eps)
        y = gamma * x + beta
        return jnp.maximum(y, tau)
