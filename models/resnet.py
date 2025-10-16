"""
ResNet implementation in Flax.

By default uses Filter Response Norm

Inspired by the official Flax ImageNet example:
https://github.com/google/flax/blob/main/examples/imagenet/models.py

"""

from typing import Any
from collections.abc import Callable, Sequence
from functools import partial

import flax.linen as nn
import jax.numpy as jnp

from .filter_response_norm import FilterResponseNorm


ModuleDef = Any


def identity(x):
    return x


class BasicBlock(nn.Module):
    filters: int
    norm: ModuleDef
    activation: Callable
    use_bias: bool
    strides: tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x

        out = nn.Conv(
            features=self.filters,
            kernel_size=(3, 3),
            strides=self.strides,
            use_bias=self.use_bias,
            padding="SAME",
        )(x)
        out = self.norm()(out)

        out = self.activation(out)

        out = nn.Conv(
            features=self.filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            use_bias=self.use_bias,
            padding="SAME",
        )(out)
        out = self.norm()(out)

        if residual.shape != out.shape:
            residual = nn.Conv(
                features=self.filters,
                kernel_size=(1, 1),
                strides=self.strides,
                use_bias=self.use_bias,
                padding="SAME",
                name="conv_proj",
            )(residual)
            residual = self.norm(name="norm_proj")(residual)

        return self.activation(out + residual)


class ResNet(nn.Module):
    stage_sizes: Sequence[int]
    block: ModuleDef
    num_classes: int
    num_filters: int
    activation: Callable
    use_bias: bool
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        norm = partial(FilterResponseNorm)

        x = nn.Conv(
            features=self.num_filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=self.use_bias,
        )(x)

        x = norm()(x)

        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if (i > 0 and j == 0) else (1, 1)
                x = self.block(
                    self.num_filters * (2**i),
                    strides=strides,
                    norm=norm,
                    use_bias=self.use_bias,
                    activation=self.activation,
                )(x)

        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        x = jnp.asarray(x, self.dtype)
        return x


ResNet20 = partial(
    ResNet,
    stage_sizes=(3, 3, 3),
    block=BasicBlock,
    num_filters=16,
    use_bias=True,
    activation=identity,
)

ResNet18 = partial(
    ResNet,
    stage_sizes=(2, 2, 2, 2),
    block=BasicBlock,
    num_filters=64,
    use_bias=True,
    activation=identity,
)
