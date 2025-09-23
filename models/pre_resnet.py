from typing import Any
from collections.abc import Callable, Sequence
from functools import partial

import flax.linen as nn
import jax.numpy as jnp


ModuleDef = Any


class BasicBlockPreResNet(nn.Module):
    filters: int
    norm: ModuleDef
    activation: Callable
    strides: tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x

        out = self.norm()(x)
        out = self.activation(out)
        out = nn.Conv(
            features=self.filters,
            kernel_size=(3, 3),
            strides=self.strides,
            use_bias=False,
            padding="SAME",
        )(out)

        out = self.norm()(out)
        out = self.activation(out)
        out = nn.Conv(
            features=self.filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            use_bias=False,
            padding="SAME",
        )(out)

        if residual.shape != out.shape:
            residual = nn.Conv(
                features=self.filters,
                kernel_size=(1, 1),
                strides=self.strides,
                use_bias=False,
                padding="SAME",
                name="conv_proj",
            )(residual)

        return out + residual


class PreResNet(nn.Module):
    stage_sizes: Sequence[int]
    block: ModuleDef
    num_classes: int
    num_filters: int
    activation: Callable = nn.relu
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool = True):

        norm = partial(
            nn.BatchNorm, use_running_average=not train, momentum=0.9, epsilon=1e-5
        )

        x = nn.Conv(
            features=self.num_filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )(x)
        x = norm()(x)
        x = self.activation(x)

        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if (i > 0 and j == 0) else (1, 1)
                x = self.block(
                    self.num_filters * (2**i),
                    strides=strides,
                    norm=norm,
                    activation=self.activation,
                )(x)

        x = norm()(x)
        x = self.activation(x)

        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        x = jnp.asarray(x, self.dtype)
        return x


PreResNet110 = partial(
    PreResNet, stage_sizes=(18, 18, 18), block=BasicBlockPreResNet, num_filters=24
)
