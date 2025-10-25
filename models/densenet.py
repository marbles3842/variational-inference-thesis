from typing import Any
from collections.abc import Sequence
from functools import partial, reduce
import flax.linen as nn
import jax.numpy as jnp

from .filter_response_norm import FilterResponseNorm

ModuleDef = Any


class BasicBlock(nn.Module):
    growth_rate: int
    norm: ModuleDef
    use_bias: bool = False

    @nn.compact
    def __call__(self, x):
        out = self.norm()(x)
        out = nn.Conv(
            features=4 * self.growth_rate,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            use_bias=self.use_bias,
        )(out)

        out = self.norm()(out)
        out = nn.Conv(
            features=self.growth_rate,
            kernel_size=(3, 3),
            padding=1,
            use_bias=self.use_bias,
        )(out)
        return jnp.concatenate([out, x], axis=-1)


class Transition(nn.Module):
    features: int
    norm: ModuleDef
    use_bias: bool = False

    @nn.compact
    def __call__(self, x):
        out = self.norm()(x)
        out = nn.Conv(
            features=self.features, kernel_size=(1, 1), use_bias=self.use_bias
        )(out)
        return nn.avg_pool(out, window_shape=(2, 2), strides=(2, 2))


class DenseNet(nn.Module):
    block: ModuleDef
    n_blocks: Sequence
    num_classes: int
    growth_rate: int = 12
    reduction: float = 0.5
    use_bias: bool = False

    @nn.compact
    def __call__(self, x):
        norm = partial(FilterResponseNorm)
        num_planes = 2 * self.growth_rate

        out = nn.Conv(
            features=num_planes,
            kernel_size=(3, 3),
            padding=1,
            use_bias=self.use_bias,
        )(x)

        for i, n_block in enumerate(self.n_blocks):
            out = self._dense_layer(BasicBlock, norm, n_block, out)
            num_planes += n_block * self.growth_rate
            if i < len(self.n_blocks) - 1:
                out_planes = int(num_planes * self.reduction)
                out = Transition(out_planes, norm)(out)
                num_planes = out_planes

        out = norm()(out)
        pool_size = out.shape[1]
        out = nn.avg_pool(
            nn.relu(out),
            window_shape=(pool_size, pool_size),
            strides=(pool_size, pool_size),
        )
        out = out.reshape((out.shape[0], -1))
        out = nn.Dense(self.num_classes)(out)
        return out

    def _dense_layer(self, block, norm, n_blocks, x):
        return reduce(
            lambda out, _: block(self.growth_rate, norm)(out), range(n_blocks), x
        )


DenseNet121 = partial(
    DenseNet, block=BasicBlock, n_blocks=(6, 12, 24, 16), growth_rate=12
)
