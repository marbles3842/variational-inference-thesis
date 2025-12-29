"""
Inspired by https://huggingface.co/blog/annotated-diffusion
and partly by https://github.com/andylolu2/jax-diffusion
"""

from typing import Tuple, Collection, Sequence
from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn
import chex


class UpSample(nn.Module):
    features: int
    kernel_size: Tuple | int
    strides: Tuple | int = (2, 2)

    @nn.compact
    def __call__(self, x: jax.Array):
        return nn.ConvTranspose(
            features=self.features, kernel_size=self.kernel_size, strides=self.strides
        )(x)


class DownSample(nn.Module):
    features: int
    kernel_size: Tuple | int
    strides: Tuple | int = (2, 2)

    @nn.compact
    def __call__(self, x: jax.Array):
        return nn.Conv(
            features=self.features, kernel_size=self.kernel_size, strides=self.strides
        )(x)


class SinusoidalPositionEmbedding(nn.Module):
    features: int

    @nn.compact
    def __call__(self, time: jax.Array):
        chex.assert_rank(time, 2)
        half_features = self.features // 2
        embeddings = jnp.log(10000) / (half_features - 1)
        embeddings = jnp.exp(jnp.arange(half_features) * -embeddings)
        embeddings = time * embeddings[None, :]
        return jnp.concat([jnp.sin(embeddings), jnp.cos(embeddings)], axis=-1)


class TimeEmbedding(nn.Module):
    features: int
    sin_embedding_features: int

    @nn.compact
    def __call__(self, time: jax.Array):
        return nn.Sequential(
            [
                SinusoidalPositionEmbedding(features=self.sin_embedding_features),
                nn.Dense(features=self.features),
                nn.gelu,
                nn.Dense(features=self.features),
            ]
        )(time)


class Block(nn.Module):
    features: int
    kernel_size: Tuple | int
    groups: int

    @nn.compact
    def __call__(self, x: jax.Array, scale_shift: Tuple = None):
        out = nn.Sequential(
            [
                nn.Conv(features=self.features, kernel_size=self.kernel_size),
                nn.GroupNorm(num_groups=self.groups),
            ]
        )(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            out = out * (scale + 1) + shift

        return nn.silu(out)


class ResnetBlock(nn.Module):
    features: int
    kernel_size: Tuple | int
    groups: int

    @nn.compact
    def __call__(self, x: jax.Array, time_emb: jax.Array = None):
        block = partial(
            Block,
            features=self.features,
            kernel_size=self.kernel_size,
            groups=self.groups,
        )

        mlp = nn.Sequential([nn.silu, nn.DenseGeneral(self.features * 2)])

        scale_shift = None
        if time_emb is not None:
            time_emb = mlp(time_emb)
            time_emb = jnp.expand_dims(time_emb, axis=(1, 2))
            scale, shift = jnp.split(time_emb, 2, axis=-1)

            scale_shift = (scale, shift)

        out = block()(x, scale_shift)
        out = block()(out)

        if x.shape[-1] != self.features:
            x = nn.Conv(self.features, kernel_size=(1, 1))(x)

        return x + out


class Attention(nn.Module):
    features: int
    heads: int
    groups: int

    @nn.compact
    def __call__(self, x: jax.Array):
        b, w, h, d = x.shape
        out = nn.Sequential(
            [
                nn.GroupNorm(self.groups),
                lambda i: jnp.reshape(i, (b, w * h, d)),
                nn.SelfAttention(num_heads=self.heads),
                lambda i: jnp.reshape(i, (b, w, h, self.features)),
            ]
        )(x)
        return x + out


class Unet(nn.Module):
    features: int
    kernel_size: Tuple | int
    feature_mults: Sequence[int]

    attention_resolutions: Collection[int]
    attention_num_heads: int
    num_res_blocks: int

    sinusoidal_embed_features: int
    time_embed_features: int

    num_groups: int

    @nn.compact
    def __call__(self, x: jax.Array, time: jax.Array):
        time = time[:, None]

        res_block = partial(
            ResnetBlock, kernel_size=self.kernel_size, groups=self.num_groups
        )
        attention = partial(
            Attention, heads=self.attention_num_heads, groups=self.num_groups
        )

        t = TimeEmbedding(self.time_embed_features, self.sinusoidal_embed_features)(
            time
        )
        out = nn.Conv(self.features, self.kernel_size)(x)

        h = [out]
        for i, feature_mult in enumerate(self.feature_mults):
            block_features = self.features * feature_mult

            for _ in range(self.num_res_blocks):
                out = res_block(block_features)(out, time_emb=t)
                if out.shape[1] in self.attention_resolutions:
                    out = attention(block_features)(out)

                h.append(out)

            if i < len(self.feature_mults) - 1:
                out = DownSample(block_features, self.kernel_size)(out)
                h.append(out)

        mid_features = self.features * self.feature_mults[-1]
        out = res_block(mid_features)(out, time_emb=t)
        out = attention(mid_features)(out)
        out = res_block(mid_features)(out, time_emb=t)

        for i, feature_mult in enumerate(reversed(self.feature_mults)):
            block_features = self.features * feature_mult

            for _ in range(self.num_res_blocks + 1):
                out = jnp.concatenate((out, h.pop()), axis=-1)
                out = res_block(block_features)(out, time_emb=t)
                if out.shape[1] in self.attention_resolutions:
                    out = attention(block_features)(out)

            if i < len(self.feature_mults) - 1:
                out = UpSample(block_features, self.kernel_size)(out)

        out = res_block(self.features)(out, time_emb=t)
        return nn.Conv(x.shape[-1], kernel_size=(1, 1))(out)


Unet_MNIST = partial(
    Unet,
    features=48,
    feature_mults=(1, 2, 2, 2),
    attention_resolutions=(16,),
    attention_num_heads=4,
    num_res_blocks=2,
    sinusoidal_embed_features=8,
    time_embed_features=32,
    kernel_size=(3, 3),
    num_groups=4,
)
