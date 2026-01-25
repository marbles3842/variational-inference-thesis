# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax.random as jr
from flax import linen as nn
import jax.numpy as jnp
from data_loaders import mnist

from core.serialization import load_model_params


class MNISTSemanticLikelihood(nn.Module):
    """A simple CNN model that can be used as MNIST semantic likelihood
    Inspired by https://github.com/google/flax/blob/main/examples/mnist/train.py
    """

    @nn.compact
    def __call__(self, x, return_embedding: bool = False):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=256)(x)
        embeddings = nn.relu(x)
        return embeddings if return_embedding else nn.Dense(features=10)(embeddings)


def init_semantic_likelihood(
    rng: jr.PRNGKey, filename: str = "trained/semantic-likelihood.msgpack"
):
    likelihood = MNISTSemanticLikelihood()
    variables = likelihood.init(rng, jnp.zeros((1,) + mnist.MNISTInfo.shape))

    return likelihood, load_model_params(filename, variables["params"])
