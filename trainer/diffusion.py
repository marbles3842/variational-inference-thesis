from typing import Tuple
from functools import partial

import optax
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.lax import scan, cond
import flax.linen as nn
from flax.struct import dataclass, field
from flax.training.train_state import TrainState
from flax.typing import FrozenVariableDict

from core.ivon import sample_parameters, accumulate_gradients


def _linear_beta_schedule(
    timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2
):
    return jnp.linspace(beta_start, beta_end, timesteps)


@dataclass
class DiffusionModelSchedule:
    beta: jax.Array
    alpha: jax.Array
    alpha_cumprod: jax.Array
    timesteps: int = field(pytree_node=False)

    @classmethod
    def create(cls, timesteps: int):
        beta = _linear_beta_schedule(timesteps=timesteps)
        alpha = 1 - beta
        alpha_cumprod = jnp.cumprod(alpha)
        return cls(
            beta=beta,
            alpha=alpha,
            alpha_cumprod=alpha_cumprod,
            timesteps=timesteps,
        )


@jax.jit
def diffusion_train_step(
    state: TrainState,
    batch: jax.Array,
    key: jr.PRNGKey,
    schedule: DiffusionModelSchedule,
):
    key_t, key_esp, key = jr.split(key, num=3)

    def negative_elbo(params):
        batch_size = batch.shape[0]
        t = jr.randint(
            key=key_t, minval=0, maxval=schedule.timesteps, shape=(batch_size,)
        )
        epsilon = jr.normal(key_esp, shape=batch.shape)
        alpha_bar_t = schedule.alpha_cumprod[t].reshape(batch_size, 1, 1, 1)
        x_t = jnp.sqrt(alpha_bar_t) * batch + jnp.sqrt(1 - alpha_bar_t) * epsilon
        variables = {"params": params}
        epsilon_theta = state.apply_fn(variables, x_t, t)
        loss = (epsilon_theta - epsilon) ** 2

        weight = (1 - schedule.alpha_cumprod[t]).reshape(batch_size, 1, 1, 1)
        weighted_loss = weight * loss
        neg_elbo = weighted_loss.mean()
        return neg_elbo

    loss, grads = jax.value_and_grad(negative_elbo)(state.params)
    new_state = state.apply_gradients(grads=grads)
    metric_update = new_state.metrics.single_from_model_output(loss=loss)
    metrics = new_state.metrics.merge(metric_update)
    return new_state.replace(metrics=metrics), key


def negative_elbo(
    batch: jax.Array,
    params: FrozenVariableDict,
    state: TrainState,
    schedule: DiffusionModelSchedule,
    key_t: jr.PRNGKey,
    key_eps: jr.PRNGKey,
):
    batch_size = batch.shape[0]
    t = jr.randint(key=key_t, minval=0, maxval=schedule.timesteps, shape=(batch_size,))
    epsilon = jr.normal(key_eps, shape=batch.shape)
    alpha_bar_t = schedule.alpha_cumprod[t].reshape(batch_size, 1, 1, 1)
    x_t = jnp.sqrt(alpha_bar_t) * batch + jnp.sqrt(1 - alpha_bar_t) * epsilon
    variables = {"params": params}
    epsilon_theta = state.apply_fn(variables, x_t, t)
    loss = (epsilon_theta - epsilon) ** 2

    weight = (1 - schedule.alpha_cumprod[t]).reshape(batch_size, 1, 1, 1)
    weighted_loss = weight * loss
    neg_elbo = weighted_loss.mean()
    return neg_elbo


@jax.jit
def _compute_hessian_metrics(hessian):
    all_hess_values = jnp.concatenate([jnp.ravel(h) for h in jax.tree.leaves(hessian)])
    return jnp.mean(all_hess_values), jnp.min(all_hess_values)


@partial(jax.jit, static_argnames=["train_mcsamples"])
def diffusion_train_step_with_ivon(
    state: TrainState,
    batch: jax.Array,
    train_key: jr.PRNGKey,
    schedule: DiffusionModelSchedule,
    mc_key: jr.PRNGKey,
    train_mcsamples: int,
):
    def loss(params, key_t, key_eps):
        return negative_elbo(
            batch=batch,
            params=params,
            state=state,
            schedule=schedule,
            key_t=key_t,
            key_eps=key_eps,
        )

    mc_key, *mc_keys = jax.random.split(mc_key, train_mcsamples + 1)
    opt_state = state.opt_state
    params = state.params

    for i, key in enumerate(mc_keys):
        key_t, key_eps, key = jr.split(key, num=3)
        psample, opt_state = sample_parameters(key, params, opt_state)

        updates = jax.grad(loss)(psample, key_t, key_eps)

        if i == train_mcsamples - 1:
            updates, opt_state = state.tx.update(updates, opt_state, params)
        else:
            opt_state = accumulate_gradients(updates, opt_state)

    new_params = optax.apply_updates(params, updates)
    new_state = state.replace(params=new_params, opt_state=opt_state)
    key_t, key_eps, train_key = jr.split(train_key, num=3)
    loss = negative_elbo(
        batch=batch,
        params=new_params,
        state=new_state,
        schedule=schedule,
        key_t=key_t,
        key_eps=key_eps,
    )
    hess_mean, hess_min = _compute_hessian_metrics(opt_state[0].hess)
    metric_update = new_state.metrics.single_from_model_output(
        loss=loss, hess_mean=hess_mean, hess_min=hess_min
    )
    metrics = new_state.metrics.merge(metric_update)
    return new_state.replace(metrics=metrics), train_key


def diffusion_sample(
    model: nn.Module,
    variables: FrozenVariableDict,
    shape: Tuple,
    key: jr.PRNGKey,
    schedule: DiffusionModelSchedule,
):
    batch_size = shape[0]

    key, subkey = jr.split(key)
    x_t = jr.normal(key=subkey, shape=shape)

    def reverse_step(carry, t):
        x_t, key = carry
        t_arr = jnp.full((batch_size,), t)
        key, subkey = jr.split(key)
        z = cond(
            t > 0,
            lambda _: jr.normal(key=subkey, shape=shape),
            lambda _: jnp.zeros(shape),
            None,
        )

        predicted_noise = model.apply(variables, x_t, t_arr)
        alpha_t = schedule.alpha[t]
        alpha_t_bar = schedule.alpha_cumprod[t]
        beta_t = schedule.beta[t]

        x_t = (
            1
            / jnp.sqrt(alpha_t)
            * (x_t - predicted_noise * (1 - alpha_t) / jnp.sqrt(1 - alpha_t_bar))
        )

        sigma_t = jnp.sqrt(beta_t)
        x_t = x_t + sigma_t * z
        return (x_t, key), None

    (x_t, key), _ = scan(
        reverse_step, (x_t, key), jnp.arange(schedule.timesteps - 1, -1, -1)
    )

    return x_t
